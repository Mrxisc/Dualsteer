import io
import json
import math
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from diffusers.utils.import_utils import is_xformers_available

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast_object_list
from datasets import Array2D, Dataset, Features, Value
from datasets.fingerprint import generate_fingerprint
from huggingface_hub import HfApi
from tqdm import tqdm
from Dualsteer_Code/Text2Image.Scripts.train.config import CacheActivationsRunnerConfig


try:
    import pandas as pd
except ImportError:
    pd = None


torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


TORCH_STRING_DTYPE_MAP = {torch.float16: "float16", torch.float32: "float32"}


class CacheActivationsRunner:
    def __init__(self, cfg: CacheActivationsRunnerConfig):
        self.cfg = cfg
        self.accelerator = Accelerator()
        if self.cfg.hook_names is not None:
            from Dualsteer_Code/Text2Image.Scripts.collect.hooked_sd_noised_pipeline import (
                HookedStableDiffusionPipeline,
            )
            pipe_kwargs = {}
            if getattr(self.cfg, "denoiser_attr", None):
                pipe_kwargs["denoiser_attr"] = self.cfg.denoiser_attr
            if self.cfg.use_dit:
                pipe_kwargs["use_dit"] = True
                if self.cfg.dit_config:
                    pipe_kwargs["dit_config"] = self.cfg.dit_config
            if self.cfg.use_flux:
                pipe_kwargs["use_flux"] = True
                if self.cfg.flux_config:
                    pipe_kwargs["flux_config"] = self.cfg.flux_config
            if self.cfg.use_sd35_backbone:
                pipe_kwargs["use_sd35_backbone"] = True
                if self.cfg.sd35_config:
                    pipe_kwargs["sd35_config"] = self.cfg.sd35_config
            self.pipe = HookedStableDiffusionPipeline.from_pretrained(
                self.cfg.model_name,
                torch_dtype=self.cfg.dtype,
                safety_checker=None,
                low_cpu_mem_usage=True,
                **pipe_kwargs,
            )
            if self.cfg.use_flux and hasattr(self.pipe, "transformer"):
                transformer = self.pipe.transformer
                self.pipe.register_modules(transformer=transformer.cpu())
                if hasattr(self.pipe, "flux_model"):
                    self.pipe.flux_model.to(self.accelerator.device, dtype=self.cfg.dtype)
            if self.cfg.use_sd35_backbone and hasattr(self.pipe, "transformer"):
                self.pipe.transformer.to(self.accelerator.device, dtype=self.cfg.dtype)
            if is_xformers_available():
                print("Enabling xFormers memory efficient attention")


                if hasattr(self.pipe, 'unet') and hasattr(self.pipe.unet, 'enable_xformers_memory_efficient_attention'):
                    self.pipe.unet.enable_xformers_memory_efficient_attention()
                elif hasattr(self.pipe, 'dit_model') and hasattr(self.pipe.dit_model, 'enable_xformers_memory_efficient_attention'):
                    self.pipe.dit_model.enable_xformers_memory_efficient_attention()
                elif hasattr(self.pipe, 'flux_model') and hasattr(self.pipe.flux_model, 'enable_xformers_memory_efficient_attention'):
                    self.pipe.flux_model.enable_xformers_memory_efficient_attention()
                elif (
                    self.cfg.use_sd35_backbone
                    and hasattr(self.pipe, 'transformer')
                    and hasattr(self.pipe.transformer, 'enable_xformers_memory_efficient_attention')
                ):
                    self.pipe.transformer.enable_xformers_memory_efficient_attention()
            self.pipe.to(self.accelerator.device)
            self.pipe.vae.to("cpu")
            self.pipe.set_progress_bar_config(disable=True)
            self.scheduler = self.pipe.scheduler
            self.scheduler.set_timesteps(self.cfg.num_inference_steps, device="cpu")
            self.scheduler_timesteps = self.scheduler.timesteps
            self.features_dict = {hookpoint: None for hookpoint in self.cfg.hook_names}
            self.token_indices: torch.Tensor | None = None
            self.channel_proj_matrix: torch.Tensor | None = None
            if not self.cfg.csv_path:
                raise ValueError(
                    "csv_path is required. UnlearnCanvas fallback prompts have been removed; "
                    "please provide --csv_path explicitly."
                )
            prompts = self._load_prompts_from_csv()
            self.dataset = Dataset.from_dict({"caption": prompts})
            self.dataset = self.dataset.shuffle(self.cfg.seed)
            if limit := self.cfg.max_num_examples:
                self.dataset = self.dataset.select(range(limit))
            self.num_examples = len(self.dataset)
            self.dataloader = self.get_batches(self.dataset, self.cfg.batch_size)
            self.n_buffers = len(self.dataloader)


    @staticmethod
    def get_batches(items, batch_size):
        num_batches = (len(items) + batch_size - 1) // batch_size
        batches = []
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(items))
            batch = items[start_index:end_index]
            batches.append(batch)
        return batches


    def _load_prompts_from_csv(self) -> list[str]:
        if pd is None:
            raise ImportError(
                "pandas is required to load prompts from CSV. Please install pandas or remove csv_path."
            )
        cfg = self.cfg
        df = pd.read_csv(cfg.csv_path)
        if cfg.csv_category_column and cfg.csv_filter_categories:
            if cfg.csv_category_match_all:
                mask = pd.Series(True, index=df.index)
                for category in cfg.csv_filter_categories:
                    mask &= df[cfg.csv_category_column].fillna("").str.contains(
                        category, case=False, na=False
                    )
            else:
                mask = pd.Series(False, index=df.index)
                for category in cfg.csv_filter_categories:
                    mask |= df[cfg.csv_category_column].fillna("").str.contains(
                        category, case=False, na=False
                    )
            df = df[mask]
        if cfg.csv_require_hard and cfg.csv_hard_column and cfg.csv_hard_column in df.columns:
            df = df[df[cfg.csv_hard_column].fillna(0).astype(int) > 0]
        if cfg.csv_max_rows is not None:
            df = df.head(cfg.csv_max_rows)


        if cfg.csv_prompt_column not in df.columns:
            raise ValueError(
                f"Prompt column '{cfg.csv_prompt_column}' not found in CSV {cfg.csv_path}."
            )


        prompts = df[cfg.csv_prompt_column].fillna("").astype(str).tolist()
        processed = []
        seen = set()
        for prompt in prompts:
            prompt = prompt.strip()
            if cfg.csv_strip_period and prompt.endswith("."):
                prompt = prompt[:-1]
            if cfg.csv_deduplicate:
                if prompt in seen or not prompt:
                    continue
                seen.add(prompt)
            if prompt:
                processed.append(prompt)
        if not processed:
            raise ValueError(
                f"No prompts loaded from CSV {cfg.csv_path} after filtering."
            )
        if self.accelerator.is_main_process:
            print(f"Loaded {len(processed)} prompts from CSV: {cfg.csv_path}")
        return processed


    @staticmethod
    def _consolidate_shards(
        source_dir: Path, output_dir: Path, copy_files: bool = True
    ) -> Dataset:
 
        first_shard_dir_name = "shard_00000"
        assert source_dir.exists() and source_dir.is_dir()
        assert (
            output_dir.exists()
            and output_dir.is_dir()
            and not any(p for p in output_dir.iterdir() if not p.name == ".tmp_shards")
        )
        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")
        transfer_fn = shutil.copy2 if copy_files else shutil.move
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            output_dir / "dataset_info.json",
        )


        arrow_files = []
        file_count = 0


        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue
            state = json.loads((shard_dir / "state.json").read_text())
            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(list(source_dir.iterdir())):05d}.arrow"
                dst = output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }
        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()


        if not copy_files:
            shutil.rmtree(source_dir)


        return Dataset.load_from_disk(output_dir)


    @staticmethod
    def _dataset_is_finalized(path: Path) -> bool:
        has_metadata = (path / "dataset_info.json").exists() and (path / "state.json").exists()
        has_arrow_files = any(path.glob("data-*.arrow"))
        return has_metadata and has_arrow_files


    @staticmethod
    def _existing_shard_indices(tmp_dir: Path) -> list[int]:

        if not tmp_dir.exists():
            return []

        shard_indices: list[int] = []
        for shard_dir in tmp_dir.iterdir():
            if shard_dir.is_dir() and shard_dir.name.startswith("shard_"):
                try:
                    shard_indices.append(int(shard_dir.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue

        return sorted(shard_indices)


    def _ensure_token_indices_path(self) -> Path:
        if self.cfg.token_indices_path is None:
            default_path = (
                Path(self.cfg.new_cached_activations_path)
                / "token_indices.pt"
            )
            self.cfg.token_indices_path = str(default_path)
        target_path = Path(self.cfg.token_indices_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        return target_path


    def _ensure_channel_proj_path(self) -> Path:
        if self.cfg.channel_proj_path is None:
            default_path = (
                Path(self.cfg.new_cached_activations_path)
                / "channel_proj.pt"
            )
            self.cfg.channel_proj_path = str(default_path)
        target_path = Path(self.cfg.channel_proj_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        return target_path


    def _prepare_token_indices(self, num_tokens: int) -> torch.Tensor | None:
        cfg = self.cfg
        if cfg.token_subsample is None or cfg.token_subsample >= num_tokens:
            return None
        if self.token_indices is None:
            if cfg.token_indices_path is not None and Path(cfg.token_indices_path).exists():
                self.token_indices = torch.load(cfg.token_indices_path).to(torch.long)
                return self.token_indices
            gen = torch.Generator().manual_seed(cfg.token_subsample_seed)
            indices = torch.randperm(num_tokens, generator=gen)[: cfg.token_subsample]
            indices, _ = torch.sort(indices)
            self.token_indices = indices.to(torch.long)
            if self.accelerator.is_main_process:
                torch.save(self.token_indices.cpu(), self._ensure_token_indices_path())
        return self.token_indices


    def _prepare_channel_projection(self, in_dim: int, dtype: torch.dtype) -> torch.Tensor | None:
        cfg = self.cfg
        target_dim = cfg.channel_proj_dim
        if target_dim is None or target_dim >= in_dim:
            return None
        if self.channel_proj_matrix is None:
            if cfg.channel_proj_path is not None and Path(cfg.channel_proj_path).exists():
                self.channel_proj_matrix = torch.load(cfg.channel_proj_path).to(torch.float32)
                return self.channel_proj_matrix.to(dtype=dtype)
            gen = torch.Generator().manual_seed(cfg.channel_proj_seed)
            matrix = torch.randn(
                (in_dim, target_dim),
                generator=gen,
                dtype=torch.float32,
            )
            matrix = torch.nn.functional.normalize(matrix, dim=0)
            self.channel_proj_matrix = matrix
            if self.accelerator.is_main_process:
                torch.save(
                    self.channel_proj_matrix.to(dtype=torch.float16).cpu(),
                    self._ensure_channel_proj_path(),
                )
        return self.channel_proj_matrix.to(dtype=dtype)


    def _apply_transforms(self, acts: torch.Tensor) -> torch.Tensor:
        if acts.ndim != 4:
            return acts


        token_indices = self._prepare_token_indices(acts.shape[-2])
        if token_indices is not None:
            token_indices = token_indices.to(acts.device)
            acts = torch.index_select(acts, -2, token_indices)


        proj = self._prepare_channel_projection(acts.shape[-1], acts.dtype)
        if proj is not None:
            proj = proj.to(device=acts.device, dtype=acts.dtype)
            with torch.autocast(device_type=acts.device.type, dtype=acts.dtype):
                acts = torch.matmul(acts, proj)

        return acts


    def _write_transform_metadata(self, hook_path: Path) -> None:
        metadata = {
            "cache_every_n_timesteps": self.cfg.cache_every_n_timesteps,
            "token_subsample": self.cfg.token_subsample,
            "token_indices_path": self.cfg.token_indices_path,
            "channel_proj_dim": self.cfg.channel_proj_dim,
            "channel_proj_path": self.cfg.channel_proj_path,
        }
        with open(hook_path / "transforms.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


    @torch.no_grad()
    def _create_shard(
        self,
        buffer: torch.Tensor,
        hook_name: str,
    ) -> Dataset:
        batch_size, n_steps, d_sample_size, d_in = buffer.shape


        buffer = buffer[:, :: self.cfg.cache_every_n_timesteps, :, :]
        _, n_steps, _, _ = buffer.shape


        activations = buffer.reshape(-1, d_sample_size, d_in)
        scheduler_subset = self.scheduler_timesteps[:: self.cfg.cache_every_n_timesteps]
        if n_steps != len(scheduler_subset):
            if len(scheduler_subset) == 0:
                raise ValueError("Scheduler timesteps are empty; cannot label activations.")
            repeats = math.ceil(n_steps / len(scheduler_subset))
            scheduler_subset = scheduler_subset.repeat(repeats)
        timesteps = scheduler_subset[:n_steps].clone()
        timesteps = timesteps.repeat(batch_size)
        timesteps = timesteps.cpu().numpy().astype("uint16")


        shard = Dataset.from_dict(
            {
                "activations": activations,
                "timestep": timesteps,
            },
            features=self.features_dict[hook_name],
        )
        return shard


    def create_dataset_feature(self, hook_name, d_in, d_out):
        self.features_dict[hook_name] = Features(
            {
                "activations": Array2D(
                    shape=(
                        d_in,
                        d_out,
                    ),
                    dtype=TORCH_STRING_DTYPE_MAP[self.cfg.dtype],
                ),
                "timestep": Value(dtype="uint16"),
            }
        )


    @torch.no_grad()
    def run(self) -> dict[str, Dataset]:

        assert self.cfg.new_cached_activations_path is not None
        final_cached_activation_paths = {
            n: Path(os.path.join(self.cfg.new_cached_activations_path, n))
            for n in self.cfg.hook_names
        }


        hooks_to_process: list[str] = []
        resume_shard_counts: list[int] = []
        resume_start_index = 0
        if self.accelerator.is_main_process:
            for hook_name, path in final_cached_activation_paths.items():
                path.mkdir(exist_ok=True, parents=True)
                tmp_dir = path / ".tmp_shards"
                dataset_finalized = self._dataset_is_finalized(path)

                if not self.cfg.resume:
                    if any(path.iterdir()):
                        raise Exception(
                            f"Activations directory ({path}) is not empty. Please delete it or specify a different path."
                        )
                    tmp_dir.mkdir(exist_ok=False, parents=False)
                    hooks_to_process.append(hook_name)
                    resume_shard_counts.append(0)
                    continue


                if dataset_finalized and not tmp_dir.exists():
                    print(f"Hook {hook_name} already cached; skipping due to resume flag.")
                    continue


                if not tmp_dir.exists():
                    tmp_dir.mkdir(exist_ok=True, parents=False)


                existing_shards = self._existing_shard_indices(tmp_dir)
                resume_count = existing_shards[-1] + 1 if existing_shards else 0
                hooks_to_process.append(hook_name)
                resume_shard_counts.append(resume_count)


            if len(set(resume_shard_counts)) > 1:
                raise ValueError(
                    "Resume requested but hooks have cached a different number of shards. Please rerun each hook separately."
                )


            if hooks_to_process:
                self.cfg.hook_names = hooks_to_process
                resume_start_index = resume_shard_counts[0]


        shared_objects = broadcast_object_list(
            [hooks_to_process, resume_start_index]
            if self.accelerator.is_main_process
            else [None, None]
        )
        hooks_to_process, resume_start_index = shared_objects


        if not hooks_to_process:
            if self.accelerator.is_main_process:
                print("Nothing to do; all requested hooks already cached.")
            return {}


        tmp_cached_activation_paths = {
            n: final_cached_activation_paths[n] / ".tmp_shards/"
            for n in hooks_to_process
        }


        self.cfg.hook_names = hooks_to_process
        self.features_dict = {hookpoint: None for hookpoint in self.cfg.hook_names}
        self.accelerator.wait_for_everyone()


        if self.accelerator.is_main_process:
            print(f"Started caching {self.num_examples} activations")


        for i, batch in tqdm(
            enumerate(self.dataloader),
            desc="Caching activations",
            total=self.n_buffers,
            initial=resume_start_index,
            disable=not self.accelerator.is_main_process,
        ):
            if i < resume_start_index:
                continue
            with self.accelerator.split_between_processes(batch) as prompt:
                prompt = prompt[self.cfg.column]
                _, acts_cache = self.pipe.run_with_cache(
                    prompt=prompt,
                    output_type="latent",
                    num_inference_steps=self.cfg.num_inference_steps,
                    save_input=True if self.cfg.output_or_diff == "diff" else False,
                    save_output=True,
                    positions_to_cache=self.cfg.hook_names,
                    guidance_scale=self.cfg.guidance_scale,
                )


            self.accelerator.wait_for_everyone()


            gathered_buffer = {}
            for hook_name in self.cfg.hook_names:
                if self.cfg.output_or_diff == "diff":
                    gathered_buffer[hook_name] = (
                        acts_cache["output"][hook_name] - acts_cache["input"][hook_name]
                    )
                else:
                    gathered_buffer[hook_name] = acts_cache["output"][hook_name]
            gathered_buffer = gather_object([gathered_buffer])


            if self.accelerator.is_main_process:
                for hook_name in self.cfg.hook_names:
                    gathered_buffer_acts = torch.cat(
                        [
                            gathered_buffer[i][hook_name]
                            for i in range(len(gathered_buffer))
                        ],
                        dim=0,
                    )
                    gathered_buffer_acts = self._apply_transforms(gathered_buffer_acts)
                    if self.features_dict[hook_name] is None:
                        self.create_dataset_feature(
                            hook_name,
                            gathered_buffer_acts.shape[-2],
                            gathered_buffer_acts.shape[-1],
                        )


                    print(f"{hook_name=} {gathered_buffer_acts.shape=}")
                    shard = self._create_shard(gathered_buffer_acts, hook_name)
                    shard.save_to_disk(
                        f"{tmp_cached_activation_paths[hook_name]}/shard_{i:05d}",
                        num_shards=1,
                    )
                    del gathered_buffer_acts, shard
                del gathered_buffer


        datasets = {}


        if self.accelerator.is_main_process:
            for hook_name, path in tmp_cached_activation_paths.items():
                datasets[hook_name] = self._consolidate_shards(
                    path, final_cached_activation_paths[hook_name], copy_files=False
                )
                self._write_transform_metadata(final_cached_activation_paths[hook_name])
                print(f"Consolidated the dataset for hook {hook_name}")


            if self.cfg.hf_repo_id:
                print("Pushing to hub...")
                for hook_name, dataset in datasets.items():
                    dataset.push_to_hub(
                        repo_id=f"{self.cfg.hf_repo_id}_{hook_name}",
                        num_shards=self.cfg.hf_num_shards or self.n_buffers,
                        private=self.cfg.hf_is_private_repo,
                        revision=self.cfg.hf_revision,
                    )


                meta_io = io.BytesIO()
                meta_contents = json.dumps(
                    asdict(self.cfg), indent=2, ensure_ascii=False
                ).encode("utf-8")
                meta_io.write(meta_contents)
                meta_io.seek(0)


                api = HfApi()
                api.upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )


        return datasets


    def load_and_push_to_hub(self) -> None:
        assert self.cfg.new_cached_activations_path is not None
        dataset = Dataset.load_from_disk(self.cfg.new_cached_activations_path)
        if self.accelerator.is_main_process:
            print("Loaded dataset from disk")
            if self.cfg.hf_repo_id:
                print("Pushing to hub...")
                dataset.push_to_hub(
                    repo_id=self.cfg.hf_repo_id,
                    num_shards=self.cfg.hf_num_shards
                    or (len(dataset) // self.cfg.batch_size),
                    private=self.cfg.hf_is_private_repo,
                    revision=self.cfg.hf_revision,
                )


                meta_io = io.BytesIO()
                meta_contents = json.dumps(
                    asdict(self.cfg), indent=2, ensure_ascii=False
                ).encode("utf-8")
                meta_io.write(meta_contents)
                meta_io.seek(0)


                api = HfApi()
                api.upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )
