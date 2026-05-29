import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
try:
    from Dualsteer_Code/Text2Image._Backup_code.dit_backbone import DiTBackbone
except ImportError:
    DiTBackbone = None
try:
    from Dualsteer_Code/Text2Image._Backup_code.flux_backbone import FluxBackbone
except ImportError:
    FluxBackbone = None
try:
    from Dualsteer_Code/Text2Image._Backup_code.sd35_backbone import SD35Backbone
except ImportError:
    SD35Backbone = None
from .hooked_scheduler import HookedNoiseScheduler


def retrieve(io, unconditional: bool = False):
    if isinstance(io, tuple):
        if len(io) == 1:
            io = io[0].detach().cpu()
            if io.shape[0] >= 2 and io.shape[0] % 2 == 0:
                io_uncond, io_cond = io.chunk(2)
                if unconditional:
                    return io_uncond
                return io_cond
            return io
        else:
            raise ValueError("A tuple should have length of 1")
    elif isinstance(io, torch.Tensor):
        io = io.detach().cpu()
        if io.shape[0] >= 2 and io.shape[0] % 2 == 0:
            io_uncond, io_cond = io.chunk(2)
            if unconditional:
                return io_uncond
            return io_cond
        return io
    elif hasattr(io, "last_hidden_state"):


        return retrieve(io.last_hidden_state, unconditional)
    elif hasattr(io, "sample"):


        return retrieve(io.sample, unconditional)
    else:
        raise ValueError("Input/Output must be a tensor, or 1-element tuple")


class HookedDiffusionAbstractPipeline:
    parent_cls = None
    pipe = None
    def __init__(
        self,
        pipe: DiffusionPipeline,
        use_hooked_scheduler: bool = False,
        *,
        denoiser_attr: str = "unet",
        use_ddim_scheduler: bool = True,
        use_dit: bool = False,
        dit_config: dict | None = None,
        use_flux: bool = False,
        flux_config: dict | None = None,
        use_sd35_backbone: bool = False,
        sd35_config: dict | None = None,
    ):
        adapters_enabled = [use_dit, use_flux, use_sd35_backbone]
        if sum(bool(flag) for flag in adapters_enabled) > 1:
            raise ValueError("Only one custom backbone adapter (DiT, Flux, SD3.5) can be enabled at a time.")


        object.__setattr__(self, "use_hooked_scheduler", use_hooked_scheduler)
        object.__setattr__(self, "use_ddim_scheduler", use_ddim_scheduler)
        object.__setattr__(self, "use_dit", use_dit)
        object.__setattr__(self, "use_flux", use_flux)
        object.__setattr__(self, "use_sd35_backbone", use_sd35_backbone)
        object.__setattr__(self, "dit_model", None)
        object.__setattr__(self, "flux_model", None)
        object.__setattr__(self, "sd35_model", None)
        object.__setattr__(self, "_flux_config", flux_config)
        object.__setattr__(self, "_sd35_config", sd35_config)


        if use_hooked_scheduler:
            pipe.scheduler = HookedNoiseScheduler(pipe.scheduler)
        if use_ddim_scheduler:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            print("Using DDIM Scheduler")


        target_denoiser_attr = denoiser_attr


        if use_dit:
            if DiTBackbone is None:
                raise ImportError("DiT support unavailable; install required modules.")
            dit_model = DiTBackbone(**(dit_config or {}))
            object.__setattr__(self, "dit_model", dit_model)
            setattr(pipe, "dit_model", dit_model)
            target_denoiser_attr = denoiser_attr or "dit_model"
            print("[SAeUron] DiT backbone loaded.")


        if use_flux:
            if FluxBackbone is None:
                raise ImportError("FluxBackbone is unavailable; ensure models/flux_backbone.py is importable.")
            flux_model = self._initialize_flux_model(pipe, flux_config)
            object.__setattr__(self, "flux_model", flux_model)
            setattr(pipe, "flux_model", flux_model)
            target_denoiser_attr = "flux_model"
        if use_sd35_backbone:
            if SD35Backbone is None:
                raise ImportError("SD35Backbone is unavailable; ensure models/sd35_backbone.py is importable.")
            sd35_target = denoiser_attr or "transformer"
            sd35_model = self._initialize_sd35_model(pipe, sd35_config, sd35_target)
            object.__setattr__(self, "sd35_model", sd35_model)
            target_denoiser_attr = sd35_target
        object.__setattr__(self, "denoiser_attr", target_denoiser_attr)
        self.__dict__["pipe"] = pipe


    def _initialize_flux_model(self, pipe, flux_config: dict | None):
        if flux_config is None:
            raise ValueError("Flux support requires `flux_config` with at least sample_size and in_channels.")


        cfg = dict(flux_config)
        required = {"sample_size", "in_channels"}
        missing = [key for key in required if key not in cfg]
        if missing:
            raise ValueError(f"Flux config missing required keys: {missing}")


        transformer_in = None
        try:
            transformer = getattr(pipe, "transformer", None)
            transformer_in = getattr(getattr(transformer, "config", None), "in_channels", None)
        except Exception:
            transformer_in = None


        latent_channels = cfg.pop("latent_channels", None)
        if transformer_in is not None:
            latent_channels = int(transformer_in)
            cfg["in_channels"] = int(transformer_in)
            cfg["out_dim"] = int(transformer_in)
        elif latent_channels is None:
            latent_channels = max(cfg["in_channels"] // 4, 1)


        backbone_kwargs = dict(cfg)
        flux_model = FluxBackbone(**backbone_kwargs)
        setattr(pipe, "flux_model", flux_model)
        resolved_cfg = {**backbone_kwargs, "latent_channels": latent_channels}
        flux_model.config = SimpleNamespace(
            sample_size=resolved_cfg["sample_size"],
            in_channels=resolved_cfg["in_channels"],
            latent_channels=latent_channels,
        )
        object.__setattr__(self, "_flux_config", resolved_cfg)
        return flux_model


    def _initialize_sd35_model(self, pipe, sd35_config: dict | None, target_attr: str):
        if sd35_config is None:
            raise ValueError("SD3.5 backbone requires `sd35_config` describing latent/text dims.")
        cfg = dict(sd35_config)
        required = {"sample_size", "in_channels"}
        missing = [key for key in required if key not in cfg]
        if missing:
            raise ValueError(f"SD3.5 config missing required keys: {missing}")


        latent_channels = cfg.pop("latent_channels", None)
        if latent_channels is None:
            latent_channels = cfg["in_channels"]


        sd35_model = SD35Backbone(**cfg)
        original = getattr(pipe, target_attr, None)
        if original is not None:
            object.__setattr__(self, f"original_{target_attr}", original.cpu())
        setattr(pipe, target_attr, sd35_model)
        resolved_cfg = {**cfg, "latent_channels": latent_channels, "target_attr": target_attr}
        sd35_model.config = SimpleNamespace(**resolved_cfg)
        object.__setattr__(self, "_sd35_config", resolved_cfg)
        return sd35_model


    def _prepare_flux_condition(self, prompt_embeds: torch.Tensor | None):
        if prompt_embeds is None or self.flux_model is None:
            return None
        cond = prompt_embeds.mean(dim=1, keepdim=True)
        target = self.flux_model.cond_dim
        current = cond.shape[-1]
        if current < target:
            cond = F.pad(cond, (0, target - current))
        elif current > target:
            cond = cond[..., :target]
        return cond


    def _num_latent_channels(self) -> int:
        if getattr(self, "use_flux", False):
            try:
                transformer = getattr(getattr(self, "pipe", None), "transformer", None)
                transformer_in = getattr(getattr(transformer, "config", None), "in_channels", None)
                if transformer_in is not None:
                    return int(transformer_in) // 4
            except Exception:
                pass
            if getattr(self, "_flux_config", None):
                return self._flux_config.get("latent_channels", self.denoiser.config.in_channels)
        if getattr(self, "use_sd35_backbone", False) and getattr(self, "_sd35_config", None):
            return self._sd35_config.get("latent_channels", self.denoiser.config.in_channels)
        return self.denoiser.config.in_channels


    def _run_flux_model(self, latents: torch.Tensor, prompt_embeds: torch.Tensor | None):
        if self.flux_model is None:
            raise RuntimeError("Flux model is not initialized.")
        if latents.ndim == 4:
            b, c, h, w = latents.shape
            tokens = latents.permute(0, 2, 3, 1).reshape(b, h * w, c)
        elif latents.ndim == 3:
            tokens = latents
            h = w = None
        else:
            raise ValueError(
                f"Flux backbone expects latents with 3 or 4 dims, got shape {latents.shape}"
            )
        cond_tokens = self._prepare_flux_condition(prompt_embeds)
        out = self.flux_model(tokens, cond_tokens=cond_tokens)
        if h is not None and w is not None:
            out = out.reshape(b, h, w, out.shape[-1]).permute(0, 3, 1, 2).contiguous()
        return out


    def _encode_prompt_with_optional_streams(
        self,
        prompt: Union[str, List[str]],
        *,
        device: torch.device,
        num_images_per_prompt: int,
        **extra_kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        encode_fn = getattr(self.pipe, "encode_prompt", None)
        if encode_fn is None:
            raise AttributeError("Pipeline does not expose encode_prompt()")
        sig = inspect.signature(encode_fn)
        kwargs = dict(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            lora_scale=None,
        )
        for optional_key in ("prompt_2", "prompt_3"):
            if optional_key in sig.parameters:
                kwargs[optional_key] = None
        for key, value in extra_kwargs.items():
            if key in sig.parameters and value is not None:
                kwargs[key] = value
        return encode_fn(**kwargs)


    @staticmethod
    def _unpack_prompt_outputs(
        prompt_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(prompt_output, tuple):


            if len(prompt_output) == 4:
                prompt_embeds = prompt_output[0]
                negative_prompt_embeds = prompt_output[1]
                pooled_prompt_embeds = prompt_output[2]
                negative_pooled_prompt_embeds = prompt_output[3]
            elif len(prompt_output) == 2:
                prompt_embeds = prompt_output[0]
                negative_prompt_embeds = None
                pooled_prompt_embeds = prompt_output[1]
                negative_pooled_prompt_embeds = None
            elif len(prompt_output) == 3:
                prompt_embeds = prompt_output[0]
                negative_prompt_embeds = None
                pooled_prompt_embeds = prompt_output[1]
                negative_pooled_prompt_embeds = None
            else:
                prompt_embeds = prompt_output[0]
                negative_prompt_embeds = prompt_output[1] if len(prompt_output) > 1 else None
                pooled_prompt_embeds = prompt_output[2] if len(prompt_output) > 2 else None
                negative_pooled_prompt_embeds = prompt_output[3] if len(prompt_output) > 3 else None
        else:
            prompt_embeds = prompt_output
            negative_prompt_embeds = None
            pooled_prompt_embeds = None
            negative_pooled_prompt_embeds = None
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )


    def _prepare_sd35_condition(self, prompt_embeds: torch.Tensor | None):
        if prompt_embeds is None or self.sd35_model is None:
            return None
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(1)
        return prompt_embeds


    def _run_sd35_model(self, latents: torch.Tensor, prompt_embeds: torch.Tensor | None):
        if self.sd35_model is None:
            raise RuntimeError("SD3.5 backbone is not initialized.")
        if latents.ndim == 4:
            b, c, h, w = latents.shape
            tokens = latents.permute(0, 2, 3, 1).reshape(b, h * w, c)
            spatial = (h, w)
        elif latents.ndim == 3:
            b = latents.shape[0]
            tokens = latents
            spatial = None
        else:
            raise ValueError(f"SD3.5 backbone expects latents with 3 or 4 dims, got shape {latents.shape}")
        cond_tokens = self._prepare_sd35_condition(prompt_embeds)
        out = self.sd35_model(tokens, cond_tokens=cond_tokens)
        if spatial is not None:
            h, w = spatial
            out = out.reshape(b, h, w, out.shape[-1]).permute(0, 3, 1, 2).contiguous()
        return out


    @classmethod
    def from_pretrained(
        cls,
        *args,
        use_hooked_scheduler: bool = False,
        use_ddim_scheduler: bool = True,
        denoiser_attr: str | None = None,
        use_dit: bool = False,
        dit_config: dict = None,
        use_flux: bool = False,
        flux_config: dict = None,
        use_sd35_backbone: bool = False,
        sd35_config: dict = None,
        **kwargs,
    ):
        pipe = cls.parent_cls.from_pretrained(*args, **kwargs)
        init_kwargs = {}
        if denoiser_attr is not None:
            init_kwargs["denoiser_attr"] = denoiser_attr
        if use_dit:
            init_kwargs["use_dit"] = use_dit
            init_kwargs["dit_config"] = dit_config
        if use_flux:
            init_kwargs["use_flux"] = use_flux
            init_kwargs["flux_config"] = flux_config
        if use_sd35_backbone:
            init_kwargs["use_sd35_backbone"] = use_sd35_backbone
            init_kwargs["sd35_config"] = sd35_config
        return cls(
            pipe,
            use_hooked_scheduler=use_hooked_scheduler,
            use_ddim_scheduler=use_ddim_scheduler,
            **init_kwargs,
        )


    @property
    def denoiser(self):
        return getattr(self.pipe, self.denoiser_attr)


    @torch.no_grad()
    def run_with_hooks(
        self,
        *args,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):
        hooks = []
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))
        hooks = [hook for hook in hooks if hook is not None]
        try:
            req_height = kwargs.pop("height", None)
            req_width = kwargs.pop("width", None)
            (
                prompt_embeds,
                pooled_prompt_embeds,
                timesteps,
                latents,
                extra_step_kwargs,
                added_cond_kwargs,
                height,
                width,
            ) = self._prepare_prompt(
                prompt,
                device,
                num_images_per_prompt,
                guidance_scale,
                num_inference_steps,
                generator,
                latents,
                height=req_height,
                width=req_width,
            )


            latents = self._denoise_loop(
                timesteps,
                latents,
                guidance_scale,
                extra_step_kwargs,
                added_cond_kwargs,
                prompt_embeds,
                pooled_prompt_embeds,
            )
            image = self._postprocess_latents(latents, output_type, generator, height=height, width=width)
        finally:
            for hook in hooks:
                hook.remove()
            if self.use_hooked_scheduler:
                self.pipe.scheduler.pre_hooks = []
                self.pipe.scheduler.post_hooks = []
        return image


    @torch.no_grad()
    def run_with_cache(
        self,
        *args,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        positions_to_cache: List[str],
        output_type: Optional[str] = "pil",
        save_input: bool = False,
        save_output: bool = True,
        unconditional: bool = False,
        **kwargs,
    ):
        cache_input, cache_output = (
            dict() if save_input else None,
            dict() if save_output else None,
        )
        hooks = [
            self._register_cache_hook(
                position, cache_input, cache_output, unconditional
            )
            for position in positions_to_cache
        ]
        hooks = [hook for hook in hooks if hook is not None]


        req_height = kwargs.pop("height", None)
        req_width = kwargs.pop("width", None)


        (
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps,
            latents,
            extra_step_kwargs,
            added_cond_kwargs,
            height,
            width,
        ) = self._prepare_prompt(
            prompt,
            device,
            num_images_per_prompt,
            guidance_scale,
            num_inference_steps,
            generator,
            latents,
            height=req_height,
            width=req_width,
        )


        latents = self._denoise_loop(
            timesteps,
            latents,
            guidance_scale,
            extra_step_kwargs,
            added_cond_kwargs,
            prompt_embeds,
            pooled_prompt_embeds,
        )


        for hook in hooks:
            hook.remove()
        if self.use_hooked_scheduler:
            self.pipe.scheduler.pre_hooks = []
            self.pipe.scheduler.post_hooks = []


        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict["input"] = cache_input


        if save_output:
            for position, block in cache_output.items():
                cache_output[position] = torch.stack(block, dim=1)
            cache_dict["output"] = cache_output


        image = self._postprocess_latents(latents, output_type, generator, height=height, width=width)
        return image, cache_dict


    @torch.no_grad()
    def run_with_cache_intermediate(
        self,
        *args,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        positions_to_cache: List[str],
        output_type: Optional[str] = "pil",
        save_input: bool = False,
        save_output: bool = True,
        **kwargs,
    ):
        assert isinstance(self.pipe.scheduler, DDIMScheduler)
        req_height = kwargs.pop("height", None)
        req_width = kwargs.pop("width", None)
        (
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps,
            latents,
            extra_step_kwargs,
            added_cond_kwargs,
            height,
            width,
        ) = self._prepare_prompt(
            prompt,
            device,
            num_images_per_prompt,
            guidance_scale,
            num_inference_steps,
            generator,
            latents,
            height=req_height,
            width=req_width,
        )


        cache_input, cache_output = (
            dict() if save_input else None,
            dict() if save_output else None,
        )
        all_intermediate_latents = []
        hooks = [
            self._register_cache_hook(position, cache_input, cache_output)
            for position in positions_to_cache
        ]
        hooks = [hook for hook in hooks if hook is not None]


        denoiser_signature = inspect.signature(self.denoiser.forward)
        self._num_timesteps = len(timesteps)
        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            )
            latent_model_input = self.pipe.scheduler.scale_model_input(
                latent_model_input, t
            )

            timestep_tensor = t
            if not torch.is_tensor(timestep_tensor):
                timestep_tensor = torch.tensor([timestep_tensor], device=latent_model_input.device)
            else:
                timestep_tensor = timestep_tensor.to(latent_model_input.device)
            if timestep_tensor.ndim == 0:
                timestep_tensor = timestep_tensor[None]
            timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])


            denoiser_kwargs = dict(
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )
            if pooled_prompt_embeds is not None and "pooled_projections" in denoiser_signature.parameters:
                denoiser_kwargs["pooled_projections"] = pooled_prompt_embeds
            if added_cond_kwargs is not None and "added_cond_kwargs" in denoiser_signature.parameters:
                denoiser_kwargs["added_cond_kwargs"] = added_cond_kwargs
            if "timestep_cond" in denoiser_signature.parameters:
                denoiser_kwargs["timestep_cond"] = None


            noise_pred = self.denoiser(latent_model_input, **denoiser_kwargs)[0]
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )


            scheduler_out = self.pipe.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=True
            )
            latents = scheduler_out.prev_sample
            all_intermediate_latents.append(scheduler_out.pred_original_sample)


        for hook in hooks:
            hook.remove()
        if self.use_hooked_scheduler:
            self.pipe.scheduler.pre_hooks = []
            self.pipe.scheduler.post_hooks = []


        cache_dict = {}
        if save_input and cache_input is not None:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict["input"] = cache_input


        if save_output and cache_output is not None:
            for position, block in cache_output.items():
                cache_output[position] = torch.stack(block, dim=1)
            cache_dict["output"] = cache_output


        if output_type != "latent":
            image = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            if all_intermediate_latents:
                for i in range(len(all_intermediate_latents)):
                    all_intermediate_latents[i] = self.pipe.vae.decode(
                        all_intermediate_latents[i]
                        / self.pipe.vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]
        else:
            image = latents
        do_denormalize = [True] * image.shape[0]


        image = self.pipe.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )
        if all_intermediate_latents:
            for i in range(len(all_intermediate_latents)):
                all_intermediate_latents[i] = self.pipe.image_processor.postprocess(
                    all_intermediate_latents[i],
                    output_type=output_type,
                    do_denormalize=do_denormalize,
                )


        if output_type == "latent":
            image = image.cpu().numpy()


        return image, all_intermediate_latents, cache_dict


    def run_with_hooks_and_cache(
        self,
        *args,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        positions_to_cache: List[str] = [],
        save_input: bool = False,
        save_output: bool = True,
        **kwargs,
    ):
        cache_input, cache_output = (
            dict() if save_input else None,
            dict() if save_output else None,
        )
        hooks = [
            self._register_cache_hook(position, cache_input, cache_output)
            for position in positions_to_cache
        ]


        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))


        hooks = [hook for hook in hooks if hook is not None]
        output = self.pipe(*args, **kwargs)
        for hook in hooks:
            hook.remove()
        if self.use_hooked_scheduler:
            self.pipe.scheduler.pre_hooks = []
            self.pipe.scheduler.post_hooks = []


        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict["input"] = cache_input


        if save_output:
            for position, block in cache_output.items():
                cache_output[position] = torch.stack(block, dim=1)
            cache_dict["output"] = cache_output


        return output, cache_dict


    def _locate_block(self, position: str):
        block = self.pipe
        for step in position.split("."):
            if step.isdigit():
                step = int(step)
                block = block[step]
            else:
                if hasattr(block, step):
                    block = getattr(block, step)
                    continue

                if step == "double_transformer_blocks":
                    if hasattr(block, "double_blocks"):
                        block = getattr(block, "double_blocks")
                        continue
                    if hasattr(block, "transformer_blocks"):
                        block = getattr(block, "transformer_blocks")
                        continue
                if step == "double_blocks" and hasattr(block, "transformer_blocks"):
                    block = getattr(block, "transformer_blocks")
                    continue
                if step == "single_transformer_blocks":
                    if hasattr(block, "single_blocks"):
                        block = getattr(block, "single_blocks")
                        continue
                    if hasattr(block, "single_transformer_blocks"):
                        block = getattr(block, "single_transformer_blocks")
                        continue
                if step == "single_blocks" and hasattr(block, "single_transformer_blocks"):
                    block = getattr(block, "single_transformer_blocks")
                    continue


                if (
                    step == "text_encoder"
                    and hasattr(self.pipe, "text_encoder")
                    and (block is getattr(self.pipe, "flux_model", None) or block.__class__.__name__ == "FluxBackbone")
                ):
                    block = getattr(self.pipe, "text_encoder")
                    continue


                block = getattr(block, step)
        return block


    def _register_cache_hook(
        self,
        position: str,
        cache_input: Dict,
        cache_output: Dict,
        unconditional: bool = False,
    ):
        block = self._locate_block(position)


        def hook(module, input, kwargs, output):
            if cache_input is not None:
                if position not in cache_input:
                    cache_input[position] = []
                input_to_cache = retrieve(input, unconditional)
                if len(input_to_cache.shape) == 4:
                    input_to_cache = input_to_cache.view(
                        input_to_cache.shape[0], input_to_cache.shape[1], -1
                    ).permute(0, 2, 1)
                cache_input[position].append(input_to_cache)


            if cache_output is not None:
                if position not in cache_output:
                    cache_output[position] = []
                output_to_cache = retrieve(output, unconditional)
                if len(output_to_cache.shape) == 4:
                    output_to_cache = output_to_cache.view(
                        output_to_cache.shape[0], output_to_cache.shape[1], -1
                    ).permute(0, 2, 1)
                cache_output[position].append(output_to_cache)


        return block.register_forward_hook(hook, with_kwargs=True)


    def _register_general_hook(self, position, hook):
        if position == "scheduler_pre":
            if not self.use_hooked_scheduler:
                raise ValueError(
                    "Cannot register hooks on scheduler without using hooked scheduler"
                )
            self.pipe.scheduler.pre_hooks.append(hook)
            return
        elif position == "scheduler_post":
            if not self.use_hooked_scheduler:
                raise ValueError(
                    "Cannot register hooks on scheduler without using hooked scheduler"
                )
            self.pipe.scheduler.post_hooks.append(hook)
            return


        block = self._locate_block(position)
        return block.register_forward_hook(hook)


    def _prepare_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        guidance_scale,
        num_inference_steps,
        generator,
        latents,
        *,
        height: int | None = None,
        width: int | None = None,
    ):

        if height is None or width is None:
            sample_size = getattr(self.denoiser.config, "sample_size", None)
            if sample_size is None:
                raise ValueError(
                    "Denoiser config does not expose `sample_size`. Please supply height/width explicitly."
                )
            vae_scale = getattr(self.pipe, "vae_scale_factor", 1)
            height = sample_size * vae_scale
            width = sample_size * vae_scale
        else:
            height = int(height)
            width = int(width)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)


        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        negative_prompt_list = None
        if do_classifier_free_guidance and prompt_list is not None:
            negative_prompt_list = ["" for _ in range(len(prompt_list))]


        prompt_output = self._encode_prompt_with_optional_streams(
            prompt_list,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt_list,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._unpack_prompt_outputs(prompt_output)


        if prompt_embeds is None:
            raise ValueError("encode_prompt() did not return prompt embeddings.")


        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                uncond_prompt_list = ["" for _ in range(len(prompt_list))]
                uncond_output = self._encode_prompt_with_optional_streams(
                    uncond_prompt_list,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=False,
                )
                (
                    negative_prompt_embeds,
                    _,
                    negative_pooled_prompt_embeds,
                    _,
                ) = self._unpack_prompt_outputs(uncond_output)


            if isinstance(prompt_embeds, torch.Tensor) and prompt_embeds.ndim == 2:
                prompt_embeds = prompt_embeds.unsqueeze(1)
            if isinstance(negative_prompt_embeds, torch.Tensor) and negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1)


            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if (
                pooled_prompt_embeds is not None
                and negative_pooled_prompt_embeds is not None
            ):
                pooled_prompt_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds]
                )
            else:
                pooled_prompt_embeds = None


        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler, num_inference_steps, device, None, None
        )


        num_channels_latents = self._num_latent_channels()
        prepare_latents_out = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if isinstance(prepare_latents_out, tuple):
            latents, added_cond_kwargs = prepare_latents_out
        else:
            latents = prepare_latents_out
            added_cond_kwargs = None


        if hasattr(self.pipe, "prepare_extra_step_kwargs"):
            extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, 0.0)
        else:
            extra_step_kwargs = {}
        return (
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps,
            latents,
            extra_step_kwargs,
            added_cond_kwargs,
            height,
            width,
        )


    def _denoise_loop(
        self,
        timesteps,
        latents,
        guidance_scale,
        extra_step_kwargs,
        added_cond_kwargs,
        prompt_embeds,
        pooled_prompt_embeds,
    ):
        denoiser_signature = inspect.signature(self.denoiser.forward)
        self._num_timesteps = len(timesteps)
        for i, t in enumerate(timesteps):


            latent_model_input = (
                torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            )
            latent_model_input = self.pipe.scheduler.scale_model_input(
                latent_model_input, t
            )
            timestep_tensor = t
            if not torch.is_tensor(timestep_tensor):
                timestep_tensor = torch.tensor([timestep_tensor], device=latent_model_input.device)
            else:
                timestep_tensor = timestep_tensor.to(latent_model_input.device)
            if timestep_tensor.ndim == 0:
                timestep_tensor = timestep_tensor[None]
            timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])


            if getattr(self, "use_flux", False) and self.flux_model is not None:
                noise_pred = self._run_flux_model(latent_model_input, prompt_embeds)
            elif getattr(self, "use_sd35_backbone", False) and self.sd35_model is not None:
                noise_pred = self._run_sd35_model(latent_model_input, prompt_embeds)
            elif getattr(self, "use_dit", False) and self.dit_model is not None:
                noise_pred = self.dit_model(latent_model_input)
                if isinstance(noise_pred, tuple):
                    noise_pred = noise_pred[0]
            else:


                denoiser_kwargs = dict(
                    timestep=timestep_tensor,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
                if pooled_prompt_embeds is not None and "pooled_projections" in denoiser_signature.parameters:
                    denoiser_kwargs["pooled_projections"] = pooled_prompt_embeds
                if added_cond_kwargs is not None and "added_cond_kwargs" in denoiser_signature.parameters:
                    denoiser_kwargs["added_cond_kwargs"] = added_cond_kwargs
                if "timestep_cond" in denoiser_signature.parameters:
                    denoiser_kwargs["timestep_cond"] = None
                noise_pred = self.denoiser(
                    latent_model_input,
                    **denoiser_kwargs,
                )[0]


            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )


            latents = self.pipe.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]
        return latents


    def _postprocess_latents(self, latents, output_type, generator, *, height: int, width: int):
        if not output_type == "latent":
            latents_to_decode = latents
            if (
                getattr(self, "use_flux", False)
                and hasattr(self.pipe, "_unpack_latents")
                and isinstance(latents_to_decode, torch.Tensor)
                and latents_to_decode.ndim == 3
            ):
                latents_to_decode = self.pipe._unpack_latents(
                    latents_to_decode, height, width, getattr(self.pipe, "vae_scale_factor", 1)
                )
            image = self.pipe.vae.decode(
                latents_to_decode / self.pipe.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
        else:
            image = latents
        do_denormalize = [True] * image.shape[0]


        image = self.pipe.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )


        if output_type == "latent":
            image = image.cpu().numpy()
        return image


    def to(self, *args, **kwargs):
        self.pipe = self.pipe.to(*args, **kwargs)
        if getattr(self, "flux_model", None) is not None:
            self.flux_model = self.flux_model.to(*args, **kwargs)
            setattr(self.pipe, "flux_model", self.flux_model)
        if getattr(self, "sd35_model", None) is not None:
            self.sd35_model = self.sd35_model.to(*args, **kwargs)
        if getattr(self, "dit_model", None) is not None:
            self.dit_model = self.dit_model.to(*args, **kwargs)
        return self


    def __getattr__(self, name):
        return getattr(self.pipe, name)


    def __setattr__(self, name, value):
        return setattr(self.pipe, name, value)


    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)


class HookedStableDiffusionPipeline(HookedDiffusionAbstractPipeline):
    parent_cls = DiffusionPipeline
