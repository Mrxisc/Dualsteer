import torch
from pathlib import Path
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


class Flux:
    def __init__(self, model_name=str(WORKSPACE_ROOT / "model" / "FLUX.1-dev"), special_token='', strength=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        from diffusers import FluxPipeline
        self.pipeline = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            local_files_only=True,
        )
        if device == "cuda":
            self.pipeline.enable_model_cpu_offload()
        self.pipeline.set_progress_bar_config(disable=True)
        self.model_name = Path(model_name).name
        self.images_per_gen = (2, 5)
        self.images_per_prompt = self.images_per_gen[0] * self.images_per_gen[1]
        self.generator = torch.Generator(device=device)
        self.special_token = special_token
        self.device = device


    def __call__(self, prompt, seed, scale):
        images = []
        self.generator.manual_seed(int(seed))
        for _ in range(self.images_per_gen[0]):
            result = self.pipeline(
                prompt=prompt + self.special_token,
                num_images_per_prompt=self.images_per_gen[1],
                guidance_scale=float(scale),
                height=512,
                width=512,
                num_inference_steps=40,
                generator=self.generator,
                max_sequence_length=256,
            )
            images.extend(result.images)
        return images
