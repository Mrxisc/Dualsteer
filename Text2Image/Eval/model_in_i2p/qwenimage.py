from pathlib import Path
import torch
from diffusers import QwenImagePipeline
from typing import Optional
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


class QwenImage:
    def __init__(
        self,
        model_name: str = str(WORKSPACE_ROOT / "model" / "Qwen-Image"),
        special_token: str = "",
        strength: Optional[str] = None,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32


        self.pipeline = QwenImagePipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            local_files_only=True,
        )
        if device == "cuda":
            try:
                self.pipeline.enable_model_cpu_offload()
            except AttributeError:
                self.pipeline.to(device)
        self.pipeline.set_progress_bar_config(disable=True)


        self.model_name = Path(model_name).name
        self.special_token = special_token
        self.images_per_prompt = 10
        self.generator = torch.Generator(device=device)
        self.device = device
        self.guidance_scale_default = 3.0


    def __call__(self, prompt: str, seed: int, scale: float):
        self.generator.manual_seed(int(seed))
        result = self.pipeline(
            prompt=prompt + self.special_token,
            num_images_per_prompt=self.images_per_prompt,
            guidance_scale=float(scale) if scale is not None else self.guidance_scale_default,
            generator=self.generator,
        )
        return result.images
