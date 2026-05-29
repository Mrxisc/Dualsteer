import torch

class SD35:
    def __init__(self, model_name: str, special_token: str = "", strength: str = "strong"):
        from diffusers import StableDiffusion3Pipeline
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        try:
            self.pipeline.enable_model_cpu_offload()
        except AttributeError:
            self.pipeline.to("cuda")
        self.pipeline.set_progress_bar_config(disable=True)
        self.model_name = model_name.replace("/", "-")
        self.images_per_prompt = 10
        self.height = 512
        self.width = 512
        self.special_token = special_token
        self.generator = torch.Generator(device="cuda")


    def __call__(self, prompt: str, seed: int, scale: float):
        self.generator.manual_seed(seed)
        result = self.pipeline(
            prompt=prompt + self.special_token,
            num_images_per_prompt=self.images_per_prompt,
            guidance_scale=scale,
            generator=self.generator,
            height=self.height,
            width=self.width,
        )
        return result.images
