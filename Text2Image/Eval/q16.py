import os
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
WORKSPACE_ROOT = Path(__file__).resolve().parents[2].parent


def _resolve_q16_prompts_path() -> str:
    env_path = os.environ.get("Q16_PROMPTS")
    if env_path:
        return env_path


    fallback = WORKSPACE_ROOT / "i2p-main" / "checkpoints" / "Q16-prompt" / "prompts.p"
    if fallback.exists():
        return str(fallback)


    raise FileNotFoundError(
        "Q16 prompts file not found. Set Q16_PROMPTS or place prompts.p at "
        f"{fallback}."
    )


def _resolve_q16_clip_path() -> str:
    env_path = os.environ.get("Q16_CLIP_MODEL")
    if env_path:
        return env_path


    fallback = WORKSPACE_ROOT / "model" / "clip-vit-large-patch14-336"
    if fallback.exists():
        return str(fallback)


    raise FileNotFoundError(
        "Q16 CLIP model path not found. Set Q16_CLIP_MODEL or place model at "
        f"{fallback}."
    )


file_path = _resolve_q16_prompts_path()
def load_prompts(device):
    return torch.FloatTensor(pickle.load(open(file_path, 'rb'))).to(device)


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


class Q16():
    def __init__(self):
        self.torch_device = "cuda"
        self.safety_prompts = load_prompts(device=self.torch_device)


        clip_path = _resolve_q16_clip_path()
        self.model = CLIPVisionModelWithProjection.from_pretrained(clip_path, local_files_only=True).to(self.torch_device)
        self.processor = CLIPImageProcessor.from_pretrained(clip_path, local_files_only=True)


    def q16_classifier(self, embeddings, verbose=False):
        safety_prompts_norm = self.safety_prompts / self.safety_prompts.norm(dim=-1, keepdim=True)
        image_features_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ safety_prompts_norm.T)


        probs = similarity.squeeze().softmax(dim=-1)
        if verbose:
            print(probs)
        prediction_score, pred_label_idx = torch.topk(probs.float(), 1)
        return pred_label_idx.squeeze()


    @torch.no_grad()
    def detect(self, image: Image):
        clip_input = self.processor(images=image, return_tensors="pt").to(self.torch_device)
        image_embeds = self.model(clip_input.pixel_values).image_embeds
        q16_safety_classfier_res = self.q16_classifier(image_embeds)
        return bool(q16_safety_classfier_res.item())


if __name__ == "__main__":
    m = Q16()
    paths = [
        "./b10_p139_19.jpg",
        "./b10_p146_13.jpg",
        "./b11_p176_6.jpg",
        "./b4_p78_18.jpg",
        "./b5_p82_6.jpg",
        "./b8_p115_17.jpg"
    ]
    for p in paths:
        img = Image.open(p)
        print(p, m.detect(img))
