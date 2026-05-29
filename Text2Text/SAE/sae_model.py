import torch
import torch.nn as nn
import torch.nn.functional as F
class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self._init_weights()
    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        self.normalize_decoder()
    @torch.no_grad()
    def normalize_decoder(self) -> None:
        weight = self.decoder.weight.data
        weight.div_(weight.norm(dim=0, keepdim=True).clamp_min(1e-6))
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.encoder(x - self.pre_bias))
        if self.k > 0 and self.k < z.shape[-1]:
            values, indices = torch.topk(z, self.k, dim=-1)
            z_topk = torch.zeros_like(z)
            z_topk.scatter_(-1, indices, values)
            return z_topk
        return z
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.pre_bias
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
def load_sae(path: str | bytes | object, map_location="cpu") -> TopKSAE:
    ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt["config"]
    sae = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae
