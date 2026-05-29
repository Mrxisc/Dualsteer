from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute collapse metrics from a manifest CSV")
    p.add_argument("--manifest", required=True, help="Path to manifest CSV (must contain image_path)")
    p.add_argument(
        "--images-root",
        default=".",
        help="Root directory that image_path is relative to (default: current working dir)",
    )
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument(
        "--fft-highfreq-radius",
        type=float,
        default=0.35,
        help="High-frequency cutoff as fraction of Nyquist radius (default: 0.35)",
    )
    p.add_argument("--max-rows", type=int, default=0, help="Debug: limit to first N rows (0 = all)")
    return p


def _rgb_to_hsv_saturation(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    s = np.zeros_like(cmax)
    mask = cmax > 1e-8
    s[mask] = delta[mask] / cmax[mask]
    return s


def _laplacian_variance(gray: np.ndarray) -> float:
    up = np.roll(gray, 1, axis=0)
    down = np.roll(gray, -1, axis=0)
    left = np.roll(gray, 1, axis=1)
    right = np.roll(gray, -1, axis=1)
    lap = (up + down + left + right) - 4.0 * gray
    return float(np.var(lap))


def _highfreq_ratio(gray: np.ndarray, radius_frac: float) -> float:
    h, w = gray.shape
    fft = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(fft))
    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = np.sqrt(cy**2 + cx**2) + 1e-8
    rrn = rr / rmax
    hf = mag[rrn >= float(radius_frac)]
    lf = mag[rrn < float(radius_frac)]
    hf_energy = float(np.mean(hf**2)) if hf.size else 0.0
    lf_energy = float(np.mean(lf**2)) if lf.size else 0.0
    denom = hf_energy + lf_energy + 1e-12
    return float(hf_energy / denom)


def _load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    return rgb, gray


def main() -> None:
    args = _build_parser().parse_args()
    manifest = Path(args.manifest)
    images_root = Path(args.images_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest)
    if "image_path" not in df.columns:
        raise ValueError(f"manifest missing image_path column: {manifest}")
    if args.max_rows and args.max_rows > 0:
        df = df.head(int(args.max_rows)).copy()
    rows = []
    for idx, r in df.iterrows():
        rel = str(r["image_path"])
        img_path = images_root / rel
        try:
            rgb, gray = _load_image(img_path)
            sat = _rgb_to_hsv_saturation(rgb)
            rows.append(
                {
                    "row": int(idx),
                    "image_path": rel,
                    "sat_mean": float(np.mean(sat)),
                    "sat_std": float(np.std(sat)),
                    "lap_var": _laplacian_variance(gray),
                    "hf_ratio": _highfreq_ratio(gray, float(args.fft_highfreq_radius)),
                }
            )
        except Exception as e:
            rows.append({"row": int(idx), "image_path": rel, "error": str(e)})


    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows -> {out_path}")
if __name__ == "__main__":
    main()
