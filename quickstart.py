#!/usr/bin/env python
"""
Quick Start: DehazeSNN Inference using Hugging Face Hub

This script demonstrates how to load a pretrained DehazeSNN model from 
Hugging Face Hub and run inference on a local paired dataset under ./datasets.

Usage:
    python quickstart.py
    python quickstart.py --dataset reside6k --revision l-reside-6k
    python quickstart.py --dataset reside6k --revision m-reside-6k
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loader import PairLoader
from models.hub import DehazeSNNHub
from utils import chw_to_hwc, write_img


def _get_hf_hub_cache() -> str | None:
    """Best-effort: resolve Hugging Face hub cache directory."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return HF_HUB_CACHE
    except Exception:
        return None


def _clear_hf_locks(repo_id: str, cache_dir: str | None = None) -> int:
    """
    Remove huggingface_hub lock folders for a specific repo.
    This is useful when a stale process holds a lock and `from_pretrained` hangs.
    """
    import shutil

    hub_cache = cache_dir or _get_hf_hub_cache()
    if not hub_cache:
        return 0

    # huggingface_hub lock layout: <HF_HUB_CACHE>/.locks/<repo_type>--<org>--<name>/...
    safe_repo = repo_id.replace("/", "--")
    lock_dir = os.path.join(hub_cache, ".locks", f"models--{safe_repo}")
    if not os.path.exists(lock_dir):
        return 0

    try:
        shutil.rmtree(lock_dir, ignore_errors=True)
        return 1
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="DehazeSNN Quick Start Inference")
    parser.add_argument(
        "--dataset",
        type=str,
        default="reside6k",
        help="Dataset folder under ./datasets/ (e.g., indoor, outdoor, reside6k, rshaze)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="l-reside-6k",
        help="HuggingFace Hub revision (main, l-reside-its, m-reside-6k, etc.)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/",
        help="Path to datasets directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/quickstart/",
        help="Directory to save dehazed images",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU id to use",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache dir (useful to avoid cache lock issues on Windows).",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download model files from Hugging Face Hub.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Do not hit network; only use local Hugging Face cache.",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default=None,
        help="Override Hugging Face endpoint for this run (PowerShell alternative to $env:HF_ENDPOINT).",
    )
    parser.add_argument(
        "--clear_hf_locks",
        action="store_true",
        help="Delete Hugging Face hub lock folder for this repo (fixes 'stuck on WeakFileLock').",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    # Create output directory
    result_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print("DehazeSNN Quick Start")
    print("=" * 60)
    print(f"Loading model from: FengShaner/DehazeSNN (revision: {args.revision})")
    if os.environ.get("HF_ENDPOINT"):
        print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
    hf_cache = _get_hf_hub_cache()
    if args.hf_cache_dir:
        print(f"HF cache_dir override: {args.hf_cache_dir}")
    elif hf_cache:
        print(f"HF_HUB_CACHE: {hf_cache}")

    if args.clear_hf_locks:
        removed = _clear_hf_locks("FengShaner/DehazeSNN", cache_dir=args.hf_cache_dir)
        if removed:
            print("Cleared Hugging Face lock folder for FengShaner/DehazeSNN.")

    # Load model from Hugging Face Hub
    model = DehazeSNNHub.from_pretrained(
        "FengShaner/DehazeSNN",
        revision=args.revision,
        cache_dir=args.hf_cache_dir,
        force_download=args.force_download,
        local_files_only=args.local_files_only,
    )
    model.cuda().eval()

    print(f"Model loaded: variant={model.variant}, dataset={model.dataset}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {result_dir}")
    print("=" * 60)

    # Load test dataset
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset not found at {dataset_dir}")
        print("Please download datasets and place them under ./datasets/")
        return

    test_dataset = PairLoader(dataset_dir, "test", "test")
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, pin_memory=True)

    print(f"Found {len(test_dataset)} test images\n")

    # Run inference
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing"):
            input_img = batch["source"].cuda()
            target_img = batch["target"].cuda()
            filename = batch["filename"][0]

            # Forward pass
            output = model(input_img).clamp(-1, 1)

            # Convert from [-1, 1] to [0, 1]
            output_01 = output * 0.5 + 0.5
            target_01 = target_img * 0.5 + 0.5

            # Calculate PSNR
            mse = F.mse_loss(output_01, target_01)
            psnr = 10 * torch.log10(1.0 / mse).item()
            total_psnr += psnr

            # Save output image
            out_img = chw_to_hwc(output_01.squeeze(0).cpu().numpy())
            write_img(os.path.join(result_dir, filename), out_img)

    avg_psnr = total_psnr / len(test_dataset)

    print("\n" + "=" * 60)
    print(f"Results saved to: {result_dir}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()

