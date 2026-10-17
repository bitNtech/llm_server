"""Download a Hugging Face model into the local cache before the engine starts.

Prints the total weight size in bytes on stdout (for run.sh's memory fit check). Run with the
engine virtualenv's interpreter (it ships huggingface_hub). Downloading up front gives a clear
progress/failure signal instead of a silent multi-minute engine startup, and skips repo files
vLLM never reads (evaluation dumps, images, GGUF/ONNX exports...).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

PATTERNS = ["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt", "*.tiktoken", "tokenizer*", "*.py"]
WEIGHTS_FALLBACK = ["*.bin", "*.pt"]


def weights_bytes(path: str) -> int:
    """Total size of the weight files in a snapshot (symlinks into the blob store are followed)."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith((".safetensors", ".bin", ".pt")):
                total += os.path.getsize(os.path.join(root, name))
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--download-dir", default="")
    parser.add_argument("--code", action="store_true", help="model needs trust_remote_code (*.py)")
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    revision = args.revision or None
    cache_dir = args.download_dir or None
    patterns = list(PATTERNS)
    try:
        info = HfApi().model_info(args.model, revision=revision)
        files = [s.rfilename for s in info.siblings or []]
        if not any(f.endswith(".safetensors") for f in files):
            patterns += WEIGHTS_FALLBACK
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        print(f"cannot access {args.model}: {exc.__class__.__name__}. "
              "Check the model id, accept its license on huggingface.co and set HF_TOKEN.", file=sys.stderr)
        return 1
    except Exception as exc:  # offline: fall back to whatever is cached
        print(f"warning: could not query the Hub ({exc.__class__.__name__}); using the local cache", file=sys.stderr)
        try:
            path = snapshot_download(args.model, revision=revision, cache_dir=cache_dir, local_files_only=True)
        except Exception:
            print(f"model {args.model} is not cached and the Hub is unreachable", file=sys.stderr)
            return 1
        print(weights_bytes(path))
        return 0

    started = time.monotonic()
    path = snapshot_download(args.model, revision=revision, cache_dir=cache_dir, allow_patterns=patterns)
    size = weights_bytes(path)
    print(f"{args.model} ready at {path} ({size / 1024**3:.1f} GiB, {time.monotonic() - started:.0f}s)",
          file=sys.stderr)
    print(size)  # stdout: weight bytes, consumed by run.sh's fit check
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
