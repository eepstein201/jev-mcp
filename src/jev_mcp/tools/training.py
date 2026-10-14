"""The jev_train_lora tool."""

import json
import os
import re
import shlex
import subprocess
import sys
from typing import Literal, Optional

from mcp.types import ToolAnnotations
from pydantic import Field

from jev_mcp import app
from jev_mcp.app import logger, mcp


def _target_refusal(target_gguf_path: str) -> Optional[str]:
    """Why a train run may not replace target_gguf_path under the active profile, if so.

    No active profile means no check. Otherwise the target must be the profile's smart
    model. A smart model with reasoning = "off" is refused outright: its training rows
    would need the empty think block, and this tool does not build rows (they come
    from the caller's dataset) nor know the model's chat template, so it cannot add or
    verify that block."""
    profile = app.active_engine_profile()
    if profile is None:
        return None
    smart = profile.smart
    name = os.path.basename(target_gguf_path)
    if name != smart.filename:
        return (
            f"Refusing to replace {name}: the active profile {profile.name!r} serves "
            f"{smart.filename} as its smart model. Pass that file as target_gguf_path, "
            f"or switch profiles with `make start PROFILE=...`."
        )
    if smart.reasoning == "off":
        return (
            f'Refusing to replace {name}: it runs with reasoning = "off", so every '
            f"training row must carry the empty think block its chat template expects. "
            f"jev_train_lora does not build rows or know that template, so it cannot "
            f"add or verify the block (fine-tuning this model family is not supported "
            f"yet). Train without target_gguf_path to get a standalone adapter instead."
        )
    return None


@mcp.tool(
    name="jev_train_lora",
    title="Fine-tune or restore a local model",
    description="Trains a local MLX LoRA adapter on your Apple Silicon GPU. Options: fuse=True (default) merges the adapter into a Q4_K_M .gguf file (llama.cpp's converter is fetched automatically for the installed llama-server commit), and if target_gguf_path is provided, replaces that model (keeping the original as <target>.bak) and restarts the launchd agent serving it. fuse=False keeps the standalone MLX adapter. iters sets the number of training iterations (default 500). action='restore' with target_gguf_path puts the .bak back and restarts that agent.",
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=False,
        open_world_hint=True,
    ),
)
def jev_train_lora(
    dataset_path: str = "",
    model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
    engine: str = "mlx",
    fuse: bool = True,
    target_gguf_path: Optional[str] = None,
    iters: int = 500,
    action: Literal["train", "restore"] = "train",
) -> str:
    """
    Launches mlx_lm.lora fine-tuning in a detached background process
    and returns the live pid plus adapter/log locations.
    """
    import importlib.util
    import time
    from jev_mcp import lora_training
    from jev_mcp.security import denial_message, is_safe_path

    try:
        if target_gguf_path:
            if not is_safe_path(target_gguf_path):
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "message": denial_message(target_gguf_path),
                    }
                )
            if not target_gguf_path.endswith(".gguf"):
                return json.dumps(
                    {
                        "status": "ERROR",
                        "message": f"target_gguf_path must be a .gguf file: {target_gguf_path}",
                    }
                )
            target_gguf_path = os.path.abspath(os.path.expanduser(target_gguf_path))

        if action == "restore":
            if not target_gguf_path:
                return json.dumps(
                    {
                        "status": "ERROR",
                        "message": "restore needs target_gguf_path (the model whose .bak should be put back).",
                    }
                )
            return json.dumps(lora_training.restore_model(target_gguf_path), indent=2)

        if target_gguf_path:
            refusal = _target_refusal(target_gguf_path)
            if refusal:
                return json.dumps({"status": "ERROR", "message": refusal})

        if not is_safe_path(dataset_path):
            return json.dumps(
                {"status": "BLOCKED", "message": denial_message(dataset_path)}
            )

        dataset_path = os.path.expanduser(dataset_path)

        if not os.path.isfile(dataset_path) or not dataset_path.endswith(".jsonl"):
            return json.dumps(
                {
                    "status": "ERROR",
                    "message": f"Dataset not found or not a .jsonl file: {dataset_path}",
                }
            )

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", model_name):
            return json.dumps(
                {
                    "status": "ERROR",
                    "message": f"Invalid model_name (expected a HuggingFace id like 'org/model-name'): {model_name}",
                }
            )

        if not isinstance(iters, int) or iters < 1:
            return json.dumps(
                {
                    "status": "ERROR",
                    "message": f"iters must be a positive integer, got: {iters}",
                }
            )

        # Reuse a single training dir under ~/.jev (no mkdtemp litter) and link
        # the dataset in under the name mlx_lm.lora expects. Always restage: a
        # train.jsonl left there by an earlier run may be stale or dangling.
        run_root = os.path.expanduser("~/.jev/training")
        os.makedirs(run_root, exist_ok=True)
        link = os.path.join(run_root, "train.jsonl")
        if os.path.abspath(dataset_path) != link:
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.abspath(dataset_path), link)

        # Adapters and logs get a fresh per-run directory instead of the CWD.
        run_id = time.strftime("%Y%m%d-%H%M%S")
        restart_label: Optional[str] = None
        adapter_path = os.path.expanduser(f"~/.jev/adapters/{run_id}")
        os.makedirs(adapter_path, exist_ok=True)
        log_path = os.path.join(adapter_path, "train.log")

        if engine == "llama.cpp":
            # Experimental pure GGUF training using llama-finetune
            if not model_name.endswith(".gguf"):
                return json.dumps(
                    {
                        "status": "ERROR",
                        "message": "llama.cpp engine requires a .gguf model_name path.",
                    }
                )

            cmd = [
                "llama-finetune",
                "--model",
                model_name,
                "--train-data",
                dataset_path,
                "--lora-out",
                os.path.join(adapter_path, "adapter.gguf"),
                "--epochs",
                "2",
            ]
        else:
            if importlib.util.find_spec("mlx_lm") is None:
                return json.dumps(
                    {
                        "status": "NOT_AVAILABLE",
                        "message": "mlx_lm is not installed in this environment. Run: pip install mlx-lm",
                    }
                )

            # The default approach: MLX for training, then convert to gguf via llama.cpp
            run_script_path = os.path.join(adapter_path, "run_train.sh")
            q = shlex.quote
            if fuse:
                problem = lora_training.ensure_converter()
                if problem:
                    return json.dumps(
                        {
                            "status": "NOT_AVAILABLE",
                            "message": f"{problem} Or pass fuse=False to train the adapter only.",
                        }
                    )
                restart_label = (
                    lora_training.find_launchd_label(target_gguf_path)
                    if target_gguf_path
                    else None
                )
                fuse_cmd = lora_training.build_fuse_script(
                    python=sys.executable,
                    model_name=model_name,
                    adapter_path=adapter_path,
                    converter_src=str(lora_training.llama_cpp_src()),
                    target_gguf_path=target_gguf_path,
                    label=restart_label,
                )
            else:
                fuse_cmd = (
                    "\necho "
                    + q(
                        f"Training complete. Adapter saved to {adapter_path} (fusion skipped)."
                    )
                    + "\n"
                )

            with open(run_script_path, "w") as f:
                f.write(f"""#!/bin/bash
set -e
echo "Starting MLX LoRA training..."
{q(sys.executable)} -m mlx_lm.lora \
    --model {q(model_name)} \
    --data {q(run_root)} \
    --adapter-path {q(adapter_path)} \
    --train \
    --iters {iters}
{fuse_cmd}
""")
            os.chmod(
                run_script_path, 0o600
            )  # launched as `bash <path>`, so no exec bit
            cmd = ["bash", run_script_path]

        with open(log_path, "a") as log_f:
            proc = subprocess.Popen(
                cmd, stdout=log_f, stderr=log_f, start_new_session=True
            )

        return json.dumps(
            {
                "status": "TRAINING_STARTED",
                "dataset": dataset_path,
                "data_dir": run_root,
                "adapter_path": adapter_path,
                "base_model": model_name,
                "pid": proc.pid,
                "log": log_path,
                "restart_agent": restart_label,
                "converter_commit": lora_training.converter_commit()
                if fuse and engine != "llama.cpp"
                else None,
                "command": " ".join(cmd),
                "note": f"Training runs in a detached background process. Monitor with: tail -f {log_path}",
            },
            indent=2,
        )
    except Exception as e:
        logger.error(f"jev_train_lora failed: {e}")
        return json.dumps({"status": "ERROR", "message": str(e)})
