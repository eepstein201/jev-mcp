"""Fuse, convert and swap steps behind the `train` tool.

The route was measured end to end on Qwen2.5-7B (mlx_lm 0.31.3, llama.cpp d81235049):
mlx_lm fuse --dequantize -> convert_hf_to_gguf.py (Q8_0) -> llama-quantize (Q4_K_M).
"""

import importlib.util
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_LLAMA_CPP_SRC = "~/.jev/llama.cpp-src"
CONVERTER_SCRIPT = "convert_hf_to_gguf.py"
CONVERTER_MODULES = ("torch", "transformers")
QUANT_TYPE = "Q4_K_M"
AGENT_GLOB = "com.jev.*.plist"

# mlx_lm.fuse rejects a hub id whose cached snapshot lacks README.md, which is how
# mlx_lm.lora leaves it; the snapshot directory itself loads fine.
_RESOLVE_MODEL = (
    "import os, sys; from huggingface_hub import try_to_load_from_cache as f; "
    "p = f(sys.argv[1], 'config.json'); print(os.path.dirname(p) if isinstance(p, str) else sys.argv[1])"
)

# Runs the converter with only its own directory and bundled gguf-py on the path.
_RUN_CONVERTER = (
    "import sys, runpy; src = sys.argv[1]; sys.path[:0] = [src, src + '/gguf-py']; "
    f"sys.argv = ['{CONVERTER_SCRIPT}', sys.argv[2], '--outfile', sys.argv[3], '--outtype', 'q8_0']; "
    f"runpy.run_path(src + '/{CONVERTER_SCRIPT}', run_name='__main__')"
)


def llama_cpp_src() -> Path:
    """Directory holding llama.cpp's converter at the commit of the installed build."""
    return Path(os.path.expanduser(os.getenv("JEV_LLAMA_CPP_SRC", "") or DEFAULT_LLAMA_CPP_SRC))


def converter_problem() -> Optional[str]:
    """Why the GGUF conversion cannot run, or None when it can."""
    src = llama_cpp_src()
    if not (src / CONVERTER_SCRIPT).is_file():
        return (
            f"{CONVERTER_SCRIPT} not found in {src}. Homebrew's llama.cpp does not ship it: extract it "
            "(with conversion/ and gguf-py/) from the llama.cpp commit shown by `llama-server --version` "
            "into that directory, or set JEV_LLAMA_CPP_SRC. See the README, 'Train LoRA'."
        )
    missing = [name for name in CONVERTER_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        return f"The converter needs {', '.join(missing)}. Run: pip install -e \".[engines]\""
    return None


def find_launchd_label(model_path: str) -> Optional[str]:
    """Label of the Jev launchd agent whose plist serves model_path, if any."""
    agents_dir = Path(os.path.expanduser("~/Library/LaunchAgents"))
    if not agents_dir.is_dir():
        return None
    for plist in sorted(agents_dir.glob(AGENT_GLOB)):
        try:
            if f">{model_path}<" in plist.read_text():
                return plist.stem
        except OSError as e:
            logger.warning(f"Could not read {plist}: {e}")
    return None


def build_fuse_script(
    python: str,
    model_name: str,
    adapter_path: str,
    converter_src: str,
    target_gguf_path: Optional[str],
    label: Optional[str],
) -> str:
    """Shell steps (run under `set -e`) that turn a trained adapter into a quantized GGUF."""
    q = shlex.quote
    fused_dir = os.path.join(adapter_path, "fused")
    q8_path = os.path.join(adapter_path, "fused-q8_0.gguf")
    out_path = os.path.join(adapter_path, f"fused-{QUANT_TYPE}.gguf")

    steps = [
        'echo "Training complete. Fusing the adapter into a dequantized copy of the base model..."',
        f"MODEL_DIR=$({q(python)} -c {q(_RESOLVE_MODEL)} {q(model_name)})",
        f'{q(python)} -m mlx_lm fuse --model "$MODEL_DIR" --adapter-path {q(adapter_path)} '
        f"--dequantize --save-path {q(fused_dir)}",
        'echo "Converting to GGUF..."',
        f"(cd {q(adapter_path)} && {q(python)} -I -c {q(_RUN_CONVERTER)} {q(converter_src)} {q(fused_dir)} {q(q8_path)})",
        f"rm -rf {q(fused_dir)}",
        f"llama-quantize --allow-requantize {q(q8_path)} {q(out_path)} {QUANT_TYPE}",
        f"rm -f {q(q8_path)}",
    ]

    if not target_gguf_path:
        steps.append(f'echo "Fused model saved to {out_path}. No target_gguf_path given, so nothing was replaced."')
        return "\n".join(steps) + "\n"

    backup = f"{target_gguf_path}.bak"
    steps += [
        # Keep the first backup: it is the original, not an earlier fine-tune.
        f"if [ -f {q(target_gguf_path)} ] && [ ! -f {q(backup)} ]; then",
        f"    mv {q(target_gguf_path)} {q(backup)}",
        f'    echo "Backed up the original model to {backup}"',
        "fi",
        f"mv {q(out_path)} {q(target_gguf_path)}",
    ]
    if label:
        steps += [
            f"launchctl kickstart -k gui/$(id -u)/{q(label)}",
            f'echo "Replaced the model and restarted {label}."',
        ]
    else:
        steps.append('echo "Replaced the model. No launchd agent serves it, so nothing was restarted."')
    return "\n".join(steps) + "\n"


def restart_agent(label: str) -> Optional[str]:
    """Restart one launchd agent in place. Returns an error message, or None on success."""
    result = subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return str(result.stderr).strip() or f"launchctl exited with {result.returncode}"
    return None


def restore_model(target_gguf_path: str) -> dict[str, Any]:
    """Put `<target>.bak` back in place of the fine-tuned model and restart its agent."""
    target = os.path.expanduser(target_gguf_path)
    backup = f"{target}.bak"
    if not os.path.isfile(backup):
        return {"status": "ERROR", "message": f"No backup to restore: {backup} does not exist."}

    os.replace(backup, target)
    label = find_launchd_label(target)
    error = restart_agent(label) if label else None
    result: dict[str, Any] = {
        "status": "RESTORED",
        "model": target,
        "restarted_agent": label if label and error is None else None,
    }
    if error:
        logger.error(f"Restored {target} but could not restart {label}: {error}")
        return {**result, "restart_error": error}
    return result
