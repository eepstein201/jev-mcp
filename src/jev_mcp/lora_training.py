"""Fuse, convert and swap steps behind the `train` tool.

The route was measured end to end on Qwen2.5-7B (mlx_lm 0.31.3, llama.cpp d81235049):
mlx_lm fuse --dequantize -> convert_hf_to_gguf.py (Q8_0) -> llama-quantize (Q4_K_M).
"""

import importlib.util
import logging
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_LLAMA_CPP_SRC = "~/.jev/llama.cpp-src"
CONVERTER_SCRIPT = "convert_hf_to_gguf.py"
CONVERTER_MODULES = ("torch", "transformers")
CONVERTER_PARTS = (CONVERTER_SCRIPT, "conversion", "gguf-py")
COMMIT_MARKER = ".jev-llama-commit"
ARCHIVE_URL = "https://github.com/ggml-org/llama.cpp/archive/{commit}.tar.gz"
DOWNLOAD_TIMEOUT_SECONDS = 120
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
            f"{CONVERTER_SCRIPT} not found in {src}, and it could not be fetched automatically because "
            "`llama-server --version` did not report a commit. Extract it (with conversion/ and gguf-py/) "
            "from the matching llama.cpp commit into that directory, or set JEV_LLAMA_CPP_SRC. "
            "See the README, 'Train LoRA'."
        )
    missing = [name for name in CONVERTER_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        return f"The converter needs {', '.join(missing)}. Run: pip install -e \".[engines]\""
    return None


def installed_llama_commit() -> Optional[str]:
    """Commit the installed llama-server was built from, or None when it cannot be read."""
    try:
        out = subprocess.run(["llama-server", "--version"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(f"Could not run llama-server --version: {e}")
        return None
    match = re.search(r"commit ([0-9a-f]{7,40})\b", f"{out.stdout}\n{out.stderr}")
    return match.group(1) if match else None


def converter_commit() -> Optional[str]:
    """Commit the converter in llama_cpp_src() was fetched from, if this module fetched it."""
    try:
        return (llama_cpp_src() / COMMIT_MARKER).read_text().strip() or None
    except OSError:
        return None


def fetch_converter(commit: str, dest: Path) -> None:
    """Replace dest with the converter files from llama.cpp at exactly this commit."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".llama.cpp-src-", dir=dest.parent))
    try:
        with urllib.request.urlopen(ARCHIVE_URL.format(commit=commit), timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    # Paths look like "llama.cpp-<sha>/conversion/qwen.py"; keep only the converter's parts.
                    parts = PurePosixPath(member.name).parts[1:]
                    if not parts or parts[0] not in CONVERTER_PARTS or ".." in parts:
                        continue
                    member.name = "/".join(parts)
                    archive.extract(member, staging, filter="data")
        if not (staging / CONVERTER_SCRIPT).is_file():
            raise ValueError(f"{CONVERTER_SCRIPT} is not in the llama.cpp archive for {commit}")
        (staging / COMMIT_MARKER).write_text(commit + "\n")
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(staging, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def ensure_converter() -> Optional[str]:
    """Bring the converter in step with the installed llama.cpp, then report any remaining problem.

    Only the default directory is managed; one named by JEV_LLAMA_CPP_SRC belongs to the user.
    """
    if os.getenv("JEV_LLAMA_CPP_SRC", ""):
        return converter_problem()

    commit = installed_llama_commit()
    src = llama_cpp_src()
    has_converter = (src / CONVERTER_SCRIPT).is_file()
    if commit and (not has_converter or converter_commit() != commit):
        try:
            logger.info(f"Fetching the llama.cpp converter for commit {commit} into {src}")
            fetch_converter(commit, src)
        except Exception as e:
            if not has_converter:
                return f"Could not download the llama.cpp converter for commit {commit}: {e}"
            logger.warning(f"Could not update the converter to llama.cpp {commit} ({e}); using the one in {src}")
    return converter_problem()


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
