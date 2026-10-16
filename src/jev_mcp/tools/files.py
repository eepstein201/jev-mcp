"""The jev_read_file, jev_compact_context, and jev_scan_repo tools."""

import json
import os
from typing import Annotated, Any, Dict, List, Optional

from mcp.types import ToolAnnotations
from pydantic import Field

from jev_mcp import app
from jev_mcp.compaction import chunk_state, plan_scoring, score_chunk
from jev_mcp.app import json_errors, logger, mcp
from jev_mcp.provider import (
    EngineUnavailableError,
    NoulQuestion,
    escalation_failures,
    mark_degraded,
)
from jev_mcp.routing_provider import RoutingProvider

# Each chunk of a file costs an engine call, so the size of the file bounds the work one read can cause.
MAX_READ_CHARS = 1_048_576


@mcp.tool(
    name="jev_read_file",
    title="Read a file, filtered for relevance",
    description="File Context Filter: Reads a file locally and uses the Dual-Engine logit cascade to mathematically evaluate if the file is relevant to the current task. Prevents context bloat by blocking irrelevant files or filtering them down to only the relevant chunks. A file judged irrelevant to task_description comes back BLOCKED, so say what you are looking for there; this tool never returns content unfiltered and cannot list a directory.",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
@json_errors
def jev_read_file(
    file_path: str = Field(
        description="The absolute path to the file you want to read."
    ),
    task_description: str = Field(
        description="A description of the current task or goal. Used to filter the file content."
    ),
    filter_by_chunk: bool = Field(
        default=True,
        description="If True, the tool will chunk the file and return only the relevant chunks. If False, it evaluates the file as a whole and either returns the entire file or blocks it completely.",
    ),
    # TODO: Expose `confidence_threshold: float = 0.50` parameter here later to allow dynamic strictness
) -> str:
    import os
    import json
    from jev_mcp.security import denial_message, is_safe_path

    if not is_safe_path(file_path):
        return json.dumps({"status": "BLOCKED", "message": denial_message(file_path)})

    file_path = os.path.expanduser(file_path)

    if not os.path.exists(file_path):
        return json.dumps(
            {"status": "ERROR", "message": f"File not found: {file_path}"}
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # One character past the cap is enough to know the file is over it.
            file_content = f.read(MAX_READ_CHARS + 1)
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": f"Failed to read file: {e}"})

    if len(file_content) > MAX_READ_CHARS:
        return json.dumps(
            {
                "status": "ERROR",
                "message": f"File is over the {MAX_READ_CHARS}-character read limit.",
            }
        )

    if not file_content.strip():
        return json.dumps({"status": "ERROR", "message": "File is empty."})

    if filter_by_chunk:
        logger.info(f"AST Chunk-filtering {file_path} for task: {task_description}")
        from jev_mcp.chunker import SemanticChunker
        from jev_mcp.routing_provider import RoutingProvider

        chunker = SemanticChunker()
        file_chunks = chunker.get_semantic_chunks(file_path, file_content)

        provider = RoutingProvider()
        kept_chunks = []
        dropped = 0
        failed_escalations: List[str] = []

        for i, chunk in enumerate(file_chunks):
            q = NoulQuestion(
                key=f"chunk_{i}",
                prompt=f"Does this specific code block contain logic or variables highly relevant to the task: '{task_description}'?",
            )
            res = provider.evaluate_batch(chunk, [q])
            failed_escalations.extend(escalation_failures(res))
            true_prob = (
                res.get(f"chunk_{i}", {}).get("probabilities", {}).get("true", 0.0)
            )

            if true_prob >= 0.50:
                kept_chunks.append(chunk)
            else:
                dropped += 1

        if not kept_chunks:
            return json.dumps(
                mark_degraded(
                    {
                        "status": "BLOCKED",
                        "message": f"Jev evaluated {file_path} and found 0 relevant chunks for the task. The file has been blocked to save context window.",
                    },
                    failed_escalations,
                    keep_status=True,
                )
            )

        raw_content = "\n\n".join(kept_chunks)
        safe_content, was_truncated = app._safe_truncate(raw_content)

        msg = f"File {file_path} filtered successfully using AST chunking. Dropped {dropped} irrelevant chunks."
        if was_truncated:
            msg += " WARNING: Output was truncated to protect the context window."

        return json.dumps(
            mark_degraded(
                {"status": "SUCCESS", "message": msg, "content": safe_content},
                failed_escalations,
            )
        )

    else:
        logger.info(f"Whole-file filtering {file_path} for task: {task_description}")
        from jev_mcp.routing_provider import RoutingProvider

        provider = RoutingProvider()

        eval_content = file_content[:16000]

        q = NoulQuestion(
            key="file_eval",
            prompt=f"Is this file highly relevant to achieving the goal: '{task_description}'?",
        )
        res = provider.evaluate_batch(eval_content, [q])
        true_prob = res.get("file_eval", {}).get("probabilities", {}).get("true", 0.0)

        if true_prob >= 0.50:
            safe_content, was_truncated = app._safe_truncate(file_content)
            msg = (
                f"File passed relevance check ({round(true_prob * 100, 1)}% confident)."
            )
            if was_truncated:
                msg += " WARNING: Output was truncated to protect the context window. Use filter_by_chunk=True to extract only relevant parts."

            return json.dumps(
                mark_degraded(
                    {"status": "SUCCESS", "message": msg, "content": safe_content},
                    escalation_failures(res),
                )
            )
        else:
            return json.dumps(
                mark_degraded(
                    {
                        "status": "BLOCKED",
                        "message": f"Jev determined this file is irrelevant to the task ({round(true_prob * 100, 1)}% confident). Blocked to save context.",
                    },
                    escalation_failures(res),
                    keep_status=True,
                )
            )


@mcp.tool(
    name="jev_compact_context",
    title="Compact context to what is relevant",
    description="Context Compressor: Slices massive state contexts into chunks and uses Confidence-Gated Logits to keep only the verbatim chunks strictly relevant to the user's goal. Eliminates hallucination risk of generative summarization.",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def jev_compact_context(
    state: Dict[str, Any] = Field(
        description="The massive JSON state or text to compress."
    ),
    goal: str = Field(
        description="The user's current goal or the objective the context is needed for."
    ),
    confidence_threshold: float = Field(
        default=0.3,
        description="The relevance score (0-1) a chunk needs to be kept. The fast engine scores relevant chunks roughly 0.3-0.9 and irrelevant ones below 0.35, so values above about 0.4 discard relevant context. With goal_aware, the default is replaced by the goal type's own threshold.",
    ),
    goal_aware: Annotated[
        bool,
        Field(
            description="Experimental. Classify the goal (open items, find a fact, continue work) and score chunks with a question and threshold tuned for that type; any other goal keeps the generic question. Off by default."
        ),
    ] = False,
) -> str:
    import json

    logger.info("Starting Confidence-Gated Context Compaction...")

    chunks = chunk_state(state)
    if not chunks:
        return json.dumps({"status": "ERROR", "message": "State is empty."})

    kept_chunks = []
    total_chunks = len(chunks)
    dropped_chunks = 0
    chunk_scores: List[Optional[float]] = []
    prompt, confidence_threshold, goal_type = plan_scoring(
        app.provider, goal, confidence_threshold, goal_aware
    )

    # We will evaluate these in batches if needed, but for now sequentially
    # to avoid context overflow on the provider.
    for i, chunk in enumerate(chunks):
        try:
            score = score_chunk(app.provider, i, chunk, prompt)
            chunk_scores.append(round(score, 3))

            if score >= confidence_threshold:
                kept_chunks.append(chunk)
            else:
                dropped_chunks += 1

        except Exception as e:
            logger.warning(
                f"Failed to evaluate chunk {i}: {e}. Keeping chunk by default."
            )
            kept_chunks.append(chunk)
            chunk_scores.append(None)

    raw_text = "\n\n".join(kept_chunks)
    safe_text, was_truncated = app._safe_truncate(raw_text)
    compression_ratio = (
        (1.0 - (len(kept_chunks) / total_chunks)) * 100 if total_chunks > 0 else 0
    )

    status_msg = "COMPACTION_COMPLETE"
    if chunk_scores and all(score is None for score in chunk_scores):
        # Nothing was scored, so everything was kept: say so rather than report a compaction.
        status_msg = "ENGINE_UNAVAILABLE"
    if was_truncated:
        status_msg += " (TRUNCATED TO PROTECT CONTEXT WINDOW)"

    goal_fields = {"goal_type": goal_type} if goal_type is not None else {}
    return json.dumps(
        {
            "status": status_msg,
            "original_chunks": total_chunks,
            "kept_chunks": len(kept_chunks),
            "dropped_chunks": dropped_chunks,
            "compression_ratio": f"{compression_ratio:.1f}%",
            "chunk_scores": chunk_scores,
            **goal_fields,
            "compacted_state": safe_text,
        },
        indent=2,
    )


@mcp.tool(
    name="jev_scan_repo",
    title="Scan a repository for relevant code",
    description="Repository Scanner: Walks a codebase, uses Tree-sitter to break code into semantic blocks (functions/classes), and mathematically filters them using the Dual-Engine cascade to find only the code relevant to the task. Only blocks judged relevant to task_description are returned, and a scan that finds none is BLOCKED; it does not enumerate files, so it cannot be used to list a directory.",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
@json_errors
def jev_scan_repo(
    directory_path: str = Field(
        description="The absolute path to the directory to scan."
    ),
    task_description: str = Field(
        description="The goal or bug description used to filter the codebase."
    ),
) -> str:
    import os
    import json
    from jev_mcp.scanner import walk_repository, generate_repo_map
    from jev_mcp.chunker import SemanticChunker
    from jev_mcp.routing_provider import RoutingProvider
    from jev_mcp.security import denial_message, is_safe_path

    if not is_safe_path(directory_path):
        return json.dumps(
            {"status": "BLOCKED", "message": denial_message(directory_path)}
        )

    directory_path = os.path.expanduser(directory_path)

    if not os.path.isdir(directory_path):
        return json.dumps(
            {"status": "ERROR", "message": f"Directory not found: {directory_path}"}
        )

    logger.info(f"Scanning repository at {directory_path} for task: {task_description}")

    files = walk_repository(directory_path)
    if not files:
        return json.dumps(
            {"status": "BLOCKED", "message": "No valid files found in directory."}
        )

    target_files = files

    if len(files) > 20:
        logger.info(
            f"Repository is massive ({len(files)} files). Escalating to Smart Model for Surgical Pointing..."
        )
        repo_map = generate_repo_map(files, directory_path)

        prompt = f"Given the following task: '{task_description}'\n\nHere is the repository structure:\n{repo_map}\n\nRespond with ONLY a comma-separated list of the 5-10 file paths most likely to contain the code needed for this task. Do not include any other text."

        try:
            content = app._chat_completion(
                prompt,
                temperature=0.0,
                max_tokens=200,
                timeout=120,
            )

            suggested_paths = [
                p.strip() for p in content.replace("\n", ",").split(",") if p.strip()
            ]

            target_files = []
            for sp in suggested_paths:
                clean_sp = sp.lstrip("- ").strip()
                abs_path = os.path.abspath(os.path.join(directory_path, clean_sp))
                if abs_path in files:
                    target_files.append(abs_path)

            if not target_files:
                logger.warning(
                    "Smart model failed to select valid paths. Falling back to first 20 files."
                )
                target_files = files[:20]
        except Exception as e:
            logger.warning(
                f"Surgical pointing failed: {e}. Falling back to first 20 files."
            )
            target_files = files[:20]

    chunker = SemanticChunker()
    all_chunks = []

    for fpath in target_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                code = f.read()

            file_chunks = chunker.get_semantic_chunks(fpath, code)
            for c in file_chunks:
                rel_path = os.path.relpath(fpath, directory_path)
                contextualized_chunk = f"### FILE: {rel_path}\n{c}"
                all_chunks.append(contextualized_chunk)
        except Exception as e:
            logger.warning(f"Skipping {fpath}: could not read or chunk it: {e}")

    if not all_chunks:
        return json.dumps(
            {
                "status": "BLOCKED",
                "message": "Failed to extract chunks from repository.",
            }
        )

    provider = RoutingProvider()
    kept_chunks = []
    dropped = 0

    from concurrent.futures import ThreadPoolExecutor

    def eval_chunk(args):
        i, chunk = args
        q = NoulQuestion(
            key=f"chunk_{i}",
            prompt=f"Does this specific code block contain logic or variables highly relevant to the task: '{task_description}'?",
        )
        res = provider.evaluate_batch(chunk, [q])
        true_prob = res.get(f"chunk_{i}", {}).get("probabilities", {}).get("true", 0.0)
        return (chunk, true_prob >= 0.50, escalation_failures(res))

    failed_escalations: List[str] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for chunk, keep, failed in executor.map(eval_chunk, enumerate(all_chunks)):
            failed_escalations.extend(failed)
            if keep:
                kept_chunks.append(chunk)
            else:
                dropped += 1

    if not kept_chunks:
        return json.dumps(
            mark_degraded(
                {
                    "status": "BLOCKED",
                    "message": f"Scanned {len(target_files)} files and {len(all_chunks)} semantic chunks. Jev determined 0 chunks were relevant to the task. Blocked to save context window.",
                },
                failed_escalations,
                keep_status=True,
            )
        )

    raw_content = "\n\n".join(kept_chunks)
    safe_content, was_truncated = app._safe_truncate(raw_content)

    msg = f"Surgically scanned {len(target_files)} files. Extracted {len(kept_chunks)} highly relevant chunks. Dropped {dropped} noisy chunks."
    if was_truncated:
        msg += " WARNING: Output was truncated to protect the context window. Refine task_description or use jev_read_file."

    return json.dumps(
        mark_degraded(
            {"status": "SUCCESS", "message": msg, "content": safe_content},
            failed_escalations,
        )
    )
