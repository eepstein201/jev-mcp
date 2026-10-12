"""The evaluate, explain-decision, and generate-data tools."""

import json
import subprocess
from typing import Any, Dict, List

from pydantic import Field

from jev_mcp import app
from jev_mcp.app import logger, mcp
from jev_mcp.provider import ChoiceQuestion, NoulQuestion, QuestionType, ScoreQuestion


def _error(message: str) -> str:
    return json.dumps({"status": "ERROR", "message": message})


@mcp.tool(
    name="evaluate",
    description=(
        "Evaluate a complex state against a batch of polymorphic System-1 questions.\n"
        "WARNING: The 'state' payload MUST be under 2048 tokens. If exceeded, QFE middleware will attempt to compress it."
    ),
)
def jev_evaluate_batch(
    state: Dict[str, Any] = Field(
        description="JSON object representing the context state."
    ),
    questions: List[QuestionType] = Field(
        description="Array of questions (noul, choice, score)."
    ),
    auto_apply_fixes: bool = Field(
        default=False,
        description="If True, auto-evaluates the LLM's fixed prompts. If False, returns REQUIRES_APPROVAL with suggestions.",
    ),
) -> str:
    """Evaluate questions and return JSON results."""
    logger.info(f"Received batch of {len(questions)} questions.")

    token_count = app.provider.check_token_limit(state)
    was_compressed = False

    if token_count > app.provider.max_tokens:
        logger.info(
            f"State token count {token_count} exceeds limit {app.provider.max_tokens}. Triggering Query-Focused Extraction (QFE)..."
        )
        try:
            q_prompts = []
            for q in questions:
                q_prompts.append(q.prompt)

            qfe_prompt = (
                "You are a strict data extraction middleware. "
                "The user is about to evaluate the following questions against a massive context state. "
                "Your job is to read the state and EXTRACT ONLY the exact sentences, facts, and entities that are necessary to answer the questions. "
                "Do NOT answer the questions. Do NOT write a generic summary. "
                "Just output a dense, condensed version of the original context retaining all relevant evidence.\n\n"
                f"QUESTIONS TO BE ANSWERED:\n{json.dumps(q_prompts, indent=2)}"
            )

            compressed_state = {
                "qfe_extracted_context": app._chat_completion(
                    f"MASSIVE STATE:\n{json.dumps(state)}",
                    system=qfe_prompt,
                    temperature=0.1,
                    max_tokens=1500,
                )
            }
            logger.info("QFE Compression successful. Re-checking token count...")

            new_token_count = app.provider.check_token_limit(compressed_state)
            if new_token_count > app.provider.max_tokens:
                raise Exception("QFE output still exceeded max tokens.")

            state = compressed_state
            was_compressed = True
            logger.info(
                f"State compressed from {token_count} to {new_token_count} tokens."
            )

        except Exception as e:
            logger.error(f"QFE Failed: {str(e)}")
            return _error(
                f"CRITICAL SYSTEM ERROR: 'state' payload was {token_count} tokens and QFE compression failed: {str(e)}"
            )

    def run_linter(qs):
        findings = []

        # --- MODEL-BASED LINTING (Fuzzy Semantic Check) ---
        # Uses SemIf itself to detect if the user's questions are generative!
        lint_qs: List[QuestionType] = []
        for i, q in enumerate(qs):
            q_prompt = q.get("prompt", "") if isinstance(q, dict) else q.prompt
            lint_qs.append(
                NoulQuestion(
                    key=f"q_{i}",
                    prompt=f"Does the user's question require the AI to generate text, write a summary, extract a list, or provide an open-ended answer? Question: '{q_prompt}'",
                )
            )

        try:
            # Evaluate the questions against a tiny dummy state
            lint_state = {"task": "Determine generative intent."}
            lint_results = app.provider.evaluate_batch(lint_state, lint_qs)

            for i, q in enumerate(qs):
                res = lint_results.get(f"q_{i}", {})
                q_key = q.key
                if res.get("noul", 0.0) > 0.85:
                    findings.append(
                        {
                            "question_key": q_key,
                            "severity": "ERROR",
                            "code": "SEMANTIC_GENERATIVE_INTENT",
                            "message": "Model-based linter detected generative or extraction intent.",
                            "suggestion": "Fast decision models cannot generate text. Reframe as a boolean, multiple choice, or Likert scale.",
                        }
                    )
        except Exception as e:
            logger.warning(f"Model-based linting failed: {e}")

        # --- STATIC STRUCTURAL LINTING ---
        for i, q in enumerate(qs):
            is_score = isinstance(q, ScoreQuestion)
            if isinstance(q, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
                opts = q.to_llm_options()
            else:
                opts = []

            prompt = q.prompt
            key = q.key

            report = app.linter.lint(state, prompt, opts, is_score=is_score)
            for f in report.findings:
                # Deduplicate generative errors if model-based linter already caught it
                if f.code == "GENERATIVE_INTENT_DETECTED" and any(
                    x["code"] == "SEMANTIC_GENERATIVE_INTENT"
                    and x["question_key"] == key
                    for x in findings
                ):
                    continue

                findings.append(
                    {
                        "question_key": key,
                        "severity": f.severity,
                        "code": f.code,
                        "message": f.message,
                        "suggestion": f.suggestion,
                    }
                )
        return findings

    all_findings = run_linter(questions)
    was_autofixed = False
    autofix_details = []

    if any(f["severity"] == "ERROR" for f in all_findings):
        logger.info("Linter errors detected. Hitting fast background autofixer...")

        q_dicts: List[Dict[str, Any]] = []
        for q in questions:
            if isinstance(q, NoulQuestion):
                q_dicts.append({"type": "noul", "key": q.key, "prompt": q.prompt})
            elif isinstance(q, ChoiceQuestion):
                q_dicts.append(
                    {
                        "type": "choice",
                        "key": q.key,
                        "prompt": q.prompt,
                        "options": q.options,
                    }
                )
            elif isinstance(q, ScoreQuestion):
                q_dicts.append(
                    {
                        "type": "score",
                        "key": q.key,
                        "prompt": q.prompt,
                        "labels": q.labels,
                    }
                )

        try:
            fixed_qs_raw = app.call_fast_autofixer(state, q_dicts, all_findings)

            if len(fixed_qs_raw) != len(questions):
                raise ValueError(
                    f"Autofixer returned {len(fixed_qs_raw)} questions for {len(questions)}."
                )

            fixed_qs: List[QuestionType] = []
            for original_q, fq in zip(questions, fixed_qs_raw):
                orig_prompt = original_q.prompt
                autofix_details.append({"original": orig_prompt, "fixed": fq["prompt"]})

                # Results stay under the caller's key, whatever the autofixer called the question.
                key = original_q.key
                if fq["type"] == "noul":
                    fixed_qs.append(NoulQuestion(prompt=fq["prompt"], key=key))
                elif fq["type"] == "choice":
                    fixed_qs.append(
                        ChoiceQuestion(
                            prompt=fq["prompt"],
                            key=key,
                            options=fq.get("options", []),
                        )
                    )
                elif fq["type"] == "score":
                    fixed_qs.append(
                        ScoreQuestion(
                            prompt=fq["prompt"],
                            key=key,
                            labels=fq.get("labels", []),
                        )
                    )
                else:
                    raise ValueError(
                        f"Autofixer returned an unknown question type: {fq['type']!r}"
                    )

            new_findings = run_linter(fixed_qs)
            if any(f["severity"] == "ERROR" for f in new_findings):
                raise Exception("Auto-fixed questions failed validation.")

            # Interactive Approval Gate
            if not auto_apply_fixes:
                return json.dumps(
                    {
                        "status": "REQUIRES_APPROVAL",
                        "message": "Your questions violate System 1 framing rules. Please review the AI's suggested fixes below. If you approve, resubmit with the fixed questions.",
                        "original_errors": all_findings,
                        "suggested_fixes": fixed_qs_raw,
                        "transformations": autofix_details,
                    },
                    indent=2,
                )

            questions = fixed_qs
            all_findings = new_findings
            was_autofixed = True

        except Exception as e:
            logger.warning(f"Rejecting payload. Autofix failed: {str(e)}")
            return json.dumps(
                {
                    "status": "REJECTED_BY_LINTER",
                    "reason": "Questions contained critical anti-patterns.",
                    "original_findings": all_findings,
                },
                indent=2,
            )

    try:
        results = app.provider.evaluate_batch(state, questions)

        response = {
            "model": getattr(app.provider, "current_model_id", "unknown_model"),
            "data": results,
        }
        if all_findings:
            response["status"] = "SUCCESS_WITH_WARNINGS"
            response["warnings"] = all_findings
        else:
            response["status"] = "SUCCESS"

        if was_autofixed:
            response["was_autofixed"] = True
            response["autofix_transformations"] = autofix_details

        if was_compressed:
            response["was_compressed"] = True
            response["qfe_state"] = state

        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"Engine failure: {str(e)}")
        return _error(f"INTERNAL ENGINE ERROR: {str(e)}")


@mcp.tool(
    name="explain-decision",
    description="Evidence Highlighting: Generates a Post-Hoc Rationale for a decision by extracting the exact, verbatim sentence from the state that proves the decision.",
)
def jev_explain_decision(
    state: Dict[str, Any] = Field(
        description="The context state used for the decision."
    ),
    question: QuestionType = Field(description="The question that was evaluated."),
    decision: str = Field(
        description="The decision output returned by the model (e.g., 'True', 'False', or a Choice label)."
    ),
) -> str:
    import json
    from jev_mcp.security import sanitize_payload

    logger.info("Starting Evidence Extraction...")

    q_prompt = question.prompt
    if len(q_prompt) > 1000:
        q_prompt = q_prompt[:1000]
    q_prompt = sanitize_payload(q_prompt)

    state_str = json.dumps(state) if not isinstance(state, str) else state
    if len(state_str) > 16000:
        state_str = state_str[:16000]
    state_str = sanitize_payload(state_str)

    decision_str = str(decision)
    if len(decision_str) > 100:
        decision_str = decision_str[:100]
    decision_str = sanitize_payload(decision_str)

    system_prompt = (
        "You are an Evidence Extraction engine for a classification model. "
        "The model was asked a question about a state, and it returned a specific decision. "
        "Your task is to find the exact, verbatim sentence or phrase in the state that proves this decision. "
        "RULES:\n"
        "1. You must COPY AND PASTE a direct quote from the STATE text. It must be an EXACT SUBSTRING. Do not rephrase or summarize.\n"
        "2. If no explicit evidence exists (i.e., it was a false positive or inferred), explain why in the 'reasoning' field and leave 'evidence_sentence' null.\n"
        "3. Output ONLY a valid JSON object.\n"
        'Format: {"evidence_sentence": "...", "reasoning": "..."}'
    )

    user_prompt = f"STATE:\n{state_str[:3000]}...\n\nQUESTION:\n{q_prompt}\n\nDECISION:\n{decision_str}\n\nReturn ONLY the JSON object."

    try:
        data = app._chat_completion_json(
            user=user_prompt,
            system=system_prompt,
            temperature=0.1,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        return json.dumps(
            {
                "status": "EVIDENCE_EXTRACTED",
                "decision": decision,
                "evidence": data.get("evidence_sentence", None),
                "reasoning": data.get("reasoning", ""),
            },
            indent=2,
        )
    except Exception as e:
        return _error(f"Evidence Extraction Failed: {str(e)}")


@mcp.tool(
    name="generate-data",
    description="Dataset Generator: Automatically generates a Golden Dataset using the local 7B model if available, otherwise returns instructions for the primary LLM to generate it.",
)
def jev_generate_synthetic_dataset(
    question: QuestionType = Field(
        description="The target question you need a dataset for."
    ),
    num_cases: int = Field(
        default=10, description="Number of synthetic edge cases to generate."
    ),
) -> str:
    import json
    import random
    from jev_mcp.security import sanitize_payload

    # 6. MCP Tool Boundary Defenses (DoS & OOM Prevention)
    num_cases = min(max(num_cases, 1), 50)

    q_prompt = question.prompt
    if len(q_prompt) > 1000:
        q_prompt = q_prompt[:1000]
    q_prompt = sanitize_payload(q_prompt)

    logger.info(f"Checking if 7B model is loaded for local dataset generation...")

    expected_type = "boolean (true/false)"
    if isinstance(question, ChoiceQuestion):
        expected_type = "one of the choice options"
    elif isinstance(question, ScoreQuestion):
        expected_type = "one of the score labels"

    # Define the fallback prompt for Claude
    prompt_to_primary_llm = f"""
INSTRUCTION TO PRIMARY LLM:
You must now generate a Golden Dataset of {num_cases} synthetic edge cases to test the following logit-based decision model question:
"{q_prompt}"

RULES FOR GENERATION:
1. Half the cases should be clear, obvious expected outcomes (True Positives and True Negatives).
2. Half the cases MUST be 'Malicious False Positives'—tricky edge cases designed to confuse the model by using target keywords in a negated, hypothetical, past-tense, or irrelevant context.
3. The 'expected' field must be a strict {expected_type} representing the actual ground truth.
4. Output your response as a raw, valid JSON array containing exactly {num_cases} objects. Do not wrap it in markdown block quotes.

REQUIRED JSON SCHEMA:
[
  {{
    "state": {{"text": "The highly realistic user context or transcript..."}},
    "expected": <{expected_type}>
  }}
]

Once you have generated this JSON array, you must immediately pass it into the `calibrate` tool to mathematically evaluate the model's performance on your synthetic data.
"""

    is_local_model = False
    try:
        out = subprocess.check_output(["pgrep", "-fl", "llama-server"]).decode()
        if "llama-server" in out:
            is_local_model = True
    except Exception:
        pass

    if is_local_model:
        logger.info(
            f"Local model detected! Attempting local generation of {num_cases} cases at Temp 0.2..."
        )
        dataset_accumulator: List[Any] = []
        batch_size = 10

        system_prompt = (
            "You are an elite QA Engineer designing test cases for an AI decision model. "
            f"The model will be evaluated on this question: '{q_prompt}'\n"
            f"Generate cases to test this question. "
            "RULES:\n"
            "1. Half the cases should be clear, obvious expected outcomes.\n"
            "2. Half the cases MUST be 'Malicious False Positives'—tricky edge cases designed to confuse the model by using target keywords in a negated, hypothetical, past-tense, or irrelevant context.\n"
            f"3. The 'expected' field must be a {expected_type}.\n"
            "4. Output ONLY a valid JSON array of objects.\n"
            'Format: [{"state": {"text": "..."}, "expected": ...}]'
        )

        try:
            while len(dataset_accumulator) < num_cases:
                remaining = num_cases - len(dataset_accumulator)
                current_batch = min(batch_size, remaining)

                seed = random.randint(1000, 9999)
                batch_data = app._chat_completion_json(
                    user=f"Generate exactly {current_batch} cases now. Ensure these cases are highly unique. [Random Seed: {seed}]",
                    system=system_prompt,
                    temperature=0.2,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                if isinstance(batch_data, dict) and "dataset" in batch_data:
                    batch_data = batch_data["dataset"]

                if not isinstance(batch_data, list):
                    raise ValueError("Local 7B Model did not return a valid JSON list.")
                if not batch_data:
                    # An empty batch makes no progress; asking again could loop forever.
                    raise ValueError("Local 7B Model returned no cases.")
                dataset_accumulator.extend(batch_data)

            dataset_accumulator = dataset_accumulator[:num_cases]
            logger.info("Local dataset generation successful!")
            return json.dumps(
                {
                    "status": "DATASET_GENERATED",
                    "size": len(dataset_accumulator),
                    "dataset": dataset_accumulator,
                },
                indent=2,
            )

        except Exception as e:
            logger.warning(
                f"Local generation failed: {e}. Falling back to Frontier Model."
            )
            # Fall through to returning prompt

    else:
        logger.info(
            "Local model NOT detected. Delegating generation to Frontier Model."
        )

    return prompt_to_primary_llm
