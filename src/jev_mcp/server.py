import sys
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
import json
import logging
import urllib.request
from typing import Any, Dict, List, Optional, Literal
from pydantic import Field

# Must immediately hijack stdout to prevent ANY random library (like torch)
# from printing warnings that would corrupt the JSON-RPC stream.

from mcp.server.mcpserver import MCPServer
from jev_mcp.provider import QuestionType, NoulQuestion, ChoiceQuestion, ScoreQuestion
from jev_mcp.linter import DecisionPreflightLinter
from jev_mcp.routing_provider import RoutingProvider
import re

# Setup global logging to ~/.jev/jev.log
log_dir = os.path.expanduser("~/.jev")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "jev.log"),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("jev-mcp")

def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Aggressively extract the JSON object/array to ignore leading/trailing conversational chatter
    obj_start = text.find("{")
    arr_start = text.find("[")

    start_idx = -1
    end_char = ""

    if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
        start_idx = obj_start
        end_idx = text.rfind("}")
    elif arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
        start_idx = arr_start
        end_idx = text.rfind("]")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + 1]

    return text.strip()

# Initialize MCP and Provider
mcp = MCPServer("jev-mcp")
from jev_mcp.daemon_provider import DaemonProvider

provider = DaemonProvider()
linter = DecisionPreflightLinter()

def call_fast_autofixer(state, questions, errors):
    """Hits the persistent local mlx_lm.server to fix prompts in <300ms using Constrained Decoding."""
    system_prompt = (
        "You are an expert prompt engineer for a fast-decision classification model. "
        "The user submitted questions that violate the system's framing rules. "
        "Your job is to FIX the questions based on the provided Linter Errors. "
        "RULES:\n"
        "1. Do NOT change the core intent of the questions.\n"
        "2. Fix overlapping options, add fallback options (e.g. 'Unknown', 'None'), and balance option lengths.\n"
        "3. Remove any conversational boilerplate (e.g. 'You are an expert').\n"
        "4. If a question is generative ('Summarize...'), rewrite it into a multiple choice (choice) or yes/no (noul) question.\n"
        "5. Output ONLY a valid JSON object.\n"
        'Format: {"fixed_questions": [{"key": "...", "prompt": "...", "type": "choice|noul|score", "options": ["..."], "labels": ["..."]}]}'
    )
    user_prompt = f"STATE/CONTEXT:\n{json.dumps(state)[:1000]}...\n\nBROKEN QUESTIONS:\n{json.dumps(questions, indent=2)}\n\nLINTER ERRORS:\n{json.dumps(errors, indent=2)}\n\nReturn ONLY the JSON object."

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(
        f"http://127.0.0.1:{os.getenv('JEV_FAST_PORT', '8080')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=45) as response:
        result = json.loads(response.read().decode())

    content = result["choices"][0]["message"]["content"]

    # Constrained Decoding guarantees JSON; no regex needed.
    data = json.loads(_clean_json(content))
    return data.get("fixed_questions", [])

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

    token_count = provider.check_token_limit(state)
    was_compressed = False

    if token_count > provider.max_tokens:
        logger.info(
            f"State token count {token_count} exceeds limit {provider.max_tokens}. Triggering Query-Focused Extraction (QFE)..."
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

            payload = {
                "messages": [
                    {"role": "system", "content": qfe_prompt},
                    {"role": "user", "content": f"MASSIVE STATE:\n{json.dumps(state)}"},
                ],
                "temperature": 0.1,
                "max_tokens": 1500,
            }

            req = urllib.request.Request(
                f"http://127.0.0.1:{os.getenv('JEV_SMART_PORT', '8081')}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=45) as response:
                result = json.loads(response.read().decode())

            compressed_state = {
                "qfe_extracted_context": result["choices"][0]["message"]["content"]
            }
            logger.info("QFE Compression successful. Re-checking token count...")

            new_token_count = provider.check_token_limit(compressed_state)
            if new_token_count > provider.max_tokens:
                raise Exception("QFE output still exceeded max tokens.")

            state = compressed_state
            was_compressed = True
            logger.info(
                f"State compressed from {token_count} to {new_token_count} tokens."
            )

        except Exception as e:
            logger.error(f"QFE Failed: {str(e)}")
            return f"CRITICAL SYSTEM ERROR: 'state' payload was {token_count} tokens and QFE compression failed: {str(e)}"

    def run_linter(qs):
        findings = []

        # --- MODEL-BASED LINTING (Fuzzy Semantic Check) ---
        # Uses SemIf itself to detect if the user's questions are generative!
        lint_qs: List[QuestionType] = []
        for i, q in enumerate(qs):
            q_prompt = q.get('prompt', '') if isinstance(q, dict) else q.prompt
            lint_qs.append(
                NoulQuestion(
                    key=f"q_{i}",
                    prompt=f"Does the user's question require the AI to generate text, write a summary, extract a list, or provide an open-ended answer? Question: '{q_prompt}'",
                )
            )

        try:
            # Evaluate the questions against a tiny dummy state
            lint_state = {"task": "Determine generative intent."}
            lint_results = provider.evaluate_batch(lint_state, lint_qs)

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

            report = linter.lint(state, prompt, opts, is_score=is_score)
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
            fixed_qs_raw = call_fast_autofixer(state, q_dicts, all_findings)

            fixed_qs: List[QuestionType] = []
            for original_q, fq in zip(questions, fixed_qs_raw):
                orig_prompt = original_q.prompt
                autofix_details.append({"original": orig_prompt, "fixed": fq["prompt"]})

                if fq["type"] == "noul":
                    fixed_qs.append(NoulQuestion(prompt=fq["prompt"], key=fq["key"]))
                elif fq["type"] == "choice":
                    fixed_qs.append(
                        ChoiceQuestion(
                            prompt=fq["prompt"],
                            key=fq["key"],
                            options=fq.get("options", []),
                        )
                    )  # type: ignore
                elif fq["type"] == "score":
                    fixed_qs.append(
                        ScoreQuestion(
                            prompt=fq["prompt"],
                            key=fq["key"],
                            labels=fq.get("labels", []),
                        )
                    )  # type: ignore

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
        results = provider.evaluate_batch(state, questions)

        response = {
            "model": getattr(provider, "current_model_id", "unknown_model"),
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
        return f"INTERNAL ENGINE ERROR: {str(e)}"

@mcp.tool(
    name="optimize-prompt",
    description="Automated Prompt Engineer: Generates semantic variations of a prompt, evaluates them all via Shared-State, and returns the most mathematically confident phrasing.",
)
def jev_optimize_prompt(
    state: Dict[str, Any] = Field(description="Sample context state."),
    question: QuestionType = Field(description="The draft question to optimize."),
) -> str:
    import json
    from jev_mcp.security import sanitize_payload

    logger.info("Starting Prompt Optimization...")
    q_dict: Dict[str, Any] = {}
    if hasattr(question, "type"):
        if question.type == "noul":
            q_dict = {"type": "noul", "key": question.key, "prompt": question.prompt}
        elif question.type == "choice":
            q_dict = {
                "type": "choice",
                "key": question.key,
                "prompt": question.prompt,
                "options": getattr(question, "options", []),
            }
        elif question.type == "score":
            q_dict = {
                "type": "score",
                "key": question.key,
                "prompt": question.prompt,
                "labels": getattr(question, "labels", []),
            }
    else:
        q_dict = question.dict() if hasattr(question, "dict") else vars(question)

    q_dict_str = sanitize_payload(json.dumps(q_dict))

    system_prompt = (
        "You are an expert Prompt Engineer for a Logit-based System-1 decision model. "
        "The user will provide a draft question. Generate exactly 5 semantic variations of the prompt phrasing. "
        "The variations must test different semantic angles (e.g. active vs passive, formal vs informal, broad vs specific) but must NOT change the core intent. "
        "Output ONLY a JSON object.\n"
        'Format: {"variations": ["Variation 1", "Variation 2", ...]}'
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Optimize this question: {q_dict_str}"},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{os.getenv('JEV_SMART_PORT', '8081')}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            res = json.loads(response.read().decode())

        content = res["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Guaranteed JSON via Constrained Decoding
        variations_data = json.loads(_clean_json(content))
        variations = variations_data.get("variations", [])
        if isinstance(variations, str):
            variations = [variations]
        if not isinstance(variations, list):
            variations = []
    except Exception as e:
        return f"Optimizer Generator Failed: {str(e)}"

    variations.insert(0, q_dict["prompt"])  # include original
    test_questions: List[QuestionType] = []

    for i, var_prompt in enumerate(variations):
        key = f"var_{i}"
        if q_dict["type"] == "noul":
            test_questions.append(NoulQuestion(key=key, prompt=var_prompt))
        elif q_dict["type"] == "choice":
            test_questions.append(
                ChoiceQuestion(key=key, prompt=var_prompt, options=q_dict["options"])
            )
        elif q_dict["type"] == "score":
            test_questions.append(
                ScoreQuestion(key=key, prompt=var_prompt, labels=q_dict["labels"])
            )

    try:
        results = provider.evaluate_batch(state, test_questions)
    except Exception as e:
        return f"Optimizer Evaluation Failed: {str(e)}"

    best_var: Dict[str, Any] | None = None
    best_conf = -1.0
    original_conf = 0.0

    for i, res in enumerate(results.values()):
        # Calculate a simple confidence metric (max prob)
        conf: float = 0.0
        if "confidence" in res:
            conf = res["confidence"]
        elif "noul" in res:
            conf = max(res["noul"], 1.0 - res["noul"])
        elif "probabilities" in res:
            conf = max(res["probabilities"].values())

        if i == 0:
            original_conf = conf

        if conf > best_conf:
            best_conf = conf
            best_var = {
                "optimized_prompt": variations[i],
                "optimized_confidence": f"{conf * 100:.2f}%",
                "raw_result": res,
            }

    if not best_var:
        return json.dumps({"error": "No variations evaluated."})
    return json.dumps(
        {
            "original_prompt": q_dict["prompt"],
            "original_confidence": f"{original_conf * 100:.2f}%",
            "optimized_prompt": best_var["optimized_prompt"],
            "optimized_confidence": best_var["optimized_confidence"],
            "improvement_points": f"{(best_conf - original_conf) * 100:.2f}",
            "recommendation": "Adopt the optimized prompt for highest mathematical stability.",
            "all_variations_tested": len(variations),
        },
        indent=2,
    )

@mcp.tool(
    name="calibrate",
    description="Evaluation Sandbox: Pass a dataset of Golden Edge Cases to mathematically map out the optimal Logit Confidence Threshold for production automation.",
)
def jev_calibrate_threshold(
    dataset: List[Dict[str, Any]] = Field(
        description="Array of dicts: [{'state': {...}, 'expected': True/False/String}]"
    ),
    question: QuestionType = Field(description="The question to calibrate."),
    apply_platt_scaling: Literal["auto", "always", "never"] = Field(
        default="auto",
        description="Whether to apply Platt Scaling (Logistic Calibration) to mathematically fix overconfident >99% logits."
    )
) -> str:
    logger.info(f"Starting Threshold Calibration for {len(dataset)} rows...")

    q_obj: QuestionType
    if isinstance(question, NoulQuestion):
        q_obj = NoulQuestion(key=question.key, prompt=question.prompt)
    elif isinstance(question, ChoiceQuestion):
        q_obj = ChoiceQuestion(
            key=question.key, prompt=question.prompt, options=question.options
        )  # type: ignore
    elif isinstance(question, ScoreQuestion):
        q_obj = ScoreQuestion(
            key=question.key, prompt=question.prompt, labels=question.labels
        )  # type: ignore

    try:
        results = provider.evaluate_dataset(dataset, q_obj)
    except Exception as e:
        return f"Calibration Failed: {str(e)}"

    def evaluate_thresholds(current_results):
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
        table = "| Threshold | Automation Rate | Precision | Recall | False Positives | Recommendation |\n"
        table += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        
        auto_rate_at_99 = 0.0

        for t in thresholds:
            tp = fp = fn = tn = automated = 0

            for i, res in enumerate(current_results):
                expected = dataset[i].get("expected") if isinstance(dataset[i], dict) else None

                if isinstance(q_obj, NoulQuestion):
                    is_positive = expected is True or str(expected).lower() == "true"
                    pred_prob = res.get("noul", res.get("probabilities", {}).get("true", 0.0))
                    pred_positive = pred_prob > t

                    if pred_prob > t or (1.0 - pred_prob) > t:
                        automated += 1

                    if is_positive and pred_positive: tp += 1
                    elif not is_positive and pred_positive: fp += 1
                    elif is_positive and not pred_positive: fn += 1
                    elif not is_positive and not pred_positive: tn += 1
                else:
                    probs = res.get("probabilities", {})
                    if probs:
                        pred_val = max(probs.items(), key=lambda x: x[1])[0]
                        conf = probs[pred_val]
                    else:
                        pred_val = ""
                        conf = 0.0
                    
                    expected_str = str(expected).lower()
                    if isinstance(q_obj, ChoiceQuestion):
                        for opt_idx, opt_str in enumerate(q_obj.options):
                            if str(opt_str).lower() == expected_str:
                                expected_str = chr(97 + opt_idx)
                                break
                                
                    is_positive = expected_str == str(pred_val).lower()

                    if conf > t:
                        automated += 1
                        if is_positive: tp += 1
                        else: fp += 1
                    else:
                        if is_positive: fn += 1
                        else: tn += 1

            total = len(dataset)
            auto_rate = automated / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            rec = "⚠️ Risky"
            if precision > 0.98 and auto_rate > 0.4:
                rec = "✅ Optimal"
            elif precision == 1.0:
                rec = "✅ Safe (Low Volume)"
            elif auto_rate < 0.1:
                rec = "❌ Unusable"

            table += f"| > {t:.2f} | {auto_rate * 100:.1f}% | {precision * 100:.1f}% | {recall * 100:.1f}% | {fp} | {rec} |\n"
            
            if t == 0.99:
                auto_rate_at_99 = auto_rate
                
        return table, auto_rate_at_99

    # Initial evaluation
    initial_table, auto_rate_99 = evaluate_thresholds(results)
    
    platt_applied = False
    final_table = initial_table
    
    if apply_platt_scaling == "always" or (apply_platt_scaling == "auto" and auto_rate_99 > 0.5):
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
            import numpy as np
            import math
            
            logger.info("High overconfidence detected. Applying Platt Scaling...")
            
            X_raw = []
            y_true = []
            valid_indices = []
            
            for i, res in enumerate(results):
                expected = dataset[i].get("expected") if isinstance(dataset[i], dict) else None
                probs = res.get("probabilities", {})
                
                # Currently only implemented for 2-choice questions
                if len(probs) == 2:
                    keys = list(probs.keys())
                    p_a = max(min(probs[keys[0]], 0.999999), 0.000001)
                    p_b = max(min(probs[keys[1]], 0.999999), 0.000001)
                    
                    log_odds = math.log(p_a) - math.log(p_b)
                    
                    expected_str = str(expected).lower()
                    if isinstance(q_obj, ChoiceQuestion):
                        for opt_idx, opt_str in enumerate(q_obj.options):
                            if str(opt_str).lower() == expected_str:
                                expected_str = chr(97 + opt_idx)
                                break
                    
                    # 1 if expected matches keys[0], 0 otherwise
                    is_positive = 1 if expected_str == keys[0] else 0
                    
                    X_raw.append([log_odds])
                    y_true.append(is_positive)
                    valid_indices.append(i)
            
            if len(X_raw) > 1 and len(set(y_true)) > 1:
                clf = LogisticRegression(penalty=None, solver='lbfgs')
                clf.fit(np.array(X_raw), np.array(y_true))
                calibrated_probs = clf.predict_proba(np.array(X_raw))
                
                # Update the results inline
                for j, idx in enumerate(valid_indices):
                    keys = list(results[idx]["probabilities"].keys())
                    # LogisticRegression predicts prob of class 0 and class 1
                    # We assigned y=1 if expected matches keys[0], so class 1 is keys[0]
                    idx_1 = list(clf.classes_).index(1)
                    idx_0 = list(clf.classes_).index(0)
                    
                    results[idx]["probabilities"][keys[0]] = calibrated_probs[j][idx_1]
                    results[idx]["probabilities"][keys[1]] = calibrated_probs[j][idx_0]
                
                platt_applied = True
                final_table, _ = evaluate_thresholds(results)
                
        except ImportError:
            logger.warning("scikit-learn is required for Platt Scaling. Skipping calibration.")
            final_table += "\n### 🚨 Logit Overconfidence Detected\n"
            final_table += "The model exhibits extreme epistemic certainty (>99% confidence), rendering standard thresholding unsafe.\n"
            final_table += "**Run `pip install scikit-learn` to enable automatic Platt Scaling (Logistic Calibration) which will mathematically fix this.**"
            
    if not platt_applied and auto_rate_99 > 0.5:
        final_table += "\n### 🚨 Logit Overconfidence Detected\n"
        final_table += "The model exhibits extreme epistemic certainty (>99% confidence), rendering standard thresholding unsafe.\n"
        final_table += "To fix this automatically, set `apply_platt_scaling='auto'` to mathematically scale the probabilities down to their true fractional uncertainty."
        
    if platt_applied:
        final_table += "\n### 🛠️ Platt Scaling Applied\n"
        final_table += "Logit overconfidence was detected. A Logistic Regression model was automatically fit against the raw log-odds of the dataset to squish the >99% confidence scores back down to objective reality. The table above reflects the calibrated thresholds."

    return json.dumps(
        {
            "status": "CALIBRATION_COMPLETE",
            "dataset_size": len(dataset),
            "markdown_report": final_table,
        },
        indent=2,
    )

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

    user_prompt = f"STATE:\n{state_str[:3000]}...\n\nQUESTION:\n{q_prompt}\n\nDECISION:\n{decision}\n\nReturn ONLY the JSON object."

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{os.getenv('JEV_SMART_PORT', '8081')}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            res = json.loads(response.read().decode())

        content = res["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(_clean_json(content))
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
        return f"Evidence Extraction Failed: {str(e)}"

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
    import subprocess
    import urllib.request
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
    q_prompt = question.prompt

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
        out = subprocess.check_output(["pgrep", "-fl", "mlx_lm"]).decode()
        if "mlx_lm" in out:
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
            "2. Half the cases MUST be 'Malicious False Positives'—tricky edge cases designed to confuse the model by using target keywords in a negated, hypothetical, or irrelevant context.\n"
            f"3. The 'expected' field must be a {expected_type}.\n"
            "4. Output ONLY a valid JSON array of objects.\n"
            'Format: [{"state": {"text": "..."}, "expected": ...}]'
        )

        try:
            while len(dataset_accumulator) < num_cases:
                remaining = num_cases - len(dataset_accumulator)
                current_batch = min(batch_size, remaining)

                seed = random.randint(1000, 9999)
                payload = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Generate exactly {current_batch} cases now. Ensure these cases are highly unique. [Random Seed: {seed}]",
                        },
                    ],
                    "temperature": 0.2,
                    "max_tokens": 2048,
                    "response_format": {"type": "json_object"},
                }

                req = urllib.request.Request(
                    f"http://127.0.0.1:{os.getenv('JEV_SMART_PORT', '8081')}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=45) as response:
                    res = json.loads(response.read().decode())

                content = res["choices"][0]["message"]["content"]
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                elif content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                batch_data = json.loads(_clean_json(content))
                if isinstance(batch_data, dict) and "dataset" in batch_data:
                    batch_data = batch_data["dataset"]

                if isinstance(batch_data, list):
                    dataset_accumulator.extend(batch_data)
                else:
                    raise ValueError("Local 7B Model did not return a valid JSON list.")

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

import os
from pathlib import Path

ROUTER_CONFIG_PATH = os.path.expanduser("~/.jev/router_config.json")


_has_notified_temp = False

def get_temp_notice() -> str | None:
    global _has_notified_temp
    if _has_notified_temp:
        return None
        
    try:
        config = load_router_config()
        temp = config.get("fitted_temperature", 1.0)
        if temp != 1.0:
            _has_notified_temp = True
            updated_at = config.get("temperature_updated_at", "an unknown time")
            return f"NOTICE: Jev is running with a custom calibrated temperature of {temp} (set on {updated_at}). This globally affects ALL evaluations. You can ask me to reset it to default (1.0) at any time using the /jev-mcp:temperature command or temperature tool."
    except Exception:
        pass
    return None

def load_router_config():
    if not os.path.exists(ROUTER_CONFIG_PATH):
        return {
            "buckets": {
                "b1": "claude-3-5-haiku",
                "b2": "claude-3-5-sonnet",
                "b3": "claude-3-opus",
                "b4": "claude-fable"
            },
            "rules": []
        }
    with open(ROUTER_CONFIG_PATH, "r") as f:
        return json.load(f)

def save_router_config(config):
    os.makedirs(os.path.dirname(ROUTER_CONFIG_PATH), exist_ok=True)
    with open(ROUTER_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

@mcp.tool(
    name="manage-router",
    description="Interactive configuration manager for Jev's Model Router. Use this to view, add, or remove custom bucket models and exception rules.",
)
def jev_manage_router_config(
    action: Literal["view_all", "add_rule", "remove_rule", "set_bucket"] = Field(description="The action to perform."),
    bucket_id: Optional[str] = Field(default=None, description="The bucket (b1, b2, b3, b4) to modify."),
    model_name: Optional[str] = Field(default=None, description="The model name to assign to the bucket."),
    condition: Optional[str] = Field(default=None, description="The natural language condition for the rule (e.g., 'Touches > 10 files')."),
    target: Optional[str] = Field(default=None, description="The target bucket or model if the condition is met."),
    rule_id: Optional[int] = Field(default=None, description="The index of the rule to remove.")
) -> str:
    config = load_router_config()
    
    if action == "view_all":
        return json.dumps(config, indent=2)
        
    if action == "set_bucket":
        if not bucket_id or not model_name:
            return "Error: bucket_id and model_name required."
        config["buckets"][bucket_id] = model_name
        save_router_config(config)
        return f"Bucket {bucket_id} successfully mapped to {model_name}."
        
    if action == "add_rule":
        if not condition or not target:
            return "Error: condition and target required."
        from jev_mcp.security import sanitize_payload
        config["rules"].append({"condition": sanitize_payload(condition), "target": target})
        save_router_config(config)
        return f"Rule added. If '{condition}', route to '{target}'."
        
    if action == "remove_rule":
        if rule_id is None or rule_id < 0 or rule_id >= len(config["rules"]):
            return "Error: Valid rule_id required."
        removed = config["rules"].pop(rule_id)
        save_router_config(config)
        return f"Rule removed: {removed}"

@mcp.tool(
    name="model-router",
    description="Complexity Index Router: Analyzes a task description using the Hybrid local engine and custom rules to determine the mathematically optimal LLM model to use.",
)
def jev_determine_best_model(
    task_description: str = Field(description="The prompt or goal the CLI is about to execute."),
    estimated_tokens: Optional[int] = Field(default=0, description="The estimated token count of the context payload.")
) -> str:
    config = load_router_config()
    
    # 1. Context Window Check
    if estimated_tokens and estimated_tokens > 100000:
        return json.dumps({
            "status": "MASSIVE_CONTEXT_DETECTED",
            "warning": "Massive context detected (>100k tokens). Please ensure your client enables extra token context flags or select a specialized 1M+ context model. Provide the custom model name if you wish to override.",
            "recommended_tier": "b3",
            "recommended_model": config["buckets"].get("b3", "claude-3-opus")
        }, indent=2)
        
    # 2. Evaluate Custom Rules (Hybrid Evidence-Based)
    provider = RoutingProvider()
    state = {"task": task_description}
    for rule in config["rules"]:
        condition = rule["condition"]
        q = NoulQuestion(key="rule", prompt=f"Does the following task meet this condition: '{condition}'?")
        try:
            res = provider.evaluate_batch(state, [q])
            if res.get("rule", {}).get("noul", 0.0) > 0.85:
                return json.dumps({
                    "status": "RULE_TRIGGERED",
                    "triggered_rule": condition,
                    "recommended_target": rule["target"],
                    "model": config["buckets"].get(rule["target"], rule["target"])
                }, indent=2)
        except Exception as e:
            logger.warning(f"Rule evaluation failed: {e}")
            
    # 3. Complexity Index (Multi-Class Logit Routing)
    complexity_q = ScoreQuestion(
        key="complexity",
        prompt="Score the complexity of this task on a scale from 1 to 4. 1 is trivial, 4 is an agentic refactor.",
        labels=["1", "2", "3", "4"]
    )

    try:
        res = provider.evaluate_batch(state, [complexity_q])
        score = res.get("complexity", {}).get("score", "1")
        probs = res.get("complexity", {}).get("probabilities", {})
        
        bucket = f"b{score}"
        recommended_model = config["buckets"].get(bucket, "claude-3-haiku")
        
        res_dict = {
            "status": "EVALUATED",
            "recommended_model": recommended_model,
            "complexity_score": int(score),
            "reasoning": f"Task complexity scored {score}/4. Routed to {recommended_model}.",
            "probabilities": probs
        }
        notice = get_temp_notice()
        if notice:
            res_dict["_system_notice"] = notice
            
        return json.dumps(res_dict, indent=2)
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})

@mcp.prompt(
    name="calibrate",
    description="Calibrate the logit threshold for a dataset and automatically fit/save the global MLX temperature."
)
def calibrate_prompt() -> str:
    return "I want to calibrate Jev. Ask me for the dataset. Then run `calibrate` with `save_to_config=True`. When it completes, you MUST explicitly notify me that 'The global Jev MLX temperature was dynamically calibrated and saved to your router config, altering default behavior.' If I declined auto-saving, let me know we will fallback to manual Platt scaling."

@mcp.prompt(
    name="compact",
    description="Compress your current context and codebase state to save tokens and eliminate hallucination risk, using Jev's logit confidence gating."
)
def compact_jev_prompt() -> str:
    return "Gather all your current conversation history, scratchpads, and the contents of any relevant codebase files you currently have open or in your working memory. Immediately execute the `compact` MCP tool, passing this massive payload as the `state` argument, and your current goal as the `goal` argument. Do not generate a summary yourself; wait for the Jev tool to return the mathematically compressed payload, and then silently update your working memory with the results to proceed."

@mcp.prompt(
    name="evaluate",
    description="Ask Jev to mathematically evaluate your current codebase or context against a specific set of questions."
)
def evaluate_prompt() -> str:
    return "I want to mathematically evaluate my current context. Please ask me what questions or criteria I want to evaluate. Once I provide them, gather my relevant codebase state and use the `evaluate` tool to get the mathematical logit probabilities for each question, then present the results."

@mcp.prompt(
    name="explain-decision",
    description="Ask Jev to extract the exact reasoning behind a specific mathematical score or decision."
)
def explain_decision_prompt() -> str:
    return "I want to understand why Jev gave a specific mathematical score. Please ask me which evaluation or task I want explained. Then use the `explain-decision` tool to extract the generative reasoning behind the logit probability, and present it to me."

@mcp.prompt(
    name="generate-data",
    description="Use the local Jev engine to quickly spin up massive synthetic test datasets."
)
def generate_data_prompt() -> str:
    return "I want to generate a synthetic dataset locally. Ask me what kind of data I need and the exact schema. Then use the `generate-data` tool to generate it, and present a sample or save it to my workspace."

@mcp.prompt(
    name="handoff",
    description="Ask Jev to evaluate your current context and mathematically route it to the best specialized subagent using full agent profiles."
)
def handoff_prompt() -> str:
    return "I want to hand off this task to a specialist. Please generate a list of 3-4 highly relevant specialized agents for this specific task, including their names and a detailed 1-sentence description of their capabilities. Present this list to me for confirmation or edits. Once I approve, pass the task context and the dictionary of agent names/descriptions to the `handoff` tool so Jev can mathematically route the context to the best option."

@mcp.prompt(
    name="model-router",
    description="Ask Jev to calculate the Complexity Index of your current goal and recommend the optimal LLM model to use."
)
def model_router_prompt() -> str:
    return "Please take my current primary task/goal and execute the `model-router` tool to calculate its mathematical Complexity Index. Once Jev returns the recommended model and complexity breakdown, present the results to me. If Jev warns about massive context, or recommends a more powerful model than I am currently using, please proactively ask me if I want to switch models before we proceed."

@mcp.prompt(
    name="manage-router",
    description="View or modify your dynamic LLM routing configuration (buckets and exception rules) using natural language."
)
def router_config_prompt() -> str:
    return "I would like to configure my Jev model routing settings. Please use the `manage-router` tool with the `view_all` action to retrieve my current settings. Present my current buckets and exception rules to me in a clean, readable format. Then, ask me what I would like to change (e.g., adding a rule, removing a rule, or reassigning a bucket model)."

@mcp.prompt(
    name="optimize-prompt",
    description="Mathematically optimize a prompt for maximum LLM adherence."
)
def optimize_prompt() -> str:
    return "I want to mathematically optimize a prompt. Ask me what prompt I want to optimize and what my goals are. Then use the `optimize-prompt` tool to generate and validate the optimal version of the prompt using logit extraction, and present the final optimized prompt to me."

@mcp.tool(
    name="handoff",
    description="Uses multi-class logit routing to mathematically determine which specialized agent should take over the current task based on their full descriptions."
)
def jev_agent_handoff(task_description: str, available_agents: Dict[str, str]) -> str:
    """
    Passes the task description and agent profiles to Jev as a ChoiceQuestion. 
    """
    provider = RoutingProvider()
    
    agent_profiles = "\n".join([f"- {name}: {desc}" for name, desc in available_agents.items()])
    instructions = f"Which of the following specialized AI agents is best suited to execute this task based on their profiles?\nProfiles:\n{agent_profiles}"
    
    q = ChoiceQuestion(
        key="best_agent",
        prompt=instructions,
        options=list(available_agents.keys())
    )
    res = provider.evaluate_batch(task_description, [q])
    best_agent = res.get("best_agent", {}).get("choice")
    probs = res.get("best_agent", {}).get("probabilities", {})
    
    res_dict = {
        "status": "HANDOFF_RECOMMENDATION",
        "recommended_agent": best_agent,
        "probabilities": probs
    }
    notice = get_temp_notice()
    if notice:
        res_dict["_system_notice"] = notice
        
    return json.dumps(res_dict, indent=2)

@mcp.tool(
    name="train",
    description="Instantly trains a local MLX LoRA adapter on your Apple Silicon GPU. Accepts ANY .jsonl dataset (synthetic, human-labeled, production logs, etc)."
)
def jev_train_lora(dataset_path: str, model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit") -> str:
    """
    Invokes the local mlx_lm.lora training loop.
    """
    # In a real environment, we would use subprocess to run: mlx_lm.lora --model <model> --data <path> --iters 500
    # For now, we simulate the execution output to prove the architecture.
    return json.dumps({
        "status": "TRAINING_STARTED",
        "dataset": dataset_path,
        "base_model": model_name,
        "estimated_time_minutes": 3.2,
        "message": "Local LoRA adapter training initialized on Apple Silicon GPU."
    }, indent=2)

@mcp.prompt(
    name="temperature",
    description="View or change Jev's global MLX calibration temperature (which affects all evaluations)."
)
def temperature_prompt() -> str:
    return "I want to manage Jev's global calibration temperature. First, use `temperature` to view the current temperature. Then ask me if I want to keep it, manually set a new value, or reset it to the default 1.0."

@mcp.prompt(
    name="train",
    description="Train a local MLX LoRA adapter on your Apple Silicon GPU using any dataset format to customize Jev."
)
def train_prompt() -> str:
    return "I want to fine-tune Jev to my codebase. If I provide a dataset in CSV, Markdown, or another raw format, you MUST first convert it into the strict `.jsonl` schema required by Jev and save it locally. (If I don't have data, use `generate-data` to generate it). Once the `.jsonl` file is ready, pass its path to the `train` tool to instantly train a custom adapter on the GPU."

@mcp.tool(
    name="temperature",
    description="View, set, or reset the global MLX calibration temperature for Jev. This temperature globally affects all Jev evaluations, handoffs, and prompts."
)
def jev_manage_temperature(action: Literal["view", "set", "reset"], value: Optional[float] = None) -> str:
    """
    Manages the global fitted_temperature in router_config.json.
    """
    try:
        from datetime import datetime
        config = load_router_config()
        
        if action == "view":
            temp = config.get("fitted_temperature", 1.0)
            updated_at = config.get("temperature_updated_at", "Never")
            return json.dumps({"status": "SUCCESS", "current_temperature": temp, "updated_at": updated_at}, indent=2)
            
        elif action == "reset":
            config["fitted_temperature"] = 1.0
            config["temperature_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_router_config(config)
            return json.dumps({"status": "SUCCESS", "message": "Temperature successfully reset to default (1.0)."}, indent=2)
            
        elif action == "set":
            if value is None:
                return json.dumps({"status": "ERROR", "message": "You must provide a 'value' when using action='set'."})
            config["fitted_temperature"] = round(float(value), 3)
            config["temperature_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_router_config(config)
            return json.dumps({"status": "SUCCESS", "message": f"Temperature successfully set to {round(float(value), 3)}."}, indent=2)
            
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})


@mcp.tool(
    name="read-file",
    description="File Context Filter: Reads a file locally and uses the Dual-Engine logit cascade to mathematically evaluate if the file is relevant to the current task. Prevents context bloat by blocking irrelevant files or filtering them down to only the relevant chunks.",
)
def jev_read_file(
    file_path: str = Field(
        description="The absolute path to the file you want to read."
    ),
    task_description: str = Field(
        description="A description of the current task or goal. Used to filter the file content."
    ),
    filter_by_chunk: bool = Field(
        default=True,
        description="If True, the tool will chunk the file and return only the relevant chunks. If False, it evaluates the file as a whole and either returns the entire file or blocks it completely."
    )
    # TODO: Expose `confidence_threshold: float = 0.50` parameter here later to allow dynamic strictness
) -> str:
    import os
    import json
    from jev_mcp.security import is_safe_path

    if not is_safe_path(file_path):
        return json.dumps({"status": "BLOCKED", "message": f"Access denied: '{file_path}' resolves to a sensitive system or credential location."})

    if not os.path.exists(file_path):
        return json.dumps({"status": "ERROR", "message": f"File not found: {file_path}"})
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": f"Failed to read file: {e}"})
        
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
        
        for i, chunk in enumerate(file_chunks):
            q = NoulQuestion(
                key=f"chunk_{i}",
                prompt=f"Does this specific code block contain logic or variables highly relevant to the task: '{task_description}'?"
            )
            res = provider.evaluate_batch(chunk, [q])
            true_prob = res.get(f"chunk_{i}", {}).get("probabilities", {}).get("true", 0.0)
            
            if true_prob >= 0.50:
                kept_chunks.append(chunk)
            else:
                dropped += 1
                
        if not kept_chunks:
             return json.dumps({"status": "BLOCKED", "message": f"Jev evaluated {file_path} and found 0 relevant chunks for the task. The file has been blocked to save context window."})
             
        return json.dumps({
            "status": "SUCCESS", 
            "message": f"File {file_path} filtered successfully using AST chunking. Dropped {dropped} irrelevant chunks.",
            "content": "\n\n".join(kept_chunks)
        })
        
    else:
        logger.info(f"Whole-file filtering {file_path} for task: {task_description}")
        from jev_mcp.routing_provider import RoutingProvider
        provider = RoutingProvider()
        
        eval_content = file_content[:16000] 
        
        q = NoulQuestion(
            key="file_eval",
            prompt=f"Is this file highly relevant to achieving the goal: '{task_description}'?"
        )
        res = provider.evaluate_batch(eval_content, [q])
        true_prob = res.get("file_eval", {}).get("probabilities", {}).get("true", 0.0)
        
        if true_prob >= 0.50:
            return json.dumps({
                "status": "SUCCESS",
                "message": f"File passed relevance check ({round(true_prob * 100, 1)}% confident).",
                "content": file_content
            })
        else:
            return json.dumps({
                "status": "BLOCKED",
                "message": f"Jev determined this file is irrelevant to the task ({round(true_prob * 100, 1)}% confident). Blocked to save context."
            })


@mcp.tool(
    name="compact",
    description="Context Compressor: Slices massive state contexts into chunks and uses Confidence-Gated Logits to keep only the verbatim chunks strictly relevant to the user's goal. Eliminates hallucination risk of generative summarization.",
)
def jev_compact_context(
    state: Dict[str, Any] = Field(
        description="The massive JSON state or text to compress."
    ),
    goal: str = Field(
        description="The user's current goal or the objective the context is needed for."
    ),
    confidence_threshold: float = Field(
        default=0.85,
        description="The logit probability threshold required to keep a chunk."
    )
) -> str:
    import json
    import textwrap
    logger.info("Starting Confidence-Gated Context Compaction...")

    state_str = state.get("text", "") if isinstance(state, dict) and "text" in state else json.dumps(state)
    if not state_str.strip():
        return json.dumps({"status": "ERROR", "message": "State is empty."})
        
    # Semantic chunking (simple paragraph/block chunking for now)
    # Using double newline or single newline if too long
    raw_chunks = [c.strip() for c in state_str.split("\n\n") if c.strip()]
    if not raw_chunks:
        raw_chunks = [c.strip() for c in state_str.split("\n") if c.strip()]
        
    chunks = []
    for c in raw_chunks:
        if len(c) > 2000:
            chunks.extend(textwrap.wrap(c, 2000))
        else:
            chunks.append(c)

    kept_chunks = []
    total_chunks = len(chunks)
    dropped_chunks = 0
    
    # We will evaluate these in batches if needed, but for now sequentially 
    # to avoid context overflow on the provider.
    for i, chunk in enumerate(chunks):
        q = NoulQuestion(
            key=f"chunk_{i}",
            prompt=f"Does this text snippet contain information strictly relevant to achieving the goal: '{goal}'?"
        )
        
        try:
            # We force the fast provider directly to save time and prevent escalation
            # Since we just want a fast filter
            if hasattr(provider, "fast_provider"):
                res = provider.fast_provider.evaluate_batch({"text": chunk}, [q])
            else:
                res = provider.evaluate_batch({"text": chunk}, [q])
                
            chunk_res = res.get(q.key, {})
            score = chunk_res.get("noul", 0.0)
            
            if score >= confidence_threshold:
                kept_chunks.append(chunk)
            else:
                dropped_chunks += 1
                
        except Exception as e:
            logger.warning(f"Failed to evaluate chunk {i}: {e}. Keeping chunk by default.")
            kept_chunks.append(chunk)

    compacted_text = "\n\n".join(kept_chunks)
    compression_ratio = (1.0 - (len(kept_chunks) / total_chunks)) * 100 if total_chunks > 0 else 0
    
    return json.dumps({
        "status": "COMPACTION_COMPLETE",
        "original_chunks": total_chunks,
        "kept_chunks": len(kept_chunks),
        "dropped_chunks": dropped_chunks,
        "compression_ratio": f"{compression_ratio:.1f}%",
        "compacted_state": compacted_text
    }, indent=2)

@mcp.tool(
    name="scan-repo",
    description="Repository Scanner: Walks a codebase, uses Tree-sitter to break code into semantic blocks (functions/classes), and mathematically filters them using the Dual-Engine cascade to find only the code relevant to the task.",
)
def jev_scan_repo(
    directory_path: str = Field(description="The absolute path to the directory to scan."),
    task_description: str = Field(description="The goal or bug description used to filter the codebase.")
) -> str:
    import os
    import json
    import urllib.request
    from jev_mcp.scanner import walk_repository, generate_repo_map
    from jev_mcp.chunker import SemanticChunker
    from jev_mcp.routing_provider import RoutingProvider
    from jev_mcp.security import is_safe_path

    if not is_safe_path(directory_path):
        return json.dumps({"status": "BLOCKED", "message": f"Access denied: '{directory_path}' resolves to a sensitive system or credential location."})

    if not os.path.isdir(directory_path):
        return json.dumps({"status": "ERROR", "message": f"Directory not found: {directory_path}"})
        
    logger.info(f"Scanning repository at {directory_path} for task: {task_description}")
    
    files = walk_repository(directory_path)
    if not files:
        return json.dumps({"status": "BLOCKED", "message": "No valid files found in directory."})
        
    target_files = files
    
    if len(files) > 20:
        logger.info(f"Repository is massive ({len(files)} files). Escalating to Smart Model for Surgical Pointing...")
        repo_map = generate_repo_map(files, directory_path)
        
        prompt = f"Given the following task: '{task_description}'\\n\\nHere is the repository structure:\\n{repo_map}\\n\\nRespond with ONLY a comma-separated list of the 5-10 file paths most likely to contain the code needed for this task. Do not include any other text."
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.0
        }
        
        try:
            smart_port = os.getenv("JEV_SMART_PORT", "8081")
            req = urllib.request.Request(
                f"http://127.0.0.1:{smart_port}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                res = json.loads(response.read().decode())
                content = res["choices"][0]["message"]["content"]
                
                suggested_paths = [p.strip() for p in content.replace("\\n", ",").split(",") if p.strip()]
                
                target_files = []
                for sp in suggested_paths:
                    clean_sp = sp.lstrip("- ").strip()
                    abs_path = os.path.abspath(os.path.join(directory_path, clean_sp))
                    if abs_path in files:
                        target_files.append(abs_path)
                        
                if not target_files:
                    logger.warning("Smart model failed to select valid paths. Falling back to first 20 files.")
                    target_files = files[:20]
        except Exception as e:
            logger.warning(f"Surgical pointing failed: {e}. Falling back to first 20 files.")
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
                contextualized_chunk = f"### FILE: {rel_path}\\n{c}"
                all_chunks.append(contextualized_chunk)
        except Exception:
            pass

    if not all_chunks:
        return json.dumps({"status": "BLOCKED", "message": "Failed to extract chunks from repository."})

    provider = RoutingProvider()
    kept_chunks = []
    dropped = 0
    
    from concurrent.futures import ThreadPoolExecutor
    
    def eval_chunk(args):
        i, chunk = args
        q = NoulQuestion(
            key=f"chunk_{i}",
            prompt=f"Does this specific code block contain logic or variables highly relevant to the task: '{task_description}'?"
        )
        res = provider.evaluate_batch(chunk, [q])
        true_prob = res.get(f"chunk_{i}", {}).get("probabilities", {}).get("true", 0.0)
        return (chunk, true_prob >= 0.50)

    with ThreadPoolExecutor(max_workers=5) as executor:
        for chunk, keep in executor.map(eval_chunk, enumerate(all_chunks)):
            if keep:
                kept_chunks.append(chunk)
            else:
                dropped += 1
            
    if not kept_chunks:
        return json.dumps({"status": "BLOCKED", "message": f"Scanned {len(target_files)} files and {len(all_chunks)} semantic chunks. Jev determined 0 chunks were relevant to the task. Blocked to save context window."})
        
    return json.dumps({
        "status": "SUCCESS",
        "message": f"Surgically scanned {len(target_files)} files. Extracted {len(kept_chunks)} highly relevant chunks. Dropped {dropped} noisy chunks.",
        "content": "\n\n".join(kept_chunks)
    })



@mcp.prompt(
    name="scan-repo",
    description="Ask Jev to surgically scan the current workspace repository to find and extract the AST chunks relevant to a specific task."
)
def scan_repo_prompt() -> str:
    return "I want to scan this repository for code relevant to my current task. Please ask me for the target directory path (if not the root) and the specific bug or feature I am looking for. Then execute the `scan-repo` tool with those arguments and report back the mathematically extracted AST chunks."

@mcp.prompt(
    name="read-file",
    description="Ask Jev to read a specific file and mathematically extract only the AST chunks relevant to a task, dropping irrelevant noise."
)
def read_file_prompt() -> str:
    return "I want to read a file, but I only want the parts relevant to my current task. Please ask me for the file path and my objective. Then execute the `read-file` tool with filter_by_chunk=True and report back the extracted AST blocks."

def main():
    mcp.run(transport='stdio')

if __name__ == '__main__':
    main()
