import re
from dataclasses import dataclass, field
from typing import Any, List, Dict

MAX_STATE_TOKENS = 3500
MIN_OPTIONS = 2
MAX_OPTIONS = 10
MAX_OPTION_LENGTH_RATIO = 3.5


@dataclass
class LintFinding:
    severity: str  # "ERROR" or "WARNING"
    code: str
    message: str
    suggestion: str = ""


@dataclass
class PreflightReport:
    is_valid: bool
    findings: List[LintFinding] = field(default_factory=list)


class DecisionPreflightLinter:
    def __init__(self):
        self.generative_patterns = re.compile(
            r"\b(write|draft|generate|summarize|summarise|sumarize|summary|sumary|explain|explaining|explanation|compose|create|extract|extracting|extraction|rewrite|translate|tell me|describe|list out|produce)\b"
            r"|\b(in your own words|give a breakdown|provide an explanation)\b",
            re.IGNORECASE,
        )
        self.arithmetic_patterns = re.compile(
            r"\b(how many|calculate|count the number of|total sum of|compute the average)\b",
            re.IGNORECASE,
        )
        self.compound_patterns = re.compile(
            r"\b(and also|as well as)\b|\b(and|or)\b.*\?", re.IGNORECASE
        )
        self.chronological_patterns = re.compile(
            r"\b(happened before|happened after|occurred before|occurred after|older than|newer than|younger than|earlier than|later than|chronological|chronologically|prior to|subsequent to)\b",
            re.IGNORECASE,
        )
        self.boilerplate_patterns = re.compile(
            r"^(you are an expert|act as a|using only the supplied|carefully determine|think step-by-step)",
            re.IGNORECASE,
        )
        self.fallback_terms = {
            "other",
            "none",
            "neither",
            "n/a",
            "unknown",
            "insufficient",
            "unclear",
            "none of the above",
            "not applicable",
        }
        self.subjective_adjectives = re.compile(
            r"^(bad|poor|fair|okay|good|great|excellent|terrible|awesome|acceptable)$",
            re.IGNORECASE,
        )

    def lint(
        self,
        state: Any,
        question: str,
        options: List[Dict[str, str]],
        is_score: bool = False,
    ) -> PreflightReport:
        findings: List[LintFinding] = []

        # 1. Structural Checks
        if not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
            findings.append(
                LintFinding(
                    severity="ERROR",
                    code="INVALID_OPTION_COUNT",
                    message=f"Options count must be between {MIN_OPTIONS} and {MAX_OPTIONS}. Found {len(options)}.",
                    suggestion="Provide at least 2 mutually exclusive categorical options, and no more than 16.",
                )
            )

        # 2. Generative Intent
        if self.generative_patterns.search(question):
            findings.append(
                LintFinding(
                    severity="ERROR",
                    code="GENERATIVE_INTENT_DETECTED",
                    message=f"Question '{question}' asks for generative prose or extraction.",
                    suggestion="Fast decision models cannot generate text. Reframe your question as a boolean (noul), multiple choice (choice), or Likert scale (score).",
                )
            )

        # 3. Arithmetic & Chronological Intent
        if self.arithmetic_patterns.search(question):
            findings.append(
                LintFinding(
                    severity="ERROR",
                    code="ARITHMETIC_INTENT_DETECTED",
                    message="Question asks for counting or arithmetic calculation.",
                    suggestion="Single forward-pass models cannot perform precise token-level math. Compute counts in python, and ask semantic questions.",
                )
            )

        if self.chronological_patterns.search(question):
            findings.append(
                LintFinding(
                    severity="WARNING",
                    code="CHRONOLOGICAL_INTENT_DETECTED",
                    message="Question asks for a chronological or date comparison.",
                    suggestion="Models treat dates as literal strings, not temporal integers. Parse timestamps in your code to do logic, not inside the prompt.",
                )
            )

        # 4. Boilerplate & Meta-Prompting
        if self.boilerplate_patterns.search(question):
            findings.append(
                LintFinding(
                    severity="WARNING",
                    code="META_PROMPTING_BOILERPLATE",
                    message="Question contains conversational boilerplate ('Act as an expert', 'Think step-by-step', etc.).",
                    suggestion="Remove all conversational boilerplate! System 1 models suffer from attention distortion when given boilerplate instructions. Frame it as a direct, literal claim or question.",
                )
            )

        # 5. Compound Question
        if self.compound_patterns.search(question) and question.count("?") <= 1:
            findings.append(
                LintFinding(
                    severity="WARNING",
                    code="COMPOUND_QUESTION",
                    message="Question combines multiple criteria.",
                    suggestion="Split into multiple atomic questions to preserve calibration. E.g., instead of 'Is the user angry and requesting a refund?', ask two separate noul questions.",
                )
            )

        # 6. Missing Fallback Option
        has_fallback = any(
            any(
                term in opt.get("id", "").lower()
                or term in opt.get("description", "").lower()
                for term in self.fallback_terms
            )
            for opt in options
        )
        if not has_fallback and len(options) < MAX_OPTIONS:
            findings.append(
                LintFinding(
                    severity="WARNING",
                    code="MISSING_FALLBACK_OPTION",
                    message="No fallback option ('Other', 'None of the above', or 'Insufficient') was detected.",
                    suggestion="Always include a fallback option like 'Insufficient evidence to determine' so the closed-world softmax isn't forced to hallucinate when the answer isn't in the state.",
                )
            )

        # 7. Unbalanced Option Length
        descriptions = [
            opt.get("description", "") for opt in options if isinstance(opt, dict)
        ]
        lens = [max(len(d.split()), 1) for d in descriptions]
        if lens:
            max_len, min_len = max(lens), min(lens)
            if (max_len / min_len) > MAX_OPTION_LENGTH_RATIO and max_len > 12:
                findings.append(
                    LintFinding(
                        severity="WARNING",
                        code="UNBALANCED_OPTION_LENGTH",
                        message=f"Disproportionate option lengths detected (ratio {max_len / min_len:.1f}x).",
                        suggestion="Keep option descriptions roughly the same length. A highly verbose option can artificially attract probability mass away from shorter options.",
                    )
                )

        # 8. BARS / Subjective Scale Check (For Score Questions)
        if is_score:
            invalid_labels = [
                opt.get("id", "") for opt in options if len(str(opt.get("id", ""))) > 1
            ]
            if invalid_labels:
                findings.append(
                    LintFinding(
                        severity="ERROR",
                        code="MULTI_TOKEN_LABELS",
                        message=f"Score labels {invalid_labels} exceed 1 character.",
                        suggestion="Score labels must be single tokens (e.g., 1-9 or A-F) to prevent logprob fragmentation.",
                    )
                )

            if len(options) > 5:
                findings.append(
                    LintFinding(
                        severity="WARNING",
                        code="EXCESSIVE_SCALE_POINTS",
                        message="Score scale has more than 5 points.",
                        suggestion="Limit Likert scales to 3-5 points (e.g., 0-4). Open-weight models suffer from adjacent logit bleed on 7+ point scales.",
                    )
                )

            subjective_count = sum(
                1
                for d in descriptions
                if len(d.split()) <= 2 and self.subjective_adjectives.match(d.strip())
            )
            if subjective_count > 0:
                findings.append(
                    LintFinding(
                        severity="WARNING",
                        code="SUBJECTIVE_ADJECTIVE_SCALE",
                        message="Score labels use subjective adjectives (e.g., 'Good', 'Bad') without structural criteria.",
                        suggestion="Use Behaviorally Anchored Rating Scales (BARS). Instead of 'Level 3: Good', write 'Level 3: Resolves the customer issue but misses a follow-up constraint.' This prevents logit entropy.",
                    )
                )

            if descriptions and "excellent" in descriptions[0].lower():
                findings.append(
                    LintFinding(
                        severity="WARNING",
                        code="REVERSED_POLARITY",
                        message="Score labels appear to start with the highest/best value.",
                        suggestion="Always arrange levels monotonically from least/absence (Index 0) to most/complete to ensure the mathematical expected value behaves correctly.",
                    )
                )

        # 9. Check for overlapping Choice descriptions
        if not is_score and len(descriptions) >= 2:
            for i, d1 in enumerate(descriptions):
                for j, d2 in enumerate(descriptions):
                    if i != j and len(d1) > 10 and d1.lower() in d2.lower():
                        findings.append(
                            LintFinding(
                                severity="WARNING",
                                code="OVERLAPPING_OPTIONS",
                                message=f"Option '{d1}' is entirely subsumed by another option.",
                                suggestion="Ensure strict Mutual Exclusivity (ME). If options overlap, the softmax probability will cannibalize itself and suppress confidence. Use exclusionary bounds (e.g., 'Billing (excluding refunds)').",
                            )
                        )
                        break

        is_valid = not any(f.severity == "ERROR" for f in findings)
        return PreflightReport(is_valid=is_valid, findings=findings)
