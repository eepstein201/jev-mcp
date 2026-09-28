import argparse
import json
import sys
from jev_mcp.linter import DecisionPreflightLinter


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time IDE Linter for System 1 Prompts"
    )
    parser.add_argument("file", help="Path to JSON file containing Jev questions")
    args = parser.parse_args()

    linter = DecisionPreflightLinter()

    try:
        with open(args.file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"{args.file}:1: ERROR: FAILED_TO_PARSE - {str(e)}")
        sys.exit(1)

    questions = data if isinstance(data, list) else data.get("questions", [])
    if not questions:
        print(
            f"{args.file}:1: WARNING: NO_QUESTIONS - No valid questions array found in JSON."
        )
        sys.exit(0)

    has_errors = False

    # Simple IDE output mapping
    for i, q in enumerate(questions):
        q_type = q.get("type", "noul")
        prompt = q.get("prompt", "")
        options = (
            q.get("options", [])
            if q_type == "choice"
            else q.get("labels", [])
            if q_type == "score"
            else []
        )
        is_score = q_type == "score"

        # Format options as dicts for linter
        opts = [{"id": str(i), "description": str(o)} for i, o in enumerate(options)]
        if q_type == "noul":
            opts = [
                {"id": "true", "description": "True"},
                {"id": "false", "description": "False"},
            ]

        report = linter.lint({}, prompt, opts, is_score=is_score)
        for f in report.findings:
            if f.severity == "ERROR":
                has_errors = True

            # Print in standard compiler format: file:line: SEVERITY: CODE - Message
            print(
                f"{args.file}:0: {f.severity}: {f.code} - {f.message} (Suggestion: {f.suggestion})"
            )

    if has_errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
