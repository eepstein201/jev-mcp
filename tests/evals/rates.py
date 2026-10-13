"""Pass rates for evals that repeat a case, because one run of a model-driven case proves little."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PassRate:
    runs: int
    passes: int

    @property
    def rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0


def _passed(case: Callable[[], bool]) -> bool:
    try:
        return bool(case())
    except Exception:
        # A run that raised did not pass; the remaining runs still count.
        return False


def run_repeated(case: Callable[[], bool], repeats: int) -> PassRate:
    """Run `case` `repeats` times; a run that raises is a failed run."""
    outcomes = [_passed(case) for _ in range(repeats)]
    return PassRate(runs=len(outcomes), passes=sum(outcomes))


def format_table(rows: Mapping[str, PassRate]) -> str:
    """One line per case: name, passes/runs, percentage."""
    width = max((len(name) for name in rows), default=0)
    return "\n".join(
        f"{name:<{width}}  {rate.passes}/{rate.runs}  {rate.rate:.0%}"
        for name, rate in rows.items()
    )
