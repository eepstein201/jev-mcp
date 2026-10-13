from tests.evals.rates import PassRate, format_table, run_repeated


def test_rate_is_passes_over_runs():
    assert PassRate(runs=4, passes=1).rate == 0.25


def test_rate_of_zero_runs_is_zero_not_a_division_error():
    assert PassRate(runs=0, passes=0).rate == 0.0


def test_run_repeated_counts_each_outcome():
    outcomes = iter([True, False, True])
    assert run_repeated(lambda: next(outcomes), repeats=3) == PassRate(runs=3, passes=2)


def test_an_exception_in_one_run_is_a_failure_and_the_rest_still_run():
    calls = []

    def case():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("engine dropped")
        return True

    assert run_repeated(case, repeats=4) == PassRate(runs=4, passes=3)
    assert len(calls) == 4


def test_table_shows_each_case_as_a_fraction_and_a_percentage():
    table = format_table({"T3": PassRate(4, 1), "D4": PassRate(4, 4)})
    assert "T3" in table and "1/4" in table and "25%" in table
    assert "D4" in table and "4/4" in table and "100%" in table
