# CRITICAL RULE: Test-Driven Commit Gate

You are strictly forbidden from committing code to version control (e.g., executing `git commit`) if there are unresolved test failures in the repository. 

Whenever you modify code and run tests:
1. You MUST explicitly verify that the test command exits with a `0` exit code.
2. If tests fail, you MUST fix the failures and re-run the tests.
3. You may ONLY execute a `git commit` command after explicitly observing a fully green, successful test run. Do not attempt to "batch" commit partial fixes while other tests are still failing.
4. When writing new tests, you MUST successfully and permanently integrate them into the primary test suite (e.g., `integration_test.py`). 
5. NEVER rely solely on isolated, temporary scratch scripts (e.g., `test_read_file.py`) to bypass the test gate. If your automated string-replacement script fails to inject the test into the suite, you must manually fix the suite and verify the specific new test actually executes before committing.
6. **Strict Code Coverage Limits:** You are strictly forbidden from executing `git commit` unless you run a coverage report (e.g. `pytest --cov=src`) and mathematically verify two thresholds:
   - **Individual File Coverage:** Every individual file modified or added must have >= 95% coverage.
   - **Overall Coverage:** The overall average coverage of the codebase must be >= 85%.
   If the coverage falls below these thresholds, you MUST halt, write additional tests, and re-verify the coverage until the thresholds are met before committing.
