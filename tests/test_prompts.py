from jev_mcp.server import (
    calibrate_prompt, compact_jev_prompt, evaluate_prompt,
    explain_decision_prompt, generate_data_prompt, handoff_prompt,
    model_router_prompt, router_config_prompt, optimize_prompt,
    temperature_prompt, train_prompt, scan_repo_prompt, read_file_prompt
)

def test_all_prompts():
    assert isinstance(calibrate_prompt(), str)
    assert isinstance(compact_jev_prompt(), str)
    assert isinstance(evaluate_prompt(), str)
    assert isinstance(explain_decision_prompt(), str)
    assert isinstance(generate_data_prompt(), str)
    assert isinstance(handoff_prompt(), str)
    assert isinstance(model_router_prompt(), str)
    assert isinstance(router_config_prompt(), str)
    assert isinstance(optimize_prompt(), str)
    assert isinstance(temperature_prompt(), str)
    assert isinstance(train_prompt(), str)
    assert isinstance(scan_repo_prompt(), str)
    assert isinstance(read_file_prompt(), str)
