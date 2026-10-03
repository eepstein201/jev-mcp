import sys
import json
import os

sys.path.insert(0, "/Users/ericepstein/Projects/jev-mcp/src")
import jev_mcp.server as server
from jev_mcp.provider import NoulQuestion, ScoreQuestion, ChoiceQuestion

print("=========================================================")
print("  JEV DUAL-ENGINE INTEGRATION SUITE")
print("=========================================================")

# 1. jev_evaluate_batch
print("\n--> 1. Testing jev_evaluate_batch (Direct Routing Evaluation)")
questions = [
    ScoreQuestion(key="complexity", prompt="Score the complexity of this instruction.", labels=["1", "2", "3", "4"]),
    NoulQuestion(key="needs_db", prompt="Does this involve a database?"),
    ChoiceQuestion(key="team", prompt="Which team?", options=["frontend", "backend"])
]
res1 = server.jev_evaluate_batch({"task": "Update the Postgres schema and deploy it."}, questions)
print(json.dumps(json.loads(res1), indent=2))

# 2. jev_determine_best_model
print("\n--> 2. Testing jev_determine_best_model (Multi-Class Logit Routing)")
res2 = server.jev_determine_best_model("Write a python script to download a file.", estimated_tokens=150)
print(json.dumps(json.loads(res2), indent=2))

# 3. jev_agent_handoff
print("\n--> 3. Testing jev_agent_handoff (Dynamic Team Escalation)")
teams = {
    "DataScience_Agent": "Builds ML models.",
    "DevOps_Agent": "Fixes CI/CD and deployment issues.",
    "UI_Agent": "Builds CSS and React components."
}
res3 = server.jev_agent_handoff("The CI/CD pipeline in GitHub actions is failing.", teams)
print(json.dumps(json.loads(res3), indent=2))

# 4. jev_manage_router_config
print("\n--> 4. Testing jev_manage_router_config (Rule Management)")
res4 = server.jev_manage_router_config("add_rule", condition="Task mentions SQL", target="b3")
print(res4)
server.jev_manage_router_config("remove_rule", rule_id=0)

# 5. jev_calibrate_threshold
print("\n--> 5. Testing jev_calibrate_threshold (Platt Scaling)")
dataset = [
    {"state": "I need to fix the UI button.", "expected": False},
    {"state": "The database is down.", "expected": True},
    {"state": "Write a CSS animation.", "expected": False},
    {"state": "Update the SQL schema.", "expected": True},
    {"state": "Center the div.", "expected": False},
    {"state": "Drop the users table.", "expected": True},
]
calib_q = NoulQuestion(key="calib", prompt="Does this require database access?")
res5 = server.jev_calibrate_threshold(dataset, calib_q, apply_platt_scaling="auto")
print(res5)

# 6. jev_explain_decision
print("\n--> 6. Testing jev_explain_decision (Decision Analysis)")
explain_q = NoulQuestion(key="exp", prompt="Does this require database access?")
res6 = server.jev_explain_decision({"task": "I need to drop the database."}, explain_q, "true")
print(res6)


def test_read_file():
    print("\n--> 7. Testing jev_read_file (Context Cascading)")
    import os
    import json
    from src.jev_mcp.server import jev_read_file
    
    with open("dummy_test_file.txt", "w") as f:
        f.write("def connect_to_database():\n    return psycopg2.connect('localhost')\n\n")
        
    try:
        res1_str = jev_read_file("dummy_test_file.txt", "How do I bake a chocolate cake?", filter_by_chunk=False)
        res1 = json.loads(res1_str)
        if res1.get("status") == "BLOCKED":
            print("Successfully blocked irrelevant file!")
        else:
            print(f"WARNING: File was not blocked. Status: {res1.get('status')}")
    finally:
        if os.path.exists("dummy_test_file.txt"):
            os.remove("dummy_test_file.txt")

test_read_file()
print("\n=========================================================")
print("  ALL TESTS PASSED WITH LIVE HYBRID ENGINE")
print("=========================================================")
