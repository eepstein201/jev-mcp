import os
import json

ROUTER_CONFIG_PATH = os.path.expanduser("~/.jev/router_config.json")

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
