import pytest
import os
import json
from jev_mcp.ultrafast_adapter import local_choose

def test_local_choose_adapter(monkeypatch):
    # Mock RoutingProvider to return predictable responses
    from jev_mcp.routing_provider import RoutingProvider
    
    class MockRoutingProvider:
        def evaluate_batch(self, state, questions):
            results = {}
            for q in questions:
                if q.key == "operation":
                    results[q.key] = {
                        "choice": "CLICK",
                        "confidence": 0.95,
                        "probabilities": {"CLICK": 0.95, "TYPE_TEXT": 0.05}
                    }
                elif q.key == "click_target":
                    results[q.key] = {
                        "choice": "1",
                        "confidence": 0.88,
                        "probabilities": {"1": 0.88, "2": 0.12}
                    }
            return results

    monkeypatch.setattr("jev_mcp.ultrafast_adapter.RoutingProvider", MockRoutingProvider)
    
    state = {"actions": [{"kind": "click", "id": "btn1", "node": "n1", "label": "button 1"}]}
    goal = "Click button 1"
    history = []
    
    # We pass operations, targets, controls as they would appear in jev-ultrafast's choose()
    operations = {"CLICK": "Click an element", "TYPE_TEXT": "Type text"}
    targets = {"CLICK": {"1": {"element": "button 1"}, "2": {"element": "button 2"}}}
    controls = {}
    
    result = local_choose(state, goal, history, operations, targets, controls)
    
    # Ensure it returns the format expected by model.py's validate_choice
    assert result["answers"]["operation"]["choice"] == "CLICK"
    assert result["answers"]["operation"]["confidence"] == 0.95
    assert result["answers"]["operation"]["probabilities"] == {"CLICK": 0.95, "TYPE_TEXT": 0.05}
    
    assert result["answers"]["click_target"]["choice"] == "1"
    assert result["answers"]["click_target"]["confidence"] == 0.88
    
    assert result["model"] == "jev-mcp-local"
