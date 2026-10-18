import pytest
import time
from src.mitigation import execute_mitigation_action, get_mitigation_history, _SERVICE_STATE, COOLDOWN_SECONDS

@pytest.fixture(autouse=True)
def reset_service_state():
    """Clear in-memory service states between tests."""
    _SERVICE_STATE.clear()

def test_mitigation_healthy_reset():
    svc = "test-cart-service"
    # Transition to healthy
    res = execute_mitigation_action(svc, "NO_ACTION", 0.1, 0.05)
    assert res["mitigation_tier"] == "TIER_0_HEALTHY"
    assert res["status"] == "HEALTHY"
    assert res["circuit_breaker"] == "STANDBY"
    assert res["command_issued"] is None

def test_mitigation_progressive_escalation():
    svc = "test-payment-service"
    
    # 1. Trigger Load Shedding at Tier 0 -> Escalates to Tier 1
    res1 = execute_mitigation_action(svc, "TRIGGER_LOAD_SHEDDING", 0.75, 0.2)
    assert _SERVICE_STATE[svc]["tier"] == 1
    assert "rate-limit" in res1["command_issued"]
    assert res1["status"] == "SUCCESS"

    # Fast-forward cooldown by resetting timestamp
    _SERVICE_STATE[svc]["last_action_ts"] = time.time() - (COOLDOWN_SECONDS + 10)

    # 2. Trigger Load Shedding again while at Tier 1 -> Escalates to Tier 2 (Scale Out)
    res2 = execute_mitigation_action(svc, "TRIGGER_LOAD_SHEDDING", 0.85, 0.3)
    assert _SERVICE_STATE[svc]["tier"] == 2
    assert "scale deployment" in res2["command_issued"]

def test_mitigation_critical_tier3_emergency():
    svc = "test-auth-service"
    res = execute_mitigation_action(svc, "CRITICAL_RESTART_AND_TRAFFIC_REROUTE", 0.95, 0.9)
    assert _SERVICE_STATE[svc]["tier"] == 3
    assert "rollout restart" in res["command_issued"]
    assert "drain" in res["command_issued"]

def test_mitigation_cooldown_suppression():
    svc = "test-order-service"
    # Action 1 executes
    res1 = execute_mitigation_action(svc, "TRIGGER_LOAD_SHEDDING", 0.8, 0.2)
    assert res1["status"] == "SUCCESS"

    # Action 2 immediately after -> Must be suppressed by cooldown
    res2 = execute_mitigation_action(svc, "TRIGGER_LOAD_SHEDDING", 0.85, 0.25)
    assert res2["status"] == "COOLDOWN_ACTIVE"
    assert res2["action_executed"] == "SUPPRESSED_COOLDOWN_ACTIVE"
    assert res2["command_issued"] is None
    assert res2["cooldown_remaining_sec"] > 0
