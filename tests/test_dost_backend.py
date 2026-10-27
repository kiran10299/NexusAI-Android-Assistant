import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.dost_server import app, detect_visual_widget, extract_clean_topic

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["assistant"] == "Dost.ai"


def test_extract_clean_topic():
    topic = extract_clean_topic("Hey Dost explain how a 4-20mA current loop works in PLC")
    assert "4-20mA" in topic or "current" in topic


def test_detect_visual_widget_420ma():
    w = detect_visual_widget("Design a 4-20mA sensor scaling circuit")
    assert w is not None
    assert w.widget_type == "circuit_calc"
    assert w.params["mode"] == "4_20ma"


def test_detect_visual_widget_rc():
    w = detect_visual_widget("Calculate RC low pass filter cutoff frequency")
    assert w is not None
    assert w.widget_type == "circuit_calc"
    assert w.params["mode"] == "rc_filter"


def test_chat_endpoint_synthesis():
    resp = client.post(
        "/api/chat",
        json={
            "message": "How does a 4-20mA current loop transmitter scale to a 12-bit ADC?",
            "history": [],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["assistant_name"] == "Dost.ai"
    assert "4–20mA" in data["reply_markdown"]
    assert data["visual_widget"]["widget_type"] == "circuit_calc"
    assert len(data["suggestions"]) >= 2


if __name__ == "__main__":
    test_health_endpoint()
    test_extract_clean_topic()
    test_detect_visual_widget_420ma()
    test_detect_visual_widget_rc()
    test_chat_endpoint_synthesis()
    print("ALL 5 DOST.AI FASTAPI BACKEND TESTS PASSED 100%!")
