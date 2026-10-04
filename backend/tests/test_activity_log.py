import pytest
import tempfile
import os
import json
from app.services.activity_log import initialize_activity_log, log_activity, verify_activity_log

@pytest.fixture
def temp_log_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "activity.jsonl")
        yield path

def modify_event_in_file(path, index, modifier_func):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    event = json.loads(lines[index])
    event = modifier_func(event)
    lines[index] = json.dumps(event) + "\n"
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def test_initial_empty_log_verification(temp_log_path):
    initialize_activity_log(temp_log_path)
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is True
    assert res["event_count"] == 0

def test_first_event_has_empty_previous_hash(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    with open(temp_log_path, 'r', encoding='utf-8') as f:
        event = json.loads(f.readline())
    assert event["previous_hash"] == ""

def test_second_event_references_first_self_hash(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    log_activity("QUERY", log_path=temp_log_path)
    with open(temp_log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    event1 = json.loads(lines[0])
    event2 = json.loads(lines[1])
    assert event2["previous_hash"] == event1["self_hash"]

def test_third_event_references_second_self_hash(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    log_activity("QUERY", log_path=temp_log_path)
    log_activity("DELETE", log_path=temp_log_path)
    with open(temp_log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    event2 = json.loads(lines[1])
    event3 = json.loads(lines[2])
    assert event3["previous_hash"] == event2["self_hash"]

def test_repeated_initialization_preserves_validity(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    log_activity("QUERY", log_path=temp_log_path)
    initialize_activity_log(temp_log_path)
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is True
    assert res["event_count"] == 2

def test_modifying_event_details_fails(temp_log_path):
    log_activity("INDEX", details={"a": 1}, log_path=temp_log_path)
    def mod(e): e["details"]["a"] = 2; return e
    modify_event_in_file(temp_log_path, 0, mod)
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is False
    assert "self_hash mismatch" in res["error"]

def test_modifying_event_type_fails(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    def mod(e): e["event_type"] = "DELETE"; return e
    modify_event_in_file(temp_log_path, 0, mod)
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is False
    assert "self_hash mismatch" in res["error"]

def test_modifying_document_id_fails(temp_log_path):
    log_activity("INDEX", document_id="doc1", log_path=temp_log_path)
    def mod(e): e["document_id"] = "doc2"; return e
    modify_event_in_file(temp_log_path, 0, mod)
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is False
    assert "self_hash mismatch" in res["error"]

def test_modifying_timestamp_fails(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    def mod(e): e["timestamp"] = "2099-01-01T00:00:00+00:00"; return e
    modify_event_in_file(temp_log_path, 0, mod)
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is False
    assert "self_hash mismatch" in res["error"]

def test_modifying_previous_hash_fails(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    log_activity("QUERY", log_path=temp_log_path)
    def mod(e): e["previous_hash"] = "fakehash"; return e
    modify_event_in_file(temp_log_path, 1, mod)
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is False
    assert "previous_hash mismatch" in res["error"]

def test_modifying_self_hash_fails(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    def mod(e): e["self_hash"] = "fakehash"; return e
    modify_event_in_file(temp_log_path, 0, mod)
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is False
    assert "self_hash mismatch" in res["error"]

def test_removing_middle_event_fails(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    log_activity("QUERY", log_path=temp_log_path)
    log_activity("DELETE", log_path=temp_log_path)
    with open(temp_log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines.pop(1) # remove middle event
    with open(temp_log_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is False
    assert "previous_hash mismatch" in res["error"]

def test_reordering_events_fails(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    log_activity("QUERY", log_path=temp_log_path)
    with open(temp_log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines[0], lines[1] = lines[1], lines[0] # swap
    with open(temp_log_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is False
    assert "previous_hash mismatch" in res["error"]

def test_appending_fabricated_event_fails(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    fake_event = {
        "timestamp": "2099-01-01T00:00:00+00:00",
        "event_type": "FAKE",
        "document_id": None,
        "details": None,
        "previous_hash": "wrong",
        "self_hash": "dummy"
    }
    with open(temp_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(fake_event) + "\n")
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is False
    assert "previous_hash mismatch" in res["error"]

def test_all_normal_events_pass(temp_log_path):
    log_activity("INDEX", log_path=temp_log_path)
    log_activity("QUERY", details={"q": "hi"}, log_path=temp_log_path)
    log_activity("DELETE", document_id="doc1", log_path=temp_log_path)
    res = verify_activity_log(temp_log_path)
    assert res["valid"] is True
    assert res["event_count"] == 3
