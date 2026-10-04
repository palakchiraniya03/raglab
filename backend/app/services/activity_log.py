import json
import os
import datetime
import hashlib
from typing import Optional, Dict, Any

LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "activity_logs", "activity.jsonl")

def calculate_self_hash(event: dict) -> str:
    canonical_json = json.dumps(
        event,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
def initialize_activity_log(log_path: Optional[str] = None) -> None:
    log_path = log_path or LOG_FILE_PATH
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, 'a', encoding='utf-8') as f:
            pass

def log_activity(event_type: str, document_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None, log_path: Optional[str] = None) -> None:
    log_path = log_path or LOG_FILE_PATH
    initialize_activity_log(log_path)
    
    previous_hash = ""
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        last_event = json.loads(line)
                        previous_hash = last_event.get("self_hash", "")
                        break
                    except json.JSONDecodeError:
                        continue
                        
    event_without_self_hash = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event_type": event_type,
        "document_id": document_id,
        "details": details,
        "previous_hash": previous_hash
    }
    
    self_hash = calculate_self_hash(event_without_self_hash)
    final_event = dict(event_without_self_hash)
    final_event["self_hash"] = self_hash
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(final_event) + "\n")

def verify_activity_log(log_path: Optional[str] = None) -> Dict[str, Any]:
    log_path = log_path or LOG_FILE_PATH
    if not os.path.exists(log_path):
        return {"valid": True, "event_count": 0, "error": None}
        
    expected_previous_hash = ""
    event_count = 0
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return {"valid": False, "event_count": event_count, "error": f"Line {line_number} is not valid JSON."}
                
            required_fields = {"timestamp", "event_type", "document_id", "details", "previous_hash", "self_hash"}
            if not required_fields.issubset(event.keys()):
                return {"valid": False, "event_count": event_count, "error": f"Line {line_number} missing required fields."}
                
            if event["previous_hash"] != expected_previous_hash:
                return {"valid": False, "event_count": event_count, "error": f"Line {line_number} previous_hash mismatch."}
                
            event_without_self_hash = {k: v for k, v in event.items() if k != "self_hash"}
            calculated_hash = calculate_self_hash(event_without_self_hash)
            
            if calculated_hash != event["self_hash"]:
                return {"valid": False, "event_count": event_count, "error": f"Line {line_number} self_hash mismatch (tampering detected)."}
                
            expected_previous_hash = event["self_hash"]
            event_count += 1
            
    return {"valid": True, "event_count": event_count, "error": None}
