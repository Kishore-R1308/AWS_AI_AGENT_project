from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime, timezone


# =====================================================
# IN-MEMORY ACTION STORAGE
# =====================================================

_pending_actions: Dict[str, Dict[str, Any]] = {}
_action_batches: Dict[str, Dict[str, Any]] = {}


# =====================================================
# INDIVIDUAL ACTION MANAGEMENT
# =====================================================

def create_pending_action(
    session_id: str,
    action: str,
    resource_type: str,
    parameters: Dict[str, Any],
    explanation: str,
) -> Dict[str, Any]:

    action_id = str(uuid4())

    action_data = {
        "action_id": action_id,
        "session_id": session_id,
        "action": action,
        "resource_type": resource_type,
        "parameters": parameters,
        "explanation": explanation,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }

    _pending_actions[action_id] = action_data

    return action_data


def get_pending_action(
    action_id: str,
) -> Optional[Dict[str, Any]]:

    return _pending_actions.get(action_id)


def approve_pending_action(
    action_id: str,
) -> Optional[Dict[str, Any]]:

    action = _pending_actions.get(action_id)

    if action:
        action["status"] = "approved"
        action["updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()

    return action


def delete_pending_action(
    action_id: str,
) -> bool:

    if action_id in _pending_actions:
        del _pending_actions[action_id]
        return True

    return False


# =====================================================
# RCA ACTION BATCH MANAGEMENT
# =====================================================

def create_action_batch(
    session_id: str,
    issue_id: str,
    issue_title: str,
    issue_description: str,
    actions: list,
) -> Dict[str, Any]:

    batch_id = str(uuid4())

    batch = {
        "batch_id": batch_id,
        "session_id": session_id,
        "issue_id": issue_id,
        "issue_title": issue_title,
        "issue_description": issue_description,
        "actions": actions,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    _action_batches[batch_id] = batch

    return batch


def get_action_batch(
    batch_id: str,
) -> Optional[Dict[str, Any]]:

    return _action_batches.get(batch_id)


def batch_belongs_to_session(
    batch_id: str,
    session_id: str,
) -> bool:

    batch = _action_batches.get(batch_id)

    return bool(
        batch
        and batch["session_id"] == session_id
    )


def approve_action_batch(
    batch_id: str,
) -> Optional[Dict[str, Any]]:

    batch = _action_batches.get(batch_id)

    if batch:
        batch["status"] = "approved"
        batch["updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        for action in batch["actions"]:
            if action.get("status") == "pending":
                action["status"] = "approved"
                action["updated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

    return batch


# =====================================================
# BATCH STATUS MANAGEMENT
# =====================================================

def update_batch_status(
    batch_id: str,
    status: str,
) -> Optional[Dict[str, Any]]:

    batch = _action_batches.get(batch_id)

    if batch:
        batch["status"] = status
        batch["updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()

    return batch


def update_batch_action_status(
    batch_id: str,
    action_id: str,
    status: str,
    result: Any = None,
    error: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Update the status and execution details
    of one batch action.
    """

    batch = _action_batches.get(batch_id)

    if not batch:
        return None

    for action in batch["actions"]:

        if action.get("action_id") == action_id:

            action["status"] = status

            if result is not None:
                action["result"] = result

            if error is not None:
                action["error"] = error

            action["updated_at"] = datetime.now(
                timezone.utc
            ).isoformat()

            return action

    return None