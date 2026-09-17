import uuid
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from agent import run_agent
from aws_auth import AWS_SESSIONS, connect_aws
from database import Base, engine, get_db
from models import ChatMessage
from schemas import (
    AWSConnectRequest,
    AWSConnectResponse,
    ChatRequest,
    ChatResponse,
)

from action_schemas import (
    AWSActionRequest,
    AWSActionApprovalRequest,
    RCARecommendedActionBatch,
    RCABatchApprovalRequest,
)

from action_store import (
    create_pending_action,
    get_pending_action,
    approve_pending_action,
    create_action_batch,
    get_action_batch,
    batch_belongs_to_session,
    approve_action_batch,
    update_batch_status,
    update_batch_action_status,
)

from action_validation import validate_action
from aws_action_executor import execute_aws_action
from aws_tools import (
    get_ec2_instances,
    get_s3_buckets,
    get_rds_instances,
    get_lambda_functions,
    get_vpcs,
    get_subnets,
    get_security_groups,
)


# =====================================================
# DATABASE INITIALIZATION / MIGRATION
# =====================================================

Base.metadata.create_all(bind=engine)


def migrate_chat_messages():
    """
    Add conversation_id to existing databases if it does not exist.
    """

    inspector = inspect(engine)

    if "chat_messages" not in inspector.get_table_names():
        return

    columns = {
        column["name"]
        for column in inspector.get_columns("chat_messages")
    }

    if "conversation_id" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE chat_messages "
                    "ADD COLUMN conversation_id VARCHAR(100)"
                )
            )

            connection.execute(
                text(
                    "UPDATE chat_messages "
                    "SET conversation_id = session_id "
                    "WHERE conversation_id IS NULL"
                )
            )


migrate_chat_messages()


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="AWS AI Agent",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# HELPERS
# =====================================================

def validate_aws_session(session_id: str) -> Dict[str, Any]:
    """
    Validate that an AWS session exists.
    """

    aws_session = AWS_SESSIONS.get(session_id)

    if not aws_session:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid session_id. "
                "Please connect to AWS first."
            ),
        )

    if not aws_session.get("account_id"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Account ID not found for "
                "the given session_id."
            ),
        )

    return aws_session


def model_to_dict(model: Any) -> Dict[str, Any]:
    """
    Support Pydantic v2 and v1.
    """

    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():
    return {
        "message": "AWS AI Agent API is running"
    }


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =====================================================
# AWS CONNECTION
# =====================================================

@app.post(
    "/aws/connect",
    response_model=AWSConnectResponse,
)
def aws_connect(
    request: AWSConnectRequest,
):
    session_id = str(uuid.uuid4())

    try:
        return connect_aws(
            session_id=session_id,
            access_key=request.access_key,
            secret_key=request.secret_key,
            region=request.region,
            role_arn=request.role_arn,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"AWS connection failed: {exc}",
        )


# =====================================================
# CHAT
# =====================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    aws_session = validate_aws_session(
        request.session_id
    )

    account_id = aws_session.get("account_id")

    previous_messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id == account_id,
            ChatMessage.conversation_id
            == request.conversation_id,
        )
        .order_by(
            ChatMessage.created_at.asc()
        )
        .all()
    )

    history = [
        {
            "user": record.user_message,
            "assistant": record.assistant_message,
        }
        for record in previous_messages
    ]

    try:
        result = run_agent(
            session_id=request.session_id,
            query=request.message,
            history=history,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    record = ChatMessage(
        account_id=account_id,
        session_id=request.session_id,
        conversation_id=request.conversation_id,
        user_message=request.message,
        assistant_message=result["answer"],
        intent=result["intent"],
        service=result.get("service"),
        rca=result.get("rca"),
        recommendations=result.get(
            "recommendations"
        ),
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return result


# =====================================================
# ALL CONVERSATIONS
# =====================================================

@app.get(
    "/conversations/{account_id}"
)
def conversations(
    account_id: str,
    db: Session = Depends(get_db),
):
    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id == account_id
        )
        .order_by(
            ChatMessage.created_at.desc()
        )
        .all()
    )

    conversations_data = {}
    ordered_ids = []

    for record in records:
        conversation_id = (
            record.conversation_id
            or record.session_id
        )

        if conversation_id not in conversations_data:
            conversations_data[
                conversation_id
            ] = {
                "conversation_id": conversation_id,
                "title": record.user_message,
                "created_at": str(
                    record.created_at
                ),
                "updated_at": str(
                    record.created_at
                ),
            }

            ordered_ids.append(conversation_id)

        else:
            conversations_data[
                conversation_id
            ]["updated_at"] = str(
                record.created_at
            )

    return [
        conversations_data[conversation_id]
        for conversation_id in ordered_ids
    ]


# =====================================================
# CONVERSATION HISTORY
# =====================================================

@app.get(
    "/history/{account_id}/{conversation_id}"
)
def conversation_history(
    account_id: str,
    conversation_id: str,
    db: Session = Depends(get_db),
):
    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id == account_id,
            ChatMessage.conversation_id
            == conversation_id,
        )
        .order_by(
            ChatMessage.created_at.asc()
        )
        .all()
    )

    return [
        {
            "id": record.id,
            "conversation_id": record.conversation_id,
            "user_message": record.user_message,
            "assistant_message": record.assistant_message,
            "intent": record.intent,
            "service": record.service,
            "rca": record.rca,
            "recommendations": record.recommendations,
            "created_at": str(record.created_at),
        }
        for record in records
    ]


# =====================================================
# DELETE CONVERSATION
# =====================================================

@app.delete(
    "/history/{account_id}/{conversation_id}"
)
def delete_conversation(
    account_id: str,
    conversation_id: str,
    db: Session = Depends(get_db),
):
    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id == account_id,
            ChatMessage.conversation_id
            == conversation_id,
        )
        .all()
    )

    if not records:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    for record in records:
        db.delete(record)

    db.commit()

    return {
        "success": True,
        "message": (
            "Conversation deleted successfully."
        ),
        "conversation_id": conversation_id,
    }


# =====================================================
# LEGACY ACCOUNT HISTORY
# =====================================================

@app.get(
    "/history/{account_id}"
)
def history(
    account_id: str,
    db: Session = Depends(get_db),
):
    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id == account_id
        )
        .order_by(
            ChatMessage.created_at.asc()
        )
        .all()
    )

    return [
        {
            "id": record.id,
            "conversation_id": record.conversation_id,
            "user_message": record.user_message,
            "assistant_message": record.assistant_message,
            "intent": record.intent,
            "service": record.service,
            "rca": record.rca,
            "recommendations": record.recommendations,
            "created_at": str(record.created_at),
        }
        for record in records
    ]


# =====================================================
# LIVE AWS RESOURCE LISTING
# =====================================================

@app.get("/aws/resources/{session_id}")
def list_aws_resources(session_id: str, service: str):
    """Return live resources for the Resource Management page."""

    validate_aws_session(session_id)

    loaders = {
        "ec2": get_ec2_instances,
        "s3": get_s3_buckets,
        "rds": get_rds_instances,
        "lambda": get_lambda_functions,
        "vpc": get_vpcs,
        "subnet": get_subnets,
        "security_group": get_security_groups,
    }

    loader = loaders.get(service)
    if loader is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported resource listing service: {service}",
        )

    try:
        raw_resources = loader(session_id)
        resources = []

        for item in raw_resources or []:
            if service == "ec2":
                resource_id = item.get("instance_id")
            elif service == "s3":
                resource_id = item.get("name")
            elif service == "rds":
                resource_id = item.get("identifier")
            elif service == "lambda":
                resource_id = item.get("function_name") or item.get("name")
            elif service == "vpc":
                resource_id = item.get("vpc_id")
            elif service == "subnet":
                resource_id = item.get("subnet_id")
            else:
                resource_id = item.get("group_id")

            if not resource_id:
                continue

            resources.append({
                "id": resource_id,
                "name": item.get("name") or resource_id,
                "details": item,
            })

        return {"service": service, "resources": resources}

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not load live AWS resources: {exc}",
        )


# =====================================================
# INDIVIDUAL AWS ACTION PLANNING
# =====================================================

@app.post("/aws/action/plan")
def plan_aws_action(
    request: AWSActionRequest,
):
    """
    Validate and store one pending AWS action.
    This endpoint does not execute the action.
    """

    validate_aws_session(request.session_id)

    is_valid, message = validate_action(
        request.action,
        request.resource_type,
        request.parameters,
    )

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=message,
        )

    action = create_pending_action(
        session_id=request.session_id,
        action=request.action,
        resource_type=request.resource_type,
        parameters=request.parameters,
        explanation=(
            f"Proposed {request.action} operation "
            f"for {request.resource_type}."
        ),
    )

    return {
        "message": (
            "Action created and awaiting confirmation"
        ),
        "action": action,
        "requires_confirmation": True,
    }


# =====================================================
# INDIVIDUAL AWS ACTION CONFIRMATION
# =====================================================

REQUIRED_CONFIRMATION_PHRASE = (
    "I CONFIRM THIS AWS ACTION"
)


@app.post("/aws/action/confirm")
def confirm_aws_action(
    request: AWSActionApprovalRequest,
):
    """
    Confirm and execute one pending AWS action.
    """

    validate_aws_session(request.session_id)

    action = get_pending_action(
        request.action_id
    )

    if not action:
        raise HTTPException(
            status_code=404,
            detail="Action not found",
        )

    if action["session_id"] != request.session_id:
        raise HTTPException(
            status_code=403,
            detail=(
                "Action does not belong to "
                "this session"
            ),
        )

    if action["status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Action is already "
                f"{action['status']}"
            ),
        )

    if (
        request.confirmation_phrase
        != REQUIRED_CONFIRMATION_PHRASE
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid confirmation phrase. "
                "Type exactly: "
                f"{REQUIRED_CONFIRMATION_PHRASE}"
            ),
        )

    try:
        result = execute_aws_action(
            session_id=request.session_id,
            action=action["action"],
            resource_type=action["resource_type"],
            parameters=action["parameters"],
        )

        approved_action = approve_pending_action(
            request.action_id
        )

        approved_action["status"] = "executed"

        return {
            "message": (
                "AWS action executed successfully"
            ),
            "action": approved_action,
            "execution_status": "completed",
            "result": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "AWS action execution failed: "
                f"{str(exc)}"
            ),
        )


# =====================================================
# RCA RECOMMENDED ACTION BATCH PLANNING
# =====================================================

@app.post("/aws/rca/batch/plan")
def plan_rca_action_batch(
    request: RCARecommendedActionBatch,
):
    """
    Create a batch of RCA remediation actions.

    All actions are validated before being stored.
    Nothing is executed at this stage.
    """

    validate_aws_session(request.session_id)

    if not request.actions:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least one remediation action "
                "is required."
            ),
        )

    validated_actions = []

    for action in request.actions:
        action_data = model_to_dict(action)

        is_valid, message = validate_action(
            action_data["action"],
            action_data["resource_type"],
            action_data["parameters"],
        )

        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid action "
                    f"{action_data.get('action_id')}: "
                    f"{message}"
                ),
            )

        validated_actions.append(
            action_data
        )

    batch = create_action_batch(
        session_id=request.session_id,
        issue_id=request.issue_id,
        issue_title=request.issue_title,
        issue_description=request.issue_description,
        actions=validated_actions,
    )

    return {
        "message": (
            "RCA remediation batch created. "
            "One confirmation is required "
            "for the complete batch."
        ),
        "batch": batch,
        "requires_confirmation": True,
        "confirmation_phrase": (
            REQUIRED_CONFIRMATION_PHRASE
        ),
    }


# =====================================================
# GET RCA ACTION BATCH STATUS
# =====================================================

@app.get("/aws/rca/batch/{batch_id}")
def get_rca_action_batch(
    batch_id: str,
    session_id: str,
):
    """
    Return the current status of an RCA action batch.
    """

    validate_aws_session(session_id)

    batch = get_action_batch(batch_id)

    if not batch:
        raise HTTPException(
            status_code=404,
            detail="RCA action batch not found.",
        )

    if not batch_belongs_to_session(
        batch_id,
        session_id,
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "This action batch does not "
                "belong to the current session."
            ),
        )

    return {
        "batch": batch
    }


# =====================================================
# RCA ACTION BATCH CONFIRMATION AND EXECUTION
# =====================================================

@app.post("/aws/rca/batch/confirm")
def confirm_rca_action_batch(
    request: RCABatchApprovalRequest,
):
    """
    Confirm and execute all RCA remediation actions.

    Important behavior:
    - Only one confirmation is required.
    - Actions execute sequentially.
    - Each action status is updated.
    - Execution stops if one action fails.
    - Remaining actions are marked as skipped.
    """

    validate_aws_session(request.session_id)

    batch = get_action_batch(
        request.batch_id
    )

    if not batch:
        raise HTTPException(
            status_code=404,
            detail="RCA action batch not found.",
        )

    if not batch_belongs_to_session(
        request.batch_id,
        request.session_id,
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "This action batch does not "
                "belong to the current session."
            ),
        )

    if batch["status"] not in [
        "pending",
        "approved",
    ]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch is already "
                f"{batch['status']}."
            ),
        )

    if (
        request.confirmation_phrase
        != REQUIRED_CONFIRMATION_PHRASE
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid confirmation phrase. "
                "Type exactly: "
                f"{REQUIRED_CONFIRMATION_PHRASE}"
            ),
        )

    # Validate every action again before execution.
    for action in batch["actions"]:
        is_valid, message = validate_action(
            action["action"],
            action["resource_type"],
            action["parameters"],
        )

        if not is_valid:
            update_batch_status(
                request.batch_id,
                "failed",
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    "Batch validation failed: "
                    f"{message}"
                ),
            )

    approve_action_batch(
        request.batch_id
    )

    update_batch_status(
        request.batch_id,
        "running",
    )

    execution_results = []

    for action in batch["actions"]:
        action_id = action.get(
            "action_id"
        )

        update_batch_action_status(
            batch_id=request.batch_id,
            action_id=action_id,
            status="running",
        )

        try:
            result = execute_aws_action(
                session_id=request.session_id,
                action=action["action"],
                resource_type=action[
                    "resource_type"
                ],
                parameters=action[
                    "parameters"
                ],
            )

            update_batch_action_status(
                batch_id=request.batch_id,
                action_id=action_id,
                status="completed",
                result=result,
            )

            execution_results.append(
                {
                    "action_id": action_id,
                    "resource_type": action[
                        "resource_type"
                    ],
                    "status": "completed",
                    "result": result,
                }
            )

        except Exception as exc:
            error_message = str(exc)

            update_batch_action_status(
                batch_id=request.batch_id,
                action_id=action_id,
                status="failed",
                error=error_message,
            )

            execution_results.append(
                {
                    "action_id": action_id,
                    "resource_type": action[
                        "resource_type"
                    ],
                    "status": "failed",
                    "error": error_message,
                }
            )

            # Stop execution after the first failure.
            remaining_actions = batch["actions"]

            current_index = next(
                (
                    index
                    for index, item
                    in enumerate(remaining_actions)
                    if item.get("action_id")
                    == action_id
                ),
                -1,
            )

            if current_index >= 0:
                for remaining_action in (
                    remaining_actions[
                        current_index + 1:
                    ]
                ):
                    remaining_action_id = (
                        remaining_action.get(
                            "action_id"
                        )
                    )

                    update_batch_action_status(
                        batch_id=request.batch_id,
                        action_id=remaining_action_id,
                        status="skipped",
                        error=(
                            "Skipped because a "
                            "previous action failed."
                        ),
                    )

                    execution_results.append(
                        {
                            "action_id": (
                                remaining_action_id
                            ),
                            "resource_type": (
                                remaining_action[
                                    "resource_type"
                                ]
                            ),
                            "status": "skipped",
                            "error": (
                                "Skipped because a "
                                "previous action failed."
                            ),
                        }
                    )

            update_batch_status(
                request.batch_id,
                "failed",
            )

            return {
                "message": (
                    "RCA remediation stopped "
                    "because an action failed."
                ),
                "batch_id": request.batch_id,
                "execution_status": "failed",
                "results": execution_results,
                "batch": get_action_batch(
                    request.batch_id
                ),
            }

    update_batch_status(
        request.batch_id,
        "completed",
    )

    return {
        "message": (
            "All RCA remediation actions "
            "executed successfully."
        ),
        "batch_id": request.batch_id,
        "execution_status": "completed",
        "results": execution_results,
        "batch": get_action_batch(
            request.batch_id
        ),
    }