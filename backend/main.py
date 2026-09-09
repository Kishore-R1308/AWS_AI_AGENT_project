import uuid

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


# =====================================================
# DATABASE INITIALIZATION / MIGRATION
# =====================================================

Base.metadata.create_all(bind=engine)


def migrate_chat_messages():
    """
    Add conversation_id to existing databases if it does not exist.

    Existing records are assigned their session_id as the
    conversation_id so old chat history is not lost.
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
    # This is ONLY the AWS authentication session.
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
    # -------------------------------------------------
    # Validate AWS session BEFORE running the agent
    # -------------------------------------------------

    aws_session = AWS_SESSIONS.get(
        request.session_id
    )

    if not aws_session:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid session_id. "
                "Please connect to AWS first."
            ),
        )

    account_id = aws_session.get(
        "account_id"
    )

    if not account_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Account ID not found for "
                "the given session_id."
            ),
        )

    # -------------------------------------------------
    # Load history ONLY for this conversation
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Run AI Agent
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Save conversation
    # -------------------------------------------------

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
    """
    Return a list of conversations belonging
    to the AWS account.
    """

    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id
            == account_id
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
                "conversation_id":
                    conversation_id,
                "title":
                    record.user_message,
                "created_at":
                    str(record.created_at),
                "updated_at":
                    str(record.created_at),
            }

            ordered_ids.append(
                conversation_id
            )

        else:
            conversations_data[
                conversation_id
            ]["updated_at"] = str(
                record.created_at
            )

    return [
        conversations_data[
            conversation_id
        ]
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
    """
    Return messages belonging to one conversation.
    """

    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id
            == account_id,
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
            "conversation_id":
                record.conversation_id,
            "user_message":
                record.user_message,
            "assistant_message":
                record.assistant_message,
            "intent":
                record.intent,
            "service":
                record.service,
            "rca":
                record.rca,
            "recommendations":
                record.recommendations,
            "created_at":
                str(record.created_at),
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
    """
    Delete one conversation and all its messages.
    """

    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id
            == account_id,
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
        "message":
            "Conversation deleted successfully.",
        "conversation_id":
            conversation_id,
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
    """
    Backward-compatible endpoint.

    Returns all messages for an account.
    """

    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id
            == account_id
        )
        .order_by(
            ChatMessage.created_at.asc()
        )
        .all()
    )

    return [
        {
            "id": record.id,
            "conversation_id":
                record.conversation_id,
            "user_message":
                record.user_message,
            "assistant_message":
                record.assistant_message,
            "intent":
                record.intent,
            "service":
                record.service,
            "rca":
                record.rca,
            "recommendations":
                record.recommendations,
            "created_at":
                str(record.created_at),
        }
        for record in records
    ]