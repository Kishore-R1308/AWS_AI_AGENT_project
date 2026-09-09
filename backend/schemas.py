from typing import List, Optional

from pydantic import BaseModel


class AWSConnectRequest(BaseModel):
    access_key: str
    secret_key: str
    region: str
    role_arn: str


class AWSConnectResponse(BaseModel):
    connected: bool
    account_id: str
    arn: str
    region: str
    message: str
    session_id: str


class ChatRequest(BaseModel):
    # AWS authentication/session
    session_id: str

    # Individual conversation/chat
    conversation_id: str

    # Current user message
    message: str


class ChatResponse(BaseModel):
    answer: str
    intent: str
    service: Optional[str] = None


class Plan(BaseModel):
    intent: str
    tools: List[str]