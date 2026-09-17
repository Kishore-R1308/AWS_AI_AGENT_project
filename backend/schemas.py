from typing import List, Optional

from pydantic import BaseModel


# =====================================================
# AWS CONNECTION
# =====================================================

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


# =====================================================
# CHAT
# =====================================================

class ChatRequest(BaseModel):
    session_id: str
    conversation_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    intent: str
    service: Optional[str] = None

    # RCA fields
    rca: Optional[str] = None
    recommendations: Optional[str] = None


# =====================================================
# PLAN
# =====================================================

class Plan(BaseModel):
    intent: str
    tools: List[str]