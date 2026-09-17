from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class AWSActionRequest(BaseModel):
    session_id: str
    action: Literal["create", "delete", "start", "stop", "reboot", "enable", "disable"]
    resource_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AWSActionPlan(BaseModel):
    action_id: str
    action: Literal["create", "delete", "start", "stop", "reboot", "enable", "disable"]
    resource_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""
    requires_confirmation: bool = True


class AWSActionApprovalRequest(BaseModel):
    session_id: str
    action_id: str
    confirmation_phrase: str


class RCARecommendedAction(BaseModel):
    action_id: str
    action: Literal["create", "delete", "start", "stop", "reboot", "enable", "disable"]
    resource_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""
    status: Literal[
        "pending",
        "approved",
        "running",
        "completed",
        "failed",
        "skipped",
    ] = "pending"


class RCARecommendedActionBatch(BaseModel):
    session_id: str
    issue_id: str
    issue_title: str
    issue_description: str
    actions: List[RCARecommendedAction] = Field(
        default_factory=list
    )


class RCABatchApprovalRequest(BaseModel):
    session_id: str
    batch_id: str
    confirmation_phrase: str


class RCAActionExecutionResult(BaseModel):
    action_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None


class RCABatchExecutionResponse(BaseModel):
    batch_id: str
    status: str
    results: List[RCAActionExecutionResult] = Field(
        default_factory=list
    )