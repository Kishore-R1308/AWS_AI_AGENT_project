import json
import os
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

from aws_tools import (
    get_ec2_instances,
    get_rds_instances,
    get_s3_buckets,
    get_s3_storage_summary,
    get_vpcs,
    get_subnets,
    get_internet_gateways,
    get_route_tables,
    get_security_groups,
    get_cost_by_service,
    get_cost_summary,
    get_patch_status,
    get_lambda_functions,
    get_cloudwatch_metrics,
    get_cloudtrail_events,
    get_inspector_findings,
    get_resource_tags,
    get_ec2_tags,
    get_s3_tags,
    get_lambda_tags,
    get_cloudwatch_alarms,
    get_cloudwatch_logs
)

from rag import retrieve_context


# =====================================================
# TOOL MAP
# =====================================================

TOOL_MAP = {
    "get_ec2_instances": get_ec2_instances,
    "get_s3_buckets": get_s3_buckets,
    "get_s3_storage_summary": get_s3_storage_summary,
    "get_rds_instances": get_rds_instances,
    "get_vpcs": get_vpcs,
    "get_subnets": get_subnets,
    "get_internet_gateways": get_internet_gateways,
    "get_route_tables": get_route_tables,
    "get_security_groups": get_security_groups,
    "get_cost_summary": get_cost_summary,
    "get_cost_by_service": get_cost_by_service,
    "get_patch_status": get_patch_status,
    "get_lambda_functions": get_lambda_functions,
    "get_cloudwatch_metrics": get_cloudwatch_metrics,
    "get_cloudtrail_events": get_cloudtrail_events,
    "get_inspector_findings": get_inspector_findings,
    "get_resource_tags": get_resource_tags,
    "get_ec2_tags": get_ec2_tags,
    "get_s3_tags": get_s3_tags,
    "get_lambda_tags": get_lambda_tags,
    "get_cloudwatch_alarms": get_cloudwatch_alarms,
    "get_cloudwatch_logs": get_cloudwatch_logs
}


# =====================================================
# STATE
# =====================================================

class AgentState(TypedDict, total=False):
    session_id: str
    query: str

    # Conversation memory
    history: list[dict[str, str]]
    resolved_query: str

    # Existing agent state
    intent: str
    tools: list[str]
    context: str
    service: str
    tool_result: str
    rca: str
    recommendations: str
    answer: str


# =====================================================
# CONVERSATION CONTEXT
# =====================================================

def _get_history_value(message, *keys):
    """Read a history field while supporting the existing DB/API formats."""
    for key in keys:
        value = message.get(key)
        if value:
            return str(value).strip()
    return ""


def get_last_turn(history: list[dict[str, str]]) -> tuple[str, str]:
    """Return only the immediately previous user/assistant turn."""
    if not history:
        return "", ""

    previous = history[-1]
    user_message = _get_history_value(
        previous, "user", "user_message", "question"
    )
    assistant_message = _get_history_value(
        previous, "assistant", "assistant_message", "answer"
    )

    return user_message, assistant_message


def format_last_turn(history: list[dict[str, str]]) -> str:
    """Format only the immediately previous turn for context resolution."""
    user_message, assistant_message = get_last_turn(history)

    if not user_message and not assistant_message:
        return "No previous conversation."

    if len(user_message) > 3000:
        user_message = user_message[:3000] + "..."

    if len(assistant_message) > 5000:
        assistant_message = assistant_message[:5000] + "..."

    return (
        f"Previous user question:\n{user_message}\n\n"
        f"Previous assistant answer:\n{assistant_message}"
    )


def format_relevant_context(history: list[dict[str, str]]) -> str:
    """
    Return only the previous turn.

    The planner decides whether the current query is actually a follow-up.
    We intentionally do NOT inject the complete chat history into every LLM
    prompt, because that causes unrelated old topics to leak into answers.
    """
    return format_last_turn(history)


# =====================================================
# LLM
# =====================================================

# =====================================================
# MISTRAL LLM
# =====================================================
# Set these in .env / Railway Variables:
# MISTRAL_API_KEY=your_mistral_api_key
# MISTRAL_MODEL=ministral-3b-2512

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.getenv(
    "MISTRAL_MODEL",
    "ministral-3b-2512"
).strip()

if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not configured. "
        "Add it to your .env file or Railway Variables."
    )

# Main LLM for RCA, recommendations and final answers.
llm = ChatOpenAI(
    model=MISTRAL_MODEL,
    temperature=0,
    max_tokens=4096,
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)

# Planner only needs to return a small JSON object.
# Keeping this at 1024 avoids unnecessarily reserving output tokens.
planner_llm = ChatOpenAI(
    model=MISTRAL_MODEL,
    temperature=0,
    max_tokens=1024,
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)


# =====================================================
# PLANNER NODE
# =====================================================

def planner_node(state):

    history_text = format_relevant_context(state.get("history", []))

    prompt = f"""
You are an AWS planning agent.

You are given the current user query and the previous conversation.

Use ONLY the immediately previous user/assistant turn when deciding whether
the current message is a follow-up.

IMPORTANT CONTEXT RULES:
1. A new standalone question must be treated as a new question.
2. Only use the previous turn when the current message clearly refers to it.
3. Resolve references such as "it", "this", "that", "these", "those",
   "the instance", "the metrics", "explain it", "explain this",
   "why", "how", "more details", etc. when they clearly refer to the
   immediately previous turn.
4. Do NOT carry information from the previous turn into a new standalone
   question merely because the previous topic was AWS.
5. If the current question is complete by itself, keep it unchanged.
6. Do not use old conversation turns to change the subject of a standalone
   current question.
7. Do not invent information that is not supported by the current question
   or the immediately previous turn.

Your responsibilities:

1. Determine whether the query is AWS related.
2. Block non-AWS queries.
3. Block requests for credentials, passwords, secrets, tokens or keys but allow logs of CloudWatch and CloudTrail.
4. For valid AWS queries, determine the intent:
- KNOWLEDGE
- MONITORING
- RCA
5. Allow educational security questions.

=====================================================
CONVERSATION HISTORY
=====================================================

{history_text}

=====================================================
CURRENT USER QUERY
=====================================================

{state["query"]}

=====================================================
RESOLVED QUERY
=====================================================

Create a standalone version of the current user request.

The resolved query must include context from the conversation when necessary.

If the current query is already complete, keep its meaning unchanged.

If the current query depends on previous conversation, resolve the missing context using the conversation history.

Examples:

Previous conversation:

User: What is Amazon EC2?

Assistant: Amazon EC2 is a virtual server service provided by AWS.

Current query:

What are its benefits?

The resolved query should represent:

What are the benefits of Amazon EC2?

Another example:

Previous conversation:

User: Show my EC2 instances.

Assistant: Here are the EC2 instances in the AWS account.

Current query:

Which one is running?

The resolved query should represent:

Which of the previously listed EC2 instances are currently running?

Another example:

Previous conversation:

User: What is VPC peering?

Assistant: VPC peering allows two VPCs to communicate privately.

Current query:

How does it work?

The resolved query should represent:

How does Amazon VPC peering work?

Do not invent information that is not available in the conversation.

IMPORTANT FOLLOW-UP EXAMPLES:

Previous conversation:
User: Which EC2 instance is tagged to monitor?
Assistant: The EC2 instance i-05f91e488b8956a1f is tagged Monitoring: MyMonitor.

Current query:
Explain it in 5 lines.

Resolved query:
Explain the monitoring-tagged EC2 instance i-05f91e488b8956a1f in 5 lines.

---

Previous conversation:
User: Which EC2 instance is tagged to monitor?
Assistant: The EC2 instance i-05f91e488b8956a1f is tagged Monitoring: MyMonitor.

Current query:
What is EC2?

Resolved query:
What is EC2?

IMPORTANT: "What is EC2?" is a complete standalone question.
Do NOT resolve it using the previous monitoring-instance answer.

---

Previous conversation:
User: What are the NetworkIn and NetworkOut metrics for my EC2 instances?
Assistant: NetworkIn measures data received and NetworkOut measures data sent.

Current query:
Explain them in 3 lines.

Resolved query:
Explain the NetworkIn and NetworkOut metrics for the user's EC2 instances in 3 lines.

---

Previous conversation:
User: What is VPC peering?
Assistant: VPC peering connects two VPCs privately.

Current query:
What is S3?

Resolved query:
What is S3?

IMPORTANT: This is a new standalone AWS question.

=====================================================
OUTPUT FORMAT
=====================================================

Return ONLY a valid JSON object.

For allowed AWS queries, return:

{{
    "allowed": true,
    "reason": "",
    "intent": "KNOWLEDGE",
    "tools": [],
    "services": [],
    "resolved_query": ""
}}

For blocked queries, return:

{{
    "allowed": false,
    "reason": "Clear explanation of why the request was blocked.",
    "intent": "BLOCKED",
    "tools": [],
    "services": [],
    "resolved_query": ""
}}

=====================================================
AVAILABLE TOOLS
=====================================================

get_ec2_instances
get_s3_buckets
get_s3_storage_summary
get_rds_instances
get_vpcs
get_subnets
get_internet_gateways
get_route_tables
get_security_groups
get_cost_summary
get_cost_by_service
get_patch_status
get_lambda_functions
get_cloudwatch_metrics
get_cloudtrail_events
get_inspector_findings
get_resource_tags
get_ec2_tags
get_s3_tags
get_lambda_tags
get_cloudwatch_alarms
get_cloudwatch_logs

=====================================================
INTENT DEFINITIONS
=====================================================

KNOWLEDGE:

AWS concepts
Explanations
Documentation
Architecture
Configuration
Best practices
General AWS Learning


MONITORING:

Questions requiring live data from the user's AWS account.

This includes:
- Resource inventory
- Resource counts
- Resource status
- Resource usage
- Resource configuration
- AWS costs
- CloudWatch metrics
- CPU utilization
- NetworkIn
- NetworkOut
- Disk metrics
- Status check metrics
- CloudWatch alarms
- CloudWatch logs

If the user asks for metrics, measurements, usage,
performance data, or CloudWatch data from their AWS account,
classify the request as MONITORING unless they are explicitly
asking why a problem occurred. If they ask why a problem
occurred, classify the request as RCA.


ROOT CAUSE ANALYSIS:

Root cause analysis
Incident investigation
Failure diagnosis
Performance degradation analysis
Security finding analysis
Questions asking for causes, impact, failures or anomalies
Troubleshooting
Diagnosis of AWS problems


=====================================================
EXAMPLE
=====================================================

"Why is my EC2 instance experiencing high CPU?"

must be:

{{
    "allowed": true,
    "intent": "RCA",
    "tools": [
        "get_cloudwatch_metrics",
        "get_cloudtrail_events"
    ],
    "services": [
        "EC2",
        "CloudWatch",
        "CloudTrail"
    ],
    "resolved_query": "Why is my EC2 instance experiencing high CPU?"
}}

=====================================================
RULES
=====================================================

- Detect every AWS service relevant to the request.
- Never omit a required tool.
- Return only the JSON response.
- Select all relevant tools needed for the request.
- Include supporting AWS services if they are relevant to the request.
- For EC2 CloudWatch metric questions, select get_cloudwatch_metrics.
- If the question requires identifying the user's EC2 instances before retrieving their metrics, also select get_ec2_instances.
- Include CloudWatch as a service whenever CloudWatch metrics, alarms, or logs are requested.
- Use conversation history when resolving follow-up questions.
- Do not invent missing context.
- Do not add explanations outside the JSON object.

=====================================================
EXAMPLES
=====================================================

Q: How many EC2 instances do I have?

{{
    "allowed": true,
    "intent": "MONITORING",
    "reason": "",
    "tools": ["get_ec2_instances"],
    "services": ["EC2"],
    "resolved_query": "How many EC2 instances do I have?"
}}

Q: Why is my EC2 CPU utilization high?

{{
    "allowed": true,
    "intent": "RCA",
    "reason": "",
    "tools": [
        "get_cloudwatch_metrics",
        "get_cloudtrail_events"
    ],
    "services": [
        "EC2",
        "CloudWatch",
        "CloudTrail"
    ],
    "resolved_query": "Why is my EC2 CPU utilization high?"
}}

Q: What are the NetworkIn and NetworkOut metrics for my EC2 instances over the last 24 hours?

{{
    "allowed": true,
    "intent": "MONITORING",
    "reason": "",
    "tools": [
        "get_ec2_instances",
        "get_cloudwatch_metrics"
    ],
    "services": [
        "EC2",
        "CloudWatch"
    ],
    "resolved_query": "What are the NetworkIn and NetworkOut metrics for my EC2 instances over the last 24 hours?"
}}

Q: Investigate security vulnerabilities in my AWS environment

{{
    "allowed": true,
    "intent": "RCA",
    "reason": "",
    "tools": [
        "get_inspector_findings",
        "get_cloudtrail_events"
    ],
    "services": [
        "Inspector",
        "CloudTrail"
    ],
    "resolved_query": "Investigate security vulnerabilities in my AWS environment"
}}

Q: What is VPC Peering?

{{
    "allowed": true,
    "intent": "KNOWLEDGE",
    "reason": "",
    "tools": [],
    "services": ["VPC"],
    "resolved_query": "What is VPC Peering?"
}}

Q: Show my AWS access keys

{{
    "allowed": false,
    "intent": "BLOCKED",
    "reason": "Requests for credentials or sensitive authentication information are not permitted",
    "tools": [],
    "services": [],
    "resolved_query": "Show my AWS access keys"
}}

Q: Give me my secret access key

{{
    "allowed": false,
    "intent": "BLOCKED",
    "reason": "Passwords and authentication secrets cannot be disclosed.",
    "tools": [],
    "services": [],
    "resolved_query": "Give me my secret access key"
}}

Q: Why is my friend stupid?

{{
    "allowed": false,
    "intent": "BLOCKED",
    "reason": "This query is unrelated to AWS.",
    "tools": [],
    "services": [],
    "resolved_query": "Why is my friend stupid?"
}}

Q: List all the services in my account

{{
    "allowed": true,
    "intent": "MONITORING",
    "reason": "",
    "tools": [
        "get_ec2_instances",
        "get_s3_buckets",
        "get_rds_instances",
        "get_lambda_functions",
        "get_vpcs"
    ],
    "services": [
        "EC2",
        "S3",
        "RDS",
        "Lambda",
        "VPC"
    ],
    "resolved_query": "List all the AWS services and resources available in my account"
}}

Q: Analyze latest alarm

{{
    "allowed": true,
    "intent": "RCA",
    "reason": "",
    "tools": [
        "get_cloudwatch_alarms",
        "get_cloudwatch_metrics"
    ],
    "services": ["CloudWatch"],
    "resolved_query": "Analyze the latest CloudWatch alarm"
}}

=====================================================
FINAL INSTRUCTION
=====================================================

Return ONLY the JSON object.
"""

    try:

        response = planner_llm.invoke(prompt)

        raw = response.content.strip()

        # Remove markdown code fences if the model returns them
        if raw.startswith("```"):
            raw = (
                raw
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        # Convert JSON text to Python dictionary
        plan = json.loads(raw)

        allowed = plan.get(
            "allowed",
            False
        )

        if not allowed:

            reason = plan.get(
                "reason",
                "Request Blocked"
            )

            print("Request Blocked")
            print("Reason:", reason)

            return {
                "intent": "BLOCKED",
                "answer": f"Sorry: {reason}"
            }

        intent = plan.get(
            "intent",
            ""
        ).strip().upper()

        tools = [
            tool
            for tool in plan.get(
                "tools",
                []
            )
            if tool in TOOL_MAP
        ]

        services = plan.get(
            "services",
            []
        )

        resolved_query = plan.get(
            "resolved_query",
            state["query"]
        )

        if not isinstance(
            resolved_query,
            str
        ):
            resolved_query = state["query"]

        resolved_query = resolved_query.strip()

        if not resolved_query:
            resolved_query = state["query"]

        print("\n===== PLANNER OUTPUT =====")
        print("Intent:", intent)
        print("Tools:", tools)
        print("Services:", ", ".join(services))
        print("Resolved Query:", resolved_query)
        print("==========================\n")

        return {
            "intent": intent,
            "tools": tools,
            "services": ", ".join(services),
            "resolved_query": resolved_query
        }

    except Exception as e:

        print(
            "Planner Error:",
            str(e)
        )

        return {
            "intent": "BLOCKED",
            "answer": "Sorry. Unable to classify the request."
        }


# =====================================================
# KNOWLEDGE NODE
# =====================================================

def knowledge_node(state):

    query = state.get(
        "resolved_query",
        state["query"]
    )

    context = retrieve_context(
        query
    )

    return {
        "context": context
    }


# =====================================================
# MONITORING NODE
# =====================================================

def monitoring_node(state):

    session_id = state["session_id"]

    tools = state.get(
        "tools",
        []
    )

    print(
        "\nSelected Tools:",
        tools
    )

    results = {}

    for tool_name in tools:

        try:

            print(
                f"Executing {tool_name}"
            )

            tool_function = TOOL_MAP[
                tool_name
            ]

            results[tool_name] = tool_function(
                session_id
            )

        except Exception as e:

            print(
                f"Error executing {tool_name}: {str(e)}"
            )

            results[tool_name] = {
                "error": str(e)
            }

    return {
        "tool_result": json.dumps(
            results,
            indent=2,
            default=str
        )
    }


# =====================================================
# RCA NODE
# =====================================================

def rca_node(state):

    if state["intent"] != "RCA":

        return {
            "rca": "RCA was not required for this query."
        }

    history_text = format_last_turn(state.get("history", []))

    resolved_query = state.get(
        "resolved_query",
        state["query"]
    )

    prompt = f"""
You are an AWS Root Cause Analysis engine.

Analyze the AWS evidence supplied below.

=====================================================
CONVERSATION HISTORY
=====================================================

{history_text}

=====================================================
CURRENT USER QUERY
=====================================================

{state["query"]}

=====================================================
RESOLVED QUERY
=====================================================

{resolved_query}

=====================================================
AWS EVIDENCE
=====================================================

{state.get("tool_result", "")}

Your job:

1. Identify abnormal behavior.
2. Correlate evidence across AWS services.
3. Determine the most likely root cause.
4. Do not invent missing information.
5. Clearly distinguish facts from assumptions.
6. If there is insufficient evidence, say so.
7. Assign confidence:
   - High
   - Medium
   - Low

Return:

Problem:
<problem>

Evidence:
<important evidence>

Root Cause:
<root cause>

Confidence:
<High/Medium/Low>

Impact:
<impact>
"""

    response = llm.invoke(
        prompt
    )

    return {
        "rca": str(
            response.content
        )
    }


# =====================================================
# RECOMMENDATION NODE
# =====================================================

def recommendation_node(state):

    if state["intent"] != "RCA":

        return {
            "recommendations": "Recommendations not required"
        }

    history_text = format_last_turn(state.get("history", []))

    resolved_query = state.get(
        "resolved_query",
        state["query"]
    )

    prompt = f"""
You are an AWS remediation and recommendation engine.

=====================================================
CONVERSATION HISTORY
=====================================================

{history_text}

=====================================================
CURRENT USER QUERY
=====================================================

{state["query"]}

=====================================================
RESOLVED QUERY
=====================================================

{resolved_query}

=====================================================
AWS EVIDENCE
=====================================================

{state.get("tool_result", "")}

=====================================================
ROOT CAUSE ANALYSIS
=====================================================

{state.get("rca", "")}

Generate practical recommendations.

Rules:

1. Recommendations must be based on the evidence.
2. Do not invent AWS resources.
3. Do not recommend destructive actions automatically.
4. Prioritize recommendations.
5. Explain why each recommendation is useful.
6. Separate investigation from remediation.
7. If remediation could modify infrastructure, mark it as requiring user approval.

Return:

Priority:
Action:
Reason:
Expected Result:

Provide 3-5 recommendations maximum.
"""

    response = llm.invoke(
        prompt
    )

    return {
        "recommendations": str(
            response.content
        )
    }


# =====================================================
# FINAL ANSWER NODE
# =====================================================

def final_answer_node(state):

    if state["intent"] == "BLOCKED":

        return {
            "answer": state["answer"]
        }

    history_text = format_last_turn(state.get("history", []))

    resolved_query = state.get(
        "resolved_query",
        state["query"]
    )

    # =================================================
    # KNOWLEDGE
    # =================================================

    if state["intent"] == "KNOWLEDGE":

        prompt = f"""
You are an AWS technical assistant.

=====================================================
CONVERSATION HISTORY
=====================================================

{history_text}

=====================================================
CURRENT USER QUESTION
=====================================================

{state["query"]}

=====================================================
RESOLVED QUESTION
=====================================================

{resolved_query}

=====================================================
KNOWLEDGE BASE
=====================================================

{state.get("context", "")}

Rules:

- Use only the resolved question.
- The resolved question already contains the required context when this is a follow-up.
- Do not answer an older question.
- Do not carry unrelated information from the previous turn.
- Do not invent information.
- Use only information supported by the Knowledge Base and the resolved question.
- Keep the answer concise unless the user explicitly asks for details.
"""

    # =================================================
    # MONITORING
    # =================================================

    elif state["intent"] == "MONITORING":

        prompt = f"""
You are an AWS monitoring assistant.

=====================================================
CONVERSATION HISTORY
=====================================================

{history_text}

=====================================================
CURRENT USER QUESTION
=====================================================

{state["query"]}

=====================================================
RESOLVED QUESTION
=====================================================

{resolved_query}

=====================================================
TOOLS EXECUTED
=====================================================

{state.get("tools")}

=====================================================
AWS RESULTS
=====================================================

{state.get("tool_result")}

Generate a monitoring report.

Use only the resolved question to identify the requested resource or result.

Rules:

- Answer the resolved question.
- Use the AWS results as the source of truth for account-specific information.
- Do not invent AWS resources or values.
- If the service is to be listed, provide it in tabular format.

Format:

## Summary

## AWS Findings
"""

    # =================================================
    # RCA
    # =================================================

    else:

        prompt = f"""
You are an AWS monitoring and operations assistant.

=====================================================
CONVERSATION HISTORY
=====================================================

{history_text}

=====================================================
CURRENT USER QUESTION
=====================================================

{state["query"]}

=====================================================
RESOLVED QUESTION
=====================================================

{resolved_query}

=====================================================
TOOLS EXECUTED
=====================================================

{state.get("tools")}

=====================================================
AWS RESULTS
=====================================================

{state.get("tool_result")}

=====================================================
ROOT CAUSE ANALYSIS
=====================================================

{state.get("rca", "")}

=====================================================
RECOMMENDATIONS
=====================================================

{state.get("recommendations", "")}

Generate a professional response.

Answer only the resolved question. Do not introduce unrelated information from previous turns.

Format:

## Summary

## AWS Findings

## Root Cause Analysis

## Recommendations

## Impact

Rules:

1. Use only AWS Results for account-specific findings.
2. Clearly distinguish confirmed facts from likely causes.
3. If RCA confidence is low, say additional investigation is required.
4. Recommendations must be actionable.
5. Do not invent AWS resources or evidence.
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content
    }


# =====================================================
# ROUTER
# =====================================================

def route_after_planner(state):

    if state["intent"] == "BLOCKED":

        return "final"

    if state["intent"] == "KNOWLEDGE":

        return "knowledge"

    if state["intent"] == "RCA":

        return "monitoring"

    return "monitoring"


def route_after_monitoring(state):

    if state["intent"] == "RCA":

        return "rca"

    return "final"


# =====================================================
# GRAPH
# =====================================================

builder = StateGraph(
    AgentState
)


builder.add_node(
    "planner",
    planner_node,
)


builder.add_node(
    "knowledge",
    knowledge_node,
)


builder.add_node(
    "rca",
    rca_node,
)


builder.add_node(
    "recommendation",
    recommendation_node,
)


builder.add_node(
    "monitoring",
    monitoring_node,
)


builder.add_node(
    "final",
    final_answer_node,
)


builder.add_edge(
    START,
    "planner",
)


builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "knowledge": "knowledge",
        "monitoring": "monitoring",
        "final": "final"
    },
)


builder.add_edge(
    "knowledge",
    "final",
)


builder.add_conditional_edges(
    "monitoring",
    route_after_monitoring,
    {
        "rca": "rca",
        "final": "final",
    }
)


builder.add_edge(
    "rca",
    "recommendation",
)


builder.add_edge(
    "recommendation",
    "final",
)


builder.add_edge(
    "final",
    END,
)


graph = builder.compile()


# =====================================================
# RUN AGENT
# =====================================================

def run_agent(
    session_id,
    query,
    history=None
):

    if history is None:
        history = []

    result = graph.invoke(
        {
            "session_id": session_id,
            "query": query,
            "history": history,
        }
    )

    return {
        "answer": result["answer"],
        "intent": result["intent"],
        "service": result.get("service"),
        "tools": result.get(
            "tools",
            []
        ),
        "rca": result.get("rca"),
        "recommendations": result.get(
            "recommendations"
        )
    }