import json
from typing import TypedDict

from langchain_groq import ChatGroq
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
    get_lambda_tags
)

from config import GROQ_API_KEY, GROQ_MODEL
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
    "get_lambda_tags": get_lambda_tags
}


# =====================================================
# STATE
# =====================================================

class AgentState(TypedDict, total=False):
    session_id: str
    query: str
    intent: str
    tools: list[str]
    context: str
    service: str
    tool_result: str
    rca: str
    recommendations: str
    answer: str


# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY
)

# IMPORTANT:
# Do NOT use with_structured_output here because your
# current Groq configuration is returning:
# "Tool choice is required, but model did not call a tool"
planner_llm = llm


# =====================================================
# PLANNER NODE
# =====================================================

def planner_node(state):

    prompt = f"""
You are an AWS planning agent.

Return ONLY a valid JSON object with the following keys:

{{
    "intent": "<KNOWLEDGE|MONITORING|ROOT CAUSE ANALYSIS>",
    "tools": ["tool1", "tool2"],
    "services": ["service1", "service2"]
}}

Your responsibilities:

1. Determine whether the query is KNOWLEDGE, MONITORING, or ROOT CAUSE ANALYSIS (RCA).
2. Decide whether AWS account data is required.
3. Select ALL AWS required tools.
4. Select ALL AWS services involved.
5. A query can require multiple tools.
6. Follow up questions may rely on previous context.


Available Tools:

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


Intent Definitions:

KNOWLEDGE:

AWS concepts
Explanations
Documentation
Architecture
Configuration
How-to questions
Best practices
General AWS Learning

A KNOWLEDGE query asks about AWS in general and does not require
investigating the user's actual AWS environment.


MONITORING:

Questions requiring data from the user's AWS account.

Resource Inventory
Resource Counts
Resource status
Storage usage
Existing AWS resources
Storage information
Cost information
Patch compliance information
Networking information

A MONITORING query retrieves or summarizes the current state
of resources in the user's AWS account.


ROOT CAUSE ANALYSIS:

Root cause analysis
Incident investigation
Failure diagnosis
Performance degradation analysis
Security finding analysis
CloudTrail event analysis
CloudWatch anomaly investigation
Questions asking "why"
Questions asking for causes, impact, failures or anomalies
Queries investigating unexpected behavior
Queries investigating service disruptions
Queries investigating operational issues
Troubleshooting
Diagnosis of AWS problems

IMPORTANT RCA RULE:

If the user is investigating a problem, failure, incident, anomaly,
performance issue, security issue, unexpected behavior, degradation,
or asks WHY something is happening, classify the query as
ROOT CAUSE ANALYSIS.

If the query asks to find the cause of a problem in the user's
AWS environment, classify it as ROOT CAUSE ANALYSIS.

Do NOT classify an AWS troubleshooting or investigation query
as KNOWLEDGE simply because it asks for an explanation.

For RCA queries, select the AWS tools required to investigate
the actual problem.

For example:

"Why is my EC2 instance experiencing high CPU?"

must be:

{{
    "intent": "ROOT CAUSE ANALYSIS",
    "tools": [
        "get_cloudwatch_metrics",
        "get_cloudtrail_events"
    ],
    "services": [
        "EC2",
        "CloudWatch",
        "CloudTrail"
    ]
}}


Rules:

- Detect every AWS service mentioned.
- A query can require multiple tools.
- Never omit a required tool.
- Never return invalid tool names.
- Return only the JSON response.
- Always populate services.
- For KNOWLEDGE queries, tools should be empty.
- For KNOWLEDGE queries, services should still contain the AWS services discussed if applicable.
- For MONITORING queries, select all tools required to retrieve the requested AWS account data.
- RCA queries focus on identifying causes, impact, failures, incidents, anomalies, or unexpected behavior.
- RCA queries may require multiple tools.
- Select all relevant tools needed for investigation.
- Consider all AWS services mentioned in the query.
- Include supporting AWS services if they are relevant to the investigation.
- Never assume a single tool is sufficient.
- For RCA, services should include both the affected service and related investigation services.
- Environment-wide RCA queries may require multiple tools.
- If an RCA query involves performance or anomalies, consider CloudWatch and CloudTrail.
- If an RCA query involves security findings, consider Inspector and CloudTrail.
- If an RCA query concerns the overall AWS environment, consider CloudWatch, CloudTrail, and Inspector.
- Do not add explanations outside the JSON object.


Examples:

User:
What is EC2?

Intent:
KNOWLEDGE

Tools:
[]

Services:
["EC2"]

--------------------------------

User:
Explain VPC peering.

Intent:
KNOWLEDGE

Tools:
[]

Services:
["VPC"]

--------------------------------

User:
How many EC2 instances do I have?

Intent:
MONITORING

Tools:
["get_ec2_instances"]

Services:
["EC2"]

--------------------------------

User:
Find resources without an Owner tag.

Intent:
MONITORING

Tools:
["get_resource_tags"]

Services:
["Resource Tagging"]

--------------------------------

User:
Why is my EC2 instance experiencing high CPU?

Intent:
ROOT CAUSE ANALYSIS

Tools:
[
    "get_cloudwatch_metrics",
    "get_cloudtrail_events"
]

Services:
[
    "EC2",
    "CloudWatch",
    "CloudTrail"
]

--------------------------------

User:
How many S3 buckets do I have?

Intent:
MONITORING

Tools:
["get_s3_buckets"]

Services:
["S3"]

--------------------------------

User:
How many EC2 and S3 resources do I have?

Intent:
MONITORING

Tools:
[
    "get_ec2_instances",
    "get_s3_buckets"
]

Services:
[
    "EC2",
    "S3"
]

--------------------------------

User:
Show all EC2 instances along with their tags.

Intent:
MONITORING

Tools:
[
    "get_ec2_instances",
    "get_ec2_tags"
]

Services:
["EC2"]

--------------------------------

User:
Investigate security vulnerabilities in my account.

Intent:
ROOT CAUSE ANALYSIS

Tools:
[
    "get_inspector_findings",
    "get_cloudtrail_events"
]

Services:
[
    "Inspector",
    "CloudTrail"
]

--------------------------------

User:
Show all EC2, S3 and RDS resources.

Intent:
MONITORING

Tools:
[
    "get_ec2_instances",
    "get_s3_buckets",
    "get_rds_instances"
]

Services:
[
    "EC2",
    "S3",
    "RDS"
]

--------------------------------

User:
Which S3 bucket consumes the most storage?

Intent:
MONITORING

Tools:
[
    "get_s3_storage_summary"
]

Services:
[
    "S3"
]

--------------------------------

User:
Show my VPCs.

Intent:
MONITORING

Tools:
[
    "get_vpcs"
]

Services:
[
    "VPC"
]

--------------------------------

User:
Show all networking resources.

Intent:
MONITORING

Tools:
[
    "get_vpcs",
    "get_subnets",
    "get_internet_gateways"
]

Services:
[
    "VPC",
    "Subnet",
    "Internet Gateway"
]

--------------------------------

User:
What is my AWS cost this month?

Intent:
MONITORING

Tools:
[
    "get_cost_summary"
]

Services:
[
    "Cost Explorer"
]

--------------------------------

User:
Which AWS service costs the most?

Intent:
MONITORING

Tools:
[
    "get_cost_by_service"
]

Services:
[
    "Cost Explorer"
]

--------------------------------

User:
Show patch compliance status.

Intent:
MONITORING

Tools:
[
    "get_patch_status"
]

Services:
[
    "SSM"
]

--------------------------------

User:
List all EC2 instances and tell me about their networking.

Intent:
MONITORING

Tools:
[
    "get_ec2_instances",
    "get_vpcs",
    "get_subnets",
    "get_internet_gateways"
]

Services:
[
    "EC2",
    "VPC",
    "Subnet",
    "Internet Gateway"
]

--------------------------------

User:
Analyze the health of my AWS environment.

Intent:
ROOT CAUSE ANALYSIS

Tools:
[
    "get_cloudwatch_metrics",
    "get_cloudtrail_events",
    "get_inspector_findings"
]

Services:
[
    "CloudWatch",
    "CloudTrail",
    "Inspector"
]

--------------------------------

Current User Query:

{state["query"]}

Return ONLY the JSON object.
"""

    try:

        response = planner_llm.invoke(prompt)

        raw = response.content.strip()

        # Remove markdown code fences if the model returns them
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        # Convert JSON text to Python dictionary
        plan = json.loads(raw)

        # -------------------------------------------------
        # IMPORTANT:
        # json.loads() returns a dictionary, not a Pydantic
        # Plan object.
        # -------------------------------------------------

        intent = plan.get("intent", "").strip().upper()

        # Normalize RCA variations
        if intent in [
            "ROOT CAUSE ANALYSIS",
            "ROOT_CAUSE_ANALYSIS",
            "RCA"
        ]:
            intent = "RCA"

        elif intent == "MONITORING":
            intent = "MONITORING"

        elif intent == "KNOWLEDGE":
            intent = "KNOWLEDGE"

        else:
            raise ValueError(
                f"Invalid intent returned by planner: {intent}"
            )

        tools = [
            tool
            for tool in plan.get("tools", [])
            if tool in TOOL_MAP
        ]

        services = plan.get("services", [])

        print("\n===== PLANNER OUTPUT =====")
        print("Intent:", intent)
        print("Tools:", tools)
        print("Services:", ", ".join(services))
        print("==========================\n")

        return {
            "intent": intent,
            "tools": tools,
            "service": ", ".join(services)
        }

    except Exception as e:

        print("Planner Error:", str(e))

        # IMPORTANT:
        # Do NOT convert planner failures into KNOWLEDGE.
        # That was causing RCA queries to incorrectly go to RAG.
        raise


# =====================================================
# KNOWLEDGE NODE
# =====================================================

def knowledge_node(state):

    context = retrieve_context(state["query"])

    return {
        "context": context
    }


# =====================================================
# MONITORING NODE
# =====================================================

def monitoring_node(state):

    session_id = state["session_id"]
    tools = state.get("tools", [])

    print("\nSelected Tools:", tools)

    results = {}

    for tool_name in tools:

        try:

            print(f"Executing {tool_name}")

            tool_function = TOOL_MAP[tool_name]

            results[tool_name] = tool_function(session_id)

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

    prompt = f"""
You are an AWS Root Cause Analysis engine.

Analyze the AWS evidence supplied below.

User Query:
{state["query"]}

AWS Evidence:
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

    response = llm.invoke(prompt)

    return {
        "rca": str(response.content)
    }


# =====================================================
# RECOMMENDATION NODE
# =====================================================

def recommendation_node(state):

    if state["intent"] != "RCA":

        return {
            "recommendations": "Recommendations not required"
        }

    prompt = f"""
You are an AWS remediation and recommendation engine.

User Query:
{state["query"]}

AWS Evidence:
{state.get("tool_result", "")}

Root Cause Analysis:
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

    response = llm.invoke(prompt)

    return {
        "recommendations": str(response.content)
    }


# =====================================================
# FINAL ANSWER NODE
# =====================================================

def final_answer_node(state):

    if state["intent"] == "KNOWLEDGE":

        prompt = f"""
You are an AWS technical assistant.

Question:
{state["query"]}

Knowledge Base:
{state.get("context", "")}

Rules:
- Answer only from the supplied knowledge.
- Do not invent information.
- Provide a practical explanation.
"""

    elif state["intent"] == "MONITORING":

        prompt = f"""
You are an AWS monitoring assistant.

User Question:
{state["query"]}

Tools Executed:
{state.get("tools")}

AWS Results:
{state.get("tool_result")}

Generate a monitoring report.

Rules:

1. Use only AWS Results.
2. Never invent resources.
3. Never invent counts.
4. Never invent names.
5. If multiple services were returned, summarize each service.
6. Provide totals whenever possible.
7. If no resources exist, explicitly state that.
8. Do NOT generate root cause analysis.
9. Do NOT generate recommendations.
10. Do NOT generate impact analysis.
11. If the query asks for a count, provide only the count and resource details.

Format:

## Summary

## AWS Findings
"""

    else:

        prompt = f"""
You are an AWS monitoring and operations assistant.

User Question:
{state["query"]}

Tools Executed:
{state.get("tools")}

AWS Results:
{state.get("tool_result")}

Root Cause Analysis:
{state.get("rca", "")}

Recommendations:
{state.get("recommendations", "")}

Generate a professional response.

Format:

## Summary

## AWS Findings

## Root Cause Analysis

## Recommendations

## Impact

Rules:

1. Use only AWS Results.
2. Never invent resources.
3. Never invent counts.
4. Never invent names.
5. Clearly distinguish confirmed facts from likely causes.
6. If RCA confidence is low, say additional investigation is required.
7. Recommendations must be actionable.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# =====================================================
# ROUTER
# =====================================================

def route_after_planner(state):

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

def run_agent(session_id, query):

    result = graph.invoke(
        {
            "session_id": session_id,
            "query": query,
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
        "recommendations": result.get("recommendations")
    }