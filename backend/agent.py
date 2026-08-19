import json
from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from aws_tools import (
    get_ec2_instances,
    get_rds_instances,
    get_s3_buckets,
    get_s3_storage_summary,
    get_vpcs,
    get_vpc_subnets,
    get_vpc_route_tables,
    get_vpc_internet_gateways,
    get_vpc_security_groups,
)

from config import GROQ_API_KEY, GROQ_MODEL
from rag import retrieve_context


class AgentState(TypedDict, total=False):
    session_id: str
    query: str
    intent: str
    service: str
    context: str
    tool_result: str
    answer: str


llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
)


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_intent(state):
    prompt = f"""
You are an AWS chatbot intent classifier.

Classify the query as exactly one of:

KNOWLEDGE
MONITORING

KNOWLEDGE means:
- AWS concepts
- How-to questions
- Documentation
- Configuration
- Architecture
- General explanations

MONITORING means:
- Current information from the connected AWS account
- Current EC2/S3/RDS/VPC resources
- Current resource states
- Current storage or database information
- Current networking information

Examples:

"How do I create an EC2 instance?"
=> KNOWLEDGE

"What is an S3 bucket?"
=> KNOWLEDGE

"Which EC2 instances are running?"
=> MONITORING

"Which S3 bucket has the highest storage?"
=> MONITORING

"How many VPCs do I have?"
=> MONITORING

"Show my VPCs"
=> MONITORING

"Show my subnets"
=> MONITORING

"Which VPCs do I have?"
=> MONITORING

Return ONLY JSON.

Example:
{{"intent":"KNOWLEDGE"}}

User query:
{state["query"]}
"""

    content = llm.invoke(prompt).content.strip()

    try:
        result = json.loads(content)

        intent = result.get(
            "intent",
            "KNOWLEDGE"
        ).upper()

    except Exception:
        intent = (
            "MONITORING"
            if "MONITOR" in content.upper()
            else "KNOWLEDGE"
        )

    if intent not in {
        "KNOWLEDGE",
        "MONITORING",
    }:
        intent = "KNOWLEDGE"

    return {
        "intent": intent
    }


# ============================================================
# SERVICE DETECTION
# ============================================================

def detect_service(state):
    if state["intent"] == "KNOWLEDGE":
        return {
            "service": ""
        }

    prompt = f"""
Identify the AWS service involved in this monitoring query.

Allowed services:

EC2
S3
RDS
VPC

Important:

VPC questions include:
- VPC
- VPCs
- subnet
- subnets
- CIDR
- route table
- route tables
- internet gateway
- NAT gateway
- security group
- security groups
- networking

Examples:

"Which EC2 instances are running?"
=> {{"service":"EC2"}}

"Show my S3 buckets"
=> {{"service":"S3"}}

"Which RDS databases are available?"
=> {{"service":"RDS"}}

"How many VPCs do I have?"
=> {{"service":"VPC"}}

"Show my VPCs"
=> {{"service":"VPC"}}

"Show my subnets"
=> {{"service":"VPC"}}

"Which VPC has CIDR 10.0.0.0/16?"
=> {{"service":"VPC"}}

"Show my route tables"
=> {{"service":"VPC"}}

"Show my security groups"
=> {{"service":"VPC"}}

Return ONLY JSON.

Query:
{state["query"]}
"""

    content = llm.invoke(prompt).content.strip()

    try:
        result = json.loads(content)

        service = result.get(
            "service",
            "EC2"
        ).upper()

    except Exception:

        query = state["query"].upper()

        # VPC must be checked FIRST
        # because networking questions
        # can sometimes contain EC2/S3 terms.

        if (
            "VPC" in query
            or "SUBNET" in query
            or "CIDR" in query
            or "ROUTE TABLE" in query
            or "INTERNET GATEWAY" in query
            or "NAT GATEWAY" in query
            or "SECURITY GROUP" in query
            or "NETWORKING" in query
        ):
            service = "VPC"

        elif (
            "S3" in query
            or "BUCKET" in query
            or "STORAGE" in query
        ):
            service = "S3"

        elif (
            "RDS" in query
            or "DATABASE" in query
        ):
            service = "RDS"

        else:
            service = "EC2"

    if service not in {
        "EC2",
        "S3",
        "RDS",
        "VPC",
    }:
        service = "EC2"

    return {
        "service": service
    }


# ============================================================
# KNOWLEDGE
# ============================================================

def knowledge_node(state):
    context = retrieve_context(
        state["query"]
    )

    return {
        "context": context
    }


# ============================================================
# MONITORING
# ============================================================

def monitoring_node(state):
    session_id = state["session_id"]

    service = state.get(
        "service",
        "EC2"
    )

    query = state["query"].lower()

    # --------------------------------------------------------
    # EC2
    # --------------------------------------------------------

    if service == "EC2":

        result = get_ec2_instances(
            session_id
        )

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    elif service == "S3":

        storage_words = [
            "storage",
            "size",
            "largest",
            "highest",
        ]

        if any(
            word in query
            for word in storage_words
        ):
            result = get_s3_storage_summary(
                session_id
            )

        else:
            result = get_s3_buckets(
                session_id
            )

    # --------------------------------------------------------
    # RDS
    # --------------------------------------------------------

    elif service == "RDS":

        result = get_rds_instances(
            session_id
        )

    # --------------------------------------------------------
    # VPC
    # --------------------------------------------------------

    elif service == "VPC":

        # VPC list
        if (
            "vpc" in query
            and (
                "how many" in query
                or "count" in query
                or "list" in query
                or "show" in query
                or "which" in query
            )
        ):
            result = get_vpcs(
                session_id
            )

        # Subnets
        elif "subnet" in query:

            result = get_vpc_subnets(
                session_id
            )

        # Route tables
        elif (
            "route table" in query
            or "route tables" in query
        ):

            result = get_vpc_route_tables(
                session_id
            )

        # Internet gateways
        elif (
            "internet gateway" in query
            or "internet gateways" in query
        ):

            result = get_vpc_internet_gateways(
                session_id
            )

        # Security groups
        elif (
            "security group" in query
            or "security groups" in query
        ):

            result = get_vpc_security_groups(
                session_id
            )

        # Default VPC operation
        else:

            result = get_vpcs(
                session_id
            )

    # --------------------------------------------------------
    # Unsupported service
    # --------------------------------------------------------

    else:

        result = {
            "error": (
                f"Unsupported AWS service: "
                f"{service}"
            )
        }

    return {
        "tool_result": json.dumps(
            result,
            indent=2,
            default=str,
        )
    }


# ============================================================
# FINAL ANSWER
# ============================================================

def final_answer_node(state):

    if state["intent"] == "KNOWLEDGE":

        prompt = f"""
You are an AWS technical assistant.

Answer the question using ONLY the supplied AWS knowledge.

Do not invent facts.

Question:
{state["query"]}

AWS knowledge:
{state.get("context", "")}

Give a clear and practical answer.
"""

    else:

        prompt = f"""
You are an AWS monitoring assistant.

The application queried the connected AWS account using boto3.

AWS service:
{state.get("service")}

User question:
{state["query"]}

Actual AWS result:
{state.get("tool_result", "")}

Explain the result clearly.

Important rules:

1. Use ONLY the actual AWS result.
2. Do not invent resources or values.
3. If the result is a list of VPCs, calculate the number
   of VPCs from the list.
4. If there are zero VPCs, clearly say that no VPCs
   were returned.
5. For VPC questions, mention relevant VPC IDs,
   CIDR blocks, state, and default status when available.
"""

    answer = llm.invoke(
        prompt
    ).content

    return {
        "answer": answer
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_intent(state):

    if state["intent"] == "KNOWLEDGE":
        return "knowledge"

    return "service"


# ============================================================
# LANGGRAPH
# ============================================================

builder = StateGraph(
    AgentState
)


builder.add_node(
    "classify_intent",
    classify_intent,
)


builder.add_node(
    "service",
    detect_service,
)


builder.add_node(
    "knowledge",
    knowledge_node,
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
    "classify_intent",
)


builder.add_conditional_edges(
    "classify_intent",
    route_after_intent,
    {
        "knowledge": "knowledge",
        "service": "service",
    },
)


builder.add_edge(
    "knowledge",
    "final",
)


builder.add_edge(
    "service",
    "monitoring",
)


builder.add_edge(
    "monitoring",
    "final",
)


builder.add_edge(
    "final",
    END,
)


graph = builder.compile()


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(
    session_id,
    query
):

    result = graph.invoke(
        {
            "session_id": session_id,
            "query": query,
        }
    )

    return {
        "answer": result["answer"],
        "intent": result["intent"],
        "service": result.get(
            "service"
        ) or None,
    }