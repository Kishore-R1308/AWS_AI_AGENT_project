
import os
import json
import uuid
import requests
import streamlit as st


# =====================================================
# CONFIGURATION
# =====================================================

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="AWS AI Agent",
    page_icon="☁️",
    layout="wide",
)


# =====================================================
# SESSION STATE
# =====================================================

DEFAULT_STATE = {
    "session_id": None,
    "conversation_id": str(uuid.uuid4()),
    "aws_connected": False,
    "account_id": None,
    "messages": [],
    "conversations": [],
    "pending_batch_id": None,
    "pending_batch_payload": None,
    "pending_action_id": None,
    "pending_action_payload": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# GENERAL HELPERS
# =====================================================

def show_error(response, default_message="Request failed."):
    """Display a readable backend error."""

    try:
        detail = response.json().get(
            "detail",
            response.text or default_message,
        )
    except Exception:
        detail = response.text or default_message

    st.error(detail)


def parse_json_safely(value, default=None):
    """Safely parse JSON strings or dictionaries."""

    if default is None:
        default = {}

    if not value:
        return default

    if isinstance(value, (dict, list)):
        return value

    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def parse_recommendations(value):
    """Parse recommendation payload."""

    result = parse_json_safely(value, {})

    if not isinstance(result, dict):
        return {}

    return result


def reset_pending_batch():
    st.session_state.pending_batch_id = None
    st.session_state.pending_batch_payload = None


def reset_pending_action():
    st.session_state.pending_action_id = None
    st.session_state.pending_action_payload = None


# =====================================================
# CONVERSATION FUNCTIONS
# =====================================================

def load_conversations():
    """Load conversations for the connected AWS account."""

    account_id = st.session_state.account_id

    if not account_id:
        return

    try:
        response = requests.get(
            f"{BACKEND_URL}/conversations/{account_id}",
            timeout=15,
        )

        if response.status_code == 200:
            data = response.json()

            if isinstance(data, list):
                st.session_state.conversations = data
            else:
                st.session_state.conversations = []

    except requests.exceptions.RequestException:
        st.session_state.conversations = []


def load_conversation(conversation_id):
    """Load one conversation from the backend."""

    account_id = st.session_state.account_id

    if not account_id:
        return

    try:
        response = requests.get(
            f"{BACKEND_URL}/history/"
            f"{account_id}/{conversation_id}",
            timeout=15,
        )

        if response.status_code != 200:
            show_error(
                response,
                "Could not load conversation.",
            )
            return

        history = response.json()

        st.session_state.messages = []

        for index, item in enumerate(history):
            st.session_state.messages.append(
                {
                    "user": item.get("user_message", ""),
                    "assistant": item.get(
                        "assistant_message",
                        "",
                    ),
                    "intent": item.get(
                        "intent",
                        "UNKNOWN",
                    ),
                    "service": item.get("service"),
                    "rca": item.get("rca"),
                    "recommendations": item.get(
                        "recommendations"
                    ),
                    "message_key": (
                        f"history_{conversation_id}_{index}"
                    ),
                }
            )

    except requests.exceptions.RequestException as exc:
        st.error(f"Could not load conversation: {exc}")
        st.session_state.messages = []


def create_new_chat():
    """Create a new conversation."""

    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = []

    reset_pending_batch()
    reset_pending_action()


def delete_conversation(conversation_id):
    """Delete a conversation."""

    account_id = st.session_state.account_id

    if not account_id:
        return

    try:
        response = requests.delete(
            f"{BACKEND_URL}/history/"
            f"{account_id}/{conversation_id}",
            timeout=15,
        )

        if response.status_code == 200:
            if (
                conversation_id
                == st.session_state.conversation_id
            ):
                create_new_chat()

            load_conversations()
            st.success("Conversation deleted.")
            st.rerun()

        else:
            show_error(
                response,
                "Could not delete conversation.",
            )

    except requests.exceptions.RequestException as exc:
        st.error(f"Delete failed: {exc}")


# =====================================================
# RCA RECOMMENDATION DISPLAY
# =====================================================

def display_rca_recommendations(
    recommendations,
    key_prefix,
):
    """Display recommendations and batch approval."""

    payload = parse_recommendations(recommendations)

    items = payload.get("recommendations") or []
    actions = payload.get("actions") or []

    if not items and not actions:
        return

    st.markdown("### 🛠️ Recommended Actions")

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            st.write(f"{index}. {item}")
            continue

        priority = item.get("priority", "normal")
        action_name = item.get("action", "")
        reason = item.get("reason", "")
        expected_result = item.get(
            "expected_result",
            "",
        )

        st.markdown(
            f"**{index}. [{priority}] {action_name}**"
        )

        if reason:
            st.write(f"Reason: {reason}")

        if expected_result:
            st.write(
                f"Expected result: {expected_result}"
            )

    valid_actions = [
        action
        for action in actions
        if (
            isinstance(action, dict)
            and action.get("action")
            in {"create", "delete", "start", "stop", "reboot", "enable", "disable"}
            and action.get("resource_type")
            and isinstance(
                action.get("parameters"),
                dict,
            )
        )
    ]

    if not valid_actions:
        if actions:
            st.warning(
                "The recommendations are informational "
                "because executable parameters are incomplete."
            )

        return

    st.warning(
        "These actions can modify AWS resources. "
        "Review everything before approval."
    )

    with st.expander(
        "Review executable actions",
        expanded=False,
    ):
        st.json(valid_actions)

    current_batch_id = st.session_state.pending_batch_id

    if (
        current_batch_id is None
        and st.button(
            "✅ Approve & Prepare Batch",
            key=f"prepare_batch_{key_prefix}",
            use_container_width=True,
        )
    ):
        issue_title = payload.get(
            "issue_title",
            "RCA remediation",
        )

        issue_description = payload.get(
            "issue_description",
            "Recommended remediation actions",
        )

        plan_payload = {
            "session_id": st.session_state.session_id,
            "issue_id": str(uuid.uuid4()),
            "issue_title": issue_title,
            "issue_description": issue_description,
            "actions": valid_actions,
        }

        try:
            with st.spinner(
                "Creating approval batch..."
            ):
                response = requests.post(
                    f"{BACKEND_URL}/aws/rca/batch/plan",
                    json=plan_payload,
                    timeout=30,
                )

            if response.status_code != 200:
                show_error(
                    response,
                    "Could not create approval batch.",
                )
                return

            response_data = response.json()
            batch = response_data.get("batch", {})
            batch_id = batch.get("batch_id") or batch.get("id")

            if not batch_id:
                st.error(
                    "The backend did not return a batch ID."
                )
                return

            st.session_state.pending_batch_id = batch_id
            st.session_state.pending_batch_payload = (
                plan_payload
            )

            st.success(
                "Batch prepared. Confirmation is required."
            )

            st.rerun()

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Batch planning request failed: {exc}"
            )

    current_batch_id = st.session_state.pending_batch_id

    if current_batch_id:
        st.markdown("### 🔐 Confirm Batch Execution")

        st.info(
            f"Pending batch ID: `{current_batch_id}`"
        )

        confirmation = st.text_input(
            "Type exactly: I CONFIRM THIS AWS ACTION",
            key=f"batch_confirmation_{key_prefix}",
        )

        confirm_col, cancel_col = st.columns(2)

        with confirm_col:
            confirm_clicked = st.button(
                "🚀 Confirm and Run",
                key=f"run_batch_{key_prefix}",
                use_container_width=True,
            )

        with cancel_col:
            cancel_clicked = st.button(
                "❌ Cancel",
                key=f"cancel_batch_{key_prefix}",
                use_container_width=True,
            )

        if cancel_clicked:
            reset_pending_batch()
            st.warning("Batch approval cancelled.")
            st.rerun()

        if confirm_clicked:
            if (
                confirmation.strip()
                != "I CONFIRM THIS AWS ACTION"
            ):
                st.error(
                    "Incorrect confirmation phrase."
                )
                return

            try:
                with st.spinner(
                    "Executing AWS actions sequentially..."
                ):
                    response = requests.post(
                        f"{BACKEND_URL}/aws/rca/batch/confirm",
                        json={
                            "session_id": (
                                st.session_state.session_id
                            ),
                            "batch_id": current_batch_id,
                            "confirmation_phrase": (
                                confirmation.strip()
                            ),
                        },
                        timeout=300,
                    )

                if response.status_code == 200:
                    result = response.json()

                    st.success(
                        result.get(
                            "message",
                            "Batch execution completed.",
                        )
                    )

                    results = result.get(
                        "results",
                        result,
                    )

                    st.markdown(
                        "### 📊 Execution Results"
                    )
                    st.json(results)

                    reset_pending_batch()

                else:
                    show_error(
                        response,
                        "Batch execution failed.",
                    )

            except requests.exceptions.RequestException as exc:
                st.error(
                    f"Batch execution request failed: {exc}"
                )


# =====================================================
# CREATE / DELETE RESOURCE MANAGEMENT
# =====================================================

CONFIRMATION_PHRASE = "I CONFIRM THIS AWS ACTION"


def post_action_plan(action_type, resource_type, parameters, explanation):
    """Send a single action plan to the backend and store its ID."""
    payload = {
        "session_id": st.session_state.session_id,
        "action": action_type,
        "resource_type": resource_type,
        "parameters": parameters,
        "explanation": explanation,
    }

    try:
        with st.spinner("Preparing AWS action..."):
            response = requests.post(
                f"{BACKEND_URL}/aws/action/plan",
                json=payload,
                timeout=30,
            )

        if response.status_code != 200:
            show_error(response, "Could not plan AWS action.")
            return

        data = response.json()
        action = data.get("action", data)

        action_id = action.get("action_id") or action.get("id")
        if not action_id:
            st.error("Backend did not return an action ID.")
            return

        st.session_state.pending_action_id = action_id
        st.session_state.pending_action_payload = payload
        st.success("Action prepared. Review and confirm it below.")
        st.rerun()

    except requests.exceptions.RequestException as exc:
        st.error(f"Action planning request failed: {exc}")


def render_resource_fields(action_type, resource_type):
    """Render dynamic fields and return the parameters dictionary."""
    parameters = {}

    def text_field(label, key, required=True, password=False, placeholder=""):
        value = st.text_input(
            label,
            key=f"resource_{key}",
            placeholder=placeholder,
            type="password" if password else "default",
        )
        if required and not value.strip():
            st.warning(f"{label} is required.")
        if value.strip():
            parameters[key] = value.strip()
        return value.strip()

    def number_field(label, key, default=1, minimum=1):
        value = st.number_input(
            label,
            min_value=minimum,
            value=default,
            step=1,
            key=f"resource_{key}",
        )
        parameters[key] = int(value)
        return int(value)

    if resource_type == "s3_bucket":
        text_field("Bucket name", "bucket_name")

    elif resource_type == "ec2_instance":
        if action_type == "create":
            text_field("AMI / Image ID", "ami_id", placeholder="ami-xxxxxxxx")
            text_field("Instance type", "instance_type", placeholder="t3.micro")
            number_field("Minimum count", "min_count", 1)
            number_field("Maximum count", "max_count", 1)
            text_field("Subnet ID (optional)", "subnet_id", required=False)
            security_group_ids = st.text_input(
                "Security group IDs (comma-separated, optional)",
                key="resource_security_group_ids",
            )
            if security_group_ids.strip():
                parameters["security_group_ids"] = [
                    item.strip() for item in security_group_ids.split(",") if item.strip()
                ]
            text_field("Key pair name", "key_name")
        else:
            text_field("Instance ID", "instance_id")
            text_field(
                "Delete confirmation",
                "delete_confirmation",
                placeholder="Type the instance ID or required confirmation",
            )

    elif resource_type == "rds_instance":
        if action_type == "create":
            text_field("DB instance identifier", "db_instance_identifier")
            text_field("DB instance class", "db_instance_class", placeholder="db.t3.micro")
            text_field("Database engine", "engine", placeholder="mysql")
            text_field("Master username", "master_username")
            text_field(
                "Master user password",
                "master_password",
                password=True,
            )
            number_field("Allocated storage (GB)", "allocated_storage", 20)
        else:
            text_field("DB instance identifier", "db_instance_identifier")
            st.checkbox(
                "Skip final snapshot",
                value=False,
                key="resource_skip_final_snapshot",
            )
            parameters["skip_final_snapshot"] = st.session_state.resource_skip_final_snapshot
            text_field("Delete confirmation", "delete_confirmation")

    elif resource_type == "lambda_function":
        if action_type == "create":
            text_field("Function name", "function_name")
            text_field("Runtime", "runtime", placeholder="python3.12")
            text_field("Execution role ARN", "role_arn")
            text_field("Handler", "handler", placeholder="lambda_function.lambda_handler")
            text_field(
                "ZIP file path / reference",
                "zip_file",
                placeholder="Path or backend-supported ZIP reference",
            )
        else:
            text_field("Function name", "function_name")
            text_field("Delete confirmation", "delete_confirmation")

    elif resource_type == "security_group":
        if action_type == "create":
            text_field("Group name", "group_name")
            text_field("Description", "description")
            text_field("VPC ID", "vpc_id")
        else:
            text_field("Security group ID", "group_id")
            text_field("Delete confirmation", "delete_confirmation")

    elif resource_type == "vpc":
        if action_type == "create":
            text_field("CIDR block", "cidr_block", placeholder="10.0.0.0/16")
        else:
            text_field("VPC ID", "vpc_id")
            text_field("Delete confirmation", "delete_confirmation")

    elif resource_type == "subnet":
        if action_type == "create":
            text_field("VPC ID", "vpc_id")
            text_field("CIDR block", "cidr_block", placeholder="10.0.1.0/24")
            text_field("Availability zone", "availability_zone")
        else:
            text_field("Subnet ID", "subnet_id")
            text_field("Delete confirmation", "delete_confirmation")

    elif resource_type == "iam_resource":
        iam_type = st.selectbox(
            "IAM resource type",
            ["role", "user", "group"],
            key="resource_iam_type",
        )
        parameters["resource_kind"] = iam_type
        text_field("IAM resource name", "resource_name")
        if action_type == "create" and iam_type == "role":
            st.text_area(
                "Assume role policy JSON",
                value='{"Version":"2012-10-17","Statement":[]}',
                key="resource_assume_role_policy",
            )
            policy = st.session_state.resource_assume_role_policy
            if policy.strip():
                parameters["assume_role_policy"] = policy
        if action_type == "delete":
            text_field("Delete confirmation", "delete_confirmation")

    return parameters


def display_single_action_manager():
    """Dynamic create/delete AWS resource form with confirmation."""
    st.header("⚙️ Create / Delete Resource")

    action_type = st.radio(
        "Choose operation",
        ["create", "delete"],
        horizontal=True,
        key="resource_action_type",
    )

    resource_options = {
        "S3 Bucket": "s3_bucket",
        "EC2 Instance": "ec2_instance",
        "RDS Instance": "rds_instance",
        "Lambda Function": "lambda_function",
        "Security Group": "security_group",
        "VPC": "vpc",
        "Subnet": "subnet",
        "IAM Resource": "iam_resource",
    }

    resource_label = st.selectbox(
        "Choose AWS resource",
        list(resource_options.keys()),
        key="resource_type_label",
    )
    resource_type = resource_options[resource_label]

    st.info(
        "Fill in the required fields. The action will not execute until "
        "you enter the exact confirmation phrase."
    )

    with st.form("dynamic_resource_action_form", clear_on_submit=False):
        parameters = render_resource_fields(action_type, resource_type)
        explanation = st.text_area(
            "Reason for this action",
            value=f"User requested {action_type} for {resource_label}.",
            key="resource_explanation",
        )
        submitted = st.form_submit_button(
            "📋 Review Action",
            use_container_width=True,
        )

    if submitted:
        missing = [
            key for key, value in parameters.items()
            if isinstance(value, str) and not value.strip()
        ]

        if resource_type == "ec2_instance" and action_type == "create":
            for required_key in ("image_id", "instance_type"):
                if not parameters.get(required_key):
                    missing.append(required_key)

        if missing:
            st.error("Please complete all required fields before continuing.")
        else:
            post_action_plan(
                action_type,
                resource_type,
                parameters,
                explanation,
            )

    pending_action_id = st.session_state.pending_action_id
    if pending_action_id:
        st.divider()
        st.subheader("🔐 Confirm AWS Action")
        st.write("Review the planned action before execution.")

        if st.session_state.pending_action_payload:
            st.json(st.session_state.pending_action_payload)

        confirmation = st.text_input(
            f"Type exactly: {CONFIRMATION_PHRASE}",
            key="single_action_confirmation",
        )

        confirm_col, cancel_col = st.columns(2)

        with confirm_col:
            confirm_clicked = st.button(
                "🚀 Confirm and Execute",
                use_container_width=True,
                key="confirm_dynamic_action",
            )

        with cancel_col:
            cancel_clicked = st.button(
                "❌ Cancel",
                use_container_width=True,
                key="cancel_dynamic_action",
            )

        if cancel_clicked:
            reset_pending_action()
            st.rerun()

        if confirm_clicked:
            if confirmation.strip() != CONFIRMATION_PHRASE:
                st.error("Incorrect confirmation phrase.")
            else:
                try:
                    with st.spinner("Executing AWS action..."):
                        response = requests.post(
                            f"{BACKEND_URL}/aws/action/confirm",
                            json={
                                "session_id": st.session_state.session_id,
                                "action_id": pending_action_id,
                                "confirmation_phrase": confirmation.strip(),
                            },
                            timeout=300,
                        )

                    if response.status_code == 200:
                        st.success("AWS action execution completed.")
                        st.json(response.json())
                        reset_pending_action()
                    else:
                        show_error(response, "AWS action execution failed.")

                except requests.exceptions.RequestException as exc:
                    st.error(f"AWS action confirmation failed: {exc}")


# =====================================================
# LIVE AWS RESOURCE MANAGEMENT
# =====================================================

def display_resource_manager():
    """Display live AWS resources."""

    st.header("☁️ Live AWS Resources")

    service = st.selectbox(
        "AWS service",
        [
            "ec2",
            "s3",
            "rds",
            "lambda",
            "vpc",
            "subnet",
            "security_group",
        ],
        key="live_resource_service",
    )

    if st.button(
        "🔄 Load Live Resources",
        use_container_width=True,
    ):
        try:
            response = requests.get(
                f"{BACKEND_URL}/aws/resources/"
                f"{st.session_state.session_id}",
                params={
                    "service": service,
                },
                timeout=60,
            )

            if response.status_code != 200:
                show_error(
                    response,
                    "Could not load AWS resources.",
                )
                return

            resources = response.json()

            st.success(
                f"Loaded {service.upper()} resources."
            )

            if isinstance(resources, list):
                if resources:
                    st.dataframe(
                        resources,
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "No resources found."
                    )
            else:
                st.json(resources)

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Resource loading failed: {exc}"
            )


# =====================================================
# PAGE TITLE
# =====================================================

st.title("☁️ AWS AI Agent")


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.header("💬 Conversations")

    if st.session_state.aws_connected:
        if st.button(
            "➕ New Chat",
            use_container_width=True,
        ):
            create_new_chat()
            st.rerun()

        st.divider()
        load_conversations()

        if st.session_state.conversations:
            for conversation in (
                st.session_state.conversations
            ):
                conversation_id = conversation.get(
                    "conversation_id"
                )

                title = conversation.get(
                    "title",
                    "New Conversation",
                )

                if len(title) > 35:
                    title = title[:35] + "..."

                is_active = (
                    conversation_id
                    == st.session_state.conversation_id
                )

                col1, col2 = st.columns([5, 1])

                with col1:
                    if st.button(
                        (
                            f"🟢 {title}"
                            if is_active
                            else f"💬 {title}"
                        ),
                        key=f"chat_{conversation_id}",
                        use_container_width=True,
                    ):
                        st.session_state.conversation_id = (
                            conversation_id
                        )

                        load_conversation(
                            conversation_id
                        )

                        st.rerun()

                with col2:
                    if st.button(
                        "🗑️",
                        key=f"delete_{conversation_id}",
                    ):
                        delete_conversation(
                            conversation_id
                        )

        else:
            st.caption(
                "No previous conversations."
            )

    else:
        st.caption(
            "Connect to AWS to view conversations."
        )

    st.divider()

    st.header("🔗 AWS Connection")

    st.write(
        "Connect using a cross-account IAM role."
    )

    access_key = st.text_input(
        "AWS Access Key",
        type="password",
    )

    secret_key = st.text_input(
        "AWS Secret Key",
        type="password",
    )

    region = st.text_input(
        "AWS Region",
        value="us-east-1",
    )

    role_arn = st.text_input(
        "Cross Account Role ARN",
        placeholder=(
            "arn:aws:iam::123456789012:"
            "role/AIAgentReadOnlyRole"
        ),
    )

    connect_clicked = st.button(
        "🔗 Connect to AWS",
        use_container_width=True,
    )

    if connect_clicked:
        if (
            not access_key
            or not secret_key
            or not role_arn
        ):
            st.error(
                "Access key, secret key and role ARN "
                "are required."
            )

        else:
            try:
                with st.spinner(
                    "Authenticating with AWS..."
                ):
                    response = requests.post(
                        f"{BACKEND_URL}/aws/connect",
                        json={
                            "access_key": access_key,
                            "secret_key": secret_key,
                            "region": region,
                            "role_arn": role_arn,
                        },
                        timeout=30,
                    )

                if response.status_code == 200:
                    data = response.json()

                    st.session_state.session_id = (
                        data["session_id"]
                    )

                    st.session_state.account_id = (
                        data["account_id"]
                    )

                    st.session_state.aws_connected = True
                    st.session_state.conversation_id = (
                        str(uuid.uuid4())
                    )
                    st.session_state.messages = []

                    reset_pending_batch()
                    reset_pending_action()

                    load_conversations()

                    st.success(
                        "AWS Connected Successfully."
                    )

                    st.rerun()

                else:
                    show_error(
                        response,
                        "AWS connection failed.",
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI backend. "
                    "Start Uvicorn on port 8000."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "AWS connection request timed out."
                )

            except requests.exceptions.RequestException as exc:
                st.error(
                    f"AWS connection failed: {exc}"
                )

    st.divider()

    if st.session_state.aws_connected:
        st.subheader("🛠️ Resource Management")
        st.caption("Create or delete AWS resources from the main panel.")
        st.session_state.resource_page = st.radio(
            "Open",
            ["Chat", "Create / Delete Resource", "Live Resources"],
            key="resource_page_selector",
        )
        st.divider()
        st.success("🟢 AWS Connected")
        st.write(
            f"Account: `{st.session_state.account_id}`"
        )
    else:
        st.warning("🔴 AWS Not Connected")


# =====================================================
# AUTHENTICATION CHECK
# =====================================================

if not st.session_state.aws_connected:
    st.info(
        "Connect your AWS account from the sidebar "
        "to use the agent."
    )

    st.stop()


# =====================================================
# RESOURCE MANAGEMENT / MAIN PAGE
# =====================================================

resource_page = st.session_state.get("resource_page", "Chat")

if resource_page == "Create / Delete Resource":
    display_single_action_manager()
elif resource_page == "Live Resources":
    display_resource_manager()


# =====================================================
# MAIN CHAT

if resource_page != "Chat":
    st.stop()

st.divider()
st.subheader("💬 Chat")


for message in st.session_state.messages:
    with st.chat_message("user"):
        st.write(message.get("user", ""))

    with st.chat_message("assistant"):
        st.write(
            message.get("assistant", "")
        )

        recommendations = message.get(
            "recommendations"
        )

        if recommendations:
            display_rca_recommendations(
                recommendations,
                message.get(
                    "message_key",
                    str(uuid.uuid4()),
                ),
            )

        metadata = [
            f"Intent: {message.get('intent', 'UNKNOWN')}"
        ]

        if (
            message.get("intent") == "MONITORING"
            and message.get("service")
        ):
            metadata.append(
                f"Service: {message['service']}"
            )

        st.caption(" | ".join(metadata))


# =====================================================
# CHAT INPUT
# =====================================================

prompt = st.chat_input(
    "Ask about AWS..."
)

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner(
            "Agent is thinking..."
        ):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "session_id": (
                            st.session_state.session_id
                        ),
                        "conversation_id": (
                            st.session_state.conversation_id
                        ),
                        "message": prompt,
                    },
                    timeout=180,
                )

                if response.status_code != 200:
                    show_error(
                        response,
                        "Agent request failed.",
                    )
                    st.stop()

                data = response.json()

                answer = data.get(
                    "answer",
                    "No answer returned.",
                )

                intent = data.get(
                    "intent",
                    "UNKNOWN",
                )

                service = data.get("service")
                recommendations = data.get(
                    "recommendations"
                )

                st.write(answer)

                if recommendations:
                    display_rca_recommendations(
                        recommendations,
                        f"current_{uuid.uuid4().hex}",
                    )

                caption = f"Intent: {intent}"

                if (
                    intent == "MONITORING"
                    and service
                ):
                    caption += (
                        f" | Service: {service}"
                    )

                st.caption(caption)

                st.session_state.messages.append(
                    {
                        "user": prompt,
                        "assistant": answer,
                        "intent": intent,
                        "service": service,
                        "rca": data.get("rca"),
                        "recommendations": (
                            recommendations
                        ),
                        "message_key": (
                            uuid.uuid4().hex
                        ),
                    }
                )

                load_conversations()

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI backend. "
                    "Make sure Uvicorn is running."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The chat request timed out."
                )

            except requests.exceptions.RequestException as exc:
                st.error(
                    f"Chat request failed: {exc}"
                )

            except Exception as exc:
                st.error(
                    f"Unexpected error: {exc}"
                )