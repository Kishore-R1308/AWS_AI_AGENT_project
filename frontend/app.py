
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
# RESOURCE MANAGER STYLING
# =====================================================

st.markdown(
    """
    <style>
    .resource-card {
        padding: 0.2rem 0;
    }

    .resource-id {
        color: #8b949e;
        font-size: 0.85rem;
        font-family: monospace;
        margin-top: -0.45rem;
    }

    .status-running {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        background: #163b28;
        color: #6ee7a2;
        font-weight: 600;
        font-size: 0.82rem;
    }

    .status-stopped {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        background: #4a1d22;
        color: #ff9aa5;
        font-weight: 600;
        font-size: 0.82rem;
    }

    .status-other {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        background: #4a3a16;
        color: #f8d477;
        font-weight: 600;
        font-size: 0.82rem;
    }

    .detail-label {
        color: #9ca3af;
        font-size: 0.82rem;
        margin-bottom: 0.1rem;
    }

    .detail-value {
        font-size: 0.98rem;
        font-weight: 500;
        word-break: break-word;
    }

    .section-heading {
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0.4rem 0 0.7rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    "live_resources": [],
    "live_resources_service": None,
    "live_resources_refresh_needed": False,
    "live_resources_refresh_service": None,
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


def mark_live_resources_for_refresh(resource_type):
    """Mark the affected service to refresh after a successful AWS action."""
    service_map = {
        "s3_bucket": "s3",
        "ec2_instance": "ec2",
        "rds_instance": "rds",
        "lambda_function": "lambda",
        "security_group": "security_group",
        "vpc": "vpc",
        "subnet": "subnet",
    }

    service = service_map.get(resource_type)

    st.session_state.live_resources_refresh_needed = bool(service)
    st.session_state.live_resources_refresh_service = service


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

        payload = st.session_state.pending_action_payload or {}

        # Show the planned action as a human-readable confirmation card
        # instead of exposing the internal JSON payload.
        action = str(payload.get("action", "unknown")).upper()
        resource_type = str(payload.get("resource_type", "unknown"))
        parameters = payload.get("parameters") or {}
        explanation = payload.get("explanation")

        info_col, resource_col = st.columns(2)

        with info_col:
            st.markdown("**Action**")
            st.info(action)

        with resource_col:
            st.markdown("**Resource**")
            st.info(resource_type.replace("_", " ").title())

        st.markdown("**Target**")
        if parameters:
            for key, value in parameters.items():
                label = key.replace("_", " ").title()
                st.write(f"**{label}:** {value}")
        else:
            st.write("No additional parameters")

        if explanation:
            st.caption(f"Reason: {explanation}")

        st.warning(
            "This action can modify your live AWS environment. "
            "Only confirm if you want AWS to perform this operation."
        )

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

                        # Keep the confirmation workflow only on the
                        # Create / Delete page, but refresh the affected
                        # service when the user opens Live Resources.
                        mark_live_resources_for_refresh(resource_type)
                        reset_pending_action()
                        st.rerun()
                    else:
                        show_error(response, "AWS action execution failed.")

                except requests.exceptions.RequestException as exc:
                    st.error(f"AWS action confirmation failed: {exc}")


# =====================================================
# LIVE AWS RESOURCE MANAGEMENT
# =====================================================

def get_service_icon(service):
    """Return a readable icon for an AWS service."""
    return {
        "ec2": "🖥️",
        "s3": "🪣",
        "rds": "🗄️",
        "lambda": "λ",
        "vpc": "🌐",
        "subnet": "🔗",
        "security_group": "🔐",
    }.get(service, "☁️")


def get_service_title(service):
    """Return a human-readable service title."""
    return {
        "ec2": "EC2 Instances",
        "s3": "S3 Buckets",
        "rds": "RDS Databases",
        "lambda": "Lambda Functions",
        "vpc": "VPCs",
        "subnet": "Subnets",
        "security_group": "Security Groups",
    }.get(service, service.upper())


def format_label(key):
    """Convert API-style keys into readable labels."""
    replacements = {
        "id": "ID",
        "arn": "ARN",
        "ip": "IP",
        "vpc": "VPC",
        "cidr": "CIDR",
        "rds": "RDS",
        "s3": "S3",
        "ec2": "EC2",
        "db": "DB",
        "iam": "IAM",
    }

    words = str(key).replace("_", " ").split()
    result = []

    for word in words:
        lower = word.lower()
        result.append(replacements.get(lower, word.capitalize()))

    return " ".join(result)


def format_value(value):
    """Convert common API values into readable text."""
    if value is None or value == "":
        return "Not available"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (dict, list)):
        return None

    return str(value)


def get_status_class(status):
    """Return the CSS class for a resource status."""
    status = str(status or "unknown").lower()

    if status in {"running", "available", "active", "enabled", "in-use"}:
        return "status-running"

    if status in {"stopped", "failed", "inactive", "disabled", "deleting"}:
        return "status-stopped"

    return "status-other"


def display_detail_grid(items, columns=3):
    """Display flat technical details as a human-readable grid."""
    valid_items = []

    for key, value in items:
        if value is None or value == "":
            display_value = "Not available"
        else:
            display_value = format_value(value)

        if display_value is None:
            continue

        valid_items.append((format_label(key), display_value))

    if not valid_items:
        st.caption("No additional technical information is available.")
        return

    for start in range(0, len(valid_items), columns):
        row = valid_items[start:start + columns]
        cols = st.columns(columns)

        for index, (label, value) in enumerate(row):
            with cols[index]:
                st.markdown(
                    f'<div class="detail-label">{label}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="detail-value">{value}</div>',
                    unsafe_allow_html=True,
                )


def display_nested_list(title, items, key_prefix):
    """Display a list of dictionaries without exposing JSON."""
    if not items:
        st.caption(f"{title}: None")
        return

    st.markdown(f"#### {title}")

    if all(isinstance(item, dict) for item in items):
        for index, item in enumerate(items, start=1):
            with st.container(border=True):
                st.markdown(f"**{title.rstrip('s')} {index}**")
                flat_items = []
                nested_items = []

                for key, value in item.items():
                    if isinstance(value, (dict, list)):
                        nested_items.append((key, value))
                    else:
                        flat_items.append((key, value))

                display_detail_grid(flat_items, columns=3)

                for nested_key, nested_value in nested_items:
                    label = format_label(nested_key)
                    if isinstance(nested_value, list):
                        if nested_value:
                            st.markdown(f"**{label}**")
                            if all(isinstance(x, dict) for x in nested_value):
                                for nested_index, nested_item in enumerate(nested_value, start=1):
                                    st.markdown(f"*{label} {nested_index}*")
                                    display_detail_grid(list(nested_item.items()), columns=3)
                            else:
                                st.write(", ".join(str(x) for x in nested_value))
                    else:
                        st.markdown(f"**{label}:**")
                        display_detail_grid(list(nested_value.items()), columns=3)
    else:
        for index, item in enumerate(items, start=1):
            st.write(f"{index}. {item}")


def display_ec2_details(details):
    st.markdown("#### 🖥️ Instance Information")
    display_detail_grid([
        ("instance_id", details.get("instance_id")),
        ("name", details.get("name")),
        ("state", details.get("state")),
        ("instance_type", details.get("instance_type")),
    ])

    st.divider()
    st.markdown("#### 🌐 Network Information")
    display_detail_grid([
        ("private_ip", details.get("private_ip")),
        ("public_ip", details.get("public_ip")),
        ("availability_zone", details.get("availability_zone")),
        ("vpc_id", details.get("vpc_id")),
        ("subnet_id", details.get("subnet_id")),
    ])

    extra = {
        key: value
        for key, value in details.items()
        if key not in {
            "instance_id", "name", "state", "instance_type",
            "private_ip", "public_ip", "availability_zone",
            "vpc_id", "subnet_id",
        }
    }

    if extra:
        st.divider()
        st.markdown("#### ⚙️ Additional Information")
        display_detail_grid(list(extra.items()), columns=3)


def display_s3_details(details):
    st.markdown("#### 🪣 Bucket Information")
    display_detail_grid([
        ("name", details.get("name")),
        ("created", details.get("created")),
        ("region", details.get("region")),
        ("bucket_arn", details.get("bucket_arn")),
    ])

    extra = {
        key: value
        for key, value in details.items()
        if key not in {"name", "created", "region", "bucket_arn"}
    }

    if extra:
        st.divider()
        st.markdown("#### ⚙️ Additional Information")
        display_detail_grid(list(extra.items()), columns=3)


def display_rds_details(details):
    st.markdown("#### 🗄️ Database Information")
    display_detail_grid([
        ("identifier", details.get("identifier")),
        ("status", details.get("status")),
        ("engine", details.get("engine")),
        ("engine_version", details.get("engine_version")),
        ("instance_class", details.get("instance_class")),
        ("storage_gb", details.get("storage_gb")),
        ("endpoint", details.get("endpoint")),
        ("port", details.get("port")),
        ("availability_zone", details.get("availability_zone")),
    ])

    extra = {
        key: value
        for key, value in details.items()
        if key not in {
            "identifier", "status", "engine", "engine_version",
            "instance_class", "storage_gb", "endpoint", "port",
            "availability_zone",
        }
    }

    if extra:
        st.divider()
        st.markdown("#### ⚙️ Additional Information")
        display_detail_grid(
            [(key, value) for key, value in extra.items()
             if not isinstance(value, (dict, list))],
            columns=3,
        )

        for key, value in extra.items():
            if isinstance(value, list):
                display_nested_list(format_label(key), value, key)
            elif isinstance(value, dict):
                with st.expander(format_label(key), expanded=False):
                    display_detail_grid(list(value.items()), columns=3)


def display_lambda_details(details):
    st.markdown("#### λ Function Information")
    display_detail_grid([
        ("function_name", details.get("function_name")),
        ("runtime", details.get("runtime")),
        ("last_modified", details.get("last_modified")),
        ("arn", details.get("arn")),
        ("handler", details.get("handler")),
        ("memory_size", details.get("memory_size")),
        ("timeout", details.get("timeout")),
    ])

    extra = {
        key: value
        for key, value in details.items()
        if key not in {
            "function_name", "runtime", "last_modified", "arn",
            "handler", "memory_size", "timeout",
        }
    }

    if extra:
        st.divider()
        st.markdown("#### ⚙️ Additional Information")
        display_detail_grid(
            [(key, value) for key, value in extra.items()
             if not isinstance(value, (dict, list))],
            columns=3,
        )
        for key, value in extra.items():
            if isinstance(value, list):
                display_nested_list(format_label(key), value, key)
            elif isinstance(value, dict):
                with st.expander(format_label(key), expanded=False):
                    display_detail_grid(list(value.items()), columns=3)


def display_vpc_details(details):
    st.markdown("#### 🌐 VPC Information")
    display_detail_grid([
        ("vpc_id", details.get("vpc_id")),
        ("name", details.get("name")),
        ("cidr_block", details.get("cidr_block")),
        ("state", details.get("state")),
        ("is_default", details.get("is_default")),
    ])


def display_subnet_details(details):
    st.markdown("#### 🔗 Subnet Information")
    display_detail_grid([
        ("subnet_id", details.get("subnet_id")),
        ("name", details.get("name")),
        ("vpc_id", details.get("vpc_id")),
        ("cidr_block", details.get("cidr_block")),
        ("availability_zone", details.get("availability_zone")),
    ])


def display_security_group_details(details):
    st.markdown("#### 🔐 Security Group Information")
    display_detail_grid([
        ("group_id", details.get("group_id")),
        ("name", details.get("name")),
        ("description", details.get("description")),
        ("vpc_id", details.get("vpc_id")),
    ])

    st.divider()
    display_nested_list(
        "Inbound Rules",
        details.get("inbound_rules") or [],
        "inbound",
    )

    st.divider()
    display_nested_list(
        "Outbound Rules",
        details.get("outbound_rules") or [],
        "outbound",
    )


def display_generic_details(details):
    """Fallback renderer for any future resource type."""
    flat = []
    nested = []

    for key, value in details.items():
        if isinstance(value, (dict, list)):
            nested.append((key, value))
        else:
            flat.append((key, value))

    display_detail_grid(flat, columns=3)

    for key, value in nested:
        label = format_label(key)
        st.divider()
        if isinstance(value, list):
            display_nested_list(label, value, key)
        else:
            st.markdown(f"#### {label}")
            display_detail_grid(list(value.items()), columns=3)


def display_resource_technical_details(service, details):
    """Render technical details in human-readable form for every resource."""
    if service == "ec2":
        display_ec2_details(details)
    elif service == "s3":
        display_s3_details(details)
    elif service == "rds":
        display_rds_details(details)
    elif service == "lambda":
        display_lambda_details(details)
    elif service == "vpc":
        display_vpc_details(details)
    elif service == "subnet":
        display_subnet_details(details)
    elif service == "security_group":
        display_security_group_details(details)
    else:
        display_generic_details(details)


def resource_summary_fields(service, details):
    """Return the small set of fields shown directly on each card."""
    if service == "ec2":
        return [
            ("Instance Type", details.get("instance_type")),
            ("Private IP", details.get("private_ip")),
            ("Public IP", details.get("public_ip")),
            ("State", details.get("state")),
        ]

    if service == "s3":
        return [
            ("Created", details.get("created")),
            ("Region", details.get("region")),
        ]

    if service == "rds":
        return [
            ("Status", details.get("status")),
            ("Engine", details.get("engine")),
            ("Engine Version", details.get("engine_version")),
            ("Instance Class", details.get("instance_class")),
        ]

    if service == "lambda":
        return [
            ("Runtime", details.get("runtime")),
            ("Last Modified", details.get("last_modified")),
        ]

    if service == "vpc":
        return [
            ("CIDR Block", details.get("cidr_block")),
            ("State", details.get("state")),
            ("Default VPC", details.get("is_default")),
        ]

    if service == "subnet":
        return [
            ("VPC ID", details.get("vpc_id")),
            ("CIDR Block", details.get("cidr_block")),
            ("Availability Zone", details.get("availability_zone")),
        ]

    if service == "security_group":
        inbound = details.get("inbound_rules") or []
        outbound = details.get("outbound_rules") or []
        return [
            ("VPC ID", details.get("vpc_id")),
            ("Inbound Rules", len(inbound)),
            ("Outbound Rules", len(outbound)),
        ]

    return []


def get_live_resource_actions(service):
    """Return actions supported by the current backend for an existing resource."""
    return {
        "ec2": ["start", "stop", "reboot", "delete"],
        "s3": ["delete"],
        "rds": ["start", "stop", "delete"],
        "lambda": ["enable", "disable", "delete"],
        "vpc": ["delete"],
        "subnet": ["delete"],
        "security_group": ["delete"],
    }.get(service, [])


def build_live_resource_action(service, details, resource_id, resource_name, action):
    """Build parameters matching backend/action_validation.py and aws_action_executor.py."""
    if service == "ec2":
        parameters = {
            "instance_id": details.get("instance_id") or resource_id,
        }
        resource_type = "ec2_instance"

    elif service == "s3":
        parameters = {
            "bucket_name": details.get("name") or resource_id,
        }
        resource_type = "s3_bucket"

    elif service == "rds":
        parameters = {
            "db_instance_identifier": details.get("identifier") or resource_id,
        }
        resource_type = "rds_instance"
        if action == "delete":
            parameters["skip_final_snapshot"] = False

    elif service == "lambda":
        parameters = {
            "function_name": details.get("function_name") or resource_id,
        }
        resource_type = "lambda_function"

    elif service == "vpc":
        parameters = {
            "vpc_id": details.get("vpc_id") or resource_id,
        }
        resource_type = "vpc"

    elif service == "subnet":
        parameters = {
            "subnet_id": details.get("subnet_id") or resource_id,
        }
        resource_type = "subnet"

    elif service == "security_group":
        parameters = {
            "group_id": details.get("group_id") or resource_id,
        }
        resource_type = "security_group"

    else:
        raise ValueError(f"No live-resource action mapping for {service}")

    # Backend requires the resource identifier to be repeated as the
    # delete_confirmation value for delete plans.
    if action == "delete":
        identifier = (
            parameters.get("bucket_name")
            or parameters.get("instance_id")
            or parameters.get("db_instance_identifier")
            or parameters.get("function_name")
            or parameters.get("group_id")
            or parameters.get("vpc_id")
            or parameters.get("subnet_id")
        )
        parameters["delete_confirmation"] = identifier

    return resource_type, parameters


def display_resource_manager():
    """Display live AWS resources as human-readable resource cards."""
    st.header("☁️ Live AWS Resources")
    st.caption(
        "View live AWS resources in a readable format. "
        "Technical information is displayed as fields and sections, not raw JSON."
    )

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
        format_func=lambda value: (
            f"{get_service_icon(value)} {get_service_title(value)}"
        ),
        key="live_resource_service",
    )

    # A successful Create/Delete action does not display its confirmation
    # here. Instead, when the user opens Live Resources, refresh only the
    # affected service so the newly created/deleted resource is reflected.
    refresh_needed = st.session_state.get(
        "live_resources_refresh_needed", False
    )
    refresh_service = st.session_state.get(
        "live_resources_refresh_service"
    )

    if refresh_needed and refresh_service == service:
        try:
            with st.spinner(
                f"Refreshing {get_service_title(service)}..."
            ):
                response = requests.get(
                    f"{BACKEND_URL}/aws/resources/"
                    f"{st.session_state.session_id}",
                    params={"service": service},
                    timeout=60,
                )

            if response.status_code == 200:
                data = response.json()
                resources = (
                    data.get("resources", [])
                    if isinstance(data, dict)
                    else []
                )
                st.session_state.live_resources = resources
                st.session_state.live_resources_service = service
                st.session_state.live_resources_refresh_needed = False
                st.session_state.live_resources_refresh_service = None
            else:
                show_error(
                    response,
                    "Could not refresh AWS resources after the action.",
                )
        except requests.exceptions.RequestException as exc:
            st.error(f"Resource refresh failed: {exc}")

    if st.button(
        "🔄 Load Live Resources",
        use_container_width=True,
        type="primary",
    ):
        try:
            with st.spinner(
                f"Loading {get_service_title(service)}..."
            ):
                response = requests.get(
                    f"{BACKEND_URL}/aws/resources/"
                    f"{st.session_state.session_id}",
                    params={"service": service},
                    timeout=60,
                )

            if response.status_code != 200:
                show_error(
                    response,
                    "Could not load AWS resources.",
                )
                return

            data = response.json()
            resources = data.get("resources", []) if isinstance(data, dict) else []

            st.session_state.live_resources = resources
            st.session_state.live_resources_service = service

            st.success(
                f"Loaded {len(resources)} {get_service_title(service).lower()}."
            )

        except requests.exceptions.RequestException as exc:
            st.error(f"Resource loading failed: {exc}")
            return

    resources = st.session_state.get("live_resources", [])
    loaded_service = st.session_state.get("live_resources_service")

    if loaded_service != service:
        resources = []

    if not resources:
        st.info(
            "Select an AWS service and click 'Load Live Resources' "
            "to view the resources."
        )
        return

    st.divider()

    title_col, count_col = st.columns([5, 1])
    with title_col:
        st.markdown(
            f"## {get_service_icon(service)} {get_service_title(service)}"
        )
    with count_col:
        st.metric("Resources", len(resources))

    for index, resource in enumerate(resources):
        details = resource.get("details") or {}
        resource_id = resource.get("id") or "Unknown"
        resource_name = resource.get("name") or resource_id

        if not isinstance(details, dict):
            details = {"details": details}

        if service == "ec2":
            title = f"🖥️ {resource_name}"
            status = details.get("state", "unknown")
        elif service == "s3":
            title = f"🪣 {resource_name}"
            status = None
        elif service == "rds":
            title = f"🗄️ {resource_name}"
            status = details.get("status", "unknown")
        elif service == "lambda":
            title = f"λ {resource_name}"
            status = None
        elif service == "vpc":
            title = f"🌐 {resource_name}"
            status = details.get("state", "available")
        elif service == "subnet":
            title = f"🔗 {resource_name}"
            status = None
        else:
            title = f"🔐 {resource_name}"
            status = None

        with st.container(border=True):
            header_col, status_col = st.columns([5, 1])

            with header_col:
                st.markdown(f"### {title}")
                st.markdown(
                    f'<div class="resource-id">{resource_id}</div>',
                    unsafe_allow_html=True,
                )

            if status is not None:
                with status_col:
                    status_class = get_status_class(status)
                    st.markdown(
                        f'<div class="{status_class}">{str(status).title()}</div>',
                        unsafe_allow_html=True,
                    )

            st.divider()

            summary = resource_summary_fields(service, details)
            if summary:
                display_detail_grid(summary, columns=4)
            else:
                st.caption("Resource information available in Technical Details.")

            st.divider()

            details_key = f"show_resource_details_{service}_{index}_{resource_id}"

            if st.button(
                "🔍 View Technical Details"
                if not st.session_state.get(details_key, False)
                else "🔼 Hide Technical Details",
                key=f"resource_details_button_{service}_{index}_{resource_id}",
                use_container_width=True,
            ):
                st.session_state[details_key] = not st.session_state.get(
                    details_key,
                    False,
                )
                st.rerun()

            if st.session_state.get(details_key, False):
                st.divider()
                with st.container(border=True):
                    display_resource_technical_details(
                        service,
                        details,
                    )

            # RCA is deliberately not shown here.
            # RCA remains available through the main AI Agent chat.



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