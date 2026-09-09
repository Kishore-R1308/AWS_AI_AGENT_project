import uuid

import requests
import streamlit as st


# =====================================================
# CONFIGURATION
# =====================================================

BACKEND_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="AWS AI Agent",
    page_icon="☁️",
    layout="wide",
)


# =====================================================
# SESSION STATE
# =====================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "aws_connected" not in st.session_state:
    st.session_state.aws_connected = False

if "account_id" not in st.session_state:
    st.session_state.account_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversations" not in st.session_state:
    st.session_state.conversations = []


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def load_conversations():
    """
    Load the list of conversations belonging to
    the currently connected AWS account.
    """

    if not st.session_state.account_id:
        return

    try:
        response = requests.get(
            f"{BACKEND_URL}/conversations/"
            f"{st.session_state.account_id}",
            timeout=10,
        )

        if response.status_code == 200:
            st.session_state.conversations = (
                response.json()
            )

    except Exception:
        st.session_state.conversations = []


def load_conversation(conversation_id):
    """
    Load messages for one specific conversation.
    """

    if not st.session_state.account_id:
        return

    try:
        response = requests.get(
            f"{BACKEND_URL}/history/"
            f"{st.session_state.account_id}/"
            f"{conversation_id}",
            timeout=10,
        )

        if response.status_code == 200:

            history = response.json()

            st.session_state.messages = [
                {
                    "user": item["user_message"],
                    "assistant": item[
                        "assistant_message"
                    ],
                    "intent": item["intent"],
                    "service": item.get("service"),
                }
                for item in history
            ]

    except Exception as exc:
        st.error(
            f"Could not load conversation: {exc}"
        )
        st.session_state.messages = []


def create_new_chat():
    """
    Create a completely new conversation while
    keeping the existing AWS session alive.
    """

    st.session_state.conversation_id = (
        str(uuid.uuid4())
    )

    st.session_state.messages = []


def delete_conversation(conversation_id):
    """
    Delete one conversation from the backend.
    """

    if not st.session_state.account_id:
        return

    try:
        response = requests.delete(
            f"{BACKEND_URL}/history/"
            f"{st.session_state.account_id}/"
            f"{conversation_id}",
            timeout=10,
        )

        if response.status_code == 200:

            if (
                conversation_id
                == st.session_state.conversation_id
            ):
                create_new_chat()

            load_conversations()

            st.rerun()

        else:
            try:
                detail = response.json().get(
                    "detail",
                    "Could not delete conversation.",
                )
            except Exception:
                detail = response.text

            st.error(detail)

    except Exception as exc:
        st.error(
            f"Delete failed: {exc}"
        )


# =====================================================
# TITLE
# =====================================================

st.title("☁️ AWS AI Agent")


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    # -------------------------------------------------
    # CHAT HISTORY
    # -------------------------------------------------

    st.header("💬 Conversations")

    if st.session_state.aws_connected:

        if st.button(
            "➕ New Chat",
            use_container_width=True,
        ):
            create_new_chat()
            st.rerun()

        st.divider()

        # Refresh conversation list
        load_conversations()

        if st.session_state.conversations:

            for conversation in (
                st.session_state.conversations
            ):

                conversation_id = (
                    conversation[
                        "conversation_id"
                    ]
                )

                title = conversation.get(
                    "title",
                    "New Conversation",
                )

                # Keep the sidebar readable
                if len(title) > 35:
                    title = title[:35] + "..."

                is_active = (
                    conversation_id
                    == st.session_state.conversation_id
                )

                col1, col2 = st.columns(
                    [5, 1]
                )

                with col1:

                    if st.button(
                        (
                            f"🟢 {title}"
                            if is_active
                            else f"💬 {title}"
                        ),
                        key=(
                            f"chat_"
                            f"{conversation_id}"
                        ),
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
                        key=(
                            f"delete_"
                            f"{conversation_id}"
                        ),
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
            "Connect to AWS to view "
            "your conversations."
        )


    # =================================================
    # AWS CONNECTION
    # =================================================

    st.divider()

    st.header("AWS Connection")

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

    connect_button = st.button(
        "🔗 Connect to AWS",
        use_container_width=True,
    )


    # =================================================
    # AWS CONNECT
    # =================================================

    if connect_button:

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

            with st.spinner(
                "Authenticating with AWS..."
            ):

                try:

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

                        # AWS authentication session
                        st.session_state.session_id = (
                            data["session_id"]
                        )

                        st.session_state.aws_connected = (
                            True
                        )

                        st.session_state.account_id = (
                            data["account_id"]
                        )

                        # New conversation for
                        # this AWS connection
                        st.session_state.conversation_id = (
                            str(uuid.uuid4())
                        )

                        st.session_state.messages = []

                        load_conversations()

                        st.success(
                            "AWS Connected Successfully"
                        )

                        st.rerun()

                    else:

                        try:
                            detail = response.json().get(
                                "detail",
                                "AWS connection failed.",
                            )
                        except Exception:
                            detail = response.text

                        st.error(detail)

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to the FastAPI "
                        "backend. Make sure Uvicorn "
                        "is running on port 8000."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "The backend request timed out."
                    )

                except Exception as exc:

                    st.error(
                        f"Backend error: {exc}"
                    )


    # =================================================
    # CONNECTION STATUS
    # =================================================

    if st.session_state.aws_connected:

        st.success("🟢 AWS Connected")

        st.write(
            f"Account: "
            f"`{st.session_state.account_id}`"
        )

    else:

        st.warning(
            "🔴 AWS Not Connected"
        )


# =====================================================
# MAIN CHAT
# =====================================================

st.subheader("💬 Chat")


# -----------------------------------------------------
# Display current conversation
# -----------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message("user"):
        st.write(message["user"])

    with st.chat_message("assistant"):

        st.write(
            message["assistant"]
        )

        metadata = [
            f"Intent: {message['intent']}"
        ]

        if (
            message["intent"]
            == "MONITORING"
            and message.get("service")
        ):

            metadata.append(
                f"Service: "
                f"{message['service']}"
            )

        st.caption(
            " | ".join(metadata)
        )


# =====================================================
# CHAT INPUT
# =====================================================

prompt = st.chat_input(
    "Ask about AWS..."
)


if prompt:

    if not st.session_state.aws_connected:

        st.warning(
            "Please connect your AWS account first."
        )

        st.stop()


    # -------------------------------------------------
    # Display user message
    # -------------------------------------------------

    with st.chat_message("user"):
        st.write(prompt)


    # -------------------------------------------------
    # Run agent
    # -------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Agent is thinking..."
        ):

            try:

                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        # AWS session
                        "session_id":
                            st.session_state.session_id,

                        # Conversation
                        "conversation_id":
                            st.session_state.conversation_id,

                        # Current question
                        "message":
                            prompt,
                    },
                    timeout=120,
                )


                # -------------------------------------
                # Handle backend error
                # -------------------------------------

                if response.status_code != 200:

                    try:

                        detail = (
                            response.json().get(
                                "detail",
                                "Agent failed.",
                            )
                        )

                    except Exception:

                        detail = response.text

                    st.error(detail)

                    st.stop()


                # -------------------------------------
                # Successful response
                # -------------------------------------

                data = response.json()

                st.write(
                    data["answer"]
                )


                caption = (
                    f"Intent: "
                    f"{data['intent']}"
                )


                if (
                    data["intent"]
                    == "MONITORING"
                    and data.get("service")
                ):

                    caption += (
                        f" | Service: "
                        f"{data['service']}"
                    )


                st.caption(caption)


                # -------------------------------------
                # Update current UI
                # -------------------------------------

                st.session_state.messages.append(
                    {
                        "user": prompt,
                        "assistant":
                            data["answer"],
                        "intent":
                            data["intent"],
                        "service":
                            data.get("service"),
                    }
                )


                # -------------------------------------
                # Refresh sidebar conversations
                # -------------------------------------

                load_conversations()


            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to the FastAPI "
                    "backend. Make sure Uvicorn "
                    "is running on port 8000."
                )


            except requests.exceptions.Timeout:

                st.error(
                    "The chat request timed out."
                )


            except Exception as exc:

                st.error(
                    f"Error: {exc}"
                )