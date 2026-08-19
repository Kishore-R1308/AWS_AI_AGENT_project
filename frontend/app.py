import uuid

import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AWS AI Agent",
    page_icon="☁️",
    layout="wide",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "aws_connected" not in st.session_state:
    st.session_state.aws_connected = False

if "account_id" not in st.session_state:
    st.session_state.account_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def load_history():
    try:
        response = requests.get(
            f"{BACKEND_URL}/history/"
            f"{st.session_state.session_id}",
            timeout=10,
        )

        if response.status_code == 200:
            return response.json()

    except Exception:
        pass

    return []


st.title("☁️ AWS AI Agent")

st.caption(
    "RAG + LangGraph + Boto3 + Groq"
)


with st.sidebar:
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

                        st.session_state.session_id = (
                            data["session_id"]
                        )

                        st.session_state.aws_connected = (
                            True
                        )

                        st.session_state.account_id = (
                            data["account_id"]
                        )

                        st.session_state.messages = []

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
                        "Cannot connect to the FastAPI backend. "
                        "Make sure Uvicorn is running on port 8000."
                    )

                except requests.exceptions.Timeout:
                    st.error(
                        "The backend request timed out."
                    )

                except Exception as exc:
                    st.error(
                        f"Backend error: {exc}"
                    )

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

    st.divider()

    st.subheader("Example queries")

    st.markdown(
        """
**Knowledge**

- How do I create an EC2 instance?
- What is an S3 bucket?
- How does RDS Multi-AZ work?
- How do IAM roles work?

**Monitoring**

- Which EC2 instances are running?
- Show my S3 buckets.
- Which S3 bucket has the highest storage?
- Show my RDS databases.
"""
    )


if (
    not st.session_state.messages
    and st.session_state.aws_connected
):
    history = load_history()

    for item in history:
        st.session_state.messages.append(
            {
                "user": item["user_message"],
                "assistant": item["assistant_message"],
                "intent": item["intent"],
                "service": item["service"],
            }
        )


st.subheader("💬 Chat")


for message in st.session_state.messages:
    with st.chat_message("user"):
        st.write(message["user"])

    with st.chat_message("assistant"):
        st.write(message["assistant"])

        metadata = [
            f"Intent: {message['intent']}"
        ]

        if message.get("service"):
            metadata.append(
                f"Service: {message['service']}"
            )

        st.caption(
            " | ".join(metadata)
        )


prompt = st.chat_input(
    "Ask about AWS..."
)


if prompt:
    if not st.session_state.aws_connected:
        st.warning(
            "Please connect your AWS account first."
        )
        st.stop()

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
                        "message": prompt,
                    },
                    timeout=120,
                )

                if response.status_code != 200:
                    try:
                        detail = response.json().get(
                            "detail",
                            "Agent failed.",
                        )
                    except Exception:
                        detail = response.text

                    st.error(detail)
                    st.stop()

                data = response.json()

                st.write(
                    data["answer"]
                )

                caption = (
                    f"Intent: {data['intent']}"
                )

                if data.get("service"):
                    caption += (
                        f" | Service: "
                        f"{data['service']}"
                    )

                st.caption(caption)

                st.session_state.messages.append(
                    {
                        "user": prompt,
                        "assistant": data["answer"],
                        "intent": data["intent"],
                        "service": data.get(
                            "service"
                        ),
                    }
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the FastAPI backend. "
                    "Make sure Uvicorn is running on port 8000."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The chat request timed out."
                )

            except Exception as exc:
                st.error(
                    f"Error: {exc}"
                )