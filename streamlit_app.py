import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Papertrail", page_icon="P", layout="wide")
st.title("Papertrail")
st.caption("Research assistant for Attention Is All You Need")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("agent_decision"):
            st.caption(f"Agent decision: {message['agent_decision']}")
        for source in message.get("sources", []):
            page = source.get("page", "not available")
            score = source.get("score")
            title = f"Source page {page}"
            if score is not None:
                title += f" | relevance {score:.2f}"
            with st.expander(title):
                st.write(source.get("excerpt", "No excerpt was returned."))


with st.sidebar:
    st.subheader("Paper library")
    st.write("Attention Is All You Need")
    try:
        status_response = requests.get(f"{API_BASE_URL}/api/status", timeout=5)
        status_response.raise_for_status()
        status = status_response.json()
        st.metric("Indexed passages", f"{status.get('chunks', 0):,}")
        if status.get("indexed"):
            st.success("Index ready")
        else:
            st.warning("Index is empty")
        st.caption(f"Embedding model: {status.get('embedding_model', 'unknown')}")
        st.caption(f"Answer model: {status.get('llm_model', 'unknown')}")
    except requests.RequestException as error:
        st.error("FastAPI is unavailable. Start the backend on port 8000.")
        st.caption(str(error))

if "messages" not in st.session_state:
    st.session_state.messages = []

for chat_message in st.session_state.messages:
    render_message(chat_message)

question = st.chat_input("Ask a question about the paper")
if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Reviewing the paper..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/chat",
                    json={"question": question, "top_k": 4},
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()
                assistant_message = {
                    "role": "assistant",
                    "content": result["answer"],
                    "agent_decision": result.get("agent_decision"),
                    "sources": result.get("sources", []),
                }
            except requests.RequestException as error:
                assistant_message = {
                    "role": "assistant",
                    "content": f"Could not reach the chatbot API: {error}",
                    "sources": [],
                }
            except (KeyError, ValueError) as error:
                assistant_message = {
                    "role": "assistant",
                    "content": f"The chatbot API returned an invalid response: {error}",
                    "sources": [],
                }

        st.session_state.messages.append(assistant_message)
        st.markdown(assistant_message["content"])
        if assistant_message.get("agent_decision"):
            st.caption(f"Agent decision: {assistant_message['agent_decision']}")
        for source in assistant_message.get("sources", []):
            page = source.get("page", "not available")
            score = source.get("score")
            title = f"Source page {page}"
            if score is not None:
                title += f" | relevance {score:.2f}"
            with st.expander(title):
                st.write(source.get("excerpt", "No excerpt was returned."))