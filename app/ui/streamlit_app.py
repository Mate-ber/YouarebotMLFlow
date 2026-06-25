import uuid

import requests
import streamlit as st

API_URL = "http://fastapi:8001"

st.title("🤖 YouAreBot Chat — live bot-probability")

if "dialog_id" not in st.session_state:
    st.session_state.dialog_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"n": 0, "correct": 0, "prob_sum": 0.0}


def predict_bot_prob(text: str, participant_index: int):
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={
                "text": text,
                "dialog_id": st.session_state.dialog_id,
                "id": str(uuid.uuid4()),
                "participant_index": participant_index,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return float(resp.json()["is_bot_probability"])
    except Exception:
        return None


def record(prob, truth_is_bot: int):
    if prob is None:
        return
    st.session_state.metrics["n"] += 1
    st.session_state.metrics["prob_sum"] += prob
    if (1 if prob >= 0.5 else 0) == truth_is_bot:
        st.session_state.metrics["correct"] += 1


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        text = msg["content"]
        if msg.get("prob") is not None:
            text += f"  \n`P(bot) = {msg['prob']:.2f}`"
        st.markdown(text)

if user_input := st.chat_input("Напишите сообщение боту..."):
    p_user = predict_bot_prob(user_input, participant_index=0)
    record(p_user, truth_is_bot=0)
    with st.chat_message("user"):
        st.markdown(
            user_input + (f"  \n`P(bot) = {p_user:.2f}`" if p_user is not None else "")
        )
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "prob": p_user}
    )

    try:
        response = requests.post(
            f"{API_URL}/get_message",
            json={
                "dialog_id": st.session_state.dialog_id,
                "last_msg_text": user_input,
                "last_message_id": str(uuid.uuid4()),
            },
            timeout=60,
        )
        bot_reply = response.json()["new_msg_text"]
    except Exception:
        bot_reply = "Error: Could not reach FastAPI service container."

    p_bot = predict_bot_prob(bot_reply, participant_index=1)
    record(p_bot, truth_is_bot=1)
    with st.chat_message("assistant"):
        st.markdown(
            bot_reply + (f"  \n`P(bot) = {p_bot:.2f}`" if p_bot is not None else "")
        )
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply, "prob": p_bot}
    )

m = st.session_state.metrics
accuracy = (m["correct"] / m["n"]) if m["n"] else 0.0
avg_prob = (m["prob_sum"] / m["n"]) if m["n"] else 0.0
st.sidebar.header("Running metrics")
st.sidebar.metric("Messages classified", m["n"])
st.sidebar.metric("Accuracy @0.5", f"{accuracy:.0%}")
st.sidebar.metric("Avg P(bot)", f"{avg_prob:.2f}")
