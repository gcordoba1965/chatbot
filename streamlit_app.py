import streamlit as st
from huggingface_hub import InferenceClient, whoami

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Chatbot", page_icon="💬")
st.title("💬 Hugging Face Chatbot")

# -------------------------
# Load HF token
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN not found. Please set it in Streamlit secrets.")
    st.stop()

# -------------------------
# Validate token
# -------------------------
try:
    user_info = whoami(token=HF_TOKEN)
    st.sidebar.success(f"Authenticated as {user_info['name']}")
except Exception:
    st.error("Invalid Hugging Face token or missing permissions.")
    st.stop()

# -------------------------
# Create HF client
# -------------------------
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=HF_TOKEN,
)

# -------------------------
# Session state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Display chat history
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Chat input
# -------------------------
if prompt := st.chat_input("Ask something"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = client.chat_completion(
            messages=st.session_state.messages,
            max_tokens=300,
            temperature=0.6,
        )
        assistant_message = response.choices[0].message["content"]
    except Exception:
        st.error("Model temporarily unavailable. Please try again.")
        st.stop()

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_message}
    )
    with st.chat_message("assistant"):
        st.markdown(assistant_message)
