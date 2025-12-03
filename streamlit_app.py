import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# ---------- Page Setup ----------
st.set_page_config(page_title="Chatbot", page_icon="💬")
st.title("💬 Chatbot")
st.caption("Powered by Hugging Face – Free & Secure")

# ---------- LLM ----------
llm = HuggingFaceHub(
    repo_id="microsoft/phi-3-mini-4k-instruct",
    model_kwargs={
        "temperature": 0.3,
        "max_new_tokens": 512
    },
    huggingfacehub_api_token=st.secrets["HF_TOKEN"]
)

# ---------- Memory ----------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

# ---------- Chat History UI ----------
for msg in st.session_state.memory.chat_memory.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ---------- Chat Input ----------
if prompt := st.chat_input("Ask me anything…"):
    with st.chat_message("user"):
        st.markdown(prompt)

    response = conversation.predict(input=prompt)

    with st.chat_message("assistant"):
        st.markdown(response)
