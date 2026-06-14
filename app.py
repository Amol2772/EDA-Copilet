import re
import streamlit as st
import os
from agent_core import agent, extract_text, df

st.set_page_config(page_title="EDA Copilot", page_icon="📊", layout="wide")

# ---- SIDEBAR ----
with st.sidebar:
    st.title("📋 Dataset Info")
    st.markdown("**Car Rental Dataset**")
    st.metric("Rows", f"{df.shape[0]:,}")
    st.metric("Columns", df.shape[1])

    st.divider()
    st.markdown("**💡 Example Queries**")
    examples = [
        "Which vehicle type has the highest average daily rate?",
        "Plot rate.daily as a histogram",
        "Detect outliers in rate.daily",
        "Is there a correlation between renterTripsTaken and rating?",
        "Using SQL, how many vehicles per fuel type?",
        "What are the top 3 vehicle makes by listings?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.prefill = ex
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.prefill = ""
        st.rerun()

    st.divider()
    st.caption("🔧 LangChain · LangGraph · Gemini")

# ---- INIT STATE ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ---- MAIN ----
st.title("📊 EDA Copilot")
st.caption("Ask questions about the car rental dataset in plain English.")

if not st.session_state.messages:
    st.info("👈 Pick an example query from the sidebar, or type your own below.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# handle prefill from sidebar buttons
prompt = None
if st.session_state.prefill:
    prompt = st.session_state.prefill
    st.session_state.prefill = ""
else:
    prompt = st.chat_input("Ask a question about the dataset...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                result = agent.invoke({"messages": st.session_state.messages})
                response = extract_text(result)
                if not response:
                    response = "⚠️ Agent returned an empty response. Try rephrasing."
            except Exception as e:
                response = f"⚠️ Error: {str(e)[:300]}"
        st.write(response)
        for path in re.findall(r"reports/[\w\-.]+\.png", response):
            if os.path.exists(path):
                st.image(path)

    st.session_state.messages.append({"role": "assistant", "content": response})