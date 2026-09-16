import os
from dotenv import load_dotenv
import streamlit as st
from serpapi import GoogleSearch
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import yagmail
import groq

# Load environment variables from the .env file
load_dotenv()

st.set_page_config(page_title="AI Search Agent", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found! Check your .env file.")
SERP_API_KEY = os.getenv("SERP_API_KEY")
if not SERP_API_KEY:
    st.error("SERP_API_KEY not found! Check your .env file.")

# Groq decommissions models over time. Try these in order; the first that
# works wins. Override the list by setting GROQ_MODELS in .env (comma-separated).
GROQ_MODEL_CANDIDATES = [
    m.strip() for m in os.getenv(
        "GROQ_MODELS",
        "openai/gpt-oss-120b,openai/gpt-oss-20b,llama-3.3-70b-versatile"
    ).split(",") if m.strip()
]

# Email (yagmail) - initialized lazily so the app starts even if creds are missing
_email_client = None


def _get_email_client():
    global _email_client
    if _email_client is None:
        # Gmail app passwords are shown as "abcd efgh ijkl mnop" - strip spaces
        email_pass = (os.getenv("EMAIL_PASS") or "").replace(" ", "")
        _email_client = yagmail.SMTP(os.getenv("EMAIL_USER"), email_pass)
    return _email_client


def send_email_tool(recipient: str, subject: str, content: str) -> str:
    """Sends an email to recipient, with subject and body."""
    try:
        _get_email_client().send(
            to=recipient,
            subject=subject,
            contents=content
        )
        return f"Email sent to {recipient} with subject '{subject}'"
    except Exception as e:
        return f"Failed to send email: {e}"


def serpapi_search(query: str):
    """Searches for a query using serpAPI."""
    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": SERP_API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" in results:
        return [
            {"title": r["title"], "link": r["link"],
             "snippet": r.get("snippet", "")}
            for r in results["organic_results"][:5]
        ]
    return {"error": "No results found"}


memory = InMemorySaver()
Checkpointer = memory

TOOLS = [serpapi_search, send_email_tool]

SYSTEM_PROMPT = """You are an intelligent AI Search Agent designed to help users find information and take actions.

IMPORTANT: Remember and acknowledge user information:

- Refer back to previous messages in this conversation to recall user details
- Be friendly and personable by using the user's name when appropriate

Your capabilities:
1. Search the internet using serpapi_search tool - use this to find current, accurate information on any topic
2. Send emails using send_email_tool - use this when the user asks you to email someone (you need recipient address, subject and content)

give long explanation on your topic of discussion and then ask user if they want to search
"""


def _make_chat_model(model_name: str) -> ChatGroq:
    return ChatGroq(
        model=model_name,
        temperature=0.0,
        max_retries=2,
        api_key=GROQ_API_KEY
    )


def _make_agent(model_name: str):
    return create_agent(
        model=_make_chat_model(model_name),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=Checkpointer
    )


def run_agent(messages: list, thread_id: str):
    """Invoke the agent, transparently falling back if a model is unavailable."""
    last_err = None
    for model_name in GROQ_MODEL_CANDIDATES:
        try:
            return _make_agent(model_name).invoke(
                {"messages": messages},
                config={"configurable": {"thread_id": thread_id}}
            )
        except groq.NotFoundError:
            last_err = f"Model '{model_name}' not available on Groq"
            continue
    raise RuntimeError(
        f"No Groq model worked. Tried: {GROQ_MODEL_CANDIDATES}. Last error: {last_err}"
    )


st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🔍 AI Search Agent</div>
        <div class="hero-subtitle">Powered by Groq & SerpAPI</div>
    </div>
""", unsafe_allow_html=True)

if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


def on_search_input():
    st.session_state.search_triggered = True


col1, col2 = st.columns([4, 1], gap="small")

with col1:
    user_query = st.text_input(
        "Search the internet:",
        placeholder="What do you want to search for?",
        label_visibility="collapsed",
        on_change=on_search_input,
        key="user_query_input"
    )

with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

if search_button or st.session_state.search_triggered:
    st.session_state.search_triggered = False
    if user_query.strip():
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_query
        })

        with st.spinner("🔄 Searching..."):
            try:
                response = run_agent(
                    st.session_state.conversation_history,
                    thread_id="1234567"
                )
            except Exception as e:
                result = f"⚠️ Agent error: {e}"
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": result
                })
                st.error(result)
                st.stop()
            result = response["messages"][-1].content

            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": result
            })

        st.success("✅ Search completed!")
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Results:")
        st.write(result)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a search query")
