import os
from dotenv import load_dotenv
import streamlit as st
from serpapi import GoogleSearch
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
import yagmail


# Load environment variables from the .env file
load_dotenv()

yag = yagmail.SMTP("rafayumar176@gmail.com", "kymo vejr ibjj hcpg")


def send_email_tool(recipient: str, subject: str, content: str) -> str:
    """Sends an email to recipient ,with subject and body."""
    yag.send(
        to=recipient,
        subject=subject,
        contents=content
    )
    return f"Email sent to {recipient} with subject '{subject}'"


# Access your API keys

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

st.set_page_config(page_title="AI Search Agent", layout="wide")

# Custom CSS for better UI
st.markdown("""
    
""", unsafe_allow_html=True)

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_retries=2,
    api_key=GROQ_API_KEY
)

Client = MongoClient(MONGODB_URI)
Checkpointer = MongoDBSaver(Client)


def serpapi_search(query: str):
    """ Searches for a query using serpAPI"""
    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": SERP_API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    # Extract top results (titles + links)
    if "organic_results" in results:
        return [
            {"title": r["title"], "link": r["link"],
                "snippet": r.get("snippet", "")}
            for r in results["organic_results"][:5]
        ]
    return {"error": "No results found"}


memory = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[serpapi_search, send_email_tool],
    system_prompt="You are a helpful assistant",
    checkpointer=Checkpointer
)

# Main hero section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🔍 AI Search Agent</div>
        <div class="hero-subtitle">Powered by Groq & SerpAPI</div>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False


def on_search_input():
    st.session_state.search_triggered = True


# Search input in hero
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

# Results display
if search_button or st.session_state.search_triggered:
    st.session_state.search_triggered = False
    if user_query.strip():
        with st.spinner("🔄 Searching..."):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_query}]},
                config={"configurable": {"thread_id": "1234567"}}
            )
            result = response["messages"][-1].content

        st.success("✅ Search completed!")
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Results:")
        st.write(result)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a search query")
