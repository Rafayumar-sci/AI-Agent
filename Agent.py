import os
from dotenv import load_dotenv
import streamlit as st
from serpapi import GoogleSearch
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import yagmail
from pymongo import MongoClient
from datetime import datetime


# Load environment variables from the .env file
load_dotenv()

# MongoDB Connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client["ai_agent"]
searches_collection = db["searches"]
conversations_collection = db["conversations"]

# yag = yagmail.SMTP(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))


# def send_email_tool(recipient: str, subject: str, content: str) -> str:
#     """Sends an email to recipient ,with subject and body."""
#     yag.send(
#         to=recipient,
#         subject=subject,
#         contents=content
#     )
#     return f"Email sent to {recipient} with subject '{subject}'"






GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found! Check your .env file.")
SERP_API_KEY = os.getenv("SERP_API_KEY")
if not SERP_API_KEY:
    st.error("SERP_API_KEY not found! Check your .env file.")


st.set_page_config(page_title="AI Search Agent", layout="wide")

st.markdown("""
    
""", unsafe_allow_html=True)

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_retries=2,
    api_key=GROQ_API_KEY
)

memory = InMemorySaver()
Checkpointer = memory


if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


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

    if "organic_results" in results:
        return [
            {"title": r["title"], "link": r["link"],
                "snippet": r.get("snippet", "")}
            for r in results["organic_results"][:5]
        ]
    return {"error": "No results found"}


agent = create_agent(
    model=model,
    tools=[serpapi_search],
    system_prompt="""You are an intelligent AI Search Agent designed to help users find information and take actions. 

IMPORTANT: Remember and acknowledge user information:
- If the user tells you their name, remember it and use it in all future responses
- Refer back to previous messages in this conversation to recall user details
- Be friendly and personable by using the user's name when appropriate

Your capabilities:
1. Search the internet using serpapi_search tool - use this to find current, accurate information on any topic
2. Send emails using send_email_tool - use this when users explicitly request to send information via email
3. Search previous queries using search_previous_queries - use this to recall past searches
4. Store to database using store_to_database - use this to save important information

Always review the full conversation history to remember user details and provide personalized responses.
""",
    checkpointer=Checkpointer
)

st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🔍 AI Search Agent</div>
        <div class="hero-subtitle">Powered by Groq & SerpAPI</div>
    </div>
""", unsafe_allow_html=True)

if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False


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
            response = agent.invoke(
                {"messages": st.session_state.conversation_history},
                
                config={"configurable": {"thread_id": "1234567"}}
            )
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
        
