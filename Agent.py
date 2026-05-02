import os
from dotenv import load_dotenv
import streamlit as st
from serpapi import GoogleSearch
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.mongodb import MongoDBSaver
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

# Initialize MongoDB Checkpointer
checkpointer = MongoDBSaver(db, auto_index=True)

yag = yagmail.SMTP(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))


def send_email_tool(recipient: str, subject: str, content: str) -> str:
    """Sends an email to recipient ,with subject and body."""
    yag.send(
        to=recipient,
        subject=subject,
        contents=content
    )
    return f"Email sent to {recipient} with subject '{subject}'"


def get_search_history(limit: int = 10) -> list:
    """Retrieve search history from MongoDB"""
    try:
        history = list(searches_collection.find().sort(
            "timestamp", -1).limit(limit))
        return history
    except Exception as e:
        st.error(f"Error retrieving history: {e}")
        return []


def search_previous_queries(keyword: str) -> str:
    """Search for previous queries in the database"""
    try:
        results = list(searches_collection.find(
            {"query": {"$regex": keyword, "$options": "i"}}).limit(5))
        if results:
            formatted = "\n".join(
                [f"- {r['query']}: {r['response'][:100]}..." for r in results])
            return f"Found previous searches:\n{formatted}"
        return "No previous searches found with that keyword"
    except Exception as e:
        return f"Error searching: {e}"


def store_to_database(data_type: str, query: str, response: str) -> str:
    """Store data to MongoDB database"""
    try:
        if data_type == "search":
            searches_collection.insert_one({
                "query": query,
                "response": response,
                "timestamp": datetime.now(),
                "thread_id": "1234567"
            })
            return f"Stored search: {query}"
        elif data_type == "note":
            conversations_collection.insert_one({
                "query": query,
                "response": response,
                "timestamp": datetime.now(),
                "thread_id": "1234567"
            })
            return f"Stored note: {query}"
    except Exception as e:
        return f"Error storing to database: {e}"


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found! Check your .env file.")
SERP_API_KEY = os.getenv("SERP_API_KEY")
if not SERP_API_KEY:
    st.error("SERP_API_KEY not found! Check your .env file.")


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

memory = MongoDBSaver(db, auto_index=True)
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
    tools=[serpapi_search, send_email_tool,
           search_previous_queries, store_to_database],
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
        # Add user message to history
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

            # Add assistant response to history
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
