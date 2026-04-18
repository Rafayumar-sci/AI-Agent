import os
from dotenv import load_dotenv
import streamlit as st
from serpapi import GoogleSearch
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

# Load environment variables from the .env file
load_dotenv()

# Access your API keys

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

st.set_page_config(page_title="AI Search Agent", layout="wide")
st.title("🔍 AI Search Agent")
st.write("Powered by Groq & SerpAPI")


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
    tools=[serpapi_search],
    system_prompt="You are a helpful assistant",
    checkpointer=Checkpointer
)

# Streamlit UI
st.sidebar.title("Search Settings")
user_query = st.sidebar.text_input(
    "Enter your search query:",
    value="search on internet about ideoversity Arfa tower",
    placeholder="What do you want to search for?"
)

if st.sidebar.button("🔍 Search", use_container_width=True):
    with st.spinner("Searching..."):
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_query}]},
            config={"configurable": {"thread_id": "1234567"}}
        )
        result = response["messages"][-1].content
        st.success("Search completed!")
        st.markdown("### Results:")
        st.write(result)
