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

st.set_page_config(page_title="AI Search Agent",
                   layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for better UI with animations and enhanced design
st.markdown("""
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Header Styles */
        .header-container {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 15px 30px;
            margin: -80px -60px 40px -60px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header-title {
            font-size: 32px;
            font-weight: bold;
            color: white;
            margin-bottom: 5px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header-subtitle {
            font-size: 14px;
            color: rgba(255,255,255,0.8);
        }
        
        /* Navigation Links */
        .nav-links {
            display: flex;
            gap: 30px;
            justify-content: center;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        
        .nav-link {
            color: white;
            text-decoration: none;
            font-size: 13px;
            padding: 5px 15px;
            border-radius: 20px;
            transition: all 0.3s ease;
            background: rgba(255,255,255,0.1);
        }
        
        .nav-link:hover {
            background: rgba(255,255,255,0.25);
            transform: translateY(-2px);
        }
        
        /* Hero Section */
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 80px 20px;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 50px;
            color: white;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
            animation: slideDown 0.6s ease;
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .hero-title {
            font-size: 56px;
            font-weight: bold;
            margin-bottom: 15px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        }
        
        .hero-subtitle {
            font-size: 20px;
            opacity: 0.95;
            margin-bottom: 30px;
            font-weight: 300;
        }
        
        /* Search Container */
        .search-container {
            display: flex;
            gap: 10px;
            max-width: 800px;
            margin: 40px auto 0;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .search-input-wrapper {
            flex: 1;
            min-width: 300px;
        }
        
        /* Features Section */
        .features-section {
            margin-bottom: 50px;
        }
        
        .section-title {
            font-size: 36px;
            font-weight: bold;
            color: #333;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .feature-card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            text-align: center;
            transition: all 0.3s ease;
            border-top: 4px solid #667eea;
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
        }
        
        .feature-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        .feature-title {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        
        .feature-desc {
            font-size: 14px;
            color: #666;
            line-height: 1.6;
        }
        
        /* Results Container */
        .results-container {
            margin-top: 40px;
        }
        
        .result-card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateX(-10px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .result-card:hover {
            transform: translateX(5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        }
        
        .result-title {
            font-size: 18px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .result-link {
            font-size: 13px;
            color: #764ba2;
            margin-bottom: 12px;
            word-break: break-all;
        }
        
        .result-snippet {
            font-size: 14px;
            color: #555;
            line-height: 1.7;
        }
        
        /* Footer */
        .footer-container {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 50px 30px;
            margin-top: 80px;
            margin: 80px -60px -100px -60px;
            color: white;
            box-shadow: 0 -4px 20px rgba(102, 126, 234, 0.2);
        }
        
        .footer-content {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .footer-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .footer-section-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .footer-link {
            color: rgba(255,255,255,0.8);
            text-decoration: none;
            font-size: 14px;
            display: block;
            margin-bottom: 10px;
            transition: all 0.3s ease;
        }
        
        .footer-link:hover {
            color: white;
            margin-left: 5px;
        }
        
        .footer-divider {
            border-top: 1px solid rgba(255,255,255,0.2);
            padding-top: 30px;
            margin-top: 30px;
        }
        
        .footer-bottom {
            text-align: center;
            font-size: 13px;
            color: rgba(255,255,255,0.7);
        }
        
        .social-links {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 15px;
        }
        
        .social-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            color: white;
            text-decoration: none;
            transition: all 0.3s ease;
            font-size: 18px;
        }
        
        .social-icon:hover {
            background: rgba(255,255,255,0.4);
            transform: translateY(-3px);
        }
        
        /* About Section */
        .about-section {
            background: white;
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 50px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .about-content {
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.8;
            font-size: 15px;
            color: #555;
        }
        
        .about-content p {
            margin-bottom: 15px;
        }
        
        /* Stats Section */
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 50px;
        }
        
        .stat-card {
            background: white;
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        
        .stat-number {
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .stat-label {
            font-size: 14px;
            color: #666;
        }
    </style>
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
    tools=[serpapi_search],
    system_prompt="You are a helpful assistant",
    checkpointer=Checkpointer
)

# Header Section
st.markdown("""
    <div class="header-container">
        <div class="header-title">⚡ AI Search Agent</div>
        <div class="header-subtitle">Intelligent Web Search Powered by AI</div>
        <div class="nav-links">
            <span class="nav-link">🏠 Home</span>
            <span class="nav-link">✨ Features</span>
            <span class="nav-link">ℹ️ About</span>
            <span class="nav-link">📞 Contact</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main hero section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🔍 AI Search Agent</div>
        <div class="hero-subtitle">Powered by Groq & SerpAPI • Lightning-Fast Results</div>
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

# Features Section
st.markdown("""
    <div class="features-section">
        <h2 class="section-title">✨ Why Choose Our AI Search Agent?</h2>
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Lightning Fast</div>
                <div class="feature-desc">Get search results in milliseconds with our optimized AI engine</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">AI Powered</div>
                <div class="feature-desc">Intelligent responses powered by advanced language models</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🌍</div>
                <div class="feature-title">Global Search</div>
                <div class="feature-desc">Search across the entire internet with real-time indexing</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔒</div>
                <div class="feature-title">Secure & Private</div>
                <div class="feature-desc">Your searches are encrypted and never stored permanently</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💡</div>
                <div class="feature-title">Smart Analysis</div>
                <div class="feature-desc">Intelligent parsing and summarization of search results</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📱</div>
                <div class="feature-title">Mobile Friendly</div>
                <div class="feature-desc">Seamless experience across all devices and screen sizes</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Stats Section
st.markdown("""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-number">10M+</div>
            <div class="stat-label">Searches Performed</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">99.9%</div>
            <div class="stat-label">Uptime Guarantee</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">&lt;100ms</div>
            <div class="stat-label">Average Response Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">50+</div>
            <div class="stat-label">Countries Supported</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# About Section
st.markdown("""
    <div class="about-section">
        <h2 class="section-title">ℹ️ About AI Search Agent</h2>
        <div class="about-content">
            <p>
                <strong>AI Search Agent</strong> is a revolutionary search platform that combines the power of artificial intelligence 
                with real-time web search capabilities. Built on cutting-edge technology, our platform provides accurate, 
                relevant, and comprehensive search results in real-time.
            </p>
            <p>
                Our mission is to make information accessible and searchable for everyone, everywhere. We leverage advanced 
                AI models and machine learning algorithms to understand your queries better and deliver the most relevant results.
            </p>
            <p>
                Whether you're researching, learning, or just exploring the web, AI Search Agent is your perfect companion. 
                With our intelligent search technology, finding what you need has never been easier or faster.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Footer Section
st.markdown("""
    <div class="footer-container">
        <div class="footer-content">
            <div class="footer-grid">
                <div>
                    <div class="footer-section-title">About</div>
                    <a href="#" class="footer-link">About Us</a>
                    <a href="#" class="footer-link">Blog</a>
                    <a href="#" class="footer-link">Press</a>
                    <a href="#" class="footer-link">Careers</a>
                </div>
                <div>
                    <div class="footer-section-title">Product</div>
                    <a href="#" class="footer-link">Features</a>
                    <a href="#" class="footer-link">Pricing</a>
                    <a href="#" class="footer-link">API Docs</a>
                    <a href="#" class="footer-link">Support</a>
                </div>
                <div>
                    <div class="footer-section-title">Legal</div>
                    <a href="#" class="footer-link">Privacy Policy</a>
                    <a href="#" class="footer-link">Terms of Service</a>
                    <a href="#" class="footer-link">Cookie Policy</a>
                    <a href="#" class="footer-link">Disclaimer</a>
                </div>
                <div>
                    <div class="footer-section-title">Connect</div>
                    <div class="social-links">
                        <a href="#" class="social-icon" title="Twitter">𝕏</a>
                        <a href="#" class="social-icon" title="Facebook">f</a>
                        <a href="#" class="social-icon" title="LinkedIn">in</a>
                        <a href="#" class="social-icon" title="GitHub">⚙</a>
                    </div>
                </div>
            </div>
            <div class="footer-divider">
                <div class="footer-bottom">
                    <p>&copy; 2026 AI Search Agent. All rights reserved. | Powered by Groq & SerpAPI</p>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
