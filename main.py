import os
from dotenv import load_dotenv
import google.generativeai as genai
from MultiAgent.agent import root_agent

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

# Run the multi-agent system
if __name__ == "__main__":
    print("Multi-Agent Summarization System Initialized")
    print(f"Using Gemini API Key: {api_key[:8]}...")
