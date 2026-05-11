"""
LangChain chain for generating sales insights using OpenAI.
"""

import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()  # Load OPENAI_API_KEY from .env

def build_chain() -> LLMChain:
    """Build and return a LangChain LLMChain with system prompt."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=api_key)
    
    system_prompt = "You are a sales analytics expert. Analyze the customer segment data and provide actionable insights."
    human_prompt = "Segment: {segment_name}\nStatistics: {stats}\nProvide short, actionable sales recommendations."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain