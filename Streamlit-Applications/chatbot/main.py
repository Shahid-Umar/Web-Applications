'''
DESCRIPTION:
Langchain is a framework designed to simplify the process of developing applications that leverage large language models (LLMs). 
It provides tools to construct these applications and allows for functionalities like:

CONTEXT-AWARENESS:
Langchain enables developers to reason about how to respond based on context or what actions to take.
LLMs can be linked with various context sources such as prompts, instructions, and relevant data.
Reasoning: Langchain enables applications to reason about how to respond based on context or what actions to take.
Here are some potential applications of Langchain:

WHAT CAN DO WITH LANGCHAIN:
1.  Building informative chatbots
2.  Creating data-driven summaries of factual topics
3.  Developing question-answering systems that can access and process information from various sources
4.  Build reasoning applications
Overall, Langchain empowers developers to construct LLM-powered applications with more ease and efficiency.
'''
import streamlit as st
from langchain.llms import OpenAI

st.title('🔗 Interactive GPT-Based ChatBot')
# Setting up OpenAI API key
openai_api_key = st.secrets['OPENAI_API_KEY'] # This takes the API key from the secrets.toml file
# openai.api_key = st.sidebar.text_input('OpenAI API Key', type='password') # Prompt to user for API key
# openai_api_key = "Enter your OpenAI API key"


def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
  st.info(llm(input_text))

with st.form('my_form'):
  text = st.text_area('Enter text:', '')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('WARNING! Please enter your OpenAI API key that starts with "sk-"!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)