from abc import ABC, abstractmethod
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import requests

# AI Provider Classes
class AIProvider(ABC):
    @abstractmethod
    def get_completion(self, messages):
        pass

class OpenAIProvider(AIProvider):
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini-2024-07-18"

    def get_completion(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content

class GeminiProvider(AIProvider):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def get_completion(self, messages):
        # Convert message history to Gemini format
        chat = self.model.start_chat()
        for message in messages:
            if message["role"] == "user":
                chat.send_message(message["content"])
        response = chat.last.text
        return response

class WatsonProvider(AIProvider):
    def __init__(self, api_key, assistant_id, service_url):
        authenticator = IAMAuthenticator(api_key)
        self.assistant = AssistantV2(
            version='2021-11-27',
            authenticator=authenticator
        )
        self.assistant.set_service_url(service_url)
        self.assistant_id = assistant_id

    def get_completion(self, messages):
        session = self.assistant.create_session(
            assistant_id=self.assistant_id
        ).get_result()
        
        response = self.assistant.message(
            assistant_id=self.assistant_id,
            session_id=session['session_id'],
            input={'text': messages[-1]['content']}
        ).get_result()
        
        return response['output']['generic'][0]['text']

class DeepAIProvider(AIProvider):
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {'api-key': api_key}

    def get_completion(self, messages):
        response = requests.post(
            "https://api.deepai.org/api/text-generator",
            data={'text': messages[-1]['content']},
            headers=self.headers
        )
        return response.json()['output']

class ClarifaiProvider(AIProvider):
    def __init__(self, api_key, user_id, app_id, model_id):
        self.api_key = api_key
        self.user_id = user_id
        self.app_id = app_id
        self.model_id = model_id

    def get_completion(self, messages):
        headers = {
            'Authorization': f'Key {self.api_key}',
            'Content-Type': 'application/json'
        }
        url = f"https://api.clarifai.com/v2/users/{self.user_id}/apps/{self.app_id}/models/{self.model_id}/outputs"
        
        response = requests.post(
            url,
            headers=headers,
            json={
                "inputs": [
                    {
                        "data": {
                            "text": {
                                "raw": messages[-1]['content']
                            }
                        }
                    }
                ]
            }
        )
        return response.json()['outputs'][0]['data']['text']['raw']

def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'ai_provider' not in st.session_state:
        st.session_state.ai_provider = None

def setup_ai_provider():
    """Setup AI provider with Streamlit interface"""
    st.sidebar.title("AI Provider Setup")
    
    provider_choice = st.sidebar.selectbox(
        "Select AI Provider",
        ["OpenAI", "Google Gemini", "IBM Watson", "DeepAI", "Clarifai"],
        index=0
    )
    
    with st.sidebar.form("credentials_form"):
        if provider_choice == "OpenAI":
            api_key = st.text_input("OpenAI API Key", type="password")
            if st.form_submit_button("Connect"):
                try:
                    st.session_state.ai_provider = OpenAIProvider(api_key)
                    st.success("Successfully connected to OpenAI!")
                except Exception as e:
                    st.error(f"Error connecting to OpenAI: {str(e)}")
        
        elif provider_choice == "Google Gemini":
            api_key = st.text_input("Google API Key", type="password")
            if st.form_submit_button("Connect"):
                try:
                    st.session_state.ai_provider = GeminiProvider(api_key)
                    st.success("Successfully connected to Gemini!")
                except Exception as e:
                    st.error(f"Error connecting to Gemini: {str(e)}")
        
        elif provider_choice == "IBM Watson":
            api_key = st.text_input("IBM Watson API Key", type="password")
            assistant_id = st.text_input("Assistant ID")
            service_url = st.text_input("Service URL")
            if st.form_submit_button("Connect"):
                try:
                    st.session_state.ai_provider = WatsonProvider(api_key, assistant_id, service_url)
                    st.success("Successfully connected to IBM Watson!")
                except Exception as e:
                    st.error(f"Error connecting to IBM Watson: {str(e)}")
        
        elif provider_choice == "DeepAI":
            api_key = st.text_input("DeepAI API Key", type="password")
            if st.form_submit_button("Connect"):
                try:
                    st.session_state.ai_provider = DeepAIProvider(api_key)
                    st.success("Successfully connected to DeepAI!")
                except Exception as e:
                    st.error(f"Error connecting to DeepAI: {str(e)}")
        
        elif provider_choice == "Clarifai":
            api_key = st.text_input("Clarifai API Key", type="password")
            user_id = st.text_input("User ID")
            app_id = st.text_input("App ID")
            model_id = st.text_input("Model ID")
            if st.form_submit_button("Connect"):
                try:
                    st.session_state.ai_provider = ClarifaiProvider(api_key, user_id, app_id, model_id)
                    st.success("Successfully connected to Clarifai!")
                except Exception as e:
                    st.error(f"Error connecting to Clarifai: {str(e)}")

def display_conversation():
    """Display the conversation history"""
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Assistant: {message['content']}")

def main():
    st.title("Multi-AI Provider Chat Interface")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup AI provider in sidebar
    setup_ai_provider()
    
    # Main chat interface
    st.write("Welcome! Start your conversation below.")
    
    # Display conversation history
    display_conversation()
    
    # Chat input
    user_input = st.text_input("Your message:", key="user_input")
    
    # Process user input
    if st.button("Send") and user_input:
        if st.session_state.ai_provider is None:
            st.error("Please set up an AI provider first!")
            return
            
        # Add user message to conversation history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Get response from AI provider
            ai_response = st.session_state.ai_provider.get_completion(st.session_state.conversation_history)
            
            # Add assistant's response to conversation history
            st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Clear the input box (requires a rerun)
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.conversation_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()