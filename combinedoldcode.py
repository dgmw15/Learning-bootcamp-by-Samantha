from abc import ABC, abstractmethod
from getpass import getpass
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

# Setup and Main Logic
def setup_ai_provider():
    print("\nAvailable AI Providers:")
    print("1. OpenAI")
    print("2. Google Gemini")
    print("3. IBM Watson")
    print("4. DeepAI")
    print("5. Clarifai")
    
    choice = input("\nSelect your AI provider (1-5): ")
    
    if choice == "1":
        api_key = getpass("Enter your OpenAI API Key: ")
        return OpenAIProvider(api_key)
    
    elif choice == "2":
        api_key = getpass("Enter your Google API Key: ")
        return GeminiProvider(api_key)
    
    elif choice == "3":
        api_key = getpass("Enter your IBM Watson API Key: ")
        assistant_id = input("Enter your Assistant ID: ")
        service_url = input("Enter your Service URL: ")
        return WatsonProvider(api_key, assistant_id, service_url)
    
    elif choice == "4":
        api_key = getpass("Enter your DeepAI API Key: ")
        return DeepAIProvider(api_key)
    
    elif choice == "5":
        api_key = getpass("Enter your Clarifai API Key: ")
        user_id = input("Enter your User ID: ")
        app_id = input("Enter your App ID: ")
        model_id = input("Enter your Model ID: ")
        return ClarifaiProvider(api_key, user_id, app_id, model_id)
    
    else:
        print("Invalid choice. Using OpenAI as default.")
        api_key = getpass("Enter your OpenAI API Key: ")
        return OpenAIProvider(api_key)

if __name__ == "__main__":
    # Setup AI provider
    ai_provider = setup_ai_provider()
    
    # Initialize conversation history
    conversation_history = []
    
    # Initial instruction
    print("\nWelcome! You can start your conversation. To end the conversation, simply say 'thank you'.")
    
    while True:
        prompt = input("\nYou: ")
        
        # Check if user wants to end conversation
        if prompt.lower() in ["thank you", "thanks"]:
            print("Assistant: You're welcome! Goodbye!")
            break
        
        # Add user message to conversation history
        conversation_history.append({"role": "user", "content": prompt})
        
        try:
            # Get response from AI provider
            ai_response = ai_provider.get_completion(conversation_history)
            
            # Add assistant's response to conversation history
            conversation_history.append({"role": "assistant", "content": ai_response})
            
            print(f"Assistant: {ai_response}")
            print("\n(Continue asking questions or say 'thank you' to end the conversation)")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again or say 'thank you' to end the conversation") 