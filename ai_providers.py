from abc import ABC, abstractmethod
from openai import OpenAI
import google.generativeai as genai
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import requests

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