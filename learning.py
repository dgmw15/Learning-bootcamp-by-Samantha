from openai import OpenAI
from getpass import getpass

openai_key = getpass("Enter your API Key:")
client = OpenAI(api_key=openai_key)

