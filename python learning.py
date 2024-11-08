
!pip install openai --quiet

from openai import OpenAi
from getpass import getpass

openai_key = getpass("Enter your API key")
client = OpenAi(api_key=openai_key)