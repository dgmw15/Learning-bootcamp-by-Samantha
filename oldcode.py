from getpass import getpass
from ai_providers import (
    OpenAIProvider,
    GeminiProvider,
    WatsonProvider,
    DeepAIProvider,
    ClarifaiProvider
)

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