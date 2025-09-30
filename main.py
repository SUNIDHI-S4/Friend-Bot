from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

# prompt = '''

# '''

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents= prompt,
#     config= types.GenerationConfig(
#         system_instructions="You are a supportive, companion, friendly AI.",
#     )
# )

# print(response.text)

def classify_message(user_input):
    classification_prompt = f"""
    Classify the following user input into one of the categories:
    - Suggestive (user wants tips, recommendations, or advice)
    - Discussive (user wants a thoughtful discussion)
    - Humorous (user wants a witty or playful response)
    - Help (user seems distressed, seeking emergency or professional help)

    Only return one word: Suggestive, Discussive, Humorous, or Help.

    User input: "{user_input}"
    """
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=classification_prompt
    )
    return response.text.lower().strip()

def handle_suggestive_conversation(user_input):
    prompt = f"""
    You are a supportive, helpful, and empathetic AI companion.
    The user is asking for suggestions or tips.
    Provide practical, positive, and human-like advice.
    Keep the tone cheerful and encouraging.

    User: {user_input}
    """
    # print("Suggestive Conversation") # Debugging line
    return generate_ai_response(prompt)

def handle_discussive_conversation(user_input):
    prompt = f"""
    You are a thoughtful, engaging, and curious AI companion.
    The user wants a discussion.
    Respond in a conversational, insightful way — ask follow-up questions,
    show empathy, and keep it natural.

    User: {user_input}
    """
    # print("Discussive Conversation") # Debugging line
    return generate_ai_response(prompt)

def handle_humorous_conversation(user_input):
    prompt = f"""
    You are a witty, kind, and playful AI companion.
    The user wants humor.
    Respond with light, positive humor and a pinch of Gen Z/Bangalore slang,
    but remain respectful and empathetic.

    User: {user_input}
    """
    # print("Humorous Conversation") # Debugging line
    return generate_ai_response(prompt)

def handle_help_conversation(user_input):
    prompt = f"""
    The user may be in distress or needs real help.
    Respond kindly and empathetically.
    Provide relevant helplines or professional resources from the internet.
    DO NOT give medical or legal advice yourself.
    """
    # Optionally call your web scraping / API logic to get helpline info
    # print("Helpline Conversation") # Debugging line
    return generate_ai_response(prompt)


def generate_ai_response(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )
    return response.text.strip()


def chatbot_reply(user_input):
    print("Categorizing user input...") # Debugging line
    category = classify_message(user_input)
    print(f"Category: {category}") # Debugging line
    
    if category == "suggestive":
        return handle_suggestive_conversation(user_input)
    elif category == "discussive":
        return handle_discussive_conversation(user_input)
    elif category == "humorous":
        return handle_humorous_conversation(user_input)
    elif category == "help":
        return handle_help_conversation(user_input)
    else:
        return "I'm here for you 😊 Could you tell me more?"

user_input = input("You: ")
reply = chatbot_reply(user_input)
print("Bot:", reply)