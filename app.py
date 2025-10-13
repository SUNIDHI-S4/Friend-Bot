from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn
import os
import json
from google import genai
from dotenv import load_dotenv

# -----------------------
# Load API Key
# -----------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

app = FastAPI()

CONVO_FILE = "conversations.json"
USERS_FILE = "users.json"
INTERESTS_FILE = "interests.json"

# -----------------------
# Models
# -----------------------
class SignupRequest(BaseModel):
    name: str
    username: str
    password: str
    nickname: str
    gmail: str
    interests: list[str]

class SigninRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    username: str
    message: str

# -----------------------
# Utilities
# -----------------------
def load_json(file, default):
    if os.path.exists(file):
        try:
            with open(file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return default
    return default

def save_json(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def add_to_conversation(role, message):
    conversations = load_json(CONVO_FILE, [])
    conversations.append({"role": role, "message": message})
    save_json(CONVO_FILE, conversations)

def get_recent_history(category):
    conversations = load_json(CONVO_FILE, [])
    if category == "discussive":
        limit = 10
    elif category == "suggestive":
        limit = 6
    elif category == "humorous":
        limit = 3
    elif category == "classify":
        limit = 5
    else:
        limit = 0

    if limit > 0:
        recent = conversations[-limit:]
        return "\n".join([f"{c['role'].capitalize()}: {c['message']}" for c in recent])
    return ""

def classify_message(user_input, username):
    history = get_recent_history("classify")
    interests = get_user_interests(username)
    prompt = f"""
    Classify the following user input into one of the categories:
    - Suggestive (user wants tips, recommendations, or advice)
    - Discussive (user wants a thoughtful discussion)
    - Humorous (user wants a witty or playful response)
    - Help (user seems distressed, seeking emergency or professional help)

    user interests: {', '.join(interests) if interests else 'None'}
    Conversation history:
    {history}

    User input: "{user_input}"
    """
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=prompt
    )
    return response.text.lower().strip()

def generate_ai_response(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=prompt
    )
    return response.text.strip()

def get_user_interests(username):
    interests_data = load_json(INTERESTS_FILE, {})
    return interests_data.get(username, [])

# -----------------------
# Conversation Handlers
# -----------------------
def handle_suggestive_conversation(user_input, username):
    history = get_recent_history("suggestive")
    interests = get_user_interests(username)
    prompt = f"""
    You are a supportive, helpful, and empathetic AI companion.
    The user is asking for suggestions or tips.
    Provide practical, positive, and human-like advice.
    Keep the tone cheerful and encouraging.

    Give concise, actionable replies.

    User interests: {', '.join(interests) if interests else 'None'}
    Conversation history:
    {history}

    User: {user_input}
    """
    return generate_ai_response(prompt)

def handle_discussive_conversation(user_input):
    history = get_recent_history("discussive")
    prompt = f"""
    You are a thoughtful, engaging, and curious AI companion.
    The user wants a discussion.
    Respond in a conversational, insightful way — ask follow-up questions,
    show empathy, and keep it natural.
    
    Give crisp, well-reasoned replies.
    Conversation history:
    {history}

    User: {user_input}
    """
    return generate_ai_response(prompt)

def handle_humorous_conversation(user_input, username):
    history = get_recent_history("humorous")
    interests = get_user_interests(username)
    prompt = f"""
    You are a witty, kind, and playful AI companion.
    The user wants humor.
    Respond with light, positive humor and a pinch of Gen Z/Bangalore slang,
    but remain respectful and empathetic.

    Give short, snappy replies.

    User interests: {', '.join(interests) if interests else 'None'}

    Conversation history:
    {history}

    User: {user_input}
    """
    return generate_ai_response(prompt)

def handle_help_conversation(user_input):
    prompt = """
    The user may be in distress or needs real help.
    Respond kindly and empathetically.
    Provide relevant helplines or professional resources from the internet.
    DO NOT give medical or legal advice yourself.
    """
    return generate_ai_response(prompt)

def chatbot_reply(user_input, username):
    category = classify_message(user_input, username)
    if category == "suggestive":
        reply = handle_suggestive_conversation(user_input, username)
    elif category == "discussive":
        reply = handle_discussive_conversation(user_input)
    elif category == "humorous":
        reply = handle_humorous_conversation(user_input, username)
    elif category == "help":
        reply = handle_help_conversation(user_input)
    else:
        reply = "I'm here for you 😊 Could you tell me more?"

    add_to_conversation("user", user_input)
    add_to_conversation("bot", reply)
    return reply

# -----------------------
# API Endpoints
# -----------------------
@app.post("/signup")
def signup(req: SignupRequest):
    users = load_json(USERS_FILE, {})
    interests_data = load_json(INTERESTS_FILE, {})

    if req.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")

    users[req.username] = {
        "name": req.name,
        "password": req.password,  # ⚠️ Plaintext (use hashing in prod)
        "nickname": req.nickname,
        "gmail": req.gmail
    }
    interests_data[req.username] = req.interests

    save_json(USERS_FILE, users)
    save_json(INTERESTS_FILE, interests_data)

    return {"message": f"User {req.username} registered successfully!"}

@app.post("/signin")
def signin(req: SigninRequest):
    users = load_json(USERS_FILE, {})
    if req.username in users and users[req.username]["password"] == req.password:
        return {"success": True, "message": f"Welcome back {users[req.username]['nickname']}!"}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/chat")
def chat(req: ChatRequest):
    users = load_json(USERS_FILE, {})
    if req.username not in users:
        raise HTTPException(status_code=404, detail="User not found")

    reply = chatbot_reply(req.message, req.username)
    return {"reply": reply}

@app.get("/")
def root():
    return {"message": "Welcome to the chatbot API!"}
# -----------------------
# Run with Uvicorn
# -----------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
