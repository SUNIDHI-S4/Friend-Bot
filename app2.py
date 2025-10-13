# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import uvicorn
import os
import json
from dotenv import load_dotenv
import re
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if genai and api_key else None

# -----------------------
# File setup
# -----------------------
DATA_DIR = "."
USERS_FILE = os.path.join(DATA_DIR, "users.json")
INTERESTS_FILE = os.path.join(DATA_DIR, "interests.json")
CONVO_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONVO_DIR, exist_ok=True)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="AI Companion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Models
# -----------------------
class SignupRequest(BaseModel):
    name: str
    username: str
    password: str
    nickname: str
    gmail: str
    designation: str
    interests: List[str] = Field(default_factory=list)

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
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_convo_file(username: str):
    safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", username)
    return os.path.join(CONVO_DIR, f"{safe_name}.json")

def add_to_conversation(username, role, message):
    file = get_convo_file(username)
    conversations = load_json(file, [])
    conversations.append({"role": role, "message": message})
    save_json(file, conversations)

def get_recent_history(username, category):
    file = get_convo_file(username)
    conversations = load_json(file, [])
    if category == "discussive":
        limit = 20
    elif category == "suggestive":
        limit = 12
    elif category == "humorous":
        limit = 6
    elif category == "classify":
        limit = 6
    else:
        limit = 0

    if limit > 0:
        recent = conversations[-limit:]
        return "\n".join([f"{c['role'].capitalize()}: {c['message']}" for c in recent])
    return ""

def get_user_interests(username):
    data = load_json(INTERESTS_FILE, {})
    return data.get(username, [])

def get_user(username):
    users = load_json(USERS_FILE, {})
    return users.get(username)

def generate(prompt):
    if client is None:
        return "(Gemini client not configured)"
    try:
        res = client.models.generate_content(model="gemini-2.5-pro", contents=prompt)
        return res.text.strip()
    except Exception as e:
        return f"(Error: {str(e)})"

# -----------------------
# Classification & Handlers
# -----------------------
def classify_message(user_input, username):
    interests = get_user_interests(username)
    user = get_user(username) or {}
    designation = user.get("designation", "Unknown")
    history = get_recent_history(username, "classify")

    prompt = f"""
        Classify the following user input into one of the categories:
        - Suggestive (user wants tips, recommendations, or advice)
        - Discussive (user wants a thoughtful discussion)
        - Humorous (user wants a witty or playful response)
        - Help (user seems distressed, seeking emergency or professional help)

        Only return one word: Suggestive, Discussive, Humorous, or Help.


        User designation: {designation}
        User interests: {', '.join(interests) if interests else 'None'}
        Recent history:
        {history}

        Input: "{user_input}"
    """
    raw = generate(prompt)
    for label in ["suggestive", "discussive", "humorous", "help"]:
        if label in raw.lower():
            return label
    return "discussive"

def handle_suggestive(user_input, username):
    user = get_user(username)
    history = get_recent_history(username, "suggestive")
    prompt = f"""
        You are a supportive, helpful, and empathetic AI companion.
        The user is asking for suggestions or tips.
        Provide practical, positive, and human-like advice.
        Keep the tone cheerful and encouraging.

        Give concise, actionable replies.

        User nickname: {user['nickname']}
        Designation: {user['designation']}
        Interests: {', '.join(get_user_interests(username))}
        History:
        {history}

        User: {user_input}
    """
    return generate(prompt)

def handle_discussive(user_input, username):
    user = get_user(username)
    history = get_recent_history(username, "discussive")
    prompt = f"""
        You are a thoughtful, engaging, and curious AI companion.
        The user wants a discussion.
        Respond in a conversational, insightful way — ask follow-up questions,
        show empathy, and keep it natural.
            
        Give crisp, well-reasoned replies.

        Nickname: {user['nickname']}
        Designation: {user['designation']}
        History:
        {history}

        User: {user_input}
    """
    return generate(prompt)

def handle_humorous(user_input, username):
    user = get_user(username)
    history = get_recent_history(username, "humorous")
    prompt = f"""
        You are a witty, kind, and playful AI companion.
        The user wants humor.
        Respond with light, positive humor and a pinch of Gen Z/Bangalore slang,
        but remain respectful and empathetic.

        Give short, snappy replies.

        Nickname: {user['nickname']}
        Interests: {', '.join(get_user_interests(username))}
        History:
        {history}

        User: {user_input}
    """
    return generate(prompt)

def handle_help(user_input, username):
    user = get_user(username)
    prompt = f"""
        The user may be in distress or needs real help.
        Respond kindly and empathetically.
        Provide relevant helplines or professional resources from the internet.
        DO NOT give medical or legal advice yourself.
        Nickname: {user['nickname']}
        User message: {user_input}
    """
    return generate(prompt)

def chatbot_reply(user_input, username):
    category = classify_message(user_input, username)
    if category == "suggestive":
        reply = handle_suggestive(user_input, username)
    elif category == "discussive":
        reply = handle_discussive(user_input, username)
    elif category == "humorous":
        reply = handle_humorous(user_input, username)
    elif category == "help":
        reply = handle_help(user_input, username)
    else:
        reply = "I'm here for you 😊 tell me more?"

    add_to_conversation(username, "user", user_input)
    add_to_conversation(username, "bot", reply)
    return reply, category


# -----------------------
# API Endpoints
# -----------------------
@app.post("/signup")
def signup(req: SignupRequest):
    users = load_json(USERS_FILE, {})
    interests = load_json(INTERESTS_FILE, {})

    if req.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")

    users[req.username] = {
        "name": req.name,
        "password": req.password,  # plaintext (for testing only)
        "nickname": req.nickname,
        "gmail": req.gmail,
        "designation": req.designation
    }
    interests[req.username] = req.interests

    save_json(USERS_FILE, users)
    save_json(INTERESTS_FILE, interests)
    save_json(get_convo_file(req.username), [])

    return {"message": f"User {req.username} registered successfully"}

@app.post("/signin")
def signin(req: SigninRequest):
    users = load_json(USERS_FILE, {})
    if req.username not in users:
        raise HTTPException(status_code=404, detail="User not found")

    if users[req.username]["password"] == req.password:
        return {"success": True, "message": f"Welcome back {users[req.username]['nickname']}!"}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/chat")
def chat(req: ChatRequest):
    users = load_json(USERS_FILE, {})
    if req.username not in users:
        raise HTTPException(status_code=404, detail="User not found")

    reply, category = chatbot_reply(req.message, req.username)
    return {"reply": reply, "category": category}

@app.get("/")
def root():
    return {"message": "AI Companion API running successfully 🚀"}

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    uvicorn.run("app2:app", host="127.0.0.1", port=8000, reload=True)
