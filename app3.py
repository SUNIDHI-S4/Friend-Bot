from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import os
import json
import re
import uvicorn
from dotenv import load_dotenv
from google import genai

# ------------------ SETUP ------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if genai and api_key else None

DATABASE_DIR = "./database"
os.makedirs(DATABASE_DIR, exist_ok=True)

app = FastAPI(title="AI Companion API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ------------------ MODELS ------------------
class SignupRequest(BaseModel):
    username: str
    gmail: str
    password: str
    nickname: str
    designation: str
    interests: List[str] = Field(default_factory=list)

class SigninRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    username: str
    message: str

# ------------------ UTILITIES ------------------
def user_file(username):
    safe = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", username)
    return os.path.join(DATABASE_DIR, f"{safe}.json")

def load_user(username):
    path = user_file(username)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_user(username, data):
    with open(user_file(username), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def generate(prompt):
    if not client:
        return "(Gemini not configured)"
    try:
        res = client.models.generate_content(model="gemini-2.5-pro", contents=prompt)
        return res.text.strip()
    except Exception as e:
        return f"(Error: {str(e)})"

def add_conversation(username, role, message):
    user = load_user(username)
    user["conversation_history"].append({"role": role, "message": message})
    save_user(username, user)

def get_recent_history(username, category):
    user = load_user(username)
    conversations = user["conversation_history"]
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

# ------------------ CLASSIFICATION ------------------
def classify_message(user_input, username):
    user = load_user(username)
    profile = user["profile"]

    prompt = f"""
    Classify the following user message into one of the categories:
    - Suggestive (user wants tips, recommendations, or advice)
    - Discussive (user wants thoughtful or emotional discussion)
    - Humorous (user wants light humor or playful tone)
    - Help (user may be in distress or seeking real help)

    Return only one word: Suggestive, Discussive, Humorous, or Help.

    User Info:
    Nickname: {profile['nickname']}
    Designation: {profile['designation']}
    Interests: {', '.join(profile['interests'])}
    History:
    {get_recent_history(username, 'classify')}

    Input: "{user_input}"
    """
    raw = generate(prompt)
    for label in ["suggestive", "discussive", "humorous", "help"]:
        if label in raw.lower():
            return label
    return "discussive"

# ------------------ RESPONSE GENERATORS ------------------
def handle_suggestive(user_input, username):
    user = load_user(username)
    profile = user["profile"]
    history = get_recent_history(username, "suggestive")

    prompt = f"""
    You are a supportive and helpful AI companion.
    The user is asking for suggestions or advice.
    Provide concise, encouraging, and friendly tips.

    User Info:
    Nickname: {profile['nickname']}
    Designation: {profile['designation']}
    Interests: {', '.join(profile['interests'])}
    Favorites: {', '.join(profile['favorites'])}
    Events: {', '.join(profile['events'])}
    People: {', '.join(profile['people'])}

    History:
    {history}

    User: {user_input}
    """
    return generate(prompt)

def handle_discussive(user_input, username):
    user = load_user(username)
    profile = user["profile"]
    history = get_recent_history(username, "discussive")

    prompt = f"""
    You are a thoughtful, empathetic, and engaging AI companion.
    Respond naturally, showing curiosity and emotional depth.
    Ask gentle follow-up questions occasionally.

    User Info:
    Nickname: {profile['nickname']}
    Designation: {profile['designation']}
    Interests: {', '.join(profile['interests'])}
    Favorites: {', '.join(profile['favorites'])}
    Events: {', '.join(profile['events'])}
    People: {', '.join(profile['people'])}

    History:
    {history}

    User: {user_input}
    """
    return generate(prompt)

def handle_humorous(user_input, username):
    user = load_user(username)
    profile = user["profile"]
    history = get_recent_history(username, "humorous")

    prompt = f"""
    You are a witty, kind, and Bangalorean-style humorous AI companion.
    Use light, fun humor with Gen Z slang.
    Keep it positive and respectful.

    Nickname: {profile['nickname']}
    Interests: {', '.join(profile['interests'])}
    Favorites: {', '.join(profile['favorites'])}
    History:
    {history}

    User: {user_input}
    """
    return generate(prompt)

def handle_help(user_input, username):
    user = load_user(username)
    profile = user["profile"]
    prompt = f"""
    The user may be in distress or need real help.
    Respond empathetically and kindly.
    Offer contact details of professional helplines.
    Do not give medical or legal advice yourself.

    Nickname: {profile['nickname']}
    User message: {user_input}
    """
    return generate(prompt)

# # ------------------ PROFILE ENRICHMENT ------------------
# def enrich_profile(username):
#     user = load_user(username)
#     convos = user["conversation_history"]
#     text = "\n".join([f"{c['role']}: {c['message']}" for c in convos])

#     extract_prompt = f"""
#     You are an AI that analyzes user conversations.
#     Extract a summary in pure JSON format:
#     {{
#         "favorites": [things or activities user enjoys],
#         "events": [important life events mentioned],
#         "people": [names or relationships mentioned]
#     }}
#     Only return valid JSON.
#     Conversation log:
#     {text}
#     """

#     result = generate(extract_prompt)
#     try:
#         extracted = json.loads(result)
#         for key in ["favorites", "events", "people"]:
#             if key in extracted:
#                 existing = user["profile"].get(key, [])
#                 user["profile"][key] = list(set(existing + extracted[key]))
#         save_user(username, user)
#     except Exception:
#         pass

# ------------------ JSON CLEANING ------------------
def clean_json_response(raw_text: str):
    """
    Cleans Gemini's response so only the pure JSON remains.
    Removes markdown/code block wrappers like ```json, '''json, etc.
    """
    # Remove leading/trailing code fences and text like ```json or '''json
    cleaned = re.sub(r"^(`{3,}|'{3,})\s*json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"(`{3,}|'{3,})$", "", cleaned.strip())

    # Optionally remove any prefix text before the first { and after last }
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)

    return cleaned

# ------------------ PROFILE ENRICHMENT ------------------
def enrich_profile(username):
    user = load_user(username)
    convos = user["conversation_history"]
    # print(f"Convos: \n{convos}")

    # ✅ Only consider the last 30 messages
    recent_convos = convos[-30:] if len(convos) >= 30 else convos

    text = "\n".join([f"{c['role']}: {c['message']}" for c in recent_convos])

    extract_prompt = f"""
    You are an AI that analyzes user conversations with their AI companion.
    Analyze only these messages and extract a summary in valid JSON:
    {{
        "favorites": [things or activities user enjoys],
        "events": [important life events mentioned],
        "people": [names or relationships mentioned]
    }}
    Respond with only valid JSON — no explanations or text outside JSON.
    
    Conversation log:
    {text}
    """

    result = generate(extract_prompt)
    # print(f"Enrichment result: {result}")
    result = clean_json_response(result)
    print(f"Cleaned JSON: {result}")
    try:
        extracted = json.loads(result)
        for key in ["favorites", "events", "people"]:
            if key in extracted:
                existing = user["profile"].get(key, [])
                user["profile"][key] = list(set(existing + extracted[key]))
        save_user(username, user)
    except Exception:
        pass  # Ignore JSON parsing errors silently

# ------------------ CHATBOT MAIN LOGIC ------------------
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

    add_conversation(username, "user", user_input)
    add_conversation(username, "bot", reply)

    # Check message count for enrichment
    user = load_user(username)
    user_msg_count = sum(1 for m in user["conversation_history"] if m["role"] == "user")
    if user_msg_count % 15 == 0:
        enrich_profile(username)

    return reply, category

# ------------------ ROUTES ------------------
@app.post("/signup")
def signup(req: SignupRequest):
    path = user_file(req.username)
    if os.path.exists(path):
        raise HTTPException(status_code=400, detail="Username already exists")

    user_data = {
        "credentials": {
            "username": req.username,
            "gmail": req.gmail,
            "password": req.password
        },
        "profile": {
            "nickname": req.nickname,
            "designation": req.designation,
            "interests": req.interests,
            "favorites": [],
            "events": [],
            "people": []
        },
        "conversation_history": []
    }
    save_user(req.username, user_data)
    return {"message": f"User {req.username} registered successfully"}

@app.post("/signin")
def signin(req: SigninRequest):
    user = load_user(req.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user["credentials"]["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"success": True, "message": f"Welcome back {user['profile']['nickname']}!"}

@app.post("/chat")
def chat(req: ChatRequest):
    user = load_user(req.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    reply, category = chatbot_reply(req.message, req.username)
    return {"reply": reply, "category": category}

@app.get("/")
def root():
    return {"message": "AI Companion Full API running 🚀"}

# ------------------ RUN ------------------
if __name__ == "__main__":
    uvicorn.run("app3:app", host="127.0.0.1", port=8000, reload=True)
