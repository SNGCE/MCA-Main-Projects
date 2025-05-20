import os
import json
import sqlite3
import cv2
import numpy as np
import tensorflow as tf
import dlib
import pickle
from openai import OpenAI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
from authlib.integrations.flask_client import OAuth
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, flash
from werkzeug.utils import secure_filename
from fuzzywuzzy import process
from textblob import TextBlob
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
import re
from flask_socketio import SocketIO
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
load_dotenv()
app.secret_key = os.getenv("SECRET_KEY")
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
app.config['GOOGLE_CLIENT_ID'] = os.getenv('GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv('GOOGLE_CLIENT_SECRET')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# OAuth Setup
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    client_kwargs={'scope': 'openid email profile'},
)


with open('dataset/intents.json', encoding="utf-8") as file:
    intents = json.load(file)

detector = dlib.get_frontal_face_detector()
model = tf.keras.models.load_model("FacialExpressionModel.h5")  # Fixed model loading
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
chat_model = GPT2LMHeadModel.from_pretrained("gpt2")


socketio = SocketIO(app)
last_emotion = None
emotion_responses = {
    "happy": "You look cheerful today! 😊 Want to share something good?",
    "sad": "You seem a bit down 😢. I'm here for you, want to talk?",
    "angry": "Looks like you're angry 😠. Do you want to let it out?",
    "surprised": "Wow, something unexpected? 😲",
    "neutral": "Just a calm vibe today. Want to chat?"
}

def get_db_connection():
    conn = sqlite3.connect("chatfie.db")  # Use a single database
    conn.row_factory = sqlite3.Row
    return conn

def find_best_match(user_input):
    user_input = user_input.lower().strip()
    best_match, highest_score = None, 0

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            score = process.extractOne(user_input, [pattern.lower()])[1]
            if score > highest_score:
                highest_score = score
                best_match = intent

    return best_match, highest_score

def detect_text_emotion(user_input):
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity
    return "Happy" if sentiment > 0.2 else "Sad" if sentiment < -0.2 else "Neutral"

def contains_profanity(text):
    profane_words = [
        "fuck", "shit", "crap", "bitch", "damn", "asshole", "bastard",
        "hell", "dick", "piss", "slut", "whore", "retard", "nigga", "cunt"
    ]
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return any(word in words for word in profane_words)


def generate_chat_response(user_input, chat_history=None):
    emotion = detect_text_emotion(user_input)
    matched_intent, confidence = find_best_match(user_input)
    default_response = "I'm not sure how to respond to that. Can you tell me more?"

    # 💬 Profanity Handling
    if contains_profanity(user_input):
        return "I know you’re feeling overwhelmed. Let's keep this space respectful so I can support you better. 💙"

    # 🎯 High-confidence tag-based replies
    if matched_intent and confidence >= 90:
        tag = matched_intent['tag']

        if tag == "tech_issue":
            return random.choice(matched_intent.get("responses", [default_response]))

        if tag == "about":
            if not session.get("greeted_about"):
                session["greeted_about"] = True
                return random.choice(matched_intent["responses"])
            else:
                return "I'm here for you. What's on your mind today?"

        return random.choice(matched_intent.get("responses", [default_response]))

    # 🧠 Fallback to OpenAI for uncertain or emotional input
    if not emotion:  # If no emotion is detected
        emotion = "neutral"

    prompt = (
        f"You're Chat-Fie, a helpful, empathetic, and slightly sassy AI assistant. "
        "Your job is to comfort users, help them talk through their stress, and respond like a caring friend.\n"
        f"respond empathetically and supportively"
        f"Respond with warmth, emotional intelligence, and human-like relatability. "
        f"Set gentle boundaries if the user is being rude or sarcastic."
    )

    response = get_openai_response(prompt, emotion=emotion, chat_history=chat_history)

    # 💌 Emotion Prefix
    emotion_prefix = {
        "Sad": "I'm really sorry to hear that. Is there anything I can do to help? 💙",
        "Happy": "That's wonderful to hear! I'm so glad you're feeling good! 😊",
        "Angry": "I understand you're feeling upset. Want to talk about it?",
        "Fearful": "That sounds concerning. Would you like to share what's going on?",
        "Neutral": "I'm here to listen. What’s on your mind?"
    }.get(emotion, "I'm here for you. What's going on?")

    return f"{emotion_prefix} {response}".strip()

# -------------------------- OpenAI Response --------------------------

def get_openai_response(prompt, emotion=None, chat_history=None):
    try:
        if chat_history:
            history_text = "\n".join([msg['content'] for msg in chat_history])
            prompt = f"{history_text}\n{prompt}"

        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)

        output = chat_model.generate(
            inputs,
            max_length=inputs.shape[1] + 50,
            temperature=0.7-0.9,            # More creativity
            top_k=30-50,                   # Sample from top 40 words
            top_p=0.9,                  # Nucleus sampling
            repetition_penalty=1.1-1.5,     # Slightly discourage repeats
            do_sample=True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

        full_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove the input part (prompt) to extract only the generated reply
        if full_text.startswith(prompt):
            response = full_text[len(prompt):].strip()
        else:
            response = full_text.strip()

        # Extra cleanup: Sometimes GPT-2 ends with quotes or incomplete sentences
        response = response.split('\n')[0].strip().strip('"').strip("'")

        if not response:
            return "I'm here for you. Can you share a bit more?"

        return response

    except Exception as e:
        print(f"GPT-2 error: {e}")
        return "I'm having trouble thinking right now. Can you try again?"
    
def analyze_text_sentiment(text):
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0.3:
        return "happy"
    elif sentiment_score < -0.3:
        return "sad"
    else:
        return "neutral"

@app.route('/')
def home():
    return render_template('index.html', username=session.get('username', 'Login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        gender = request.form.get('gender')

        # Extra validation to avoid NULL values sneaking in
        if not all([name, username, email, password, gender]):
            return "Error: All fields are required, including gender."

        try:
            with sqlite3.connect("chatfie.db") as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO registration (name, username, email, password, gender) VALUES (?, ?, ?, ?, ?)",
                               (request.form['name'], request.form['username'], request.form['email'], request.form['password'], request.form['gender']))
                conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Error: Username or Email already exists!"
    return render_template('register.html')


# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email

users = {}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

@app.route("/login/google")
def login_google():
    return google.authorize_redirect(url_for("auth_callback", _external=True))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 🔐 Hardcoded admin login
        if username == "admin" and password == "Admin@123":
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM registration WHERE username = ? AND password = ?", 
                           (request.form['username'], request.form['password']))
            user = cursor.fetchone()
            if user:
                session["user_id"] = user[0]  # ✅ Store user ID in session after login
                session["username"] = request.form['username']  # ✅ Store username in session
                return redirect(url_for('home'))
        return "Invalid username or password!"
    return render_template('login.html')

@app.route("/check_login")
def check_login():
    return {"logged_in": "user_id" in session}

@app.route('/admin')
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("login"))
    return render_template("admin_dashboard.html")

@app.route('/profile')
def profile():
    if "user_id" not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, email, profile_pic FROM registration WHERE id = ?", (session["user_id"],))
        user = cursor.fetchone()
    
    if user:
        # If the user doesn't have a profile picture, use a default one
        profile_pic = user[2] if user[2] else 'static/default_profile.png'
        return render_template("profile.html", username=user[0], email=user[1], profile_pic=profile_pic)
    else:
        return "User not found!", 404
    
@app.route("/edit_profile", methods=["GET", "POST"])
def edit_profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    conn = sqlite3.connect("chatfie.db")
    cursor = conn.cursor()

    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        gender = request.form["gender"]
        file = request.files.get("profile_pic")

        # Get current profile pic path
        cursor.execute("SELECT profile_pic FROM registration WHERE id=?", (user_id,))
        row = cursor.fetchone()
        profile_pic_path = row[0] if row else 'static/default_profile.png'

        # If new profile picture is uploaded
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            upload_dir = os.path.join("static", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            filepath = os.path.join('static', 'uploads', filename)
            file.save(filepath)
            profile_pic_path = filepath

        cursor.execute("""
            UPDATE registration 
            SET name=?, email=?, gender=?, profile_pic=?
            WHERE id=?
        """, (name, email, gender, profile_pic_path, user_id))
        conn.commit()
        conn.close()

        flash("Profile updated successfully.")
        return redirect(url_for("profile"))

    # GET: Load existing data
    cursor.execute("SELECT name, email, gender, profile_pic FROM registration WHERE id=?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return "User not found", 404

    return render_template("edit_profile.html", user={
        "name": user[0],
        "email": user[1],
        "gender": user[2],
        "profile_pic": user[3]
    })

@app.route("/login/callback")
def auth_callback():
    # Step 1: Fetch Token
    token = google.authorize_access_token()
    
    if not token:
        return "Login failed!", 401

    # Step 2: Get User Info
    user_info = google.get("userinfo").json()
    if not user_info or "email" not in user_info:
        return "Error retrieving user info", 400

    # Step 3: Store/Check User in Database
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM registration WHERE email = ?", (user_info["email"],))
        user = cursor.fetchone()

        if not user:
            # New user - Insert into DB
            cursor.execute(
                "INSERT INTO registration (name, username, email, password, gender) VALUES (?, ?, ?, ?, ?)",
                (user_info["name"], user_info["email"], user_info["email"], "", "")
            )
            conn.commit()
            user_id = cursor.lastrowid
        else:
            user_id = user[0]

    # Step 4: Store User in Session
    session["user_id"] = user_id
    session["username"] = user_info["name"]
    session["email"] = user_info["email"]

    return redirect(url_for("home"))

@app.route('/logout')
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('home'))

@app.route('/add_intent', methods=['POST'])
def add_intent():
    if not session.get("admin"):
        return redirect(url_for("login"))

    tag = request.form["tag"]
    patterns = request.form.getlist("patterns")
    responses = request.form.getlist("responses")

    with open('dataset/intents.json', 'r+', encoding="utf-8") as f:
        data = json.load(f)
        data['intents'].append({
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        })
        f.seek(0)
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.truncate()

    return redirect(url_for("admin_dashboard"))

@app.route('/admin/feedback')
def view_feedback():
    if not session.get("admin"):
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM feedback ORDER BY timestamp DESC")
    feedback_data = cursor.fetchall()
    conn.close()

    return render_template("admin_feedback.html", feedback=feedback_data)

@app.route('/admin_logout')
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("login"))


@app.route('/chat', methods=['POST'])
def chat():
    conn = get_db_connection()
    user_input = request.form['message']
    matched_intent, confidence = find_best_match(user_input)
    detected_mood = detect_text_emotion(user_input)
    default_response = "I'm not sure how to respond to that. Can you tell me more?"

    # 💡 Always trigger OpenAI if no match OR low confidence
    use_openai = matched_intent is None or confidence < 85

    if use_openai:
        last_user_input = session.get('last_user_input', '')
        last_emotion = session.get('last_emotion', '')

        prompt = (
            f"The user feels {detected_mood}. "
            f"They previously said: '{last_user_input}' and now said: '{user_input}'. "
            f"They might be experiencing emotions like {last_emotion}. "
            f"Please respond with emotional warmth, offer comfort, and avoid sounding robotic or generic. "
            f"Ask gentle follow-up questions if needed."
        )

        response = get_openai_response(prompt)
    else:
        response = random.choice(matched_intent.get('responses', [default_response]))

    # 💢 Profanity override (only after matching)
    if contains_profanity(user_input):
        response = "I understand you're upset. I'm here to help if you want to talk."

    # Save emotion-based message if changed
    if 'emotion_changed_to' in session:
        changed_emotion = session.pop('emotion_changed_to')
        emotion_msg = emotion_messages.get(changed_emotion)
        if emotion_msg and 'user_id' in session:
            user_id = session['user_id']
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                "INSERT INTO chat_history (user_id, message, response, emotion, timestamp) VALUES (?, ?, ?, ?, ?)",
                (user_id, "[Emotion detected]", emotion_msg, changed_emotion, timestamp)
            )
            conn.commit()

    emotion_messages = {
    "happy": "You look cheerful today! 😊 Want to share something good?",
    "sad": "You seem a little down 😔. I'm here to listen if you need me.",
    "angry": "Frustrated? 😤 Let it out, I'm all ears.",
    "neutral": "Just a calm vibe today. Want to chat?",
    "surprise": "Oh! Something surprised you? 😯 Want to talk about it?",
    "fear": "You seem anxious 😟. It’s okay, I’m here for you."
    }
    # Save actual chat
    if 'user_id' in session:
        user_id = session['user_id']
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        conn.execute("INSERT INTO chat_history (user_id, message, response, timestamp) VALUES (?, ?, ?, ?, ?)",
                     (user_id, user_input, response, detected_mood, timestamp))

        conn.execute("INSERT INTO mood_tracking (user_id, mood, timestamp) VALUES (?, ?, ?)",
                     (user_id, detected_mood, timestamp))

        conn.commit()
    conn.close()

    return jsonify({"user_input": user_input, "response": response, "emotion": detected_mood})

@app.route("/get_previous_chats", methods=["GET"])
def get_previous_chats():
    if "user_id" not in session:
        return jsonify({"error": "User not logged in"}), 401

    user_id = session["user_id"]
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, user_message, bot_response, timestamp FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,)
    )
    
    chats = cursor.fetchall()
    conn.close()

    chat_list = [
        {"id": chat["id"], "user_message": chat["user_message"], "bot_response": chat["bot_response"], "timestamp": chat["timestamp"]}
        for chat in chats
    ]
    
    return jsonify(chat_list)

@app.route("/get_chat_sessions", methods=["GET"])
def get_chat_sessions():
    if "user_id" not in session:
        return jsonify({"error": "User not logged in"}), 401

    user_id = session["user_id"]
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT session_id, session_title, created_at
        FROM chat_sessions
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,)
    )

    sessions = cursor.fetchall()
    conn.close()

    session_list = [
        {
            "session_id": row["session_id"],
            "title": row["session_title"],
            "created_at": row["created_at"]
        }
        for row in sessions
    ]
    
    return jsonify(session_list)

@app.route("/get_session_messages/<session_id>", methods=["GET"])
def get_session_messages(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT user_message, bot_response, timestamp
        FROM chat_history
        WHERE session_id = ?
        ORDER BY timestamp ASC
        """,
        (session_id,)
    )
    
    messages = cursor.fetchall()
    conn.close()

    return jsonify([
        {
            "user_message": msg["user_message"],
            "bot_response": msg["bot_response"],
            "timestamp": msg["timestamp"]
        } for msg in messages
    ])


import sqlite3

from flask import request, jsonify
from datetime import datetime, timedelta
import sqlite3

@app.route('/get_mood_data')
def get_mood_data():
    try:
        conn = sqlite3.connect("chatfie.db")
        cursor = conn.cursor()

        # Query parameters
        selected_date = request.args.get("date")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        user_id = request.args.get("user_id")  # Optional filtering

        # Default to last 7 days
        if not selected_date and not (start_date and end_date):
            two_weeks_ago = (datetime.today() - timedelta(days=7)).strftime("%d-%m-%Y")
            today = datetime.today().strftime("%d-%m-%Y")
            start_date, end_date = two_weeks_ago, today

        # Construct SQL and parameters
        if selected_date:
            query = "SELECT mood, COUNT(*) FROM mood_tracking WHERE timestamp = ?"
            params = [selected_date]
        elif start_date and end_date:
            query = "SELECT mood, COUNT(*) FROM mood_tracking WHERE timestamp BETWEEN ? AND ?"
            params = [start_date, end_date]
        else:
            return jsonify({"error": "Invalid date selection"}), 400

        # Add user filter if present
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        query += " GROUP BY mood"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return jsonify({"error": "No mood data found"}), 404

        # Convert to dictionary
        mood_summary = {mood: count for mood, count in rows}

        return jsonify(mood_summary)

    except sqlite3.OperationalError as e:
        print("Database error:", e)
        return jsonify({"error": "Database error: " + str(e)}), 500
    except Exception as e:
        print("Unexpected error:", e)
        return jsonify({"error": "Unexpected error: " + str(e)}), 500


@app.route('/video_feed')
def video_feed():
    user_id = session.get('user_id')
    return Response(generate_frames(user_id), mimetype='multipart/x-mixed-replace; boundary=frame')


def load_object(name):
    with open(f"{name}.pck", "rb") as f:
        return pickle.load(f)

Le = load_object("LabelEncoder")

def ProcessImage(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [96, 96], method="bilinear")
    image = tf.expand_dims(image, 0)
    return image

def RealtimePrediction(image, model, encoder_):
    prediction = model.predict(image)
    prediction = np.argmax(prediction, axis=1)
    return encoder_.inverse_transform(prediction)[0]

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Initialize video capture
VideoCapture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

def generate_frames(user_id=None):
    global last_emotion
    while True:
        ret, frame = VideoCapture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects) >= 1:
            for rect in rects:
                (x, y, w, h) = rect_to_bb(rect)
                img = gray[y-10:y+h+10, x-10:x+w+10]

                if img.shape[0] == 0 or img.shape[1] == 0:
                    continue
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = ProcessImage(img)
                    out = RealtimePrediction(img, model, Le)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    z = y - 15 if y - 15 > 15 else y + 15
                    cv2.putText(frame, str(out), (x, z), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                    if out != last_emotion:
                        last_emotion = out
                        response = emotion_responses.get(out)
                        if response:
                            socketio.emit('bot_message', {'msg': response})

                            if user_id:
                                timestamp = datetime.now().strftime("%d-%m-%Y")
                                conn = get_db_connection()
                                conn.execute(
                                    "INSERT INTO mood_tracking (user_id, mood, timestamp) VALUES (?, ?, ?)",
                                    (user_id, out, timestamp)
                                )
                                conn.commit()
                                conn.close()

                            print(f"Emotion changed to {out} -> Sent message: {response}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/save_chat', methods=['POST'])
def save_chat():
    data = request.get_json()
    user_id = data.get('user_id')   # Frontend must send this!
    user_message = data.get('message')
    bot_response = data.get('response')
    timestamp = data.get('timestamp')  # Optional (can use default CURRENT_TIMESTAMP)

    if not all([user_id, user_message, bot_response]):
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO chat_history (user_id, user_message, bot_response, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (user_id, user_message, bot_response, timestamp))
            conn.commit()

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/feedback')
def feedback_page():
    if 'user_id' not in session:
        flash("Login first to access the feedback page.")
        return redirect('/')
    return render_template('feedback.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    rating = data.get('rating')
    feedback = data.get('comment')
    timestamp = data.get('timestamp')
    user_id = session.get('user_id')

    if not user_id:
        return jsonify({'error': 'User not logged in'}), 403

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback (user_id, rating, feedback, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (user_id, rating, feedback, timestamp))
    conn.commit()
    conn.close()

    return jsonify({'status': 'success'})


@app.route('/clear_chat_history', methods=['POST'])
def clear_chat_history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User not logged in'}), 403

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_history WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get', methods=['GET'])
def get_response():
    user_input = request.args.get('msg')
    return jsonify({'response': generate_chat_response(user_input)})

if __name__ == '__main__':
    app.run(debug=True)
