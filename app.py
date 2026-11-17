import os
import uuid
import json
import math
import sqlite3
import secrets
import urllib.request
import urllib.parse
import urllib.error

from functools import wraps
from datetime import timedelta

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    url_for,
    redirect,
    session
)

from werkzeug.utils import secure_filename
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from gradcam import generate_gradcam
from disease_info import get_disease_info
from eye_care import get_eye_care_recommendations
from chatbot import get_chatbot_response


# =====================================
# 1. FLASK CONFIGURATION
# =====================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    secrets.token_hex(32)
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Enable secure cookies when deploying with HTTPS.
app.config["SESSION_COOKIE_SECURE"] = (
    os.environ.get("COOKIE_SECURE", "False").lower() == "true"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

CONFIDENCE_THRESHOLD = 80.0

# Database configuration
INSTANCE_FOLDER = os.path.join(BASE_DIR, "instance")

os.makedirs(INSTANCE_FOLDER, exist_ok=True)

DATABASE_PATH = os.path.join(
    INSTANCE_FOLDER,
    "users.db"
)


# =====================================
# 2. DATABASE FUNCTIONS
# =====================================

def get_db_connection():

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    return connection


def create_database():

    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()

    connection.close()


# Create database automatically when app starts.
create_database()


# =====================================
# 3. LOGIN REQUIRED DECORATOR
# =====================================

def login_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:

            # API requests receive JSON instead of a login page.
            if request.path.startswith("/api/") or request.path in [
                "/analyze",
                "/chat"
            ]:

                return jsonify({
                    "success": False,
                    "error": "Please login first."
                }), 401

            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return decorated_function


# =====================================
# 4. CHECK FILE EXTENSION
# =====================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =====================================
# 5. SIGNUP PAGE
# =====================================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if "user_id" in session:
        return redirect(url_for("home"))

    error = None
    success = None

    if request.method == "POST":

        name = request.form.get("name", "").strip()

        email = request.form.get("email", "").strip().lower()

        password = request.form.get("password", "")

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not name or not email or not password:

            error = "Please fill in all required fields."

        elif len(name) > 100:

            error = "Name must be under 100 characters."

        elif len(email) > 254 or "@" not in email:

            error = "Please enter a valid email address."

        elif len(password) < 8:

            error = "Password must contain at least 8 characters."

        elif len(password) > 128:

            error = "Password is too long."

        elif password != confirm_password:

            error = "Passwords do not match."

        else:

            connection = get_db_connection()

            existing_user = connection.execute(
                "SELECT id FROM users WHERE email = ?",
                (email,)
            ).fetchone()

            if existing_user:

                error = "An account with this email already exists."

            else:

                hashed_password = generate_password_hash(password)

                connection.execute(
                    """
                    INSERT INTO users (name, email, password)
                    VALUES (?, ?, ?)
                    """,
                    (name, email, hashed_password)
                )

                connection.commit()

                success = "Account created successfully. Please login."

            connection.close()

    return render_template(
        "signup.html",
        error=error,
        success=success
    )


# =====================================
# 6. LOGIN PAGE
# =====================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:
        return redirect(url_for("home"))

    error = None
    success = None

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()

        password = request.form.get("password", "")

        if not email or not password:

            error = "Please enter your email and password."

        else:

            connection = get_db_connection()

            user = connection.execute(
                """
                SELECT * FROM users
                WHERE email = ?
                """,
                (email,)
            ).fetchone()

            connection.close()

            if user and check_password_hash(
                user["password"],
                password
            ):

                session.clear()

                session.permanent = True

                session["user_id"] = user["id"]

                session["user_name"] = user["name"]

                session["user_email"] = user["email"]

                return redirect(url_for("home"))

            else:

                error = "Incorrect email or password."

    return render_template(
        "login.html",
        error=error,
        success=success
    )


# =====================================
# 7. LOGOUT
# =====================================

@app.route("/logout")
@login_required
def logout():

    session.clear()

    return redirect(url_for("login"))


# =====================================
# 8. HOME PAGE
# =====================================

@app.route("/")
@login_required
def home():

    return render_template(
        "index.html",
        user_name=session.get("user_name")
    )


# =====================================
# 9. NEARBY EYE CENTRES PAGE
# MODULE 9
# =====================================

@app.route("/eye-centres")
@login_required
def eye_centres():

    return render_template("map.html")


# =====================================
# 10. MAP DATA API
# MODULE 9
# =====================================

OVERPASS_SERVERS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter"
]


def calculate_distance(lat1, lon1, lat2, lon2):

    radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return 2 * radius * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )


def build_eye_centres_query(lat, lon):

    radius = 10000

    return f"""
    [out:json][timeout:20];
    (
      nwr["healthcare"="ophthalmology"]
          (around:{radius},{lat},{lon});

      nwr["healthcare:speciality"~"ophthalmology",i]
          (around:{radius},{lat},{lon});

      nwr["name"~"eye hospital|eye clinic|eye centre|eye center|ophthalmology|eye specialist",i]
          (around:{radius},{lat},{lon});
    );
    out center;
    """


def fetch_overpass_data(query):

    payload = urllib.parse.urlencode({
        "data": query
    }).encode("utf-8")

    for server in OVERPASS_SERVERS:

        try:

            api_request = urllib.request.Request(
                server,
                data=payload,
                headers={
                    "User-Agent": "EyeCareAI-EducationalProject/1.0",
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                method="POST"
            )

            with urllib.request.urlopen(
                api_request,
                timeout=10
            ) as response:

                raw_data = response.read(
                    5 * 1024 * 1024 + 1
                )

            if len(raw_data) > 5 * 1024 * 1024:

                raise ValueError("Map response too large.")

            data = json.loads(
                raw_data.decode("utf-8")
            )

            if not isinstance(data.get("elements"), list):

                raise ValueError("Invalid map data.")

            return data

        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            OSError
        ) as error:

            app.logger.warning(
                "Overpass server unavailable: %s - %s",
                server,
                error
            )

    return None


@app.route("/api/eye-centres", methods=["GET"])
@login_required
def nearby_eye_centres_api():

    try:

        lat = float(request.args.get("lat", ""))

        lon = float(request.args.get("lon", ""))

        if not (
            math.isfinite(lat)
            and math.isfinite(lon)
            and -90 <= lat <= 90
            and -180 <= lon <= 180
        ):

            raise ValueError("Invalid coordinates.")

    except (ValueError, TypeError):

        return jsonify({
            "success": False,
            "error": "Please provide valid coordinates."
        }), 400

    query = build_eye_centres_query(lat, lon)

    data = fetch_overpass_data(query)

    if data is None:

        return jsonify({
            "success": False,
            "error": (
                "Live eye hospital data is temporarily "
                "unavailable. Please use the Google Maps "
                "search option."
            ),
            "clinics": []
        }), 503

    clinics = []

    seen = set()

    for item in data["elements"]:

        point = item.get("center") or item

        clinic_lat = point.get("lat")

        clinic_lon = point.get("lon")

        if not isinstance(
            clinic_lat, (int, float)
        ) or not isinstance(
            clinic_lon, (int, float)
        ):

            continue

        tags = item.get("tags") or {}

        name = tags.get(
            "name",
            "Unnamed eye-care facility"
        )

        identifier = (
            item.get("type"),
            item.get("id")
        )

        if identifier in seen:
            continue

        seen.add(identifier)

        address = ", ".join(
            str(value)
            for value in [
                tags.get("addr:housenumber"),
                tags.get("addr:street"),
                tags.get("addr:city")
            ]
            if value
        )

        distance = calculate_distance(
            lat,
            lon,
            clinic_lat,
            clinic_lon
        )

        clinics.append({
            "name": name,
            "lat": clinic_lat,
            "lon": clinic_lon,
            "address": address,
            "distance": round(distance, 1)
        })

    clinics.sort(
        key=lambda clinic: clinic["distance"]
    )

    clinics = clinics[:30]

    return jsonify({
        "success": True,
        "clinics": clinics,
        "count": len(clinics),
        "radius_km": 10
    })


# =====================================
# 11. IMAGE ANALYSIS
# MOBILE NET V2 + 80% THRESHOLD
# =====================================

@app.route("/analyze", methods=["POST"])
@login_required
def analyze_image():

    image_path = None
    heatmap_path = None

    try:

        # Validate image upload

        if "image" not in request.files:

            return jsonify({
                "error": "Please upload an image."
            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({
                "error": "No image selected."
            }), 400

        if not allowed_file(file.filename):

            return jsonify({
                "error": (
                    "Only JPG, JPEG and PNG "
                    "images are allowed."
                )
            }), 400

        original_name = secure_filename(
            file.filename
        )

        if not original_name or "." not in original_name:

            return jsonify({
                "error": "Invalid image filename."
            }), 400

        extension = original_name.rsplit(
            ".", 1
        )[1].lower()

        filename = f"{uuid.uuid4().hex}.{extension}"

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(image_path)

        # Create unique heatmap filename

        heatmap_filename = (
            f"gradcam_{uuid.uuid4().hex}.jpg"
        )

        heatmap_path = os.path.join(
            UPLOAD_FOLDER,
            heatmap_filename
        )

        # Run Fine-Tuned MobileNetV2

        result = generate_gradcam(
            image_path,
            heatmap_path
        )

        disease = result["disease"]

        confidence = float(
            result["confidence"]
        )

        # Apply confidence threshold

        threshold = float(
            result.get(
                "threshold",
                CONFIDENCE_THRESHOLD
            )
        )

        is_uncertain = (
            confidence < threshold
        )

        prediction_status = (
            "Uncertain"
            if is_uncertain
            else "Accepted"
        )

        if is_uncertain:

            warning_message = (
                "Uncertain Prediction: Model confidence "
                f"is below the {threshold:g}% threshold. "
                "This result should not be relied upon "
                "for medical decisions. Please consult "
                "an eye specialist."
            )

        else:

            warning_message = ""

        # Get disease information

        disease_info = get_disease_info(
            disease
        )

        care_info = get_eye_care_recommendations(
            disease
        )

        # Print result in terminal

        print("\n========== WEBSITE PREDICTION ==========")

        print("Disease:", disease)

        print(
            "Confidence:",
            round(confidence, 2),
            "%"
        )

        print(
            "Threshold:",
            threshold,
            "%"
        )

        print(
            "Prediction Status:",
            prediction_status
        )

        if is_uncertain:
            print("Warning:", warning_message)

        # Send results to website

        return jsonify({

            "success": True,

            "disease": disease.replace(
                "_", " "
            ).title(),

            "confidence": round(
                confidence, 2
            ),

            "is_uncertain": bool(is_uncertain),

            "threshold": threshold,

            "prediction_status": prediction_status,

            "warning_message": warning_message,

            "image_url": url_for(
                "static",
                filename=f"uploads/{filename}"
            ),

            "heatmap_url": url_for(
                "static",
                filename=f"uploads/{heatmap_filename}"
            ),

            "disease_info": disease_info,

            "eye_care": care_info

        })

    except Exception:

        app.logger.exception(
            "Image analysis error"
        )

        return jsonify({
            "error": (
                "Image analysis failed. "
                "Please check the terminal for details."
            )
        }), 500


# =====================================
# 12. GEMINI AI CHATBOT
# MODULE 8
# =====================================

@app.route("/chat", methods=["POST"])
@login_required
def chatbot_api():

    try:

        data = request.get_json(
            silent=True
        )

        if not isinstance(data, dict):

            return jsonify({
                "success": False,
                "error": "Invalid request format."
            }), 400

        message = data.get(
            "message", ""
        )

        history = data.get(
            "history", []
        )

        if not isinstance(message, str) or not message.strip():

            return jsonify({
                "success": False,
                "error": "Please enter a message."
            }), 400

        message = message.strip()

        if len(message) > 2000:

            return jsonify({
                "success": False,
                "error": "Message must be under 2000 characters."
            }), 400

        if not isinstance(history, list):

            return jsonify({
                "success": False,
                "error": "Invalid conversation history."
            }), 400

        if len(history) > 10:

            history = history[-10:]

        clean_history = []

        for item in history:

            if not isinstance(item, dict):
                continue

            role = item.get("role")

            text = item.get("text", "")

            if (
                role in ("user", "model")
                and isinstance(text, str)
                and text.strip()
            ):

                clean_history.append({
                    "role": role,
                    "text": text.strip()[:2000]
                })

        reply = get_chatbot_response(
            message,
            clean_history
        )

        return jsonify({
            "success": True,
            "reply": reply
        }), 200

    except ValueError as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except Exception:

        app.logger.exception(
            "Gemini chatbot request failed"
        )

        return jsonify({
            "success": False,
            "error": (
                "AI assistant is temporarily unavailable. "
                "Please try again."
            )
        }), 503


# =====================================
# 13. HANDLE LARGE FILES
# =====================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": (
            "Request is too large. "
            "Maximum allowed size is 10 MB."
        )
    }), 413


# =====================================
# 14. RUN FLASK SERVER
# =====================================

if __name__ == "__main__":

    app.run(debug=True)