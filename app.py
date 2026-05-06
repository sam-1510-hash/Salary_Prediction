from flask import Flask, render_template, request, jsonify
import pickle
import PyPDF2
import re
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model + columns
try:
    model = pickle.load(open("model.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model files not found! Please train the model first.")
    exit(1)

# Enhanced skills database
SKILLS_DATABASE = {
    "programming": ["python", "java", "c++", "javascript", "typescript", "rust", "ruby", "php", "swift",
                    "kotlin"],
    "data_science": ["machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
                     "scikit-learn", "pandas", "numpy", "data science", "data analysis", "data mining"],
    "web_dev": ["flask", "django", "react", "angular", "vue", "node.js", "express", "html", "css", "rest api",
                "graphql"],
    "database": ["sql", "mysql", "postgresql", "mongodb", "oracle", "redis", "cassandra"],
    "cloud_devops": ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "ci/cd", "terraform", "ansible"],
    "soft_skills": ["leadership", "communication", "teamwork", "problem solving", "project management", "agile",
                    "scrum"]
}

# Flatten skills list
ALL_SKILLS = [skill for category in SKILLS_DATABASE.values() for skill in category]

# Education level mapping with weights
EDUCATION_LEVELS = {
    "phd": {"level": 4, "name": "PhD"},
    "doctorate": {"level": 4, "name": "PhD"},
    "master": {"level": 3, "name": "Master's"},
    "mtech": {"level": 3, "name": "Master's"},
    "msc": {"level": 3, "name": "Master's"},
    "bachelor": {"level": 2, "name": "Bachelor's"},
    "btech": {"level": 2, "name": "Bachelor's"},
    "ba": {"level": 2, "name": "Bachelor's"},
    "bsc": {"level": 2, "name": "Bachelor's"},
    "diploma": {"level": 1, "name": "Diploma"},
    "high school": {"level": 0, "name": "High School"}
}


def extract_text(file):
    """Extract text from PDF with error handling"""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text.lower()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


def extract_experience(text):
    """Improved experience extraction (no false positives)"""

    # Strict patterns for professional experience
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
        r'(\d+)\+?\s*(?:years?|yrs?)\s+in\s+\w+',
        r'experience\s*[:\-]?\s*(\d+)',
        r'worked\s+for\s+(\d+)\s*(?:years?|yrs?)'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return int(matches[0])

    # Handle internships separately (optional)
    if 'internship' in text or 'intern' in text:
        # Only count if explicitly duration mentioned
        intern_match = re.findall(r'(\d+)\s*(?:months?)\s+intern', text)
        if intern_match:
            months = int(intern_match[0])
            return round(months / 12, 1)  # convert months → years

        return 0  # Don't assume experience

    return 0


def extract_education(text):
    """Enhanced education extraction with details"""
    for degree, info in EDUCATION_LEVELS.items():
        if degree in text:
            return info["level"], info["name"]
    return 0, "Not Specified"


def extract_skills(text):
    """Enhanced skill extraction with categorization"""
    skills_found = []
    for skill in ALL_SKILLS:
        if skill in text:
            skills_found.append(skill)

    # Remove duplicates
    unique_skills = list(set(skills_found))
    return len(unique_skills), unique_skills[:10]  # Return count and list of top 10 skills


def extract_certifications(text):
    """Enhanced certification extraction"""
    cert_patterns = [
        r'(\w+(?:\s+\w+)?)\s+certification',
        r'certified\s+(\w+(?:\s+\w+)?)',
        r'(?i)(aws|azure|gcp|pmp|scrum|itil|ccna|ccnp|cisco|oracle|microsoft)\s+cert',
        r'(?i)certificate\s+in\s+(\w+(?:\s+\w+)?)'
    ]

    certifications = set()
    for pattern in cert_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            certifications.add(match.lower())

    return len(certifications), list(certifications)


def extract_projects(text):
    """Extract project information"""
    project_patterns = [
        r'project[s]?\s*:?\s*([^.\n]+)',
        r'developed\s+([^.\n]+)',
        r'built\s+([^.\n]+)',
        r'created\s+([^.\n]+)'
    ]

    projects = []
    for pattern in project_patterns:
        matches = re.findall(pattern, text)
        projects.extend(matches)

    return len(set(projects))  # Unique projects


def calculate_quality_score(text, skills_count, certifications_count, projects_count):
    """Calculate overall resume quality score"""
    score = 0

    # Length score (optimal: 500-2000 words)
    word_count = len(text.split())
    if 500 <= word_count <= 2000:
        score += 20
    elif word_count > 2000:
        score += 15
    elif word_count > 200:
        score += 10

    # Skills score
    if skills_count >= 10:
        score += 30
    elif skills_count >= 5:
        score += 20
    elif skills_count >= 2:
        score += 10

    # Certifications score
    if certifications_count >= 3:
        score += 20
    elif certifications_count >= 1:
        score += 10

    # Projects score
    if projects_count >= 3:
        score += 20
    elif projects_count >= 1:
        score += 10

    # Action verbs score
    action_verbs = ['developed', 'created', 'built', 'managed', 'led', 'designed', 'implemented', 'achieved']
    action_count = sum(text.count(verb) for verb in action_verbs)
    if action_count >= 10:
        score += 10
    elif action_count >= 5:
        score += 5

    return min(score, 100)  # Cap at 100


def build_features(text):
    """Build features with enhanced extraction"""
    # Extract basic features
    experience = extract_experience(text)
    education_level, education_name = extract_education(text)
    skills_count, skills_list = extract_skills(text)
    certifications_count, cert_list = extract_certifications(text)
    projects_count = extract_projects(text)
    quality_score = calculate_quality_score(text, skills_count, certifications_count, projects_count)

    # Additional features for better prediction
    data = {
        "experience": experience,
        "education_level": education_level,
        "skills_count": skills_count,
        "certifications": certifications_count
    }

    # Additional info for display
    extra_info = {
        "education_name": education_name,
        "skills_list": skills_list,
        "certifications_list": cert_list,
        "projects_count": projects_count,
        "quality_score": quality_score
    }

    return [data[col] for col in columns], extra_info


@app.route("/", methods=["GET", "POST"])
def index():
    salary = None
    features_data = None

    if request.method == "POST":
        if 'resume' not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["resume"]

        if file.filename == '':
            return render_template("index.html", error="No file selected")

        if not file.filename.endswith('.pdf'):
            return render_template("index.html", error="Please upload a PDF file")

        try:
            text = extract_text(file)

            if not text:
                return render_template("index.html", error="Could not extract text from PDF")

            features, extra_info = build_features(text)

            # Make prediction
            prediction = model.predict([features])
            salary = round(prediction[0], 2)

            # Get confidence interval (using model's tree predictions)
            predictions = [tree.predict([features])[0] for tree in model.estimators_]
            confidence_low = round(min(predictions), 2)
            confidence_high = round(max(predictions), 2)

            features_data = {
                "experience": features[0],
                "education_level": extra_info["education_name"],
                "education_code": features[1],
                "skills_count": features[2],
                "skills_list": extra_info["skills_list"],
                "certifications": features[3],
                "certifications_list": extra_info["certifications_list"],
                "projects": extra_info["projects_count"],
                "quality_score": extra_info["quality_score"],
                "confidence_low": confidence_low,
                "confidence_high": confidence_high
            }

        except Exception as e:
            print(f"Error processing file: {e}")
            return render_template("index.html", error="Error processing resume. Please try again.")

    return render_template("index.html", salary=salary, features=features_data)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for salary prediction"""
    try:
        if 'resume' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]
        text = extract_text(file)

        if not text:
            return jsonify({"error": "Could not extract text"}), 400

        features, extra_info = build_features(text)
        prediction = model.predict([features])

        return jsonify({
            "predicted_salary": round(prediction[0], 2),
            "features": {
                "experience_years": features[0],
                "education_level": features[1],
                "skills_count": features[2],
                "certifications": features[3]
            },
            "resume_analysis": extra_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)