from flask import Flask, render_template, request, send_file, jsonify
from openai import OpenAI, OpenAIError, RateLimitError
import os
import re
import traceback
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

openai = OpenAI(api_key=api_key)

app = Flask(__name__)
app.config["DEBUG"] = True

# === Helper Functions ===

def clean_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    return text.strip()

def generate_pdf(text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x_margin = 50
    y_position = height - 50
    line_height = 15
    p.setFont("Helvetica", 12)

    for line in text.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        if y_position < 50:
            p.showPage()
            p.setFont("Helvetica", 12)
            y_position = height - 50
        p.drawString(x_margin, y_position, line)
        y_position -= line_height

    p.save()
    buffer.seek(0)
    return buffer

def generate_suggestions_with_fallback(prompt):
    models = ["gpt-4o-mini", "gpt-3.5-turbo"]
    for model in models:
        try:
            print(f"Trying model: {model}")
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You provide resume improvement suggestions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip(), model
        except RateLimitError:
            print(f"[Rate Limit] Model {model} is rate-limited, trying next...")
        except OpenAIError as e:
            print(f"[OpenAI Error with {model}]: {e}")
    return None, None

def analyze_with_fallback(messages):
    models = ["gpt-4o-mini", "gpt-3.5-turbo"]
    for model in models:
        try:
            print(f"Analyzing with model: {model}")
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip(), model
        except RateLimitError:
            print(f"[Rate Limit] {model} - trying next.")
        except OpenAIError as e:
            print(f"[OpenAI Error] {e}")
    return None, None

# === Routes ===

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    resume_file = request.files.get('resume')
    job_desc = request.form.get('job_description')
    linkedin_url = request.form.get('linkedin_url', '')

    if not resume_file or not job_desc:
        return jsonify({"error": "Missing resume or job description"}), 400

    reader = PdfReader(resume_file)
    resume_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    prompt = f"""
Analyze the following resume, LinkedIn profile, and job description.

Resume:
{resume_text}

LinkedIn:
{linkedin_url}

Job Description:
{job_desc}

Please provide the following:
1. A detailed analysis of the resume.
2. Percentage match for each category: experience, skills, education, and certifications.
3. An overall confidence score (0-100%) indicating how well the candidate fits the position.

Return the response structured clearly with these headings:
Analysis:
Match Percentage:
Confidence Score:
"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes resumes against job descriptions."},
        {"role": "user", "content": prompt}
    ]

    try:
        result_text, model_used = analyze_with_fallback(messages)

        if not result_text:
            return jsonify({"error": "All models are currently rate-limited. Please try again shortly."}), 429

        def extract_section(text, heading):
            pattern = rf"{heading}:(.*?)(?=\n[A-Z][a-z]+:|\Z)"
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else ""

        analysis = clean_markdown(extract_section(result_text, "Analysis"))
        match_percentage = clean_markdown(extract_section(result_text, "Match Percentage"))
        confidence_score = clean_markdown(extract_section(result_text, "Confidence Score"))

        return jsonify({
            "analysis": analysis,
            "match_percentage": match_percentage,
            "confidence_score": confidence_score,
            "resume_text": resume_text,
            "model_used": model_used
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Something went wrong during analysis. Please try again."}), 500

@app.route('/generate', methods=['POST'])
def generate():
    try:
        resume_text = request.form['resume']
        job_description = request.form['job']
        linkedin_url = request.form.get('linkedin_url', '')

        prompt = f"""
You are a helpful career consultant.

Based on the following resume, LinkedIn profile, and job description, provide a concise list of actionable suggestions
to improve the resume so it better matches the job requirements.

Resume:
{resume_text}

Job Description:
{job_description}

LinkedIn profile:
{linkedin_url}

Return ONLY a JSON array of suggestion strings, with no surrounding quotes or markdown formatting. Do not include any extra text or explanation. 
Please also analyse and suggest improvements to the CV, considering it should help getting the recruiter's attention and passing by automatic triage systems (ATS) using keywords based on the profile and an structure which facilitates HR automatized reading systems.
Example:
[
    "Suggestion 1",
    "Suggestion 2",
    "Suggestion 3"
]
"""

        raw_text, model_used = generate_suggestions_with_fallback(prompt)

        if not raw_text:
            return jsonify({"error": "Both models failed or were rate-limited."}), 500

        try:
            suggestions = json.loads(raw_text)
            if not isinstance(suggestions, list):
                raise ValueError("Not a list")
        except Exception:
            suggestions = [line.strip("-•* \n") for line in raw_text.split("\n") if line.strip()]

        return jsonify({"suggestions": suggestions, "model_used": model_used})

    except Exception as e:
        print(f"[ERROR] Suggestions generation failed: {e}")
        return jsonify({"error": "Failed to generate suggestions."}), 500

@app.route("/generate-cover-letter", methods=["POST"])
def generate_cover_letter():
    resume_text = request.form.get("resume", "")
    linkedin_url = request.form.get("linkedin_url", "")
    job_description = request.form.get("job", "")

    if not resume_text:
        return jsonify({"error": "Resume text is required"}), 400

    prompt = f"""
You are a helpful assistant generating a professional, honest cover letter for a job applicant for the below job description.

Use ONLY the following resume information and LinkedIn URL (do NOT invent or assume anything):

Resume:
{resume_text}

LinkedIn URL:
{linkedin_url}

Job Description:
{job_description}

Generate a concise, positive cover letter that highlights relevant skills and experience from the resume. Do not invent any information. The letter should be tailored to the job but keep it truthful.

Cover Letter:
"""

    for model in ["gpt-4o-mini", "gpt-3.5-turbo"]:
        try:
            print(f"Trying model: {model}")
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return jsonify({
                "cover_letter": response.choices[0].message.content.strip(),
                "model_used": model
            })
        except RateLimitError:
            print(f"[Rate Limit] Model {model} rate-limited.")
        except OpenAIError as e:
            print(f"[OpenAI Error] {e}")
            break

    return jsonify({"error": "Cover letter generation failed."}), 500

if __name__ == '__main__':
    app.run(debug=True)