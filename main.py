from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from typing import List, Optional
import spacy
import pytesseract
from pdf2image import convert_from_bytes
import docx2txt
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
        # Add your Vercel deployment URL here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Define request and response models
class AnalyzeRequest(BaseModel):
    resume_text: str
    job_description: str


class AnalyzeResponse(BaseModel):
    keywords: List[str]
    matchedKeywords: List[str]
    score: float
    suggestions: List[str]
    keySkillsAnalysis: str
    improvementAreas: str


# Technical skill lemmas for software engineering
skill_lemmas = {
    # Programming & Development
    "develop",
    "program",
    "code",
    "implement",
    "engineer",
    "architect",
    "deploy",
    "maintain",
    "debug",
    "compile",
    "build",
    "refactor",
    "script",
    # Web Technologies
    "design",
    "integrate",
    "configure",
    "render",
    "style",
    "animate",
    # Database
    "query",
    "model",
    "migrate",
    "normalize",
    "index",
    "cache",
    # Testing
    "test",
    "validate",
    "verify",
    "debug",
    "profile",
    "benchmark",
    # DevOps
    "deploy",
    "containerize",
    "orchestrate",
    "automate",
    "monitor",
    "scale",
    "optimize",
    # System Design
    "architect",
    "design",
    "structure",
    "model",
    "scale",
    "optimize",
    # Agile/Project Management
    "collaborate",
    "manage",
    "lead",
    "coordinate",
    "review",
    "mentor",
    "plan",
    # Problem Solving
    "solve",
    "troubleshoot",
    "improve",
    "optimize",
    "enhance",
    "streamline",
    # API & Integration
    "integrate",
    "connect",
    "sync",
    "interface",
    "transform",
    "process",
    # Security
    "secure",
    "encrypt",
    "authenticate",
    "authorize",
    "protect",
    "validate",
}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        images = convert_from_bytes(file_bytes)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        return docx2txt.process(io.BytesIO(file_bytes))
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing DOCX: {str(e)}")


def extract_keywords(text: str) -> List[str]:
    doc = nlp(text.lower())
    keywords = set()

    # Common technical terms
    technical_terms = {
        # Programming Languages
        "python",
        "javascript",
        "java",
        "typescript",
        "cpp",
        "csharp",
        "golang",
        "rust",
        "php",
        "ruby",
        "swift",
        "kotlin",
        "scala",
        # Frontend
        "react",
        "angular",
        "vue",
        "redux",
        "jquery",
        "css",
        "html",
        "sass",
        "webpack",
        "babel",
        "tailwind",
        "bootstrap",
        # Backend
        "node",
        "express",
        "django",
        "flask",
        "spring",
        "fastapi",
        "rails",
        # Database
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "redis",
        "elasticsearch",
        "oracle",
        "nosql",
        # Cloud & DevOps
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "jenkins",
        "gitlab",
        "terraform",
        "ansible",
        "prometheus",
        "grafana",
        # Testing
        "jest",
        "pytest",
        "selenium",
        "cypress",
        "junit",
        "mocha",
        # Methodologies & Concepts
        "agile",
        "scrum",
        "ci/cd",
        "tdd",
        "rest",
        "graphql",
        "microservices",
        "mvc",
        "orm",
        "api",
        # Version Control
        "git",
        "github",
        "gitlab",
        "bitbucket",
        # Architecture & Design Patterns
        "solid",
        "dry",
        "mvc",
        "mvvm",
        "clean code",
        "design patterns",
    }

    for token in doc:
        # Add technical terms
        if token.text in technical_terms:
            keywords.add(token.text)

        # Add nouns and proper nouns that are longer than 2 characters
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
            if not token.is_stop:  # Skip stop words
                keywords.add(token.text)

        # Add words related to software engineering skills
        if token.lemma_ in skill_lemmas:
            keywords.add(token.text)

        # Add compound terms
        if token.dep_ == "compound":
            compound_term = token.text + " " + token.head.text
            if compound_term in technical_terms:
                keywords.add(compound_term)

    return list(keywords)


async def get_gemini_analysis(
    job_desc: str,
    resume_text: str,
    matched_keywords: List[str],
    missing_keywords: List[str],
) -> tuple:
    try:
        prompt = f"""
        Analyze this job description and resume match. Provide analysis in the following format:

        Key Skills Analysis: [Brief analysis of the match between resume and job requirements]

        Improvement Suggestions:
        - [Suggestion 1]
        - [Suggestion 2]
        - [Suggestion 3]

        Additional Keywords: [Comma-separated list of relevant keywords]

        Resume:
        {resume_text}

        Job Description:
        {job_desc}

        Context:
        - Matched Keywords: {', '.join(matched_keywords)}
        - Missing Keywords: {', '.join(missing_keywords)}

        Please keep your response structured exactly as shown above.
        """

        response = await model.generate_content_async(prompt)
        content = response.text

        # Parse the response
        try:
            sections = content.split("\n\n")

            key_skills_section = [s for s in sections if "Key Skills Analysis:" in s][0]
            key_skills_analysis = key_skills_section.replace(
                "Key Skills Analysis:", ""
            ).strip()

            suggestions_section = [
                s for s in sections if "Improvement Suggestions:" in s
            ][0]
            suggestions = [
                s.strip("- ").strip()
                for s in suggestions_section.split("\n")[1:]
                if s.strip("- ").strip()
            ]

            keywords_section = [s for s in sections if "Additional Keywords:" in s][0]
            additional_keywords = [
                k.strip()
                for k in keywords_section.replace("Additional Keywords:", "").split(",")
                if k.strip()
            ]

        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            key_skills_analysis = "Analysis not available"
            suggestions = ["No specific suggestions available"]
            additional_keywords = []

        return key_skills_analysis, suggestions, additional_keywords

    except Exception as e:
        logger.error(f"Error getting Gemini analysis: {str(e)}")
        return (
            "Unable to generate analysis at this time",
            ["Try highlighting your relevant skills and experience"],
            [],
        )


@app.post("/upload/resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = ""

        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(content)
        elif file.filename.endswith(".docx"):
            text = extract_text_from_docx(content)
        elif file.filename.endswith(".txt"):
            text = content.decode("utf-8")
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload a PDF, DOCX, or TXT file.",
            )

        return {"text": text}

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        # Extract initial keywords from job description
        job_keywords = extract_keywords(request.job_description)

        # Find matched keywords in resume
        resume_text_lower = request.resume_text.lower()
        matched_keywords = [
            keyword for keyword in job_keywords if keyword in resume_text_lower
        ]

        missing_keywords = [k for k in job_keywords if k not in matched_keywords]

        # Get Gemini analysis
        key_skills_analysis, suggestions, additional_keywords = (
            await get_gemini_analysis(
                request.job_description,
                request.resume_text,
                matched_keywords,
                missing_keywords,
            )
        )

        # Calculate match score
        total_keywords = len(job_keywords)
        if total_keywords > 0:
            score = (len(matched_keywords) / total_keywords) * 100
        else:
            score = 0

        return AnalyzeResponse(
            keywords=list(set(job_keywords + additional_keywords)),
            matchedKeywords=matched_keywords,
            score=round(score, 2),
            suggestions=suggestions,
            keySkillsAnalysis=key_skills_analysis,
            improvementAreas="\n".join(suggestions),
        )

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing resume: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
