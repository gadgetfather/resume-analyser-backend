# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pytesseract
from pdf2image import convert_from_bytes
import docx2txt
import spacy
from typing import List
import io

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


class AnalyzeRequest(BaseModel):
    resume_text: str
    job_description: str


class AnalyzeResponse(BaseModel):
    keywords: List[str]
    matchedKeywords: List[str]
    score: float


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        # Convert PDF to images
        images = convert_from_bytes(file_bytes)
        text = ""
        for image in images:
            # Extract text from each page
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        return docx2txt.process(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing DOCX: {str(e)}")


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


def extract_keywords(text: str) -> List[str]:
    # Process the text with spaCy
    doc = nlp(text.lower())

    keywords = set()

    # Common technical terms that might not be caught by lemma analysis
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

        # Add compound terms (e.g., "unit testing", "version control")
        if token.dep_ == "compound":
            compound_term = token.text + " " + token.head.text
            if compound_term in technical_terms:
                keywords.add(compound_term)

    return list(keywords)


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
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        # Extract keywords from job description
        job_keywords = extract_keywords(request.job_description)

        # Find matched keywords in resume
        resume_text_lower = request.resume_text.lower()
        matched_keywords = [
            keyword for keyword in job_keywords if keyword in resume_text_lower
        ]

        # Calculate match score
        score = (len(matched_keywords) / len(job_keywords) * 100) if job_keywords else 0

        return AnalyzeResponse(
            keywords=job_keywords,
            matchedKeywords=matched_keywords,
            score=round(score, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
