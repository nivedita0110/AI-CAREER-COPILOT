"""
ATS Scoring and Semantic Analysis Engine
Provides vectorized batch embedding similarity, technical skill extraction,
experience duration extraction, and transparent ATS score decomposition.
"""
from typing import Dict, List, Optional, Set, Tuple
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_WORD_SIZE,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_SKILL_WEIGHT,
    SKILL_PARTIAL_THRESHOLD,
    SKILL_SIMILARITY_THRESHOLD,
)

# Comprehensive Taxonomy of Multi-word Technical Terms
COMMON_TECH_PHRASES = [
    # AI & Machine Learning
    "machine learning", "deep learning", "artificial intelligence", "data science",
    "natural language processing", "computer vision", "large language models", "prompt engineering",
    "retrieval augmented generation", "reinforcement learning", "neural networks", "transformer models",
    "generative ai", "time series analysis", "predictive modeling", "feature engineering",
    # Cloud & DevOps
    "cloud computing", "amazon web services", "google cloud platform", "google cloud",
    "microsoft azure", "docker container", "kubernetes cluster", "ci/cd", "continuous integration",
    "continuous deployment", "infrastructure as code", "terraform", "ansible", "jenkins pipeline",
    "serverless architecture", "linux administration", "bash scripting",
    # Backend & Architecture
    "spring boot", "rest api", "restful api", "graphql api", "microservices architecture",
    "distributed systems", "event driven architecture", "apache kafka", "rabbitmq", "message queue",
    "object oriented programming", "test driven development", "clean architecture",
    # Frontend & Fullstack
    "react native", "next js", "node js", "vue js", "angular js", "front end", "back end",
    "full stack", "tailwind css", "material ui", "responsive design", "web development",
    # Databases & Big Data
    "relational database", "nosql database", "postgresql", "mysql", "mongodb", "redis cache",
    "data engineering", "apache spark", "apache flink", "etl pipeline", "data warehousing",
    "snowflake", "bigquery", "data modeling", "business intelligence", "power bi", "tableau",
    # Methodologies & Security
    "agile methodology", "scrum master", "version control", "git workflow", "code review",
    "unit testing", "integration testing", "penetration testing", "cyber security",
    "data structures", "algorithms and data structures", "system design"
]

TWO_LETTER_SKILLS = {
    "c", "r", "go", "ai", "ml", "qa", "ci", "cd", "ui", "ux", "js", "ts", "os", "db", "ip", "it", "c++", "c#"
}

EXPANDED_STOPWORDS = {
    "and", "or", "the", "to", "of", "in", "for", "with", "a", "an", "is", "are", "on", "at",
    "by", "this", "that", "from", "as", "be", "was", "were", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "have", "has", "had", "do", "does", "did", "not",
    "but", "if", "so", "than", "then", "into", "over", "after", "before", "about", "above",
    "candidate", "candidates", "applicant", "applicants", "selected", "selection", "requirement",
    "requirements", "responsibility", "responsibilities", "qualification", "qualifications",
    "experience", "experiences", "years", "year", "work", "working", "worker", "job", "role",
    "position", "opportunity", "opportunities", "team", "teams", "company", "companies", "client",
    "clients", "business", "businesses", "looking", "seeking", "join", "joining", "help", "helping",
    "strong", "demonstrated", "proven", "excellent", "good", "great", "well", "ability", "abilities",
    "skilled", "skills", "skill", "proficient", "proficiency", "knowledge", "understanding",
    "familiar", "familiarity", "plus", "preferred", "bonus", "written", "verbal", "communication",
    "interpersonal", "collaborative", "collaboration", "independent", "independently", "fast",
    "paced", "environment", "support", "supporting", "manage", "managing", "management", "lead",
    "leading", "leadership", "deliver", "delivering", "delivery", "develop", "developing",
    "development", "create", "creating", "creation", "build", "building", "implement", "implementing",
    "implementation", "maintain", "maintaining", "maintenance", "ensure", "ensuring", "provide",
    "providing", "use", "using", "utilize", "utilizing", "various", "multiple", "including",
    "include", "includes", "such", "other", "etc", "new", "daily", "day", "week", "month",
    "hands", "mindset", "track", "record", "focus", "focused", "passion", "passionate", "high",
    "level", "quality", "solutions", "impact", "cross", "functional", "driven", "detail", "oriented"
}


def extract_skills(text: str) -> Set[str]:
    """
    Extracts individual and multi-word technology terms and skills from text,
    ignoring conversational and boilerplate filler words.
    """
    if not text:
        return set()

    text_lower = text.lower()
    found_skills = set()

    # 1. Match multi-word technical skills
    for phrase in COMMON_TECH_PHRASES:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.add(phrase)

    # 2. Extract technical tokens (including C++, C#, .NET)
    # Check for C++ and C# explicitly
    if re.search(r"\bc\+\+\b", text_lower):
        found_skills.add("c++")
    if re.search(r"\bc\#\b", text_lower):
        found_skills.add("c#")
    if re.search(r"\b\.net\b", text_lower):
        found_skills.add(".net")

    # Regular single tokens
    tokens = re.findall(r"\b[a-z][a-z0-9\+#\.\-]*[a-z0-9\+#]\b|\b[a-z]\b", text_lower)
    for token in tokens:
        clean_token = token.strip(". ")
        if not clean_token or clean_token.isdigit():
            continue
        if clean_token in TWO_LETTER_SKILLS:
            found_skills.add(clean_token)
        elif len(clean_token) >= 3 and clean_token not in EXPANDED_STOPWORDS:
            # Filter pure numeric values
            if not clean_token.replace(".", "").isdigit():
                found_skills.add(clean_token)

    return found_skills


def extract_experience_years(text: str) -> List[str]:
    """
    Detects years of experience mentions (e.g. '5+ years', '3-5 years', '2 years of experience').
    """
    if not text:
        return []
    pattern = r"(\b\d{1,2}(?:\+|-|\s*to\s*\d{1,2})?\s*(?:years?|yrs?)(?:\s+of\s+experience)?\b)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    # Normalize and deduplicate
    clean_matches = list(dict.fromkeys([m.strip() for m in matches if m.strip()]))
    return clean_matches[:5]


def detect_resume_sections(text: str) -> Dict[str, bool]:
    """
    Checks for the presence of standard ATS resume sections.
    """
    text_lower = text.lower()
    return {
        "Contact Information": bool(re.search(r"(@|phone|email|linkedin|github|\+?\d{10})", text_lower)),
        "Work Experience": bool(re.search(r"(experience|employment|work history|career history)", text_lower)),
        "Education": bool(re.search(r"(education|academic|university|degree|bachelor|master|b\.tech|m\.tech|phd)", text_lower)),
        "Skills Section": bool(re.search(r"(skills|technical skills|competencies|technologies|tools)", text_lower)),
        "Projects": bool(re.search(r"(projects|portfolio|personal projects|key achievements)", text_lower))
    }


def chunk_text(text: str, chunk_size: int = CHUNK_WORD_SIZE, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    """
    Splits text into overlapping word chunks to overcome the 256-token limit.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def get_document_embedding(embedder: SentenceTransformer, text: str) -> np.ndarray:
    """
    Computes a document-level embedding by averaging chunk embeddings.
    Prevents token truncation for long resumes or job descriptions.
    """
    if not text.strip():
        return np.zeros(384)

    chunks = chunk_text(text)
    if not chunks:
        return np.zeros(384)

    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    mean_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm
    return mean_embedding


def calculate_match(
    embedder: SentenceTransformer,
    resume: str,
    jd: str,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    skill_weight: float = DEFAULT_SKILL_WEIGHT,
    similarity_threshold: float = SKILL_SIMILARITY_THRESHOLD
) -> Dict:
    """
    High-performance, vectorized ATS matching engine.
    Computes chunk-pooled document semantic similarity and fast matrix skill matching.
    """
    if not resume or not jd:
        return {
            "final_score": 0.0,
            "semantic_score": 0.0,
            "skill_score": 0.0,
            "matched_skills": set(),
            "missing_skills": set(),
            "partial_skills": set(),
            "skill_details": [],
            "resume_experience": [],
            "jd_experience": [],
            "section_audit": {}
        }

    # 1. Document-level Semantic Similarity (Chunk-pooled)
    resume_emb = get_document_embedding(embedder, resume)
    jd_emb = get_document_embedding(embedder, jd)
    semantic_similarity = float(cosine_similarity([resume_emb], [jd_emb])[0][0]) * 100
    semantic_similarity = max(0.0, min(100.0, semantic_similarity))

    # 2. Skill Extraction
    r_skills = extract_skills(resume)
    j_skills = extract_skills(jd)

    matched = set()
    missing = set()
    partial = set()
    skill_details = []

    # 3. Vectorized Semantic Skill Matching
    if j_skills and r_skills:
        j_list = sorted(list(j_skills))
        r_list = sorted(list(r_skills))

        # Batch encode in 2 calls rather than O(N*M) individual forward passes
        j_vectors = embedder.encode(j_list, normalize_embeddings=True)
        r_vectors = embedder.encode(r_list, normalize_embeddings=True)

        # Fast cosine similarity matrix: shape (len(j_list), len(r_list))
        sim_matrix = np.dot(j_vectors, r_vectors.T) * 100

        for idx, skill in enumerate(j_list):
            best_idx = int(np.argmax(sim_matrix[idx]))
            best_sim = float(sim_matrix[idx][best_idx])
            best_match = r_list[best_idx]

            if best_sim >= similarity_threshold:
                matched.add(skill)
                status = "matched"
            elif best_sim >= SKILL_PARTIAL_THRESHOLD:
                partial.add(skill)
                status = "partial"
            else:
                missing.add(skill)
                status = "missing"

            skill_details.append({
                "jd_skill": skill,
                "matched_with": best_match if best_sim >= 35 else "None",
                "similarity": round(best_sim, 1),
                "status": status
            })
    elif j_skills and not r_skills:
        missing = set(j_skills)
        for skill in sorted(list(j_skills)):
            skill_details.append({
                "jd_skill": skill,
                "matched_with": "None",
                "similarity": 0.0,
                "status": "missing"
            })
    else:
        matched = set(j_skills)

    # Sort skill details by priority (missing first, then partial, then matched)
    status_order = {"missing": 0, "partial": 1, "matched": 2}
    skill_details.sort(key=lambda x: (status_order.get(x["status"], 3), -x["similarity"]))

    # 4. Skill Coverage Score (Count partial matches as 0.5)
    matched_weight = len(matched) + (len(partial) * 0.5)
    skill_score = (matched_weight / len(j_skills)) * 100 if j_skills else 0.0

    # 5. Composite Final Score
    final_score = (semantic_weight * semantic_similarity) + (skill_weight * skill_score)
    final_score = round(max(0.0, min(100.0, final_score)), 1)

    # 6. Experience & Section Audit
    resume_exp = extract_experience_years(resume)
    jd_exp = extract_experience_years(jd)
    section_audit = detect_resume_sections(resume)

    return {
        "final_score": final_score,
        "semantic_score": round(semantic_similarity, 1),
        "skill_score": round(skill_score, 1),
        "matched_skills": matched,
        "missing_skills": missing,
        "partial_skills": partial,
        "skill_details": skill_details,
        "resume_experience": resume_exp,
        "jd_experience": jd_exp,
        "section_audit": section_audit
    }
