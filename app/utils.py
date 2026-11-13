"""
Helper Utilities
Provides sample data presets for instant demoing, score categorization,
and export utilities.
"""
from typing import Dict


SAMPLE_JOB_DESCRIPTIONS = {
    "Senior Full-Stack Engineer (Python / React)": """
About the Role:
We are seeking an experienced Full-Stack Engineer to lead the design and development of scalable cloud applications.

Key Responsibilities:
- Design, build, and maintain efficient, reusable, and reliable Python (FastAPI / Django) backend services.
- Develop dynamic, high-performance web user interfaces using React, Next.js, and TypeScript.
- Architect and manage relational databases (PostgreSQL) and caching systems (Redis).
- Design and deploy containerized microservices on AWS using Docker and Kubernetes.
- Drive CI/CD pipeline automation and maintain test coverage with unit testing (pytest) and integration testing.
- Collaborate in an agile methodology / scrum framework with cross-functional teams.

Qualifications:
- 4+ years of professional full stack software development experience.
- Strong proficiency in Python, JavaScript, and TypeScript.
- Deep experience with modern web frameworks (React, FastAPI, Node.js).
- Proven track record with cloud computing (AWS or GCP), Docker, and RESTful API architecture.
- Solid understanding of data structures, algorithms, and system design principles.
- Experience with Git version control and automated CI/CD workflows.
""",
    "AI / Machine Learning Engineer": """
About the Role:
We are looking for a Machine Learning Engineer to deploy state-of-the-art NLP and Large Language Model applications.

Key Responsibilities:
- Build and fine-tune machine learning and deep learning models using PyTorch and Hugging Face.
- Develop Retrieval-Augmented Generation (RAG) systems and LLM agent pipelines using LangChain / LlamaIndex.
- Build vector search and semantic retrieval architectures using vector databases.
- Develop high-throughput REST APIs and microservices using Python and Docker.
- Implement data engineering pipelines, feature extraction, and model monitoring in production.
- Work with cloud infrastructure on Google Cloud Platform or AWS.

Qualifications:
- 3+ years of experience in AI, machine learning, and data science.
- Strong background in Python, PyTorch, Scikit-learn, and NumPy.
- Practical experience with Natural Language Processing (NLP) and transformer models.
- Familiarity with MLOps, CI/CD, Git, and Linux environments.
- Bachelor's or Master's degree in Computer Science, Data Science, or related field.
""",
    "DevOps / Cloud Infrastructure Engineer": """
About the Role:
Join our platform infrastructure team to architect high-availability cloud platforms and automated deployment pipelines.

Key Responsibilities:
- Architect and automate infrastructure as code using Terraform and Ansible on AWS.
- Manage production Kubernetes clusters (EKS) and Docker container workloads.
- Design resilient CI/CD pipelines using GitHub Actions and Jenkins.
- Monitor system health, performance, and security with Prometheus, Grafana, and Datadog.
- Implement security best practices, access controls, and network security.

Qualifications:
- 3+ years of hands-on experience in DevOps, Cloud Engineering, or SRE roles.
- Strong proficiency with Linux administration, Bash scripting, and Python.
- Deep expertise in AWS (VPC, EC2, S3, IAM, CloudWatch) and Kubernetes.
- Hands-on experience with Terraform, Docker, and Git workflows.
"""
}

SAMPLE_RESUMES = {
    "Alex Mercer - Full-Stack Developer": """
Alex Mercer
Email: alex.mercer@example.com | Phone: +1-555-0199 | Location: San Francisco, CA
LinkedIn: linkedin.com/in/alexmercer | GitHub: github.com/alexmercer

PROFESSIONAL SUMMARY
Dynamic Software Engineer with 4 years of experience building scalable web applications and microservices. Proven expertise in Python, React, and AWS cloud environments. Passionate about clean architecture, performance optimization, and agile team delivery.

CORE TECHNICAL SKILLS
- Programming Languages: Python, JavaScript, TypeScript, SQL, HTML5, CSS3
- Frameworks & Libraries: FastAPI, Django, React, Next.js, Redux, Node.js
- Cloud & DevOps: AWS (EC2, S3, Lambda), Docker, GitHub Actions, CI/CD, Linux
- Databases & Storage: PostgreSQL, MySQL, Redis, MongoDB
- Engineering Practices: REST API Design, Microservices, Unit Testing (Pytest, Jest), Git, Agile/Scrum

PROFESSIONAL EXPERIENCE
Software Engineer | NexaTech Solutions | 2022 - Present
- Architected and delivered 12+ RESTful microservices using Python and FastAPI, handling 1.5M daily requests with 99.9% uptime.
- Redesigned core client-facing dashboard using React, TypeScript, and Tailwind CSS, reducing page load latency by 42%.
- Containerized development and production services using Docker and automated CI/CD deployment pipelines via GitHub Actions.
- Optimized slow PostgreSQL queries and introduced Redis caching layer, decreasing p95 database response times from 340ms to 48ms.
- Participated in bi-weekly scrum sprints, conducted peer code reviews, and mentored 2 junior developers.

Associate Full-Stack Developer | CloudSphere Inc. | 2020 - 2022
- Developed responsive web interfaces using React and JavaScript, integrating RESTful backend endpoints.
- Maintained Django-based backend services, wrote comprehensive unit test suites achieving 88% test coverage.
- Assisted in migrating legacy monolithic application components into containerized AWS microservices.

PROJECTS
- DevPulse - Developer Metrics Dashboard: Built an open-source analytics platform with Next.js, FastAPI, and PostgreSQL tracking GitHub repository velocity.
- CloudCart - Distributed E-Commerce Microservices: Implemented event-driven ordering system utilizing Python, Docker, Redis, and Stripe API.

EDUCATION
Bachelor of Science in Computer Science | State University | Graduated 2020
"""
}


def get_score_category(score: float) -> Dict[str, str]:
    """
    Returns visual badge categorization and color coding for ATS scores.
    """
    if score >= 80.0:
        return {
            "label": "Strong Match",
            "color": "#10B981", # Emerald
            "badge": "badge-success",
            "message": "Excellent alignment with role requirements. Strong probability of passing ATS filters."
        }
    elif score >= 65.0:
        return {
            "label": "Moderate Match",
            "color": "#3B82F6", # Blue
            "badge": "badge-info",
            "message": "Good foundation. Addressing key missing skills and terminology will significantly elevate ranking."
        }
    elif score >= 50.0:
        return {
            "label": "Borderline Match",
            "color": "#F59E0B", # Amber
            "badge": "badge-warning",
            "message": "Partial alignment. Requires targeted resume tailoring and emphasis on transferable technical skills."
        }
    else:
        return {
            "label": "Low Match",
            "color": "#EF4444", # Red
            "badge": "badge-danger",
            "message": "Significant qualification gap. Needs extensive restructuring or acquisition of prerequisite core technologies."
        }

