# AGENTS.md

## Project Overview
- **Project:** German Credit Risk Platform — production-minded ML/MLOps system for German Credit loan default risk assessment.
- **Target user:** credit risk data scientists, ML engineers, MLOps engineers, and banking technical teams
- **My skill level:** intermediate / expert
- **Stack:** Python, scikit-learn, FastAPI, Pydantic, Docker, pytest, Google Cloud Platform deployment guidance

## Commands
- **Install:** `pip install -r requirements.txt`
- **Train:** `python -m credit_risk.train`
- **API:** `uvicorn api.main:app --reload`
- **Test:** `pytest -q`
- **Docker:** `docker build -f deployment/Dockerfile -t german-credit-risk-api .`

## Do
- Read existing code before modifying anything
- Match existing patterns, naming, and style
- Handle errors gracefully — no silent failures
- Keep changes small and scoped to what was asked
- Run tests after changes to verify nothing broke
- Ask clarifying questions before guessing

## Don't
- Install new dependencies without asking
- Delete or overwrite files without confirming
- Hardcode secrets, API keys, or credentials
- Rewrite working code unless explicitly asked
- Push, deploy, or force-push without permission
- Make changes outside the scope of the request

## When Stuck
- If a task is large, break it into steps and confirm the plan first
- If you can't fix an error in 2 attempts, stop and explain the issue

## Testing
- Run existing tests after any change
- Add at least one test for new features
- Never skip or delete tests to make things pass

## Git
- Small, focused commits with descriptive messages
- Never force push

## Response Style
- always respond with clear & concise messages
- use plain English when explaining to the User
- avoid long sentences, complex words, or long paragraphs
