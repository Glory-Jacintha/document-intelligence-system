# Document Intelligence System

A Streamlit + FastAPI-backed document intelligence app for uploading PDFs, indexing their content with ChromaDB, and asking Gemini-powered questions over the uploaded documents.

## Local Streamlit Run

```powershell
cd "D:\Glory\education\AI internship-code++\document-intelligence-system"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Streamlit Community Cloud Deployment

1. Push this project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Create a new app from your GitHub repository.
4. Set the entrypoint file to:

```text
streamlit_app.py
```

5. In Advanced settings, add this secret:

```toml
GEMINI_API_KEY = "your-api-key-here"
```

6. Select a Python version compatible with the project, preferably Python 3.13 if available.
7. Deploy the app.

## Notes

- Do not commit `.env` or `.streamlit/secrets.toml`.
- Uploaded files, chunks, and ChromaDB data are runtime data.
- `requirements.txt` must stay in the repository root so Streamlit Community Cloud installs dependencies.
