# PS 38 – Website Q&A Bot

**Problem statement:** Visitors can't find information on large websites. This app loads pages of a website and answers questions from their content, showing the source URL for each answer.

## How it works (RAG)
1. `WebBaseLoader` downloads the pages (5–10 URLs)
2. `RecursiveCharacterTextSplitter` splits text into chunks
3. BM25 builds a local keyword index over the chunks (no embedding API key needed)
4. For each question, the top 4 similar chunks are retrieved
5. Groq (`openai/gpt-oss-20b`) answers using only those chunks
6. Source URLs of the retrieved chunks are displayed under the answer

## Run
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env
# add your Groq API key (https://console.groq.com/keys) to .env
streamlit run app.py
```

## Screenshots
Add 1–2 screenshots here.
# Website-Q-A-Bot
