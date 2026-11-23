"""
PS 38 - Website Q&A Bot
Loads pages of a website, builds a vector index (RAG) and answers questions
from that content, showing the source URL for every answer.
"""
import os
import re

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (WebsiteQABot student project)")
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
TOP_K = 4

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a website.
Use ONLY the context below to answer. If the answer is not in the context,
reply exactly: "I couldn't find this on the website."
Keep the answer clear and short.

Previous conversation:
{history}

Context:
{context}

Question: {question}

Answer:"""

DEFAULT_URLS = """https://en.wikipedia.org/wiki/Nashik
https://en.wikipedia.org/wiki/Maharashtra
https://en.wikipedia.org/wiki/Godavari_River"""


# ---------- helpers ----------
def parse_urls(text: str):
    urls = [u.strip() for u in re.split(r"[\n,\s]+", text) if u.strip()]
    valid = [u for u in urls if u.startswith(("http://", "https://"))]
    seen, unique = set(), []
    for u in valid:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique[:10]  # project limit: 5-10 pages


def clean_text(t: str) -> str:
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", t)).strip()


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    return ChatGroq(model=CHAT_MODEL, api_key=api_key, temperature=0.2)


@st.cache_resource(show_spinner=False)
def build_cached_index(urls_tuple):
    return _build_index(list(urls_tuple))


def _build_index(urls):
    """Load pages, split them, and build a local keyword index."""
    docs, failed = [], []
    for url in urls:
        try:
            loaded = WebBaseLoader(url).load()
        except Exception:
            loaded = []

        if not loaded:
            try:
                response = requests.get(
                    url,
                    headers={"User-Agent": os.environ["USER_AGENT"]},
                    timeout=20,
                )
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                content = clean_text(soup.get_text(" "))
                if len(content) > 200:
                    loaded = [Document(page_content=content, metadata={"source": url})]
            except Exception:
                loaded = []

        if loaded:
            accepted = False
            for document in loaded:
                document.page_content = clean_text(document.page_content)
                document.metadata["source"] = url
                if len(document.page_content) > 200:
                    docs.append(document)
                    accepted = True
            if not accepted:
                failed.append(url)
        else:
            failed.append(url)

    if not docs:
        return None, 0, 0, failed

    splitter = RecursiveCharacterTextSplitter(chunk_size=2200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    retriever = BM25Retriever.from_documents(
        chunks, k=TOP_K, preprocess_func=tokenize_text
    )
    return retriever, len(docs), len(chunks), failed


def format_history(messages, max_turns=4):
    recent = messages[-max_turns * 2:]
    if not recent:
        return "(none)"
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)


def answer_question(retriever, question, messages, api_key):
    hits = retriever.invoke(question)
    context = "\n\n---\n\n".join(f"[Source: {h.metadata['source']}]\n{h.page_content}" for h in hits)
    prompt = PROMPT_TEMPLATE.format(
        history=format_history(messages), context=context, question=question
    )
    response = get_llm(api_key).invoke(prompt)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    sources = list(dict.fromkeys(h.metadata["source"] for h in hits))  # unique, ordered
    return answer.strip(), sources


# ---------- UI ----------
st.set_page_config(page_title="Website Q&A", page_icon="📖")
st.title("Website Q&A", icon=":material/menu_book:")
st.caption("A simple way to explore the information in your chosen web pages.")

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = None

with st.sidebar:
    st.title("Research setup", icon=":material/tune:")
    st.caption("Choose the pages this conversation should draw from.")

    with st.expander("Groq API key", expanded=not st.session_state.groq_api_key):
        api_key = st.text_input(
            "API key",
            value=st.session_state.groq_api_key,
            type="password",
            key="sidebar_groq_api_key",
            help="Your key is used to generate answers with Groq.",
        )
        st.session_state.groq_api_key = api_key
        if not st.session_state.groq_api_key:
            st.caption("Add a key to enable questions.")

    st.subheader("Website pages", icon=":material/language:")
    url_text = st.text_area(
        "Page URLs",
        value=DEFAULT_URLS,
        height=180,
        help="Add up to 10 page URLs, one per line.",
    )
    if st.button(
        "Load pages",
        type="primary",
        icon=":material/download:",
        key="load_website",
        width="stretch",
    ):
        urls = parse_urls(url_text)
        if not urls:
            st.warning("Please enter at least one valid URL starting with http(s)://")
        else:
            with st.spinner(f"Loading and indexing {len(urls)} page(s)..."):
                try:
                    store, n_pages, n_chunks, failed = build_cached_index(tuple(urls))
                except Exception as e:
                    store, n_pages, n_chunks, failed = None, 0, 0, []
                    st.error(f"Indexing failed: {e}")
            if store:
                st.session_state.store = store
                st.session_state.messages = []
                st.success(f"Indexed {n_pages} page(s) into {n_chunks} chunks.")
            if failed:
                st.warning("Could not load / empty content:\n" + "\n".join(failed))
    if st.button(
        "Clear conversation",
        icon=":material/delete_sweep:",
        width="stretch",
    ):
        st.session_state.messages = []
        st.rerun()

if st.session_state.store is None:
    with st.container(border=True):
        st.subheader("Start with a source", icon=":material/menu_book:")
        st.write(
            "Load a few website pages from the sidebar, then ask a question "
            "about what they contain."
        )
        st.caption('For example: “What is Nashik known for?”')

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.markdown("**Sources:**\n" + "\n".join(f"- {s}" for s in m["sources"]))

question = st.chat_input(
    "Ask about the pages you loaded...",
    disabled=st.session_state.store is None or not st.session_state.groq_api_key,
)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the website..."):
            try:
                answer, sources = answer_question(
                    st.session_state.store,
                    question,
                    st.session_state.messages,
                    st.session_state.groq_api_key,
                )
            except Exception as e:
                answer, sources = f"Error while contacting the LLM: {e}", []
        st.markdown(answer)
        if sources and "couldn't find" not in answer.lower():
            st.markdown("**Sources:**\n" + "\n".join(f"- {s}" for s in sources))
        else:
            sources = []
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
