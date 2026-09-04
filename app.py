import os

import streamlit as st
import psycopg2

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector
from openai import OpenAI


# =========================================================
# LOAD .ENV
# =========================================================

load_dotenv()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RAG PDF Q&A",
    page_icon="📄",
    layout="centered"
)


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunkText(text, size=1000):
    chunks = []

    for i in range(0, len(text), size):
        chunk = text[i:i + size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def loadModel():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = loadModel()


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

def createEmbedding(text):
    return model.encode(text)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def connectDB():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432"),

        # IMPORTANT:
        # Local PostgreSQL should not force SSL
        sslmode=os.getenv("DB_SSLMODE", "prefer")
    )

    register_vector(conn)

    return conn


# =========================================================
# INITIALIZE DATABASE
# =========================================================

def initializeDatabase():
    conn = connectDB()
    cur = conn.cursor()

    try:

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding VECTOR(384)
            );
            """
        )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()


# =========================================================
# STORE PDF CHUNKS
# =========================================================

def replaceDocument(chunks, vectors):
    conn = connectDB()
    cur = conn.cursor()

    try:

        # Remove previous document
        cur.execute(
            """
            TRUNCATE TABLE documents
            RESTART IDENTITY;
            """
        )

        # Store new document
        for chunk, vector in zip(chunks, vectors):

            cur.execute(
                """
                INSERT INTO documents
                (chunk_text, embedding)
                VALUES (%s, %s);
                """,
                (
                    chunk,
                    vector
                )
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()


# =========================================================
# VECTOR SEARCH
# =========================================================

def findRelatedVector(questionVector):
    conn = connectDB()
    cur = conn.cursor()

    try:

        cur.execute(
            """
            SELECT chunk_text
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT 3;
            """,
            (
                questionVector,
            )
        )

        return cur.fetchall()

    finally:
        cur.close()
        conn.close()


# =========================================================
# OPENROUTER CLIENT
# =========================================================

@st.cache_resource
def initModel():

    apiKey = os.getenv("OPENROUTER_API_KEY")

    if not apiKey:
        return None

    client = OpenAI(
        api_key=apiKey,
        base_url="https://openrouter.ai/api/v1"
    )

    return client


Client = initModel()


# =========================================================
# CREATE PROMPT
# =========================================================

def generatePrompt(relatedChunks, question):

    context = "\n\n".join(
        chunk[0] for chunk in relatedChunks
    )

    prompt = f"""
You are a PDF Question Answering Assistant.

Answer the user's question ONLY using the context
retrieved from the uploaded PDF.

If the answer is not available in the context,
reply exactly:

Answer not found!!!

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


# =========================================================
# SESSION STATE
# =========================================================

if "processed" not in st.session_state:
    st.session_state.processed = False

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# =========================================================
# DATABASE CHECK
# =========================================================

try:
    initializeDatabase()

except Exception as e:
    st.error("Database connection failed!")
    st.error(str(e))
    st.stop()


# =========================================================
# UI
# =========================================================

st.title("📄 Retrieval Augmented Generation")

st.caption(
    "Upload a PDF and ask questions using "
    "RAG + PostgreSQL + pgvector + OpenRouter"
)


# =========================================================
# PDF UPLOAD
# =========================================================

uploadedFile = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if st.button(
    "Process PDF",
    type="primary"
):

    if uploadedFile is None:

        st.warning("Please upload a PDF file.")

    else:

        try:

            with st.spinner("Reading PDF..."):

                reader = PdfReader(uploadedFile)

                text = ""

                for page in reader.pages:

                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"


            # Check PDF text
            if not text.strip():

                st.error(
                    "No readable text found in the PDF."
                )

                st.stop()


            # Create chunks
            chunks = chunkText(text)


            with st.spinner("Creating embeddings..."):

                vectors = createEmbedding(chunks)


            # Store in PostgreSQL
            with st.spinner(
                "Saving data to PostgreSQL..."
            ):

                replaceDocument(
                    chunks,
                    vectors
                )


            st.session_state.processed = True

            st.session_state.file_name = uploadedFile.name


            st.success(
                "✅ Document scanned successfully!"
            )

            st.info(
                f"{len(chunks)} chunks stored in PostgreSQL."
            )


        except Exception as e:

            st.error(
                "Error while processing PDF."
            )

            st.error(
                str(e)
            )


# =========================================================
# QUESTION ANSWERING
# =========================================================

if st.session_state.processed:

    st.divider()

    st.success(
        f"📄 Current PDF: {st.session_state.file_name}"
    )


    question = st.text_input(
        "Ask a question about the PDF"
    )


    if st.button(
        "Ask AI",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif Client is None:

            st.error(
                "OPENROUTER_API_KEY not found in .env file."
            )

        else:

            try:

                # Create question embedding
                questionVector = createEmbedding(
                    question
                )


                # Search similar PDF chunks
                relatedChunks = findRelatedVector(
                    questionVector
                )


                if not relatedChunks:

                    st.warning(
                        "No related information found."
                    )

                    st.stop()


                # Generate RAG prompt
                prompt = generatePrompt(
                    relatedChunks,
                    question
                )


                # Ask OpenRouter
                with st.spinner(
                    "AI is generating answer..."
                ):

                    response = Client.chat.completions.create(

                        model="openrouter/free",

                        messages=[
                            {
                                "role": "system",
                                "content":
                                "Answer questions only from the supplied PDF context."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        temperature=0.2
                    )


                answer = response.choices[0].message.content


                # Display Answer
                st.subheader(
                    "🤖 AI Response"
                )

                st.success(
                    answer
                )


            except Exception as e:

                st.error(
                    "Error while generating AI response."
                )

                st.error(
                    str(e)
                )