# USF Campus Concierge: RAG-Powered Assistant
The USF (University of South Florida) Campus Concierge is a Retrieval-Augmented Generation (RAG) assistant designed to provide students and staff with accurate, verifiable, and consistent answers to common questions about Admissions, Orientation, and Registrar processes.

It was created to solve the problem of information fragmentation across multiple USF web pages and frequent policy changes, which often leads to confusion, duplicate work, and inconsistent replies.

## Key Features
**Grounded & Transparent Answers:** Every response is traceable to an official USF document and includes inline citations and a compact list of sources with titles, categories, and verified URLs.

**Precision Retrieval:** Uses a cross encoder reranker (MS MARCO MiniLM L 6 v2) to select the most relevant text chunks, ensuring focused and accurate answers.

**Privacy-First Deployment:** The system runs locally for privacy and control, using Ollama for the LLM and a Chroma vector database. The chatbot could be converted to paid, cloud based as needed.

**Robust Guardrails:** Implements prompt injection detection, sanitizes inputs, and stores the system prompt on the server to prevent user edits and maintain consistency.

**Optimized Data Ingestion:** Documents are converted to Markdown format with heading-aware chunking for the cleanest structure and highest retrieval quality.

## System Architecture
The core of the system is built on a robust RAG pipeline:

**1. Data Ingestion**
Sources: Official USF websites for Admissions, Orientation, and Registrar.

Format: Documents are converted to Markdown.

Chunking: Heading-aware chunking is used, splitting documents into sections of about 900 characters with an overlap of roughly 150 characters to preserve meaning.

Embedding Model: Google EmbeddingGemma 300m via SentenceTransformers is used for generating sentence-level embeddings.

Vector Database: Chroma is used for storage due to its lightweight, fast, and local persistence capabilities.

**2. Retrieval Pipeline**
Guardrail: User input is sanitized and checked for prompt injections.

Initial Retrieval: Searches the Chroma vector database to retrieve the twenty most relevant text chunks.

Reranking: A cross encoder reranker (MS MARCO MiniLM L 6 v2) scores chunk similarity to the query, keeping only the top five for increased precision.

Generation: The final context is sent to the Ollama Phi 3 Mini model, which generates the answer.

**3. Output**
Interface: Built with Streamlit for the chat interface and user sessions.

Transparency: The response includes inline citations (e.g., [Source 1]) and a Sources list that displays document titles, categories, and verified URLs.

## Getting Started

### 1. Clone the repository
git clone [repository-url]
cd USF-Campus-Concierge

### 2. Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

### 3. Install dependencies, check requirements.txt

### 4. Set up Ollama (ensure it's running and the Phi 3 Mini model is pulled)
ollama pull phi3:mini

### 5. Ingest data (Please contact us for full data. Have it in 'data/raw' directory)
python ingest_data.py

### 6. Run the Streamlit application
streamlit run app.py

**Security and Responsible AI**
Security was a core consideration from the start. Implemented guardrails include:

Input Sanitization and Injection Detection: All inputs are checked for prompt injections, and the system returns a polite refusal for detected attempts.

**System Prompt Hardening:** The system prompt is stored on the server to prevent runtime edits and maintain consistent behavior.

**Privacy Guardrail:** The assistant refuses to disclose unauthorized student information, referencing regulations like FERPA.

**Source Verification:** The model is instructed to only use links verified as official USF pages and place them exclusively in the final Sources list, never in the body of the text.

## Future Work
Planned improvements for future versions include:

**Automatic Document Freshness Checks:** To detect and prioritize the latest university policies.

**Answerability Gate:** To prevent the model from responding when the confidence score is too low.

**Category Routing:** A light step to direct questions to the correct area (Registrar, Admissions, Orientation) before searching the vector database.

**Dataset Balancing:** Adding more data from Registrar and Billing to reduce orientation bias.

## AI Tools Disclosure (REQUIRED)
We used ChatGPT to assist with:
Data ingestion: drafting and refining the heading-aware Markdown splitter (chunk size/overlap, glue of short sections) and adding clean metadata fields (category, filename, canonical).
Core pipeline code: helping put together rag.py (retrieval, cross-encoder reranking: retrieve-20 → rerank → top-5, numbered context, Sources block) and wiring it cleanly into app.py.

### Demonstration Video on Youtube:
**https://www.youtube.com/watch?v=lGyfctCOQDs**
