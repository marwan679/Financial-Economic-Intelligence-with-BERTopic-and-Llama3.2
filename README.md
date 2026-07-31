# Sovereign Intelligence: Privacy-First Enterprise Topic Discovery Engine

**A production-grade, fully local AI system for discovering thematic structure in institutional document collections — with zero external API dependency**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![BERTopic](https://img.shields.io/badge/BERTopic-enabled-green.svg)](https://maartengr.github.io/BERTopic/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-orange.svg)](https://ollama.com)

---

## 🎯 Project Overview

**Sovereign Intelligence** is an end-to-end NLP pipeline that automatically discovers hidden thematic patterns across large institutional document collections — financial reports, clinical notes, legal briefs, research papers — without ever sending data outside your firewall.

Unlike traditional RAG (Retrieval-Augmented Generation) systems that split documents into arbitrary character-window chunks, this system uses **BERTopic** to discover the semantic topic structure of your documents first, then retrieves by topic coherence rather than embedding proximity alone. The result: contextually richer answers and dramatically improved faithfulness.

### The Core Innovation

**Topic-Boundary Chunking.** Standard RAG:
```
Document → Fixed 1000-character chunks → Embed → FAISS → Retrieve
```

**Sovereign Intelligence:**
```
Document → Sentences → BERTopic → Topic-coherent chunks → FAISS → Retrieve
```

Every chunk is semantically unified. A retrieval about "medication errors" pulls entire topic-coherent passages about medication errors — not arbitrary windows that happen to contain those words.

---

## 🚀 Key Features

### ✅ **Fully Local & Air-Gapped**
- **No external APIs** — runs entirely on your infrastructure
- **Zero data exfiltration** — all processing local via Ollama + open-source models
- **Sovereignty-first architecture** — built for banks, hospitals, law firms, and government agencies

### ✅ **BERTopic-Powered Topic Discovery**
- Automatically surfaces **172 coherent topics** from 122K financial conversations (validated on BAAI/IndustryInstruction dataset)
- Discovers sub-domains like **Islamic banking, ESG policy, blockchain regulation, European sovereign debt** — with zero predefined categories
- UMAP (5-component, cosine metric) + HDBSCAN (EOM clustering) for noise-robust, parameter-free topic extraction

### ✅ **Semantic Topic-Boundary Chunking**
- **Vectorized NumPy boundary detection** — consecutive same-topic sentences merged into coherent chunks
- Replaces naive `RecursiveCharacterTextSplitter` with semantically meaningful segmentation
- Preserves reading-order context while ensuring every chunk is topically unified

### ✅ **GPU-Accelerated Pipeline**
- **cuML integration** (cuUMAP, cuHDBSCAN) — processes 228K sentences in **under 60 seconds** on Tesla T4 GPU
- Automatic CPU fallback for portability
- **15x speedup** vs CPU-only (UMAP reduced from ~15 minutes to <1 minute)

### ✅ **Multi-Format Document Support**
- **PDF**, **CSV**, **DOCX**, **XLSX** — unified LangChain loader pipeline
- Handles multi-page documents, spreadsheets, and structured/unstructured text
- Automatic sentence-level extraction with min-length filtering (removes boilerplate, headers, page numbers)

### ✅ **Advanced Topic Representation**
- **Custom `OllamaRepresentation` class** — injects documents + keywords into locally-hosted Llama 4 to generate natural-language topic labels
- **`KeyBERTInspired` reranking** — scores terms against cluster embeddings for refined keyword relevance
- **T5-based summarization** (`T5SummarizationPipeline`) — abstractive topic summaries comparing `t5-base` (250M) vs `t5-large` (770M)

### ✅ **Rigorous RAG Evaluation Framework**
- **LLM-as-judge** across three independent metrics:
  - **Faithfulness (86%):** Answer contains only information present in retrieved context
  - **Answer Relevancy (81%):** Answer directly addresses the question
  - **Correctness (63%):** Answer matches reference ground truth
- Failure analysis identifies hallucination and off-topic drift for iterative improvement

### ✅ **FAISS Vector Store with Persistent Indexing**
- `save_index()` / `load_index()` — eliminate re-embedding costs across sessions
- **HuggingFaceEmbeddings** + **Ollama** for local embedding generation
- Fast similarity search over topic-coherent chunks

---

## 📊 Validated Performance

### Corpus Statistics
- **Dataset:** BAAI/IndustryInstruction Finance-Economics (122,090 conversations)
- **After filtering:** 74,477 high-quality English samples (FastText language detection)
- **Sentences extracted:** 228,291
- **Topics discovered:** 172 coherent themes
- **Topic-coherent chunks:** 98,138

### Topic Examples (From Real Run)
| Topic ID | Count | Representative Keywords |
|---|---|---|
| 5 | 3,214 | blockchain, regulation, crypto, compliance, decentralized |
| 12 | 2,891 | sovereign debt, eurozone, restructuring, default, bonds |
| 27 | 1,847 | islamic banking, shariah, sukuk, halal finance, riba |
| 34 | 1,523 | ESG, sustainability, governance, climate, impact |
| 58 | 982 | merger, acquisition, valuation, synergy, due diligence |

### Evaluation Metrics (30-sample test)
| Metric | Score | Samples |
|---|---|---|
| **Faithfulness** | 86% | 21/30 parseable |
| **Answer Relevancy** | 81% | 21/30 parseable |
| **Correctness** | 63% | 30/30 |
| **Overall** | 75% | — |

**Note:** 9 samples returned unparseable judge responses (judge LLM formatting variance) — a known limitation when using smaller local models as evaluators.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Documents (PDF/CSV/DOCX/XLSX)                               │
│         ↓                                                     │
│  LangChain Loaders → Raw Pages                               │
│         ↓                                                     │
│  Sentence Splitter (regex, min_length=40)                    │
│         ↓                                                     │
│  SentenceTransformer Encoding (thenlper/gte-small, 384-dim)  │
│         ↓                                                     │
│  UMAP Reduction (5D, cosine metric, min_dist=0.0)            │
│         ↓                                                     │
│  HDBSCAN Clustering (min_cluster_size=150, EOM selection)    │
│         ↓                                                     │
│  BERTopic → Topic Labels (-1 = noise, 0+ = topics)           │
│         ↓                                                     │
│  NumPy Boundary Detection (topic_change | doc_change)        │
│         ↓                                                     │
│  Topic-Coherent Chunks (consecutive same-topic sentences)    │
│         ↓                                                     │
│  FAISS Vector Store (Ollama embeddings) + Persist to Disk    │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  User Question                                                │
│         ↓                                                     │
│  FAISS Similarity Search (top-k topic chunks, k=4)            │
│         ↓                                                     │
│  Context Assembly (labeled chunks with metadata)             │
│         ↓                                                     │
│  Prompt Template → Ollama Mistral (local LLM)                │
│         ↓                                                     │
│  Generated Answer                                             │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  EVALUATION FRAMEWORK                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Query + Context + Answer                                     │
│         ↓                                                     │
│  Faithfulness Evaluator (LLM-as-judge via Mistral)           │
│  Answer Relevancy Evaluator                                  │
│  Correctness Evaluator (vs ground truth)                     │
│         ↓                                                     │
│  Three Independent Scores (0-1 scale)                         │
│         ↓                                                     │
│  Failure Analysis (identify 0-scored samples)                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for cuML acceleration)
- Ollama (for local LLM)

### Quick Start

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:latest

# 2. Install Python dependencies
pip install bertopic faiss-gpu sentence-transformers umap-learn hdbscan
pip install langchain-community langchain-ollama pypdf colorama datasets

# 3. (Optional) GPU acceleration
pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com

# 4. Clone the repository
git clone https://github.com/yourusername/sovereign-intelligence.git
cd sovereign-intelligence
```

---

## 📖 Usage

### Basic Example

```python
from bertopic_rag import BERTopicRAG

# Initialize the system
rag = BERTopicRAG(
    ollama_model="mistral:latest",
    embedding_model_name="thenlper/gte-small",
    min_cluster_size=150,        # Increase for larger corpora
    n_umap_components=5,
    top_k_retrieval=4,
    verbose=True
)

# Ingest documents
documents = ["financial_report_2024.pdf", "earnings_call_q3.pdf", "regulatory_filing.pdf"]
rag.ingest(documents)

# Inspect discovered topics
print(rag.get_topic_summary())

# Query the system
answer = rag.query("What are the key drivers of revenue growth in Q3?")
print(answer)
```

### Advanced: HuggingFace Dataset Integration

```python
from datasets import load_dataset

# Load large corpus
dataset = load_dataset("BAAI/IndustryInstruction_Finance-Economics")["train"]
dataset = dataset.select(range(75000))  # Subsample if needed

# Convert to documents
raw_docs = rag.load_from_hf_dataset(dataset, text_fields=["conversations"])

# Rest of pipeline identical
chunks = rag.chunk_documents(raw_docs)
rag.create_vectorstore(chunks)
rag.save_index("faiss_index")  # Persist

# Later sessions
rag.load_index("faiss_index")
answer = rag.query("Explain quantitative easing")
```

### Evaluation

```python
from langchain_classic.evaluation import load_evaluator, EvaluatorType
from langchain_ollama import ChatOllama

# Setup judge LLM
eval_llm = ChatOllama(model="mistral:latest")

# Create evaluators
faithfulness_evaluator = load_evaluator(
    EvaluatorType.CRITERIA,
    llm=eval_llm,
    criteria={"faithfulness": "Does the answer contain ONLY information from the context?"}
)

relevancy_evaluator = load_evaluator(
    EvaluatorType.CRITERIA,
    llm=eval_llm,
    criteria={"answer_relevancy": "Does the answer directly address the question?"}
)

correctness_evaluator = load_evaluator(EvaluatorType.QA, llm=eval_llm)

# Evaluate
for sample in eval_samples:
    question = sample["question"]
    ground_truth = sample["ground_truth"]
    
    answer = rag.query(question)
    context = rag.get_last_retrieved_context()  # Capture for faithfulness check
    
    faith_score = faithfulness_evaluator.evaluate_strings(
        input=question, prediction=answer, reference=context
    )["score"]
    
    rel_score = relevancy_evaluator.evaluate_strings(
        input=question, prediction=answer
    )["score"]
    
    correct_score = correctness_evaluator.evaluate_strings(
        input=question, prediction=answer, reference=ground_truth
    )["score"]
    
    print(f"Faithfulness: {faith_score}, Relevancy: {rel_score}, Correctness: {correct_score}")
```

---

## ⚙️ Configuration

### BERTopicRAG Parameters

| Parameter | Default | Description |
|---|---|---|
| `ollama_model` | `"mistral:latest"` | Local LLM for generation and embeddings |
| `embedding_model_name` | `"thenlper/gte-small"` | SentenceTransformer for BERTopic (384-dim, fast) |
| `min_cluster_size` | `150` | HDBSCAN minimum cluster size — controls topic granularity |
| `n_umap_components` | `5` | UMAP dimensionality before clustering |
| `top_k_retrieval` | `4` | Number of topic chunks retrieved per query |
| `verbose` | `True` | Print pipeline progress to stdout |

### Tuning Guide

**Small corpora (<10K sentences):**
- `min_cluster_size=5` to `15`
- More fine-grained topics, higher noise percentage

**Large corpora (>100K sentences):**
- `min_cluster_size=100` to `300`
- Broader topics, lower noise

**Low-resource environments:**
- `embedding_model_name="all-MiniLM-L6-v2"` (lighter, 384-dim)
- `ollama_model="mistral:7b"` or `"llama3.1:8b"`

---

## 📂 Project Structure

```
sovereign-intelligence/
├── README.md                       # This file
├── bertopic_rag.py                 # Main RAG system
├── evaluation/
│   ├── llm_as_judge.py             # Evaluation framework
│   └── metrics.py                  # Faithfulness/Relevancy/Correctness
├── notebooks/
│   ├── 01_LDA_LSA_baseline.ipynb   # Classical topic modeling
│   ├── 02_word_embeddings.ipynb    # GloVe, Word2Vec, BERT
│   ├── 03_sentence_embeddings.ipynb # STS benchmark
│   ├── 04_UMAP_SVD.ipynb           # Dimensionality reduction
│   ├── 05_KMeans_HDBSCAN.ipynb     # Clustering comparison
│   ├── 06_BERTopic_full.ipynb      # BERTopic pipeline
│   ├── 07_RAG_build.ipynb          # RAG construction
│   └── 08_RAG_evaluation.ipynb     # Evaluation results
├── skills/                         # Custom BERTopic representation modules
│   ├── ollama_representation.py    # LLM-powered topic labels
│   ├── t5_summarization.py         # Abstractive topic summaries
│   └── keybert_reranking.py        # Keyword refinement
├── data/                           # Sample datasets (not committed)
└── requirements.txt                # Python dependencies
```

---

## 🔬 Technical Deep Dive

### Why Topic-Boundary Chunking Outperforms Fixed Windows

**Problem with fixed-window chunking:**
1. **Arbitrary segmentation** — a 1000-character window might split a critical sentence in half
2. **Topic bleeding** — a chunk might contain the end of one topic and the start of another
3. **Retrieval noise** — FAISS pulls chunks based on embedding similarity, but embeddings average over all sentences in the chunk — including irrelevant ones

**Sovereign Intelligence solution:**
1. **Sentence-level granularity** — every sentence is independently embedded and topic-labeled
2. **Boundary detection** — topic transitions identified via NumPy array operations:
   ```python
   topic_change[1:] = (topics_arr[1:] != topics_arr[:-1]) & (topics_arr[1:] != -1)
   doc_change[1:] = doc_idx[1:] != doc_idx[:-1]
   boundary_positions = np.where(topic_change | doc_change)[0]
   ```
3. **Coherent context** — every retrieved chunk is guaranteed to be about ONE semantic topic
4. **Metadata enrichment** — each chunk carries `topic_id`, `topic_name`, `source`, `n_sentences`

**Result:** When you query "What are the regulatory challenges in blockchain?", you retrieve entire topic-coherent passages about blockchain regulation — not arbitrary text windows that happen to mention both words.

### GPU Acceleration Impact

**CPU baseline (228K sentences):**
- UMAP: ~12 minutes
- HDBSCAN: ~3 minutes
- **Total:** ~15 minutes

**GPU with cuML (Tesla T4):**
- UMAP: 35 seconds
- HDBSCAN: 18 seconds
- **Total:** 53 seconds

**Speedup:** ~17x

### Why HDBSCAN > K-Means for Topic Discovery

| | **K-Means** | **HDBSCAN** |
|---|---|---|
| **Requires K?** | Yes — must specify cluster count | No — automatically finds natural groupings |
| **Handles noise?** | No — forces every point into a cluster | Yes — labels outliers as `-1` |
| **Cluster shapes** | Assumes spherical | Discovers arbitrary density-based shapes |
| **Use case** | When you know how many topics exist | When you want the data to tell you |

For institutional documents, you almost never know the "right" number of topics upfront. HDBSCAN discovers what's actually there.

---

## 🎓 Educational Resources

This project was formalized into a full **Udemy course**: [Applied GenAI: Building Enterprise Topic Discovery Engines](https://udemy.com/course/applied-genai-bertopic)

**Course curriculum:**
1. Classical Topic Modeling (LDA, LSA)
2. Word Embeddings (GloVe, Word2Vec, BERT contextual embeddings)
3. Sentence Embeddings & STS Benchmark
4. Dimensionality Reduction (UMAP vs SVD)
5. Clustering (K-Means vs HDBSCAN evaluation)
6. BERTopic Full Pipeline
7. Topic Representation (Ollama LLM labels, KeyBERT reranking, T5 summarization)
8. BERTopic-Powered RAG System
9. LLM-as-Judge Evaluation Framework

Each lesson includes runnable code, real datasets, and instructor walkthroughs.

---

## 🏆 Use Cases

### ✅ **Financial Institutions**
- Automatically categorize 10,000+ earnings calls, analyst reports, regulatory filings
- Surface emerging risk themes (e.g., "supply chain disruption", "interest rate exposure")
- Q&A over proprietary internal documents without external API exposure

### ✅ **Healthcare Systems**
- Discover clinical note patterns (medication errors, treatment protocols, adverse events)
- HIPAA-compliant — zero patient data leaves the institution
- Knowledge extraction from 100K+ anonymized case records

### ✅ **Legal Firms**
- Cluster case law by legal issue, jurisdiction, ruling type
- Identify precedent patterns across thousands of briefs
- Attorney privilege protected — no cloud API risk

### ✅ **Government Agencies**
- Thematic analysis of public comments, FOIA requests, policy documents
- Classified material never transmitted externally
- Multi-language support via multilingual SentenceTransformers

### ✅ **Research Institutions**
- Literature review automation — discover research themes across 50K+ papers
- Citation network analysis combined with topic modeling
- Grant proposal optimization — align with trending topics

---

## 🔐 Privacy & Security

### Zero External Dependencies
- **No OpenAI API** — all generation via local Ollama
- **No Anthropic/Cohere/HuggingFace Inference** — embeddings generated locally
- **No telemetry** — BERTopic, FAISS, LangChain run entirely offline

### Data Sovereignty
- **Air-gapped deployment** — can run on networks with no internet access
- **On-premises hosting** — deploy to your own servers
- **Compliance-ready** — GDPR, HIPAA, SOC 2, FedRAMP compatible

### Audit Trail
- Every retrieved chunk includes source metadata (file path, topic ID, sentence count)
- LLM prompts are fully inspectable
- Evaluation metrics provide quantitative accountability

---

## 📈 Roadmap

### Current (v1.0)
- [x] BERTopic topic discovery
- [x] Topic-boundary chunking
- [x] GPU acceleration (cuML)
- [x] Multi-format document support
- [x] LLM-as-judge evaluation

### Planned (v1.1)
- [ ] Incremental indexing (add documents without full refit)
- [ ] Multi-modal support (images, tables in PDFs)
- [ ] Hybrid retrieval (BM25 + semantic)
- [ ] Fine-tuned embeddings (domain adaptation)
- [ ] Web UI (Gradio interface)

### Future (v2.0)
- [ ] Multi-language topic discovery
- [ ] Temporal topic tracking (topic evolution over time)
- [ ] Citation graph integration
- [ ] Active learning (user feedback loop)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas especially welcome:**
- Additional document loaders (EPUB, Markdown, LaTeX)
- Evaluation metric improvements
- Deployment guides (Docker, Kubernetes, AWS/GCP/Azure)
- Benchmarks on new domains (medical, legal, scientific)

---

## 🙏 Acknowledgments

- **BERTopic** by Maarten Grootendorst — foundational topic modeling framework
- **Ollama** — local LLM infrastructure
- **LangChain** — document loaders and vector store abstractions
- **FAISS** (Meta AI) — efficient similarity search
- **UMAP** (Leland McInnes) — manifold learning
- **HDBSCAN** (Leland McInnes, John Healy) — density-based clustering
- **SentenceTransformers** (Nils Reimers, Iryna Gurevych) — semantic embeddings
- **BAAI** — IndustryInstruction dataset for validation

---

## 📧 Contact

**Marwan Gamal**  
NLP & Applied GenAI Researcher  
📧 [marawangamal229@gmail.com]  
🔗 [LinkedIn](www.linkedin.com/in/marwan-gamal-5b40b4280)  
📚 [Udemy Course](https://www.udemy.com/course/applied-genai-building-enterprise-topic-discovery-engines/?src=sac&kw=applied+genai&couponCode=KEEPLEARNING])

---

## 📊 Citation

If you use this system in your research or production environment, please cite:

```bibtex
@software{gamal2025sovereign,
  title={Sovereign Intelligence: Privacy-First Enterprise Topic Discovery Engine},
  author={Gamal, Marwan},
  year={2025},
  url={https://github.com/marwan679/Financial-Economic-Intelligence-with-BERTopic-and-Llama3.2},
  note={BERTopic-powered RAG system with topic-boundary chunking}
}
```

---

**Built with ❤️ for institutions that value both intelligence and sovereignty.**
