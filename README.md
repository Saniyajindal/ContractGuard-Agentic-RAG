# 🛡️ ContractGuard | Enterprise Legal & SLA Audit Engine

> **Next-Gen Hybrid RAG System paired with a Deterministic Arithmetic Guardrail for verifiable, zero-hallucination legal document analysis.**

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://contractguard-agentic-rag-cga8jpbunszedakgesa6pf.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?logo=chainlink)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Storage-FC4C02)](https://trychroma.com)
[![Gemini](https://img.shields.io/badge/Google_Gemini-3.6_Flash-4E75F8?logo=google&logoColor=white)](https://ai.google.dev/)

[**Explore Live Dashboard ↗**](https://contractguard-agentic-rag-cga8jpbunszedakgesa6pf.streamlit.app) • [**Architecture**](#-under-the-hood-architecture) • [**Quickstart**](#-quickstart)

</div>

---

### 💡 Why ContractGuard?

Standard RAG architectures fail in compliance and legal domains because:
* **Semantic Drift:** Vector similarity often misses exact clause numbers, penalty rates, or strict legal identifiers.
* **Arithmetic Hallucination:** LLMs are notorious for botching financial downtime calculations and penalty caps.

**ContractGuard** solves this by unifying **Dense Semantic Embeddings** with **Exact BM25 Lexical Matching**, while offloading all penalty arithmetic to a **deterministic runtime engine**.

---

### 🌟 Core Highlights

| Feature | Technical Implementation | Enterprise Impact |
| :--- | :--- | :--- |
| **Hybrid Search (Dense + Sparse)** | ChromaDB (`all-MiniLM-L6-v2`) + BM25 Token Ranking | Eliminates false negatives; catches both semantic concepts & explicit clause strings. |
| **Zero-Hallucination Guardrail** | Strict Negative-Constraint Prompting with fallback alerts | Enforces exact page citations (`[Page X, Clause Y]`); halts fabrication if clauses are omitted. |
| **Deterministic SLA Engine** | Dedicated Python Arithmetic Tool | Eliminates LLM math hallucination for outage liabilities ($/hr scaling + severity flagging). |
| **Ultra-Fast LLM Inference** | Google Gemini 3.6 Flash | Sub-second enterprise compliance reports with low operational latency. |

---

### 🏗 Under-The-Hood Architecture

```text
              ┌────────────────────────────────────────────────────────┐
              │                   Legal Contract (PDF)                 │
              └───────────────────────────┬────────────────────────────┘
                                          │
                        [Recursive Chunking: 400 chars]
                                          │
             ┌────────────────────────────┴────────────────────────────┐
             ▼                                                         ▼
    [Chroma Vector Store]                                     [BM25 Index]
  (Dense Semantic Proximity)                              (Sparse Keyword Match)
             │                                                         │
             └────────────────────────────┬────────────────────────────┘
                                          │
                           [Ranked Fusion & Deduplication]
                                          │
                                          ▼
                               [Context Injection Window]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
      [Google Gemini 3.6 Flash]                     [Deterministic Python Tool]
    - Strict Page Citations                       - Downtime Penalty Calculation
    - Zero Hallucination Guardrails               - Breach Severity Tiering
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │  Audited Compliance Findings Dashboard│
                      └───────────────────────────────────────┘
