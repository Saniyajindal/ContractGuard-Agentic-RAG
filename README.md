# 🛡️ ContractGuard: Enterprise Agentic RAG System
> **Autonomous Legal Compliance Auditing & Deterministic SLA Risk Quantification Engine**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![LLM](https://img.shields.io/badge/LLM-Meta%20Llama--3.3--70B%20(Groq)-orange.svg)](https://groq.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

Enterprise legal contracts and Master Service Agreements (MSAs) often span dozens of pages with nested clauses, stringent SLAs, and heavy financial liability terms. Standard generative search pipelines frequently suffer from **keyword fragmentation** (missing specific clause numbers) and **stochastic hallucinations** (guessing terms or mathematical figures).

**ContractGuard** is a production-grade **Agentic RAG (Retrieval-Augmented Generation)** pipeline engineered to perform automated compliance auditing, clause extraction with exact page citations, and deterministic financial liability calculations.

---

## 🏗️ System Architecture
[ Unstructured Legal PDF ]
│
▼
[ Recursive Chunking (400 chars, 50 overlap) ]
│
┌─────┴─────────────────────────┐
▼                               ▼
[ Dense Embeddings ]          [ Sparse Indexing ]
(all-MiniLM-L6-v2 + ChromaDB)      (BM25 Retriever)
└─────┬─────────────────────────┘
▼
[ Reciprocal Rank Fusion / Ensemble Retriever (50/50) ]
│
▼
[ Grounded Context + Strict Negative Prompt Guardrails ]
│
▼
[ Meta Llama-3.3-70B via Groq API (Temp = 0.0) ]
│
┌─────┴─────────────────────────┐
▼                               ▼
[ Structured Audit Report ]    [ Deterministic Python SLA Tool ]
(With Exact Page Citations)     (Mathematical Penalty Execution)
