🛡️ ContractGuard: Enterprise Legal SLA Auditor & Hybrid RAG EngineAn enterprise-grade, deterministic Legal & SLA auditing system built with Hybrid RAG (BM25 + ChromaDB), Google Gemini 3.6 Flash, and a rule-based Deterministic Penalty Execution Engine.Designed to parse complex technical spec sheets, commercial proposals, and legal service level agreements (SLAs) with zero arithmetic hallucination and verifiable page-level audit trails.🌐 Live DeploymentAccess the interactive web dashboard on Streamlit Cloud:👉 ContractGuard Live Application⚡ Key Architectural FeaturesHybrid Retrieval Mechanism (Dense + Sparse Fusion):ChromaDB (all-MiniLM-L6-v2): Captures high-dimensional contextual embeddings and semantic similarity.BM25 Retriever: Preserves exact token matches for technical identifiers, contract clauses, and specific numerical constraints.Fusion Deduplication: Merges and prioritizes unique top-$k$ contextual windows for LLM inference.Deterministic Financial Execution Tool:Mitigates LLM numerical hallucinations by delegating SLA downtime penalty computations to a standalone, deterministic Python evaluation engine.Dynamically computes financial liability thresholds and flags operational breach severity levels (MODERATE vs CRITICAL).Zero-Hallucination Audit Guardrails:Prompt-constrained extraction guarantees answers strictly reflect the contextual documents.Mandates exact citations: [Source: Page X, Heading: Y].Triggers explicit fallback alerts (RISK AUDIT ALERT) whenever queried parameters are missing from the source text.🛠️ Tech Stack & DependenciesFrontend / UI: StreamlitRAG Orchestration: LangChain Core, LangChain CommunityDense Retrieval / Vector Store: ChromaDB, HuggingFace Transformers (sentence-transformers/all-MiniLM-L6-v2)Sparse Retrieval: Rank-BM25LLM Engine: Google Gemini (gemini-3.6-flash) via google-generativeaiDocument Parsing: PyPDF📋 System ArchitecturePlaintext[PDF Upload]
      │
      ▼
[Recursive Text Splitting]
      ├──► [Dense Pipeline: MiniLM-L6-v2] ──► [ChromaDB Vector Store]
      │                                                │
      └──► [Sparse Pipeline: Exact Tokens] ────► [BM25 Index]
                                                       │
                                                       ▼
[Audit Query] ──────────────────────────► [Hybrid Retrieval Fusion Engine]
                                                       │
                                                       ▼
[Deterministic SLA Tool] ◄──── [Context Injection + Strict Audit Guardrails]
 (Rule-Based Math Engine)                              │
                                                       ▼
                                         [Gemini 3.6 Flash Inference]
                                                       │
                                                       ▼
                                     [Structured Audit Report with Citations]
🚀 Local Setup & InstallationClone the Repository:Bashgit clone https://github.com/Saniya-29/contractguard-agentic-rag.git
cd contractguard-agentic-rag
Create and Activate a Virtual Environment:Bashpython -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install Dependencies:Bashpip install -r requirements.txt
Run the Streamlit Application:Bashstreamlit run app.py
Authenticate:Provide your Gemini API Key directly in the sidebar interface to begin indexing documents and running audits.
