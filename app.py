import os
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="ContractGuard AI", page_icon="🛡️", layout="wide")

st.title("🛡️ ContractGuard: Autonomous Legal SLA Auditor")
st.caption("Hybrid RAG (BM25 + ChromaDB) + Llama-3.3-70B via Groq + Deterministic Penalty Engine")

# Sidebar for API Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    uploaded_file = st.file_uploader("Upload Legal Contract (PDF)", type=["pdf"])

# Deterministic Risk Calculation Tool
def calculate_downtime_penalty(hours_down: float, penalty_per_hour: float = 50000.0) -> dict:
    total_penalty = max(0.0, hours_down) * penalty_per_hour
    return {
        "outage_duration_hours": hours_down,
        "hourly_penalty_rate_usd": penalty_per_hour,
        "total_financial_liability_usd": f"${total_penalty:,.2f}",
        "breach_severity": "CRITICAL" if total_penalty > 100000 else "MODERATE"
    }

class EnterpriseHybridRetriever:
    def __init__(self, bm25, chroma):
        self.bm25 = bm25
        self.chroma = chroma

    def invoke(self, query: str):
        dense_docs = self.chroma.invoke(query)
        sparse_docs = self.bm25.invoke(query)
        seen, unique_docs = set(), []
        for doc in dense_docs + sparse_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs[:3]

if uploaded_file and groq_api_key:
    # Save temporary PDF
    with open("temp_contract.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    @st.cache_resource
    def process_document():
        loader = PyPDFLoader("temp_contract.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        chroma_ret = vectorstore.as_retriever(search_kwargs={"k": 2})
        bm25_ret = BM25Retriever.from_documents(chunks)
        bm25_ret.k = 2
        return EnterpriseHybridRetriever(bm25_ret, chroma_ret)

    hybrid_retriever = process_document()
    st.success("Contract successfully indexed into Hybrid RAG Engine!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔍 Legal Clause Audit")
        query = st.text_input("Enter Audit Query / Compliance Clause Check:", "What is the penalty if servers face 1 hour of unapproved downtime?")
        
        if st.button("Run Audit"):
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=groq_api_key)
            audit_prompt = PromptTemplate.from_template("""
            You are a Senior Enterprise Legal Auditor. Analyze and answer the question STRICTLY using the context below.
            [CONTEXT]:
            {context}
            [USER QUESTION]:
            {question}
            Rules:
            1. State the exact finding clearly in structured bullet points.
            2. ALWAYS cite the exact Page Number and Clause Number.
            3. If not present, output: "RISK AUDIT ALERT: Clause not specified in source document."
            Audit Report:
            """)
            qa_chain = audit_prompt | llm | StrOutputParser()
            retrieved_docs = hybrid_retriever.invoke(query)
            context_str = "\n\n".join([f"[Source: Page {d.metadata.get('page', 0) + 1}]:\n" + d.page_content for d in retrieved_docs])
            
            with st.spinner("Analyzing legal clauses..."):
                response = qa_chain.invoke({"context": context_str, "question": query})
            
            st.markdown("### 📋 Audit Findings")
            st.write(response)

    with col2:
        st.subheader("⚡ Deterministic SLA Tool")
        downtime_hours = st.number_input("Observed Outage Hours:", min_value=0.0, max_value=100.0, value=3.5, step=0.5)
        if st.button("Calculate SLA Penalty"):
            risk_result = calculate_downtime_penalty(downtime_hours)
            st.metric("Total Liability", risk_result["total_financial_liability_usd"])
            st.write(f"**Breach Severity:** `{risk_result['breach_severity']}`")
            st.json(risk_result)
else:
    st.info("👈 Upload a Contract PDF and enter your Groq API Key in the sidebar to start.")
