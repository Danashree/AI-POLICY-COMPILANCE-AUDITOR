# AI-POLICY-COMPILANCE-AUDITOR
RAG + Vector Database + Threshold-Based Hallucination Control
📌 Project Overview

The AI Policy Compliance Auditor is a Retrieval-Augmented Generation (RAG) system that checks whether employee questions are covered in official company policy documents.

It prevents hallucinated answers by using:

Vector similarity threshold filtering

Citation grounding with similarity scores

Smart “I Don’t Know” refusal mode

This ensures transparent, reliable, and policy-grounded responses.

🎯 Problem Statement

In organizations, employees frequently ask HR and compliance-related questions. Traditional chatbots often:

Hallucinate answers

Provide policy-inconsistent responses

Fail to show evidence sources

This system solves that by answering only if the policy explicitly covers the question.

🚀 Key Features
✅ R1 – Document Embedding + Chroma Vector DB

PDF policy ingestion

Text chunking with overlap

SentenceTransformer embeddings

ChromaDB persistent vector storage

✅ R2 – Threshold-Based Answer / Not Answered Logic

Cosine similarity scoring

Configurable threshold gate

Below threshold → System refuses to answer

✅ R3 – Citation Display with Scores

Displays top-k retrieved chunks

Shows similarity scores

Provides content preview for transparency

✅ R4 – Top-K Retrieval Experiments

Adjustable k parameter

Allows performance tuning and retrieval experiments

✅ R5 – Smart “I Don’t Know” Mode

If:

Similarity is below threshold
OR

Context does not clearly contain the answer

System responds with:

I DON'T KNOW – NOT COVERED IN POLICY

This prevents hallucination.
