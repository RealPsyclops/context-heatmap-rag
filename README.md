# Context Heatmap RAG 🔥

> **Stop asking the AI what's important. Watch the user.**

This system improves Retrieval Augmented Generation (RAG) by tracking **User Behavior** (highlighting, copying, hovering) to create a "Heatmap" of the conversation. It prioritizes content the user has physically interacted with, rather than relying solely on semantic similarity.

## The Core Philosophy

Standard RAG treats all historical text as equal.
* **The Problem:** If a user highlights a code block to copy it, that block is gold. If they scroll past a hallucination, that block is garbage. Standard RAG can't tell the difference.
* **The Solution:** We overlay a "Heat Layer" on the vector index.
    * **Cold:** Unread / Ignored.
    * **Warm:** Read / Hovered.
    * **Hot:** Selected / Copied / Explicitly Saved.

## Architecture

1.  **Passive Listener (`frontend/listener.js`):** A lightweight JS script that detects `selectionchange` and `copy` events. It debounces inputs to filter noise (fidgeting).
2.  **Thermal Retriever (`backend/retriever.py`):** A modified vector search that boosts scores based on Heat Density.
    * `Score = CosineSimilarity * (1 + (HeatScore * Alpha))`
3.  **Semantic Anchors:** A "Side Door" mechanism allowing users to explicitly tag messages (e.g., "Save this as 'Login Logic'").

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
