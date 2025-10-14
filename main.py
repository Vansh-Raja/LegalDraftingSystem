"""
Main entry point for the Legal Drafting System.

This file is intentionally minimal. The actual entry points are:
- process.py: For processing PDFs and extracting metadata
- app.py: For the Streamlit web interface
- chat.py: For the CLI chat interface
- ingest.py: For ingesting processed data into the vector database

To get started:
1. Run: python process.py (to process PDFs and extract metadata)
2. Run: python ingest.py (to create vector embeddings)
3. Run: streamlit run app.py (for web interface) or python chat.py (for CLI)
"""


