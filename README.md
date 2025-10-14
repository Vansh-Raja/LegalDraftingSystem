# Legal Drafting System

A sophisticated RAG (Retrieval-Augmented Generation) system for legal document analysis and Q&A. This system processes legal judgments, extracts structured metadata, and provides intelligent question-answering capabilities through both web and CLI interfaces.

## 🏗️ Architecture Overview

The system follows a modular architecture with clear separation of concerns:

### Core Components

1. **PDF Processing** (`functions.py`)
   - Converts PDF legal documents to text
   - Maintains stable file numbering with manifest system
   - Handles both "judgements" and "judgments" directory names

2. **Metadata Extraction** (`ai.py`)
   - Uses LLMs to extract structured metadata from legal text
   - Supports multiple backends: OpenAI, OpenRouter, Ollama
   - Extracts case numbers, parties, court names, dates, summaries, and legal provisions

3. **RAG Operations** (`rag.py`)
   - Document chunking and embedding generation
   - PGVector database integration for similarity search
   - Intelligent filtration and context assembly
   - Named case detection and prioritization

4. **Query Processing** (`orchestrator.py`)
   - Classifies queries (general law, new, follow-up)
   - Creates execution plans with bridging strategies
   - Handles context reuse and statute-based refills

5. **User Interfaces**
   - **Web Interface** (`app.py`): Streamlit-based chat with debug tools
   - **CLI Interface** (`chat.py`): Command-line chat with memory
   - **Debug Tools** (`debug.py`, `st_debug.py`): Database management and logging

## 🔄 System Flow

### Offline Processing Pipeline
```
PDF Files → Text Extraction → Metadata Extraction → Chunking → Embedding → Vector Database
```

1. **PDF to Text**: `functions.py` processes PDFs in `judgements/` directory
2. **Metadata Extraction**: `ai.py` uses LLMs to extract structured metadata
3. **Chunking**: Text is split into 2000-character chunks with 400-character overlap
4. **Embedding**: Chunks are embedded using Ollama's `nomic-embed-text:latest`
5. **Storage**: Embeddings stored in PostgreSQL with PGVector extension

### Online Query Processing
```
User Query → Query Classification → Retrieval → Filtration → Context Assembly → Answer Generation
```

1. **Query Classification**: Determines if query is general law, new, or follow-up
2. **Retrieval**: Searches vector database with optional filters (court, statutes)
3. **Filtration**: LLM selects most relevant chunks and full documents
4. **Context Assembly**: Builds focused context within token budget
5. **Answer Generation**: Streams response using selected LLM

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Main programming language
- **PostgreSQL + PGVector**: Vector database for embeddings
- **LangChain**: RAG framework and document processing
- **Streamlit**: Web interface framework

### AI/ML Components
- **Embeddings**: Ollama `nomic-embed-text:latest` (local)
- **Chat Models**: 
  - OpenAI `gpt-5-nano-2025-08-07` (preferred)
  - Ollama `qwen3:latest` (fallback)
- **Filtration**: OpenAI `gpt-5-nano-2025-08-07` for intelligent document selection

### External Services
- **OpenAI API**: For chat and filtration models
- **OpenRouter API**: Alternative LLM provider (free tier available)
- **Ollama**: Local LLM hosting (completely free)

## 📁 Project Structure

```
LegalDraftingSystem_VG/
├── app.py                 # Main Streamlit web interface
├── ai.py                  # Metadata extraction with multiple LLM backends
├── rag.py                 # RAG operations, vector store, filtration
├── functions.py           # PDF processing and text extraction
├── orchestrator.py        # Query planning and classification
├── ingest.py              # Data ingestion into vector database
├── chat.py                # CLI chat interface
├── process.py             # Interactive processing pipeline
├── debug.py               # Database debug utilities
├── st_debug.py            # Streamlit debug utilities
├── main.py                # Entry point documentation
├── requirements.txt       # Python dependencies
├── judgements/            # Input PDF files directory
└── processed_data/        # Processed text and metadata
    ├── txt_data/          # Extracted text files (1.txt, 2.txt, ...)
    └── metadata/          # JSON metadata files (1.json, 2.json, ...)
```

## 🚀 Quick Start

### Prerequisites
1. **PostgreSQL** with PGVector extension
2. **Python 3.8+** with pip
3. **Ollama** (optional, for local models)
4. **API Keys** (OpenAI and/or OpenRouter)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd LegalDraftingSystem_VG

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and database connection
```

### Environment Variables
```bash
# Database connection
PGVECTOR_CONNECTION=postgresql://user:password@localhost:5432/dbname
# OR individual components
DB_NAME=legaldraftingsystemdb
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# API Keys (at least one required)
OPENAI_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key

# Optional
OLLAMA_HOST=http://localhost:11434
```

### Usage

1. **Process Documents**:
   ```bash
   python process.py
   ```
   - Converts PDFs to text
   - Extracts metadata using selected LLM backend

2. **Ingest Data**:
   ```bash
   python ingest.py
   ```
   - Creates embeddings and stores in vector database

3. **Start Chat Interface**:
   ```bash
   # Web interface
   streamlit run app.py
   
   # CLI interface
   python chat.py
   ```

## 🎯 Key Features

### Intelligent Query Processing
- **Query Classification**: Automatically determines query type and strategy
- **Context Bridging**: Maintains conversation context across queries
- **Statute Filtering**: Focuses on specific legal provisions
- **Named Case Detection**: Prioritizes mentioned cases

### Advanced RAG Pipeline
- **Multi-stage Filtration**: LLM-powered document selection
- **Context Budgeting**: Manages token limits intelligently
- **Full Document Loading**: Loads complete cases when needed
- **Fallback Strategies**: Graceful degradation when components fail

### Flexible LLM Support
- **Multiple Backends**: OpenAI, OpenRouter, Ollama
- **Model Selection**: Choose between different models at runtime
- **Cost Optimization**: Use local models to reduce API costs
- **Fallback Chains**: Automatic fallback when APIs are unavailable

### User Experience
- **Real-time Streaming**: Answers stream as they're generated
- **Debug Visibility**: Detailed logging and debug information
- **Session Management**: Maintains conversation history
- **Multiple Interfaces**: Web and CLI options

## 🔧 Configuration Options

### Retrieval Settings
- **Top-K Chunks**: Number of initial chunks to retrieve (3-12)
- **Court Filtering**: Filter by specific courts or all courts
- **Statute Filtering**: Focus on specific legal provisions
- **Filtration Mode**: Chunk-based or metadata-based selection

### Model Selection
- **Chat Models**: OpenAI gpt-5-nano or Ollama qwen3
- **Embedding Model**: Ollama nomic-embed-text (configurable)
- **Filtration Model**: OpenAI gpt-5-nano (for document selection)

## 🐛 Debugging and Maintenance

### Debug Tools
```bash
# Database utilities
python debug.py

# Check vector database status
# Clear embeddings (dangerous)
# Count documents
```

### Common Issues
1. **No embeddings found**: Run `python ingest.py` after processing
2. **API rate limits**: Switch to Ollama or wait for limits to reset
3. **Database connection**: Check PostgreSQL and PGVector installation
4. **Empty responses**: Check debug logs in Streamlit interface

## 📊 Performance Considerations

### Optimization Tips
- Use local Ollama models to reduce API costs
- Adjust chunk size based on document characteristics
- Monitor token usage in debug logs
- Use statute filtering to narrow search scope

### Scalability
- Batch processing for large document sets
- Idempotent ingestion (skips existing embeddings)
- Configurable batch sizes for memory management
- Efficient vector similarity search with PGVector

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive comments to new code
4. Test with sample legal documents
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- LangChain team for the RAG framework
- OpenAI for advanced language models
- Ollama for local model hosting
- PGVector for PostgreSQL vector extensions