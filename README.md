A comprehensive end-to-end machine learning system for automatically summarizing Indian legal documents (judgements) into concise summaries, built as a Final Year Project.

---

Quick Start

Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended for training)
- GPU with CUDA (optional, recommended for training)

Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/madhusudhanant/CaseClarity-Automated-Legal-Document-Summarization.git
   cd CaseClarity-Automated-Legal-Document-Summarization
   ```

2. **Create virtual environment**
   ```powershell
   # Windows PowerShell
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

Usage

#### Option 1: Run Web Application (Recommended)

If you already have a trained model:

```bash
streamlit run app.py
```

Access at: http://localhost:8501

Option 2: Train Your Own Model

Run the complete pipeline:

```bash
python main.py
```

Or train only the model stage:

```bash
python train_model.py
```

**Training Details:**
- Dataset: 6,000 training examples, 100 test examples
- Time: ~2-3 hours on CPU, ~45 minutes on GPU
- Output: `artifacts/model_trainer/t5-legal-model/`

---

Dependencies

Core libraries:
- `transformers>=4.57.0` - HuggingFace transformers library
- `torch>=2.0.0` - PyTorch deep learning framework
- `datasets>=2.14.0` - HuggingFace datasets library
- `streamlit>=1.36.0` - Web application framework
- `pypdf>=4.0.0` - PDF text extraction

See `requirements.txt` for complete list.

---

Model Performance

**Training Configuration:**
- Model: T5-small (60M parameters)
- Epochs: 1
- Batch Size: 1 (with gradient accumulation 16)
- Learning Rate: 5e-5
- Warmup Steps: 500
- Final Training Loss: 3.085

**Tokenization:**
- Input: Max 1024 tokens (legal judgements)
- Output: Max 128 tokens (summaries)
- Tokenizer: T5Tokenizer

---
