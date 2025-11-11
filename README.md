# ⚖️ CaseClarity-Automated Legal Document Summarization

**AI-powered document summarization for legal texts using fine-tuned T5 transformer model**

A comprehensive end-to-end machine learning system for automatically summarizing Indian legal documents (judgements) into concise summaries, built as a Final Year Project.

---

## 🎯 Project Overview

This project implements a complete ML pipeline for legal document summarization:
- **Model**: Fine-tuned T5-small transformer (60M parameters)
- **Dataset**: 6,000+ Indian legal judgements with professional summaries
- **Architecture**: Modular pipeline with 4 stages (Ingestion → Validation → Transformation → Training)
- **Deployment**: Streamlit web application with map-reduce chunking for long documents
- **Training**: Successfully trained on legal corpus (~2.5 hours, 375 steps)

---

## ✨ Features

- 📄 **Multi-format Support**: Upload PDF files or paste text directly
- 🧠 **Smart Chunking**: Handles long documents (>1024 tokens) with map-reduce summarization
- ⚡ **GPU/CPU Support**: Automatic device detection (CUDA, MPS, CPU)
- 🔧 **Configurable**: YAML-based configuration for easy customization
- 📊 **MLflow Tracking**: Experiment tracking with TensorBoard integration
- 🎨 **Modern UI**: Clean Streamlit interface with legal theme

---

## 🏗️ Architecture

### Pipeline Stages

```
Stage 1: Data Ingestion     → Download and extract legal_summary.zip
Stage 2: Data Validation    → Verify CSV format and required columns
Stage 3: Data Transformation → Tokenize with T5 (judgement → summary)
Stage 4: Model Training      → Fine-tune T5-small model
```

### Project Structure

```
document_summarization_for_legal_texts/
├── app.py                     # Streamlit web application
├── main.py                    # Full pipeline orchestrator
├── train_model.py             # Standalone training script
├── datascience/
│   ├── components/            # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── config/                # Configuration management
│   ├── pipeline/              # Stage pipelines
│   └── utils/                 # Helper functions
├── config/
│   └── config.yaml            # Project configuration
├── params.yaml                # Training hyperparameters
├── research/                  # Jupyter notebooks
└── requirements.txt           # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended for training)
- GPU with CUDA (optional, recommended for training)

### Installation

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

### Usage

#### Option 1: Run Web Application (Recommended)

If you already have a trained model:

```bash
streamlit run app.py
```

Access at: http://localhost:8501

#### Option 2: Train Your Own Model

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

## 📦 Dependencies

Core libraries:
- `transformers>=4.57.0` - HuggingFace transformers library
- `torch>=2.0.0` - PyTorch deep learning framework
- `datasets>=2.14.0` - HuggingFace datasets library
- `streamlit>=1.36.0` - Web application framework
- `pypdf>=4.0.0` - PDF text extraction

See `requirements.txt` for complete list.

---

## 📊 Model Performance

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

## 🎓 Academic Context

**Final Year Project** - Computer Science Engineering

**Problem Statement**: Legal documents are often lengthy and complex, making it time-consuming for legal professionals to extract key information. This system automates the summarization process using state-of-the-art NLP.

**Dataset**: Indian Legal Documents corpus containing judgements from various courts with professional human-written summaries.

---

## 🔧 Configuration

### config/config.yaml
```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://github.com/madhusudhanant/Datasets/raw/main/legal_summary.zip
  local_data_file: artifacts/data_ingestion/legal_summary.zip
  unzip_dir: artifacts/data_ingestion

# ... (additional configuration)
```

### params.yaml
```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  weight_decay: 0.01
  logging_steps: 10
```

---

## 🐛 Troubleshooting

### Issue: TensorFlow/Keras import errors
**Solution**: Set environment variables in your script:
```python
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

### Issue: Out of memory during training
**Solution**: Reduce batch size or enable gradient checkpointing in `params.yaml`

### Issue: Streamlit app doesn't load model
**Solution**: Ensure model exists at `artifacts/model_trainer/t5-legal-model/`

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Madhusudhan**
- GitHub: [@madhusudhanant](https://github.com/madhusudhanant)
- Project: Final Year - Computer Science Engineering

---

## 🙏 Acknowledgments

- HuggingFace for the Transformers library and T5 model
- Indian Legal Documents dataset contributors
- Streamlit team for the web framework

---

## 📧 Contact

For questions or feedback about this project, please open an issue on GitHub.

---

## 🔄 Development Workflow

### Project Setup (Already Done)
1. ✅ Update config.yaml
2. ✅ Update params.yaml
3. ✅ Update entity
4. ✅ Update configuration manager
5. ✅ Update components
6. ✅ Update pipeline
7. ✅ Update app.py
8. ✅ Update main.py

---

**⭐ If this project helped you, please give it a star!**  