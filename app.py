import io
import math
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

try:
	from pypdf import PdfReader  # lightweight PDF text extraction
except Exception:  # pragma: no cover - optional
	PdfReader = None  # type: ignore


# -------------------------
# Page and Theme Setup
# -------------------------
st.set_page_config(
	page_title="CaseClarity – Automated Legal Document Summarization",
	page_icon="⚖️",
	layout="wide",
)

# Add a touch of custom styling to reinforce the legal theme
st.markdown(
	"""
	<style>
	.legal-hero {
		background: linear-gradient(135deg, rgba(182,134,44,0.08), rgba(15,23,42,0.05));
		border: 1px solid rgba(182,134,44,0.25);
		border-radius: 14px;
		padding: 18px 20px;
		margin-bottom: 16px;
	}
	.small-muted { color: #4b5563; font-size: 0.9rem; }
	.token-hint { color: #6b7280; font-size: 0.85rem; }
	</style>
	""",
	unsafe_allow_html=True,
)


# -------------------------
# Utilities
# -------------------------
def get_default_paths() -> Tuple[Path, Path, Path]:
	"""Return (workspace_root, final_model_dir, tokenizer_dir)."""
	root = Path(__file__).resolve().parent
	model_base = root / "artifacts" / "model_trainer"
	final_model = model_base / "t5-legal-model"
	checkpoint = model_base / "checkpoint-375"
	tokenizer = model_base / "tokenizer"
	# Prefer final model if present, else fallback to checkpoint
	preferred_model = final_model if final_model.exists() else checkpoint
	return preferred_model, checkpoint, tokenizer


@st.cache_resource(show_spinner=False)
def load_model(model_dir: Path, tokenizer_dir: Path):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
	model = T5ForConditionalGeneration.from_pretrained(str(model_dir))
	model.to(device)
	model.eval()
	return model, tokenizer, device


def read_pdf(file_bytes: bytes) -> str:
	if PdfReader is None:
		raise RuntimeError("pypdf is not installed. Please install 'pypdf'.")
	reader = PdfReader(io.BytesIO(file_bytes))
	texts = []
	for page in reader.pages:
		try:
			texts.append(page.extract_text() or "")
		except Exception:
			# If extraction fails for a page, skip it gracefully
			continue
	return "\n\n".join([t.strip() for t in texts if t and t.strip()])


def chunk_by_tokens(text: str, tokenizer, max_input_tokens: int, overlap: int = 50) -> List[str]:
	"""Chunk text by tokenizer tokens. Adds small overlaps to preserve context.
	Args:
		text: Full input text.
		tokenizer: HF tokenizer compatible with the model.
		max_input_tokens: Max tokens per chunk (model's max input ~512 for T5-small/base).
		overlap: Token overlap between chunks to preserve context coherence.
	Returns:
		List of chunk strings.
	"""
	tokens = tokenizer.encode(text, add_special_tokens=False)
	if not tokens:
		return []

	chunks = []
	step = max_input_tokens - overlap if max_input_tokens > overlap else max_input_tokens
	for start in range(0, len(tokens), step):
		end = min(start + max_input_tokens, len(tokens))
		chunk_ids = tokens[start:end]
		chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
		if chunk_text.strip():
			chunks.append(chunk_text)
		if end == len(tokens):
			break
	return chunks


def summarize_chunk(
	model,
	tokenizer,
	device: str,
	text: str,
	max_input_tokens: int,
	gen_max_new_tokens: int,
	num_beams: int = 4,
	length_penalty: float = 1.0,
	no_repeat_ngram_size: int = 3,
	temperature: float = 1.0,
	top_p: float = 1.0,
	top_k: int = 50,
	prefix: str = "summarize: ",
):
	chunks = chunk_by_tokens(text, tokenizer, max_input_tokens=max_input_tokens)
	summaries = []
	for c in chunks:
		input_ids = tokenizer.encode(prefix + c, return_tensors="pt").to(device)
		with torch.no_grad():
			outputs = model.generate(
				input_ids,
				max_new_tokens=gen_max_new_tokens,
				num_beams=num_beams,
				length_penalty=length_penalty,
				no_repeat_ngram_size=no_repeat_ngram_size,
				do_sample=(temperature != 1.0 or top_p < 1.0 or top_k < 50),
				temperature=temperature,
				top_p=top_p,
				top_k=top_k,
				early_stopping=True,
			)
		summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
		summaries.append(summary.strip())
	return summaries


def map_reduce_summarize(
	model,
	tokenizer,
	device: str,
	text: str,
	max_input_tokens: int,
	gen_max_new_tokens: int,
	map_num_beams: int,
	reduce_num_beams: int,
	max_reduce_passes: int = 2,
) -> str:
	"""Two-stage map-reduce summarization for very long texts."""
	# Map stage
	map_summaries = summarize_chunk(
		model,
		tokenizer,
		device,
		text,
		max_input_tokens=max_input_tokens,
		gen_max_new_tokens=gen_max_new_tokens,
		num_beams=map_num_beams,
	)

	combined = "\n".join(map_summaries)
	# If combined still exceeds the model window, iterate a reduce pass
	for _ in range(max_reduce_passes):
		token_count = len(tokenizer.encode(combined, add_special_tokens=False))
		if token_count <= max_input_tokens:
			break
		map_summaries = summarize_chunk(
			model,
			tokenizer,
			device,
			combined,
			max_input_tokens=max_input_tokens,
			gen_max_new_tokens=gen_max_new_tokens,
			num_beams=reduce_num_beams,
		)
		combined = "\n".join(map_summaries)
	return combined


# -------------------------
# Sidebar Controls
# -------------------------
default_model, checkpoint_dir, tokenizer_dir = get_default_paths()

st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio(
	"Model weights",
	options=["Final model", "Checkpoint (375)"],
	index=0 if default_model.name == "t5-legal-model" else 1,
)

model_dir = default_model if model_choice == "Final model" else checkpoint_dir

max_input_tokens = st.sidebar.slider("Max input tokens per chunk", 256, 1024, 512, step=64)
gen_max_new_tokens = st.sidebar.slider("Max summary tokens (per chunk)", 64, 512, 160, step=16)
num_beams = st.sidebar.slider("Beam search width", 1, 8, 4)
length_penalty = st.sidebar.slider("Length penalty", 0.1, 2.0, 1.0, step=0.1)
no_repeat_ngram_size = st.sidebar.slider("No-repeat n-gram size", 1, 6, 3)
sampling = st.sidebar.checkbox("Enable sampling (temperature/top-p)", value=False)
temperature = st.sidebar.slider("Temperature", 0.3, 2.0, 1.0, 0.1, disabled=not sampling)
top_p = st.sidebar.slider("Top-p (nucleus)", 0.1, 1.0, 1.0, 0.05, disabled=not sampling)
top_k = st.sidebar.slider("Top-k", 10, 200, 50, 10, disabled=not sampling)

use_map_reduce = st.sidebar.checkbox("Map-Reduce for long texts", value=True)
map_beams = st.sidebar.slider("Map stage beams", 1, 8, max(2, num_beams), key="map_beams")
reduce_beams = st.sidebar.slider("Reduce stage beams", 1, 8, max(2, num_beams), key="reduce_beams")


# -------------------------
# Load Model
# -------------------------
with st.sidebar:
	st.caption("Model directory:")
	st.code(str(model_dir), language="text")
	st.caption("Tokenizer directory:")
	st.code(str(tokenizer_dir), language="text")

with st.spinner("Loading model… this may take a moment on first run"):
	if not model_dir.exists():
		st.error(f"Model directory not found: {model_dir}")
		st.stop()
	if not tokenizer_dir.exists():
		st.error(f"Tokenizer directory not found: {tokenizer_dir}")
		st.stop()
	model, tokenizer, device = load_model(model_dir, tokenizer_dir)
st.sidebar.success(f"Model loaded on: {device.upper()}")


# -------------------------
# Main Content
# -------------------------
st.markdown(
	"""
	<div class="legal-hero">
	  <h1>⚖️ CaseClarity</h1>
	  <div class="small-muted">Automated Legal Document Summarization powered by fine-tuned T5 model.</div>
	</div>
	""",
	unsafe_allow_html=True,
)

left, right = st.columns([2, 1])

with left:
	st.subheader("Input document")
	uploaded = st.file_uploader("Upload a legal document (PDF or TXT)", type=["pdf", "txt"]) 
	text_input = st.text_area(
		"Or paste legal text here",
		height=250,
		placeholder="Paste a judgment, contract, or legal brief…",
	)

	input_text = ""
	if uploaded is not None:
		if uploaded.type == "application/pdf":
			try:
				input_text = read_pdf(uploaded.getvalue())
			except Exception as e:
				st.error(f"Failed to read PDF: {e}")
		else:
			try:
				input_text = uploaded.getvalue().decode("utf-8", errors="ignore")
			except Exception as e:
				st.error(f"Failed to read text file: {e}")
	elif text_input.strip():
		input_text = text_input.strip()

	if input_text:
		tok_count = len(tokenizer.encode(input_text, add_special_tokens=False))
		st.caption(f"Input length ≈ {tok_count} tokens")

with right:
	st.subheader("Summary")
	run = st.button("Summarize", type="primary", use_container_width=True)
	output_area = st.empty()
	download_area = st.empty()

if 'last_summary' not in st.session_state:
	st.session_state['last_summary'] = ""

if run:
	if not input_text:
		st.warning("Please upload a file or paste text to summarize.")
	else:
		with st.spinner("Summarizing…"):
			if use_map_reduce:
				summary = map_reduce_summarize(
					model,
					tokenizer,
					device,
					input_text,
					max_input_tokens=max_input_tokens,
					gen_max_new_tokens=gen_max_new_tokens,
					map_num_beams=map_beams,
					reduce_num_beams=reduce_beams,
				)
			else:
				parts = summarize_chunk(
					model,
					tokenizer,
					device,
					input_text,
					max_input_tokens=max_input_tokens,
					gen_max_new_tokens=gen_max_new_tokens,
					num_beams=num_beams,
					length_penalty=length_penalty,
					no_repeat_ngram_size=no_repeat_ngram_size,
					temperature=temperature if sampling else 1.0,
					top_p=top_p if sampling else 1.0,
					top_k=top_k if sampling else 50,
				)
				summary = "\n".join(parts)

		st.session_state['last_summary'] = summary
		output_area.write(summary)
		download_area.download_button(
			"Download Summary (TXT)",
			data=summary.encode("utf-8"),
			file_name="case-clarity-summary.txt",
			mime="text/plain",
			use_container_width=True,
		)


# Helpful tips
st.markdown(
	"""
	<div class="token-hint">
	Tips:
	<ul>
	  <li>If your document is very long, keep Map-Reduce enabled for better coherence.</li>
	  <li>Increase <b>Max input tokens per chunk</b> cautiously; T5 models are commonly trained with 512-token windows.</li>
	  <li>Use <b>Beam search</b> for deterministic summaries or enable sampling for more diverse phrasing.</li>
	</ul>
	</div>
	""",
	unsafe_allow_html=True,
)

