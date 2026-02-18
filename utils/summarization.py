from transformers import pipeline, AutoTokenizer
import torch
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUMMARY_MODEL = "Falconsai/text_summarization"


@st.cache_resource
def _load_summarizer(device_int):
    """Load and cache the summarization pipeline."""
    logger.info(f"Loading summarization model on device {device_int}")
    return pipeline("summarization", model=SUMMARY_MODEL, device=device_int)


@st.cache_resource
def _load_summary_tokenizer():
    """Load and cache the summarization tokenizer."""
    return AutoTokenizer.from_pretrained(SUMMARY_MODEL)


def chunk_text(text, max_tokens, tokenizer):
    """
    Splits text into chunks by tokenizing once, then splitting by token windows.
    Much faster than the per-word tokenization approach.
    """
    all_ids = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
    content_ids = all_ids[1:-1]  # strip BOS/EOS
    usable_max = max_tokens - 2  # leave room for special tokens

    chunks = []
    for i in range(0, len(content_ids), usable_max):
        chunk_ids = content_ids[i : i + usable_max]
        decoded = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if decoded:
            chunks.append(decoded)

    if not chunks:
        chunks.append(text)

    return chunks


def summarize_text(text, use_gpu=True, memory_fraction=0.8):
    """
    Summarize text using a Hugging Face pipeline with chunking support.
    
    Args:
        text (str): Text to summarize
        use_gpu (bool): Whether to use GPU if available
        memory_fraction (float): Fraction of GPU memory to use
    
    Returns:
        str: Summarized text
    """
    device = -1
    if use_gpu and torch.cuda.is_available():
        device = 0
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    logger.info(f"Using device {device} for summarization")
    
    try:
        summarizer = _load_summarizer(device)
        tokenizer = _load_summary_tokenizer()
        
        max_tokens = 512
        tokens = tokenizer(text, return_tensors='pt')
        num_tokens = len(tokens['input_ids'][0])
        
        if num_tokens > max_tokens:
            chunks = chunk_text(text, max_tokens, tokenizer)
            summaries = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
                summary_output = summarizer(
                    "summarize: " + chunk,
                    max_length=150,
                    min_length=30,
                    do_sample=False
                )
                summaries.append(summary_output[0]['summary_text'])
            
            if len(summaries) > 1:
                logger.info("Generating final summary from chunk summaries")
                combined_text = " ".join(summaries)
                return summarizer(
                    "summarize: " + combined_text,
                    max_length=150,
                    min_length=30,
                    do_sample=False
                )[0]['summary_text']
            return summaries[0]
        else:
            return summarizer(
                "summarize: " + text,
                max_length=150,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        if device != -1:
            logger.info("Falling back to CPU")
            return summarize_text(text, use_gpu=False, memory_fraction=memory_fraction)
        raise
