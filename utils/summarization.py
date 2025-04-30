from transformers import pipeline, AutoTokenizer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUMMARY_MODEL = "Falconsai/text_summarization"

def chunk_text(text, max_tokens, tokenizer):
    """
    Splits the text into a list of chunks based on token limits.
    
    Args:
        text (str): Text to chunk
        max_tokens (int): Maximum tokens per chunk
        tokenizer (AutoTokenizer): Tokenizer to use
        
    Returns:
        list: List of text chunks
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        hypothetical_length = current_length + len(tokenizer(word, return_tensors='pt')['input_ids'][0]) - 2
        if hypothetical_length <= max_tokens:
            current_chunk.append(word)
            current_length = hypothetical_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(tokenizer(word, return_tensors='pt')['input_ids'][0]) - 2
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
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
    # Determine device
    device = -1  # Default to CPU
    if use_gpu and torch.cuda.is_available():
        device = 0  # Use first GPU
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    logger.info(f"Using device {device} for summarization")
    
    try:
        # Initialize the pipeline and tokenizer
        summarizer = pipeline("summarization", model=SUMMARY_MODEL, device=device)
        tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL)
        
        # Check if text needs to be chunked
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
            
            # If multiple chunks, summarize the combined summaries
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
        # Fallback to CPU if GPU fails
        if device != -1:
            logger.info("Falling back to CPU")
            return summarize_text(text, use_gpu=False, memory_fraction=memory_fraction)
        raise
