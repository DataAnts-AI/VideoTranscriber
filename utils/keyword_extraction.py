"""
Keyword extraction utilities for the OBS Recording Transcriber.
Provides functions to extract keywords and link them to timestamps.
"""

import logging
import re
import torch
import numpy as np
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import GPU utilities, but don't fail if not available
try:
    from utils.gpu_utils import get_optimal_device
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# Default models
NER_MODEL = "dslim/bert-base-NER"


def extract_keywords_tfidf(text, max_keywords=10, ngram_range=(1, 2)):
    """
    Extract keywords using TF-IDF.
    
    Args:
        text (str): Text to extract keywords from
        max_keywords (int): Maximum number of keywords to extract
        ngram_range (tuple): Range of n-grams to consider
        
    Returns:
        list: List of (keyword, score) tuples
    """
    try:
        # Preprocess text
        text = text.lower()
        
        # Remove common stopwords
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                    'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
                    'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for',
                    'is', 'of', 'while', 'during', 'to', 'from', 'in', 'out', 'on', 'off', 'by'}
        
        # Create sentences for better TF-IDF analysis
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # Apply TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words=stopwords,
            ngram_range=ngram_range
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF score across all sentences
            avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top keywords
            keywords = [(feature_names[i], avg_tfidf[i]) for i in avg_tfidf.argsort()[::-1]]
            
            # Filter out single-character keywords and limit to max_keywords
            keywords = [(k, s) for k, s in keywords if len(k) > 1][:max_keywords]
            
            return keywords
        except ValueError as e:
            logger.warning(f"TF-IDF extraction failed: {e}")
            return []
    
    except Exception as e:
        logger.error(f"Error extracting keywords with TF-IDF: {e}")
        return []


def extract_named_entities(text, model=NER_MODEL, use_gpu=True):
    """
    Extract named entities from text.
    
    Args:
        text (str): Text to extract entities from
        model (str): Model to use for NER
        use_gpu (bool): Whether to use GPU acceleration if available
        
    Returns:
        list: List of (entity, type) tuples
    """
    # Configure device
    device = torch.device("cpu")
    if use_gpu and GPU_UTILS_AVAILABLE:
        device = get_optimal_device()
        device_arg = 0 if device.type == "cuda" else -1
    else:
        device_arg = -1
    
    try:
        # Initialize the pipeline
        ner_pipeline = pipeline("ner", model=model, device=device_arg, aggregation_strategy="simple")
        
        # Split text into manageable chunks if too long
        max_length = 512
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        else:
            chunks = [text]
        
        # Process each chunk
        all_entities = []
        for chunk in chunks:
            entities = ner_pipeline(chunk)
            all_entities.extend(entities)
        
        # Extract entity text and type
        entity_info = [(entity["word"], entity["entity_group"]) for entity in all_entities]
        
        return entity_info
    except Exception as e:
        logger.error(f"Error extracting named entities: {e}")
        return []


def find_keyword_timestamps(segments, keywords):
    """
    Find timestamps for keywords in transcript segments.
    
    Args:
        segments (list): List of transcript segments with timing info
        keywords (list): List of keywords to find
        
    Returns:
        dict: Dictionary mapping keywords to lists of timestamps
    """
    keyword_timestamps = {}
    
    # Convert keywords to lowercase for case-insensitive matching
    if isinstance(keywords[0], tuple):
        # If keywords is a list of (keyword, score) tuples
        keywords_lower = [k.lower() for k, _ in keywords]
    else:
        # If keywords is just a list of keywords
        keywords_lower = [k.lower() for k in keywords]
    
    # Process each segment
    for segment in segments:
        segment_text = segment["text"].lower()
        start_time = segment["start"]
        end_time = segment["end"]
        
        # Check each keyword
        for i, keyword in enumerate(keywords_lower):
            if keyword in segment_text:
                # Get the original case of the keyword
                original_keyword = keywords[i][0] if isinstance(keywords[0], tuple) else keywords[i]
                
                # Initialize the list if this is the first occurrence
                if original_keyword not in keyword_timestamps:
                    keyword_timestamps[original_keyword] = []
                
                # Add the timestamp
                keyword_timestamps[original_keyword].append({
                    "start": start_time,
                    "end": end_time,
                    "context": segment["text"]
                })
    
    return keyword_timestamps


def extract_keywords_from_transcript(transcript, segments, max_keywords=15, use_gpu=True):
    """
    Extract keywords from transcript and link them to timestamps.
    
    Args:
        transcript (str): Full transcript text
        segments (list): List of transcript segments with timing info
        max_keywords (int): Maximum number of keywords to extract
        use_gpu (bool): Whether to use GPU acceleration if available
        
    Returns:
        tuple: (keyword_timestamps, entities_with_timestamps)
    """
    try:
        # Extract keywords using TF-IDF
        tfidf_keywords = extract_keywords_tfidf(transcript, max_keywords=max_keywords)
        
        # Extract named entities
        entities = extract_named_entities(transcript, use_gpu=use_gpu)
        
        # Count entity occurrences and get the most frequent ones
        entity_counter = Counter([entity for entity, _ in entities])
        top_entities = [(entity, count) for entity, count in entity_counter.most_common(max_keywords)]
        
        # Find timestamps for keywords and entities
        keyword_timestamps = find_keyword_timestamps(segments, tfidf_keywords)
        entity_timestamps = find_keyword_timestamps(segments, top_entities)
        
        return keyword_timestamps, entity_timestamps
    
    except Exception as e:
        logger.error(f"Error extracting keywords from transcript: {e}")
        return {}, {}


def generate_keyword_index(keyword_timestamps, entity_timestamps=None):
    """
    Generate a keyword index with timestamps.
    
    Args:
        keyword_timestamps (dict): Dictionary mapping keywords to timestamp lists
        entity_timestamps (dict, optional): Dictionary mapping entities to timestamp lists
        
    Returns:
        str: Formatted keyword index
    """
    lines = ["# Keyword Index\n"]
    
    # Add keywords section
    if keyword_timestamps:
        lines.append("## Keywords\n")
        for keyword, timestamps in sorted(keyword_timestamps.items()):
            if timestamps:
                times = [f"{int(ts['start'] // 60):02d}:{int(ts['start'] % 60):02d}" for ts in timestamps]
                lines.append(f"- **{keyword}**: {', '.join(times)}\n")
    
    # Add entities section
    if entity_timestamps:
        lines.append("\n## Named Entities\n")
        for entity, timestamps in sorted(entity_timestamps.items()):
            if timestamps:
                times = [f"{int(ts['start'] // 60):02d}:{int(ts['start'] % 60):02d}" for ts in timestamps]
                lines.append(f"- **{entity}**: {', '.join(times)}\n")
    
    return "".join(lines)


def generate_interactive_transcript(segments, keyword_timestamps=None, entity_timestamps=None):
    """
    Generate an interactive transcript with keyword highlighting.
    
    Args:
        segments (list): List of transcript segments with timing info
        keyword_timestamps (dict, optional): Dictionary mapping keywords to timestamp lists
        entity_timestamps (dict, optional): Dictionary mapping entities to timestamp lists
        
    Returns:
        str: HTML formatted interactive transcript
    """
    # Combine keywords and entities
    all_keywords = {}
    if keyword_timestamps:
        all_keywords.update(keyword_timestamps)
    if entity_timestamps:
        all_keywords.update(entity_timestamps)
    
    # Generate HTML
    html = ["<div class='interactive-transcript'>"]
    
    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        # Format timestamp
        timestamp = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"
        
        # Add speaker if available
        speaker = segment.get("speaker", "")
        speaker_html = f"<span class='speaker'>[{speaker}]</span> " if speaker else ""
        
        # Highlight keywords in text
        highlighted_text = text
        for keyword in all_keywords:
            # Use regex to match whole words only
            pattern = r'\b' + re.escape(keyword) + r'\b'
            replacement = f"<span class='keyword' data-keyword='{keyword}'>{keyword}</span>"
            highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
        
        # Add segment to HTML
        html.append(f"<p class='segment' data-start='{start_time}' data-end='{end_time}'>")
        html.append(f"<span class='timestamp'>{timestamp}</span> {speaker_html}{highlighted_text}")
        html.append("</p>")
    
    html.append("</div>")
    
    return "\n".join(html)


def create_keyword_cloud_data(keyword_timestamps, entity_timestamps=None):
    """
    Create data for a keyword cloud visualization.
    
    Args:
        keyword_timestamps (dict): Dictionary mapping keywords to timestamp lists
        entity_timestamps (dict, optional): Dictionary mapping entities to timestamp lists
        
    Returns:
        list: List of (keyword, weight) tuples for visualization
    """
    cloud_data = []
    
    # Process keywords
    for keyword, timestamps in keyword_timestamps.items():
        weight = len(timestamps)  # Weight by occurrence count
        cloud_data.append((keyword, weight))
    
    # Process entities if provided
    if entity_timestamps:
        for entity, timestamps in entity_timestamps.items():
            weight = len(timestamps) * 1.5  # Give entities slightly higher weight
            cloud_data.append((entity, weight))
    
    return cloud_data 