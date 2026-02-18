"""
Translation utilities for the Video Transcriber.
Provides functions for language detection and translation.
"""

import logging
import torch
from pathlib import Path
from transformers import pipeline, AutoTokenizer, M2M100ForConditionalGeneration
import whisper
import iso639
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from utils.gpu_utils import get_optimal_device
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

TRANSLATION_MODEL = "facebook/m2m100_418M"
LANGUAGE_DETECTION_MODEL = "papluca/xlm-roberta-base-language-detection"


@st.cache_resource
def _load_language_detector(model_name, device_int):
    """Load and cache the language detection pipeline."""
    logger.info(f"Loading language detection model: {model_name}")
    return pipeline("text-classification", model=model_name, device=device_int)


@st.cache_resource
def _load_translation_model(model_name, device_str):
    """Load and cache the M2M100 translation model and tokenizer."""
    logger.info(f"Loading translation model: {model_name} on {device_str}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device(device_str)
    model = model.to(device)
    return model, tokenizer


def get_language_name(code):
    """Get the language name from ISO code."""
    try:
        return iso639.languages.get(part1=code).name
    except (KeyError, AttributeError):
        try:
            return iso639.languages.get(part2b=code).name
        except (KeyError, AttributeError):
            return code


def detect_language(text, model=LANGUAGE_DETECTION_MODEL, use_gpu=True):
    """
    Detect the language of a text.
    
    Args:
        text (str): Text to detect language for
        model (str): Model to use for language detection
        use_gpu (bool): Whether to use GPU acceleration if available
        
    Returns:
        tuple: (language_code, confidence)
    """
    device = torch.device("cpu")
    if use_gpu and GPU_UTILS_AVAILABLE:
        device = get_optimal_device()
        device_arg = 0 if device.type == "cuda" else -1
    else:
        device_arg = -1
    
    try:
        classifier = _load_language_detector(model, device_arg)
        
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        result = classifier(text)[0]
        return result["label"], result["score"]
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return None, 0.0


def _translate_text_with_model(text, source_lang, target_lang, trans_model, tokenizer, device):
    """Translate text using a pre-loaded model and tokenizer."""
    tokenizer.src_lang = source_lang
    
    max_length = 512
    if len(text) > max_length:
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    else:
        chunks = [text]
    
    translated_chunks = []
    for chunk in chunks:
        encoded = tokenizer(chunk, return_tensors="pt").to(device)
        generated_tokens = trans_model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang),
            max_length=max_length
        )
        translated_chunk = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        translated_chunks.append(translated_chunk)
    
    return " ".join(translated_chunks)


def translate_text(text, source_lang=None, target_lang="en", model=TRANSLATION_MODEL, use_gpu=True):
    """
    Translate text from source language to target language.
    
    Args:
        text (str): Text to translate
        source_lang (str, optional): Source language code (auto-detect if None)
        target_lang (str): Target language code
        model (str): Model to use for translation
        use_gpu (bool): Whether to use GPU acceleration if available
        
    Returns:
        str: Translated text
    """
    if source_lang is None:
        detected_lang, confidence = detect_language(text, use_gpu=use_gpu)
        if detected_lang and confidence > 0.5:
            source_lang = detected_lang
            logger.info(f"Detected language: {get_language_name(source_lang)} ({source_lang}) with confidence {confidence:.2f}")
        else:
            logger.warning("Could not reliably detect language, defaulting to English")
            source_lang = "en"
    
    if source_lang == target_lang:
        logger.info(f"Source and target languages are the same ({source_lang}), skipping translation")
        return text
    
    device = torch.device("cpu")
    if use_gpu and GPU_UTILS_AVAILABLE:
        device = get_optimal_device()
    
    try:
        trans_model, tokenizer = _load_translation_model(model, str(device))
        return _translate_text_with_model(text, source_lang, target_lang, trans_model, tokenizer, device)
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        return text


def translate_segments(segments, source_lang=None, target_lang="en", use_gpu=True):
    """
    Translate transcript segments. Loads the model once and reuses for all segments.
    
    Args:
        segments (list): List of transcript segments
        source_lang (str, optional): Source language code (auto-detect if None)
        target_lang (str): Target language code
        use_gpu (bool): Whether to use GPU acceleration if available
        
    Returns:
        list: Translated segments
    """
    if not segments:
        return []
    
    if source_lang is None:
        combined_text = " ".join([segment["text"] for segment in segments])
        detected_lang, _ = detect_language(combined_text, use_gpu=use_gpu)
        source_lang = detected_lang if detected_lang else "en"
    
    if source_lang == target_lang:
        return segments
    
    device = torch.device("cpu")
    if use_gpu and GPU_UTILS_AVAILABLE:
        device = get_optimal_device()
    
    try:
        trans_model, tokenizer = _load_translation_model(TRANSLATION_MODEL, str(device))
        
        translated_segments = []
        for segment in segments:
            translated_text = _translate_text_with_model(
                segment["text"], source_lang, target_lang, trans_model, tokenizer, device
            )
            
            translated_segment = segment.copy()
            translated_segment["text"] = translated_text
            translated_segment["original_text"] = segment["text"]
            translated_segment["source_lang"] = source_lang
            translated_segment["target_lang"] = target_lang
            translated_segments.append(translated_segment)
        
        return translated_segments
    except Exception as e:
        logger.error(f"Error translating segments: {e}")
        return segments


def transcribe_and_translate(audio_path, whisper_model="base", target_lang="en", 
                            use_gpu=True, detect_source=True):
    """
    Transcribe audio and translate to target language.
    
    Args:
        audio_path (Path): Path to the audio file
        whisper_model (str): Whisper model size to use
        target_lang (str): Target language code
        use_gpu (bool): Whether to use GPU acceleration if available
        detect_source (bool): Whether to auto-detect source language
        
    Returns:
        tuple: (original_segments, translated_segments, original_transcript, translated_transcript)
    """
    from utils.transcription import _load_whisper_model
    
    audio_path = Path(audio_path)
    
    device = torch.device("cpu")
    if use_gpu and GPU_UTILS_AVAILABLE:
        device = get_optimal_device()
    
    try:
        logger.info(f"Transcribing audio with Whisper model: {whisper_model}")
        model = _load_whisper_model(whisper_model, str(device))
        
        if detect_source:
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(device if device.type != "mps" else "cpu")
            _, probs = model.detect_language(mel)
            source_lang = max(probs, key=probs.get)
            logger.info(f"Whisper detected language: {get_language_name(source_lang)} ({source_lang})")
            result = model.transcribe(str(audio_path), language=source_lang)
        else:
            result = model.transcribe(str(audio_path))
            source_lang = result.get("language", "en")
        
        original_segments = result["segments"]
        original_transcript = result["text"]
        
        if source_lang != target_lang:
            logger.info(f"Translating from {source_lang} to {target_lang}")
            translated_segments = translate_segments(
                original_segments,
                source_lang=source_lang,
                target_lang=target_lang,
                use_gpu=use_gpu
            )
            translated_transcript = " ".join([segment["text"] for segment in translated_segments])
        else:
            logger.info(f"Source and target languages are the same ({source_lang}), skipping translation")
            translated_segments = original_segments
            translated_transcript = original_transcript
        
        return original_segments, translated_segments, original_transcript, translated_transcript
    
    except Exception as e:
        logger.error(f"Error in transcribe_and_translate: {e}")
        return None, None, None, None