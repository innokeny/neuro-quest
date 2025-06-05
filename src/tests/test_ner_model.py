import os
import pytest
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'ner')

def test_ner_model_load():
    """Проверка загрузки модели и токенизатора"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    assert tokenizer is not None
    assert model is not None

def test_ner_inference_english():
    """Проверка инференса на английском тексте"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    text = "John Smith lives in New York City."
    result = nlp(text)
    assert isinstance(result, list)
    assert any('entity_group' in ent for ent in result)

def test_ner_inference_russian():
    """Проверка инференса на русском тексте (ожидаем, что модель может не распознать сущности)"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    text = "Иван Иванов живет в Москве."
    result = nlp(text)
    assert isinstance(result, list)
    # Проверяем, что пайплайн не падает и возвращает список (даже если он пустой)
