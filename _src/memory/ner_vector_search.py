import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from src.memory.vector_store import VectorStore

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'ner')

def extract_entities(text):
    """Извлекает сущности из текста с помощью NER-модели (только text и type)"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="first")
    ner_results = nlp(text)
    entities = []
    for ent in ner_results:
        entities.append({
            "text": ent['word'],
            "type": ent.get('entity_group', 'UNKNOWN')
        })
    return entities

def search_with_ner(prompt, vector_store: VectorStore):
    """Проводит NER и ищет по векторной базе (без фильтрации по id)"""
    entities = extract_entities(prompt)
    query = prompt  # Можно доработать: query = ...
    results = vector_store.search(query=query)
    return results, entities

# Пример использования:
if __name__ == "__main__":
    store = VectorStore(storage_path="data/vector_store")
    prompt = "The Rogue sneaks behind the dragon"
    results, entities = search_with_ner(prompt, store)
    print("Извлечённые сущности:", entities)
    print("Результаты поиска:")
    for r in results:
        print(r.text)
