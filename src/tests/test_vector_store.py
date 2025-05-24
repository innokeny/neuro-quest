import pytest
import os
import shutil
from src.memory.vector_store import VectorStore


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory for tests"""
    storage_path = "data/test_vector_store"
    os.makedirs(storage_path, exist_ok=True)
    yield storage_path
    # Cleanup after tests
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)


def test_vector_store_basic_operations(temp_storage):
    # Initialize vector store with temporary storage
    store = VectorStore(storage_path=temp_storage)
    
    # Test adding entries
    test_texts = [
        "The party enters a dark dungeon",
        "A dragon appears from the shadows",
        "The wizard casts a fireball spell",
        "The rogue sneaks behind the dragon"
    ]
    
    # Add entries with metadata
    for i, text in enumerate(test_texts):
        store.add_entry(text, {"type": "game_event", "index": i})
    
    # Test search functionality
    query = "What's happening in the dungeon?"
    results = store.search(query, k=2)
    
    # Verify we got results
    assert len(results) > 0
    
    # Verify results contain our test texts
    result_texts = [r.text for r in results]
    assert any(text in result_texts for text in test_texts)
    
    # Test context retrieval
    context = store.get_context(query)
    assert isinstance(context, str)
    assert len(context) > 0
    
    # Test metadata preservation
    for result in results:
        assert "type" in result.metadata
        assert "index" in result.metadata
        assert result.metadata["type"] == "game_event"


def test_vector_store_empty(temp_storage):
    store = VectorStore(storage_path=temp_storage)
    
    # Test search with empty store
    results = store.search("test query")
    assert len(results) == 0
    
    # Test context with empty store
    context = store.get_context("test query")
    assert context == ""


def test_vector_store_similarity(temp_storage):
    store = VectorStore(storage_path=temp_storage)
    
    # Add semantically similar texts
    store.add_entry("The dragon breathes fire")
    store.add_entry("The dragon uses its fire breath")
    store.add_entry("The party finds a treasure chest")
    
    # Search for similar concept
    results = store.search("The dragon attacks with flames")
    
    # Verify that fire-related entries are ranked higher
    result_texts = [r.text for r in results]
    assert "The dragon breathes fire" in result_texts
    assert "The dragon uses its fire breath" in result_texts


def test_vector_store_tavern_context(temp_storage):
    """Test that the vector store can find relevant tavern-related context"""
    store = VectorStore(storage_path=temp_storage)
    
    # Add some historical context about the tavern
    store.add_entry("Ранее игрок украл монету в этой таверне", {"type": "historical_event"})
    store.add_entry("В таверне сидит пьяный гном", {"type": "scene_description"})
    store.add_entry("Игрок нашел сокровище в подземелье", {"type": "historical_event"})
    
    # Test search for tavern-related context
    query = "Я зашел в таверну"
    results = store.search(query, k=2)
    
    # Verify we got relevant results
    result_texts = [r.text for r in results]
    assert "Ранее игрок украл монету в этой таверне" in result_texts
    assert "В таверне сидит пьяный гном" in result_texts
    assert "Игрок нашел сокровище в подземелье" not in result_texts
    
    # Test context retrieval
    context = store.get_context(query)
    assert "таверне" in context.lower()
    assert "монету" in context.lower()
    assert "гном" in context.lower() 