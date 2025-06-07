# D&D Knowledge Processing Pipeline

A unified pipeline for processing D&D knowledge and generating contextually rich responses. The system combines Named Entity Recognition (NER), Knowledge Graph, Vector Store, and Large Language Model (LLM) components to provide deep, contextual understanding of D&D concepts.

## Architecture

The pipeline follows this flow:

1. **Input**: User prompt
2. **NER Processing**: Extracts D&D entities using BERT model
3. **Knowledge Graph**: Queries relationships between entities
4. **Vector Store**: Performs semantic search for relevant context
5. **LLM**: Generates final response using QWEN model

### Components

- **NER Processor**: Uses BERT model to identify D&D entities (characters, locations, items, etc.)
- **Knowledge Graph**: Stores and queries relationships between D&D entities
- **Vector Store**: Maintains semantic embeddings for fast similarity search
- **LLM Processor**: Generates responses using the QWEN model

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare model paths:
- NER model: `src/ner/model`
- LLM model: `src/llm/model`
- Vector store: `src/memory/vector_store.py`

## Usage

```python
from src.pipeline import UnifiedPipeline, PipelineConfig
from pathlib import Path

# Initialize pipeline
config = PipelineConfig(
    ner_model_path=Path("src/ner/model"),
    llm_model_path=Path("src/llm/model"),
    vector_store_path=Path("src/memory/vector_store.py"),
    knowledge_graph_path=Path("data/knowledge_graph.json")
)

pipeline = UnifiedPipeline(config)

# Process a query
response = pipeline.process("Tell me about the relationship between Drizzt and Bruenor")

# Add new knowledge
pipeline.add_knowledge(
    text="Drizzt Do'Urden is a drow ranger who became friends with Bruenor Battlehammer.",
    metadata={"source": "The Crystal Shard", "page": 42},
    entity_type="CHARACTER"
)

# Save state
pipeline.save_state(Path("data"))
```

## Entity Types

The system recognizes these D&D entity types:
- CHARACTER: Characters, NPCs, and creatures
- LOCATION: Places, realms, and locations
- ITEM: Magic items, equipment, and artifacts
- SPELL: Spells and magical abilities
- CLASS: Character classes and professions
- RACE: Races and species
- FACTION: Organizations and groups

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
