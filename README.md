# Neuro Quest

A unified pipeline for processing D&D knowledge and generating contextually rich responses. The system combines Named Entity Recognition (NER), Vector Store, and Large Language Model (LLM) components to provide deep, contextual understanding of D&D concepts.

## Project structure
```
.
├── app.py                          # Streamlit app 
├── setup.py                        # Project config
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies            
├── notebooks                       
│   ├── nb_session.ipynb            # Testing in notebook
│   ├── eval.ipynb                  # Model evaluation
│   ├── llm_train.ipynb             # LLM training
│   └── ner_train.ipynb             # Python dependencies            
├── parser                       
│   └── parser.py                   # Data parser
└── src                             
    ├── engine
    │   └── engine.py               # Main pipeline engine
    ├── memory
    │   ├── db
    │   │   └── storage.py          # Data storage
    │   └── short
    │       └── memory.py           # Short-term memory
    ├── ml
    │   └── inference
    │       ├── embedding.py        # Embedding logic
    │       ├── master.py           # LLM inference
    │       └── ner.py              # NER inference
    └── session
        └── notebook.py             # Notebook session logic
```

## Architecture

The pipeline follows this flow:

1. **Input**: User prompt
2. **NER Processing**: Extracts D&D entities using BERT model
4. **Vector Store**: Performs semantic search for relevant context
5. **LLM**: Generates final response using QWEN model

### Components

- **NER Processor**: Uses BERT model to identify D&D entities (characters, locations, items, etc.)
- **Vector Store**: Maintains semantic embeddings for fast similarity search
- **LLM Processor**: Generates responses using the QWEN model

## Setup

1. Clone the repository:
```bash
git clone <repo_url>
cd neuro-quest
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. (Optional) Install Jupyter for working with notebooks:
```bash
pip install jupyter
```
4. Make sure you have the required models in the `models/` folder


## Usage

You can test the app online via Streamlit:

[Try Neuro Quest on Streamlit](https://neuro-quest.streamlit.app/)

_Note: The online demo runs on CPU, so response generation may be slow._

Or run locally as shown below:

```python
from src.engine.engine import Engine, EngineConfig
from src.ml.inference.master import MasterConfig, GenerationConfig
from pathlib import Path

vector_db_path = Path('tmp/db')
vector_db_path.mkdir(parents=True, exist_ok=True)

config = EngineConfig(
    vector_db_path=Path('tmp/db'),
    number_of_remind_items=5,
    master_config=MasterConfig(
        path=Path('rar7/Qwen3-1.7B-dnd'),
        preambular=" An ancient seal weakens, freeing horrors long imprisoned. The realm trembles, its hope fading with the dying light. You must journey where others fear to tread before the final dusk falls.",
        generation_config=GenerationConfig(temperature=0.7, max_new_tokens=128),
    ),
    ner_model_path=Path('rar7/EntityBERT-dnd'),
    embedding_model_path=Path(
        'sentence-transformers/all-MiniLM-L6-v2'
    )
)

engine = Engine(config, debug=True) 

# Example user prompt
user_prompt = "Describe the magical properties of the Staff of Power found in Waterdeep."

# Get the response
response = engine.dialog(user_prompt)
print(response.text)
```

Or run the Streamlit app:
```bash
streamlit run app.py
```


## Entity Types

The system recognizes the following D&D entity types:

- **UNKNOWN**: Entities that do not fit any known category or could not be classified.
- **LOC** (Location): Places, realms, cities, dungeons, and any geographical or spatial locations in the D&D universe.
- **ITEM**: Magic items, equipment, artifacts, weapons, armor, and other objects of significance.
- **SPELL**: Spells, magical abilities, and supernatural powers used by characters or creatures.
- **ACTION**: Actions, maneuvers, or special moves performed by characters (e.g., "Attack", "Dodge").
- **STATUS**: Conditions, effects, or states that can affect characters or entities (e.g., "Poisoned", "Invisible").
- **ORG** (Organization): Factions, guilds, cults, or any organized groups within the D&D world.
- **MON** (Monster): Monsters, creatures, and non-player characters (NPCs) that are not classified as persons.
- **PER** (Person): Named characters, heroes, villains, or any individual person in the D&D setting.
