from src.engine.engine import Engine, EngineConfig
from src.ml.inference.master import MasterConfig, GenerationConfig
from pathlib import Path

config = EngineConfig(
    vector_db_path=Path('tmp/db'),
    number_of_remind_items=5,
    master_config=MasterConfig(
        path=Path('Qwen/Qwen3-0.6B').as_posix(),
        preambular="An ancient seal weakens, freeing horrors long imprisoned. The realm trembles, its hope fading with the dying light. You must journey where others fear to tread before the final dusk falls.",
        generation_config=GenerationConfig(temperature=0.7, max_new_tokens=128),
    ),
    ner_model_path=Path('models/ner'),
    embedding_model_path=Path(
        'sentence-transformers/all-MiniLM-L6-v2'
    )
)

engine = Engine(config, debug=True) 