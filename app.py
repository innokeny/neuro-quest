import streamlit as st
import time
from streamlit_agraph import agraph, Node, Edge, Config
# from src.engine.engine import Engine, EngineConfig
# from src.ml.inference.master import MasterConfig, GenerationConfig
# from pathlib import Path
# 
# config = EngineConfig(
#     vector_db_path=Path('tmp/db'),
#     number_of_remind_items=5,
#     master_config=MasterConfig(
#         path=Path('Qwen/Qwen3-0.6B'),
#         preambular=" An ancient seal weakens, freeing horrors long imprisoned. The realm trembles, its hope fading with the dying light. You must journey where others fear to tread before the final dusk falls.",
#         generation_config=GenerationConfig(temperature=0.7, max_new_tokens=128),
#     ),
#     ner_model_path=Path('models/ner'),
#     embedding_model_path=Path(
#         'sentence-transformers/all-MiniLM-L6-v2'
#     )
# )

# engine = Engine(config, debug=True) 



# FRONTEND
def chat_stream(prompt):
    response = f'You said, "{prompt}" ...interesting.'
    # with st.spinner("Wait for it...", show_time=True):
    #       response = engine.dialog(prompt)
    for char in response:
        yield char
        time.sleep(0.02)

def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]

st.title("DNDchat")

if "history" not in st.session_state:
    st.session_state.history = []

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = st.write_stream(chat_stream(prompt))
        st.feedback(
            "thumbs",
            key=f"feedback_{len(st.session_state.history)}",
            on_change=save_feedback,
            args=[len(st.session_state.history)],
        )
    st.session_state.history.append({"role": "assistant", "content": response})

with st.sidebar:
    st.title("Vector :red[Database] :sunglasses:")

    nodes = []
    edges = []
    nodes.append( Node(id="Spiderman", 
                    label="Peter Parker", 
                    size=25, 
                    shape="circularImage",
                    image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_spiderman.png") 
                ) # includes **kwargs
    nodes.append( Node(id="Captain_Marvel", 
                    size=25,
                    shape="circularImage",
                    image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_captainmarvel.png") 
                )
    edges.append( Edge(source="Captain_Marvel", 
                    label="friend_of", 
                    target="Spiderman", 
                    # **kwargs
                    ) 
                ) 

    config = Config(width=550,
                    height=950,
                    directed=True, 
                    physics=True, 
                    hierarchical=False,
                    # **kwargs
                    )

    return_value = agraph(nodes=nodes, 
                        edges=edges, 
                        config=config)