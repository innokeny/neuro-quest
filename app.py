import streamlit as st
import time
from streamlit_agraph import agraph, Node, Edge, Config
# from src.engine.engine import Engine, EngineConfig
# from src.ml.inference.master import MasterConfig, GenerationConfig
# from pathlib import Path
from setup import engine


# FRONTEND
def chat_stream(prompt):
    with st.spinner("Wait for it...", show_time=True):
        response = engine.dialog(prompt)
    for char in response.text:
        yield char
        time.sleep(0.02)
    return response

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
    st.title("Session :red[Net] :spider:")

    # Create nodes and edges for NER entities
    nodes = []
    edges = []
    node_colors = {
        "ITEM": "#FF9999",  # Light red
        "LOC": "#99FF99",   # Light green
        "PER": "#9999FF",   # Light blue
        "MON": "#FF99FF",   # Light purple
        "ORG": "#FFFF99",   # Light yellow
    }
    edge_colors = {
        "ACTION": "#FF0000",  # Red
        "STATUS": "#00FF00",  # Green
        "SPELL": "#0000FF",   # Blue
    }
    
    # Add player node
    player_node = Node(
        id="player",
        label="Player",
        size=30,
        shape="dot",
        color="#FFD700"  # Gold color for player
    )
    nodes.append(player_node)
    
    # First pass: create all nodes
    for item in engine.db._items:
        if "type" in item.meta and "text" in item.meta:
            entity_type = item.meta["type"]
            entity_text = item.meta["text"]
            
            # Create node for entity if it's a relevant type
            if entity_type in node_colors:
                node_id = f"{entity_type}_{entity_text}"
                if not any(n.id == node_id for n in nodes):
                    node = Node(
                        id=node_id,
                        label=entity_text,
                        size=25,
                        shape="dot",
                        color=node_colors[entity_type]
                    )
                    nodes.append(node)
                
                # Create edge from player to entity
                edge = Edge(
                    source="player",
                    target=node_id,
                    label="interacts",
                    color="#999999"  # Gray for basic interaction
                )
                edges.append(edge)
    
    # Second pass: create edges for actions/statuses/spells
    for item in engine.db._items:
        if "type" in item.meta and "text" in item.meta:
            entity_type = item.meta["type"]
            entity_text = item.meta["text"]
            
            # Create edges for actions/statuses/spells
            if entity_type in edge_colors:
                # Find the closest entity in the same text
                for other_item in engine.db._items:
                    if other_item.text == item.text and "type" in other_item.meta and other_item.meta["type"] in node_colors:
                        source_id = f"{other_item.meta['type']}_{other_item.meta['text']}"
                        if any(n.id == source_id for n in nodes):
                            edge = Edge(
                                source=source_id,
                                target="player",
                                label=entity_text,
                                color=edge_colors[entity_type],
                                type="CURVE_SMOOTH"
                            )
                            edges.append(edge)

    config = Config(
        width=550,
        height=950,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeSpacing=150,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={'labelProperty': 'label'},
        link={'labelProperty': 'label', 'renderLabel': True}
    )

    return_value = agraph(nodes=nodes, edges=edges, config=config)