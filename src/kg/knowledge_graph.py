"""
Knowledge Graph component for storing and querying D&D entity relationships.
"""

from typing import Dict, List, Optional
from pathlib import Path
import networkx as nx
import json

class KnowledgeGraph:
    """Manages D&D entity relationships in a graph structure."""
    
    def __init__(self, graph_path: Optional[Path] = None):
        self.graph = nx.DiGraph()
        if graph_path and graph_path.exists():
            self.load_graph(graph_path)
    
    def load_graph(self, path: Path):
        """Load graph from file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
        except Exception as e:
            print(f"Error loading graph: {e}")
            self.graph = nx.DiGraph()
    
    def save_graph(self, path: Path):
        """Save graph to file."""
        data = nx.node_link_data(self.graph)
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict):
        """Add an entity to the graph."""
        self.graph.add_node(entity_id, type=entity_type, **properties)
    
    def add_relationship(self, source_id: str, target_id: str, relationship_type: str, properties: Dict = None):
        """Add a relationship between entities."""
        if properties is None:
            properties = {}
        self.graph.add_edge(source_id, target_id, type=relationship_type, **properties)
    
    def query_entities(self, entity_ids: List[str]) -> Dict:
        """
        Query the graph for relationships between entities.
        
        Args:
            entity_ids: List of entity IDs to query
            
        Returns:
            Dictionary containing:
            - Direct relationships between queried entities
            - Common neighbors
            - Path information
        """
        result = {
            "direct_relationships": [],
            "common_neighbors": {},
            "paths": []
        }
        
        # Get direct relationships
        for source in entity_ids:
            for target in entity_ids:
                if source != target and self.graph.has_edge(source, target):
                    edge_data = self.graph.get_edge_data(source, target)
                    result["direct_relationships"].append({
                        "source": source,
                        "target": target,
                        "type": edge_data.get("type"),
                        "properties": {k: v for k, v in edge_data.items() if k != "type"}
                    })
        
        # Get common neighbors
        for i, entity1 in enumerate(entity_ids):
            for entity2 in entity_ids[i+1:]:
                common = list(nx.common_neighbors(self.graph, entity1, entity2))
                if common:
                    result["common_neighbors"][f"{entity1}-{entity2}"] = common
        
        # Get paths between entities
        for source in entity_ids:
            for target in entity_ids:
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=3))
                        if paths:
                            result["paths"].append({
                                "source": source,
                                "target": target,
                                "paths": paths
                            })
                    except nx.NetworkXNoPath:
                        continue
        
        return result
    
    def get_entity_context(self, entity_id: str, depth: int = 2) -> Dict:
        """
        Get context for an entity by exploring its neighborhood.
        
        Args:
            entity_id: ID of the entity to explore
            depth: How many hops to explore from the entity
            
        Returns:
            Dictionary containing entity context information
        """
        context = {
            "entity": self.graph.nodes[entity_id],
            "neighbors": {},
            "relationships": []
        }
        
        # Get neighbors up to specified depth
        for node in nx.single_source_shortest_path_length(self.graph, entity_id, depth).keys():
            if node != entity_id:
                context["neighbors"][node] = self.graph.nodes[node]
        
        # Get relationships
        for _, target, data in self.graph.edges(entity_id, data=True):
            context["relationships"].append({
                "target": target,
                "type": data.get("type"),
                "properties": {k: v for k, v in data.items() if k != "type"}
            })
        
        return context 