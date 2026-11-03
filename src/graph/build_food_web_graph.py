from pathlib import Path
import pandas as pd
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NODES_FILE = PROJECT_ROOT / "data/processed/food_web_nodes.csv"
EDGES_FILE = PROJECT_ROOT / "data/processed/food_web_edges.csv"

OUTPUT_DIR = PROJECT_ROOT / "data/processed/graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

nodes = pd.read_csv(NODES_FILE)
edges = pd.read_csv(EDGES_FILE)

for network_id in sorted(nodes["network_id"].unique()):

    network_nodes = nodes[nodes["network_id"] == network_id]
    network_edges = edges[edges["network_id"] == network_id]

    graph = nx.DiGraph()

    for _, row in network_nodes.iterrows():
        graph.add_node(
            row["node_id"],
            species_name=row["species_name"],
            taxonomy_id=row["taxonomy_id"],
        )

    for _, row in network_edges.iterrows():
        graph.add_edge(
            row["source_node"],
            row["target_node"],
            interaction_type=row["interaction_type"],
            value=row["value"],
        )

    output_file = OUTPUT_DIR / f"network_{network_id}.graphml"

    nx.write_graphml(graph, output_file)

    print(
        f"Network {network_id}: "
        f"{graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

print("\n✅ Food-web graphs created.")