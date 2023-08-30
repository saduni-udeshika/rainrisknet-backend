import networkx as nx
import matplotlib.pyplot as plt

def generate_knowledge_graph(assessment_data):
    G = nx.DiGraph()

    for record in assessment_data:
        disaster_type = record["disaster_type"]
        location = record["location"]
        damage_percentage = record["damage_percentage"]

        G.add_node(disaster_type)
        G.add_node(location)
        G.add_edge(disaster_type, location, damage_percentage=damage_percentage)

    return G

def visualize_graph(graph):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): f"{d['damage_percentage']}%" for u, v, d in graph.edges(data=True)})
    plt.title("Assessment Knowledge Graph")
    plt.show()
