import networkx as nx
import matplotlib.pyplot as plt

def generate_knowledge_graph(all_data, forecast_reports):
    G = nx.DiGraph()

    # Add nodes and edges for all data sources
    for record in all_data:
        disaster_type = record.get("disaster_type", "")
        location = record.get("location", "")
        damage_percentage = record.get("damage_percentage", "")
        date = record.get("date", "")
        percentage_damage = record.get("percentage_damage", "")

        if disaster_type and location:
            G.add_node(disaster_type)
            G.add_node(location)
            edge_id = f"{disaster_type}_{location}"
            G.add_edge(disaster_type, location, edge_id=edge_id, damage_percentage=damage_percentage)

        if date and disaster_type and location:
            G.add_node(date)
            G.add_node(disaster_type)
            G.add_node(location)
            edge_id = f"{date}_{disaster_type}_{location}"
            G.add_edge(date, location, edge_id=edge_id, percentage_damage=percentage_damage)

    # Process the "disaster forecast reports" data and add it to the graph
    for report in forecast_reports:
        disaster_date = report.get("Disaster Date", "")
        location = report.get("Location", "")
        severity = report.get("Severity", "")
        disaster_type = report.get("Disaster Type", "")
        humidity_day = report.get("Humidity Day (%)", "")
        rainfalls = report.get("Rainfalls (mm)", "")
        temperature_max = report.get("Temperature Max (°C)", "")

        if disaster_type:
            G.add_node(disaster_type)
            G.add_node(location)
            edge_id = f"{disaster_type}_{location}"
            G.add_edge(disaster_type, location, edge_id=edge_id, severity=severity)

        if disaster_type:
            G.add_node(disaster_type)
            G.add_node(disaster_date)
            edge_id = f"{disaster_type}_{disaster_date}"
            G.add_edge(disaster_type, disaster_date, edge_id=edge_id, humidity_day=humidity_day, rainfalls=rainfalls, temperature_max=temperature_max)

    return G

def visualize_graph(graph):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=2000, font_size=10)

    # Separate edge labels for different data sources
    damage_edge_labels = {(u, v): f"{d['damage_percentage']}%" for u, v, d in graph.edges(data=True) if 'damage_percentage' in d}
    flood_damage_edge_labels = {(u, v): f"{d['percentage_damage']}%" for u, v, d in graph.edges(data=True) if 'percentage_damage' in d}
    forecast_edge_labels = {(u, v): d['severity'] for u, v, d in graph.edges(data=True) if 'severity' in d}
    forecast_attribute_labels = {(u, v): f"Type: {d['humidity_day']}, Humidity: {d['rainfalls']}%, Rainfalls: {d['temperature_max']}°C" for u, v, d in graph.edges(data=True) if 'humidity_day' in d and 'rainfalls' in d and 'temperature_max' in d}

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=damage_edge_labels, font_color='blue')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=flood_damage_edge_labels, font_color='red')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=forecast_edge_labels, font_color='green')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=forecast_attribute_labels, font_color='purple')

    plt.title("Assessment Knowledge Graph")
    plt.show()

