import random
import networkx as nx
import pandas as pd

paradigm_list = ["Interaction and Dialogue", "Concepts and Ideas", "Character and Role", "Settings and Scenes",
                "Actions and Tasks", "Imaginative Elements", "Technical Terms", "Emotional Expressions"]

file_path = 'New_frequent_semantic_categorized.xlsx'


def build_data(data):
    data_processed = data.copy()
    for col in data_processed.columns:
        none_count = 0
        for i in range(len(data_processed)):
            if data_processed[col][i] == '[None]':
                none_count += 1
                if none_count > 1:
                    data_processed.at[i, col] = None
    return data_processed


def build_graph(G, data_processed):
    start_node = random.choice(list(G.nodes))
    random_walk_path = random_walk(G, start_node, len(G.nodes))

    # Creating a dictionary from the random walk path, ensuring all categories are covered
    random_walk_dict = {}

    # Select sentences from categories in the random walk
    for category in random_walk_path:
        sentences = data_processed[category].dropna().tolist()
        if sentences and sentences != ['[None]']:
            selected_sentence = random.choice(sentences)
        else:
            selected_sentence = '[None]'
        random_walk_dict[category] = selected_sentence

    # Ensuring all categories are included
    for category in G.nodes:
        if category not in random_walk_dict:
            sentences = data_processed[category].dropna().tolist()
            if sentences and sentences != ['[None]']:
                selected_sentence = random.choice(sentences)
            else:
                selected_sentence = '[None]'
            random_walk_dict[category] = selected_sentence

    return random_walk_dict

def random_walk(graph, start_node, steps):

    path = [start_node]
    for _ in range(steps - 1):
        neighbors = list(graph.neighbors(path[-1]))
        if neighbors:
            next_node = random.choice(neighbors)
            path.append(next_node)
        else:
            break
    return path

def add_edges_based_on_shared_content(graph, dataframe):
    for i in range(len(dataframe)):
        row = dataframe.iloc[i]
        for col1 in dataframe.columns:
            for col2 in dataframe.columns:
                if col1 != col2 and pd.notna(row[col1]) and pd.notna(row[col2]) and row[col1] != '[None]' and row[
                    col2] != '[None]':
                    graph.add_edge(col1, col2)


def random_walk_function(file_path='New_frequent_semantic_categorized.xlsx'):
    data = pd.read_excel(file_path)
    data_processed = build_data(data)
    G = nx.Graph()
    G.add_nodes_from(data.columns)
    add_edges_based_on_shared_content(G, data_processed)
    random_walk_dict = build_graph(G, data_processed)
    return random_walk_dict


if __name__ == '__main__':
    random_walk_dict = random_walk_function()