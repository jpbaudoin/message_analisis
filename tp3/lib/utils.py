import textwrap

import emoji
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from dateutil.parser import parse
# detect the laguage of the text in the field mesaage_text
from langdetect import LangDetectException, detect, detect_langs


def print_side_by_side(message1, message2, width=30, linebreak_symbol='↵'):
    # Split both messages into lines, preserving line breaks
    # replace linebreaks with a symbol and a linebreak
    message1 = message1.replace('\n', f'{linebreak_symbol}\n')
    message2 = message2.replace('\n', f'{linebreak_symbol}\n')
    lines1 = message1.splitlines()
    lines2 = message2.splitlines()

    # Apply textwrap.wrap to each individual line while preserving line breaks
    wrapped_lines1 = [textwrap.wrap(line, width=width) if line else [linebreak_symbol] for line in lines1]
    wrapped_lines2 = [textwrap.wrap(line, width=width) if line else [linebreak_symbol] for line in lines2]

    # Flatten the lists of wrapped lines
    flattened_lines1 = [item for sublist in wrapped_lines1 for item in (sublist or [''])]
    flattened_lines2 = [item for sublist in wrapped_lines2 for item in (sublist or [''])]

    # Get the maximum number of lines between both messages
    max_lines = max(len(flattened_lines1), len(flattened_lines2))

    # Pad messages with empty lines if necessary
    flattened_lines1 += [''] * (max_lines - len(flattened_lines1))
    flattened_lines2 += [''] * (max_lines - len(flattened_lines2))

    # Print both messages side by side
    for line1, line2 in zip(flattened_lines1, flattened_lines2):
        print(f'{line1:<{width}} | {line2:<{width}}')


def print_top(df, field, title, top=10):
    print(f"Top {top} {title} ({field}):")
    # get the top 10 most frequent values
    top_list = df[field].value_counts(dropna=False)
    top_list.index = top_list.index.fillna('N/A')
    top_list = top_list.sort_values(ascending=False)
    # count the total number of messages
    total = top_list.sum()
    #count the total without nan
    total_no_nan = top_list.sum() - top_list.get('N/A', 0)
    percentage = round((top_list / total) * 100, 2)
    percentage_no_nan = round((top_list / total_no_nan) * 100, 2)
    acum_percent = 0
    acum_percent_wna = 0
    order = 0
    for idx, count, percent in zip(top_list.index[0:top], top_list.values[0:top], percentage.values[0:top]):
        # add acumulated percentage
        acum_percent += percent
        order += 1 

        if total_no_nan == total:
            # print with two decimal places
            print(f"{order:02d} - {idx}: {count} \t ({percent}% - {acum_percent:.2f}%)")
        else:
            if idx == 'N/A':
                print(f"{order:02d} - {idx}: {count} \t ({percent}%  - {acum_percent:.2f}%) \t (NaN)")
            else:
                percent_wna = percentage_no_nan[idx]
                acum_percent_wna += percent_wna
                print(f"{order:02d} - {idx}: {count} \t ({percent}%  - {acum_percent:.2f}%) \t (W/NaN {percent_wna}% - {acum_percent_wna:.2f}%)")


def clean_lbr(text):
    # Replace continuous line breaks with a single line break
    try:
        cleaned_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    except Exception as e:
        print(text, type(text))
        print(e)
        cleaned_text = text
        raise e
    return cleaned_text
    

def remove_emojis(text):
    return emoji.replace_emoji(text, '')

def count_line_breaks(text):
    return text.count('\n')


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException as e:
        print(e, text) 
        return "Unknown"
    else:
        print(e, text)
        return "Error langdetect"

def detect_multiple_languages(text):
    try:
        return detect_langs(text)
    except LangDetectException as e:
        print(e, text) 
        return "Unknown"
    else:
        print(e, text)
        return "Error langdetect"


# Clustering

def plot_clusters(df_clusters, k=0.3):
    # Create a graph
    G = nx.Graph()
    # Add edges with weights based on distance and assign the cluster id
    for _, row in df_clusters.iterrows():
        if row['id_1'] not in G.nodes:
            G.add_node(row['id_1'], c_id=row['c_id'], c_grp=row['c_grp'], chat_id=row['chat_id_1'], label=row['chat_id_1'])

        if row['id_2'] not in G.nodes:
            G.add_node(row['id_2'], c_id=row['c_id'], c_grp=row['c_grp'], chat_id=row['chat_id_2'], label=row['chat_id_2'])

        G.add_edge(row['id_1'], row['id_2'], weight=(row['distance']** (1/3))*3, c_id=row['c_id'])

    # Get unique clusters and define colors
    unique_clusters = df_clusters['c_id'].unique()
    palette = sns.color_palette('hsv', len(unique_clusters))
    cluster_colors = {c_id: palette[i] for i, c_id in enumerate(unique_clusters)}

    # Get unique chat_id and define colors
    unique_chat_id1 = df_clusters['chat_id_1'].unique()
    unique_chat_id2 = df_clusters['chat_id_2'].unique()
    # Merge chat_id1 and chat_id2
    unique_chat_id = list(set(list(unique_chat_id1) + list(unique_chat_id2)))
    palette_chat_id = sns.color_palette('Paired', len(unique_chat_id))
    chat_id_colors = {unique_chat_id[i]: palette for i, palette in enumerate(palette_chat_id)}

    # Generate positions for each node with increased `k` to separate clusters
    pos = nx.spring_layout(G, seed=42, k=k)  # Increase `k` for more separation between clusters

    # Plot the graph
    plt.figure(figsize=(12, 10))
    axes = plt.gca()

    # Draw edges with widths based on distance
    for c_id in unique_clusters:
        edges_in_cluster = [(u, v) for u, v, d in G.edges(data=True) if d['c_id'] == c_id]
        similarities = [G[u][v]['weight'] for u, v in edges_in_cluster]
        widths = [d if d > 0.1 else 0.1 for d in similarities]

        nx.draw_networkx_edges(G, pos, edgelist=edges_in_cluster, width=widths, alpha=0.6)

    # Draw nodes with colors based on their chat_id
    node_colors = [chat_id_colors[G.nodes[n]['chat_id']] if 'chat_id' in G.nodes[n] else 'lightblue' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors='black')

    # Encircle clusters by drawing a shaded circle around nodes in each cluster
    for c_id in unique_clusters:
        cluster_nodes = [n for n in G.nodes if G.nodes[n]['c_id'] == c_id]
        if cluster_nodes:  # Draw the cluster boundary if there are nodes
            x_values = [pos[n][0] for n in cluster_nodes]
            y_values = [pos[n][1] for n in cluster_nodes]
            x_center = sum(x_values) / len(x_values)
            y_center = sum(y_values) / len(y_values)
            radius = max([((pos[n][0] - x_center) ** 2 + (pos[n][1] - y_center) ** 2) ** 0.5 for n in cluster_nodes])
            # Draw shaded circle (filled with transparency)
            circle = plt.Circle((x_center, y_center), radius * 1.2, color=cluster_colors[c_id], fill=True, alpha=0.15)
            axes.add_patch(circle)

    # Draw labels for the nodes
    labels = nx.get_node_attributes(G, 'chat_id')  # get labels 
    nx.draw_networkx_labels(G, pos, font_size=12, labels=labels)

    # # Create a legend for the clusters (circle legend)
    # cluster_legend = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {c_id}',
    #                              markerfacecolor=cluster_colors[c_id], markersize=10, alpha=0.5) for c_id in unique_clusters]

    # Create a legend for the chat_id colors
    chat_legend = [plt.Line2D([0], [0], marker='o', color='w', label=f'Chat ID: {chat_id}',
                            markerfacecolor=chat_id_colors[chat_id], markersize=10, alpha=1) for chat_id in unique_chat_id]

    # Add both legends using `add_artist()`
    # legend2 = plt.legend(handles=cluster_legend, title="Clusters", loc='upper right', bbox_to_anchor=(1, 1))
    #  bbox_to_anchor=(Y, X)
    legend1 = plt.legend(handles=chat_legend, title="Chat IDs", loc='upper left', bbox_to_anchor=(0, 0.5))

    # axes.add_artist(legend2)
    axes.add_artist(legend1)

    plt.title('Graph Visualization of Message Clusters by Distance and Chat ID', size=15)
    plt.tight_layout()

    plt.show()



# generate a function that prints the messages  with similarities > 0.3
def check_sim_clusters(clusters_compare_df, sim_threshold=[0.3, 0.4], width=50):
    flt = (clusters_compare_df['similarity'] >= sim_threshold[0]) & (clusters_compare_df['similarity'] <= sim_threshold[1])
    # filter by chat_id_1 != chat_id_2
    flt = flt & (clusters_compare_df['chat_id_1'] != clusters_compare_df['chat_id_2'])
    df_comp_flt = clusters_compare_df[flt].head(50).tail(10)
    # loop over df_comp_flt and print the messages
    for idx, row in df_comp_flt.iterrows():
        print(f" --- {row['c_id']} ---")
        print(f"Similarity: {row['similarity']}")
        # print 100 times the character '-'
        print("_" * (width*2+1))
        print_side_by_side(f'Chat ID: {row["chat_id_1"]}', f'Chat ID: {row["chat_id_2"]}', width=width)
        print_side_by_side(row['message_text_clean_utf8_1'], row['message_text_clean_utf8_2'], width=width)
        print("_" * (width*2+1))
        print()


# generate a function that prints the messages  with similarities > 0.3
def check_sim_chats(clusters_compare_df, chat_a, chat_b, width=50, records=10, offset=0):
    
    # filter by chat_id_1 in chat_list  and chat_id_2 in chat_list
    flt = ((clusters_compare_df['chat_id_1'] == chat_a) & (clusters_compare_df['chat_id_2'] == chat_b))
    flt = flt | ((clusters_compare_df['chat_id_1'] == chat_b) & (clusters_compare_df['chat_id_2'] == chat_a))

    total_records = offset + records
    df_comp_flt = clusters_compare_df[flt].head(total_records).tail(records)
    # loop over df_comp_flt and print the messages
    for idx, row in df_comp_flt.iterrows():
        if 'c_id' in row:
            print(f" --- {row['c_id']} ---")
        print(f"Similarity: {row['similarity']}")
        # print 100 times the character '-'
        print("_" * (width*2+1))
        print_side_by_side(f'Chat ID: {row["chat_id_1"]}', f'Chat ID: {row["chat_id_2"]}', width=width)
        print_side_by_side(f'Msg ID: {row["id_1"]} - Fecha: {row["date_id_1"]}', f'Msg ID: {row["id_2"]} - Fecha: {row["date_id_2"]}', width=width)
        print("-" * (width*2+1))
        print_side_by_side(row['message_text_clean_utf8_1'], row['message_text_clean_utf8_2'], width=width)
        print("_" * (width*2+1))
        print()


def is_date(string): # NEEDS MORE WORK
    try:
        # Try to parse the string as a date
        parse(string, fuzzy=False)
        return True
    except ValueError:
        return False
    except Exception as e:
        print(f"Error checking date with: {string}")
        print(e)
        return False

def is_number(string):
    try:
        float(string)  # Try to convert to float
        return True
    except ValueError:
        return False
    except Exception as e:
        print(f"Error checking number with: {string}")
        print(e)
        return False


### Plorting functions
# import mplcursors


def plot_chat_pair_metrics(df_clusters_grp, k=0.3):
    """
    Requires a DataFrame with the following columns:
    - chat_id_low: The lower chat_id in the pair
    - chat_id_high: The higher chat_id in the pair
    - sim_ratio_norm: The normalized similarity ratio between the pair

    """

    # Create a graph
    G = nx.Graph()

    # Add edges with weights based on distance and assign the cluster id
    for _, row in df_clusters_grp.iterrows():
        if row['chat_id_low'] not in G.nodes:
            G.add_node(row['chat_id_low'], chat_id=row['chat_id_low'], label=row['chat_id_low'])

        if row['chat_id_high'] not in G.nodes:
            G.add_node(row['chat_id_high'], chat_id=row['chat_id_high'], label=row['chat_id_high'])

        G.add_edge(row['chat_id_low'], row['chat_id_high'], weight=row['sim_ratio_norm'])

    # Get unique chat_id and define colors
    unique_chat_id1 = df_clusters_grp['chat_id_low'].unique()
    unique_chat_id2 = df_clusters_grp['chat_id_high'].unique()
    unique_chat_id = list(set(list(unique_chat_id1) + list(unique_chat_id2)))
    palette_chat_id = sns.color_palette('Paired', len(unique_chat_id))
    chat_id_colors = {unique_chat_id[i]: palette for i, palette in enumerate(palette_chat_id)}

    # Generate positions for each node with increased `k` to separate clusters
    pos = nx.spring_layout(G, seed=42, k=k)

    # Plot the graph
    plt.figure(figsize=(12, 10))
    axes = plt.gca()

    # Draw edges with widths based on distance
    edges_in_cluster = [(u, v) for u, v, d in G.edges(data=True)]
    similarities = [G[u][v]['weight'] * 5 for u, v in edges_in_cluster]
    # normalize the similarities to be between 0 and 2
    # similarities = [2 * (s - min(similarities)) / (max(similarities) - min(similarities)) for s in similarities]

    print(similarities)
    nx.draw_networkx_edges(G, pos, edgelist=edges_in_cluster, width=similarities, alpha=0.6)

    # Draw nodes with colors based on their chat_id
    node_colors = [chat_id_colors[G.nodes[n]['chat_id']] if 'chat_id' in G.nodes[n] else 'lightblue' for n in G.nodes()]
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors='black')

    # Draw labels for the nodes
    labels = nx.get_node_attributes(G, 'chat_id')
    nx.draw_networkx_labels(G, pos, font_size=12, labels=labels)

    # Create a legend for the chat_id colors
    chat_legend = [plt.Line2D([0], [0], marker='o', color='w', label=f'Chat ID: {chat_id}',
                              markerfacecolor=chat_id_colors[chat_id], markersize=10, alpha=1) for chat_id in unique_chat_id]

    legend1 = plt.legend(handles=chat_legend, title="Chat IDs", loc='upper left', bbox_to_anchor=(0, 0.5))
    axes.add_artist(legend1)

    # Use mplcursors to display the title when hovering over nodes
    # cursor = mplcursors.cursor(nodes, hover=True)

    # # This function will be triggered on hover
    # @cursor.connect("add")
    # def on_add(sel):
    #     node_id = list(G.nodes())[sel.index]  # Get the node ID (chat_id)
    #     # Find the corresponding title from df_info based on chat_id
    #     title_row = df_info[df_info['chat_id'] == G.nodes[node_id]['chat_id']]
    #     if not title_row.empty:
    #         title = title_row.iloc[0]['title']  # Get the title
    #         sel.annotation.set(text=f"Title: {title}")  # Show the title on hover
    #     else:
    #         sel.annotation.set(text="No title available")

    plt.title('Graph Visualization of Message Clusters by Distance and Chat ID', size=15)
    plt.tight_layout()

    plt.show()


def plot_msg_pairs(df_clusters, k=0.3):
    # Create a graph
    G = nx.Graph()
    # Add edges with weights based on distance and assign the cluster id
    for _, row in df_clusters.iterrows():
        if row['id_1'] not in G.nodes:
            G.add_node(row['id_1'], chat_id=row['chat_id_1'], label=row['chat_id_1'])

        if row['id_2'] not in G.nodes:
            G.add_node(row['id_2'], chat_id=row['chat_id_2'], label=row['chat_id_2'])

        # G.add_edge(row['id_1'], row['id_2'], weight=(row['distance']** (1/3))*3)
        G.add_edge(row['id_1'], row['id_2'], weight=(row['distance']))

    # Get unique clusters and define colors
    # unique_clusters = df_clusters['c_id'].unique()
    # palette = sns.color_palette('hsv', len(unique_clusters))
    # cluster_colors = {c_id: palette[i] for i, c_id in enumerate(unique_clusters)}

    # Get unique chat_id and define colors
    unique_chat_id1 = df_clusters['chat_id_1'].unique()
    unique_chat_id2 = df_clusters['chat_id_2'].unique()
    # Merge chat_id1 and chat_id2
    unique_chat_id = list(set(list(unique_chat_id1) + list(unique_chat_id2)))
    palette_chat_id = sns.color_palette('Paired', len(unique_chat_id))
    chat_id_colors = {unique_chat_id[i]: palette for i, palette in enumerate(palette_chat_id)}

    # Generate positions for each node with increased `k` to separate clusters
    pos = nx.spring_layout(G, seed=42, k=k)  # Increase `k` for more separation between clusters

    # Plot the graph
    plt.figure(figsize=(12, 10))
    axes = plt.gca()

    # Draw edges with widths based on distance
    
    edges_in_cluster = [(u, v) for u, v, d in G.edges(data=True)]
    similarities = [G[u][v]['weight'] for u, v in edges_in_cluster]
    widths = [d if d > 0.1 else 0.1 for d in similarities]
    nx.draw_networkx_edges(G, pos, edgelist=edges_in_cluster, width=widths, alpha=0.6)

    # Draw nodes with colors based on their chat_id
    node_colors = [chat_id_colors[G.nodes[n]['chat_id']] if 'chat_id' in G.nodes[n] else 'lightblue' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors='black')


    # Draw labels for the nodes
    labels = nx.get_node_attributes(G, 'chat_id')  # get labels 
    nx.draw_networkx_labels(G, pos, font_size=12, labels=labels)


    # Create a legend for the chat_id colors
    chat_legend = [plt.Line2D([0], [0], marker='o', color='w', label=f'Chat ID: {chat_id}',
                            markerfacecolor=chat_id_colors[chat_id], markersize=10, alpha=1) for chat_id in unique_chat_id]

    legend1 = plt.legend(handles=chat_legend, title="Chat IDs", loc='upper left', bbox_to_anchor=(0, 0.5))

    # axes.add_artist(legend2)
    axes.add_artist(legend1)

    plt.title('Graph Visualization of Message Clusters by Distance and Chat ID', size=15)
    plt.tight_layout()

    plt.show()
