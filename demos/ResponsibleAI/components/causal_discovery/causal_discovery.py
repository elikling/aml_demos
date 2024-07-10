import argparse
import os
import pandas as pd
import numpy as np
import mltable
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam

from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.FCMBased.lingam.utils import make_dot
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
import networkx as nx
import regex as re

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--data_mltable", type=str, help="Path to data, MLTable format expected")
    parser.add_argument("--exclude_features", type=str, help="Name of columns to exlude from the training set")
    parser.add_argument("--step_artifacts", type=str, help="path to write outputs")
    
    
    # parse args
    args = parser.parse_args()

    # return args
    return args

def parse_graph_string(graph_str):
    # Split the string into nodes and edges sections
    nodes_str, edges_str = graph_str.split("\n\nGraph Edges:")
    
    # Extract nodes
    nodes = nodes_str.split("Graph Nodes:\n")[1].split(";")
    nodes = [node.strip() for node in nodes]
    
    # Extract edges
    edge_pattern = r"(\w+)\s*(?:--->?|---)\s*(\w+)"
    edges = re.findall(edge_pattern, edges_str)
    
    return nodes, edges

def dag_plot(causal_dag, labels, fn):
    graph_string = str(causal_dag)
    nodes, edges = parse_graph_string(graph_string)

    G = nx.Graph()

    for node in nodes:
        i = int(re.search(r'\d+$', node).group()) -1
        node_lable=labels[i]
        G.add_node(node,label=node_lable)

    nx_labels = nx.get_node_attributes(G, 'label')
    #print(nx_labels)

    G.add_edges_from(edges)

    # Set up the plot
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Draw the graph
    nx.draw(G, pos, with_labels=False,node_color='orange',node_size=300,edge_color='gray',width=1,alpha=0.7)
    nx.draw_networkx_labels(G, pos, nx_labels, font_size=14,font_weight='bold')

    # Add a title
    plt.title("Network Visualization", fontsize=16)

    # Show the plot
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    plt.savefig(fn, format='png', dpi=300, bbox_inches='tight')

def main(args):
    # Read in training data
    print("<<< Reading traning data >>>")
    data_mltbl = mltable.load(args.data_mltable)
    data_df = data_mltbl.to_pandas_dataframe()

    exclude_features_str=args.exclude_features

    if exclude_features_str != "NA":
        exclude_features_list01 = exclude_features_str.split(',')
        exclude_features_list = list(exclude_features_list01)
        print("<<< Exluding: ", exclude_features_list, " >>>")
        #drop the variables and then set them anew to zero thus ensuring int type.
        data_df = data_df.drop(labels=exclude_features_list, axis="columns")

    for col in data_df.columns:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    data_df.dropna(inplace=True,how="all")

    
    labels = [f'{col}' for i, col in enumerate(data_df.columns)]
    print("<<<>>>")
    print(data_df.shape)
    column_info = data_df.dtypes
    print("<<< - >>>")
    print(column_info)
    print("<<<>>>")

    #assuming all columns are numeric
    data_np = data_df.to_numpy()

    #CG
    cg = pc(data_np)

    dag_plot(causal_dag=cg.G, labels=labels, fn=os.path.join(args.step_artifacts, "pc.png"))
    
    #Ges
    cg_ges = ges(data_np)

    dag_plot(causal_dag=cg_ges['G'], labels=labels, fn=os.path.join(args.step_artifacts, "ges.png"))

    #LINGAM
    #model_lingam = lingam.ICALiNGAM()
    #model_lingam.fit(data_np)
    #make_dot(model_lingam.adjacency_matrix_, labels=labels)

# run script
if __name__ == "__main__":
    
    print("*" * 60)
    print("\n\n")

    args = parse_args()
    main(args)

    print("*" * 60)
    print("\n\n")
