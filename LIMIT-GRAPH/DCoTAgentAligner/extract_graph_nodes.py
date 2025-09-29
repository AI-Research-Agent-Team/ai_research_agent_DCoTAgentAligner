def extract_graph_nodes(chain, graph_vocab):
    # Match reasoning steps to graph concepts
    matched = []
    for concept in graph_vocab:
        if concept in chain:
            matched.append(concept)
    return matched
