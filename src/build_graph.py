"""Relation-preserving adjacency lists for the teacher's Lesson 02 walkthrough."""


def build_graph(triples, add_inverse=True):
    graph = {}
    for head, relation, tail in triples:
        triple = (head, relation, tail)
        graph.setdefault(head, [])
        graph.setdefault(tail, [])
        graph[head].append({'target': tail, 'relation': relation,
                            'inverse': False, 'triple': triple})
        if add_inverse:
            graph[tail].append({'target': head, 'relation': relation,
                                'inverse': True, 'triple': triple})
    return graph
