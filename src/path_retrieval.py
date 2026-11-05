"""Bounded, node-simple DFS; keeps relation labels and original fact direction."""


def find_paths(graph, start, target=None, max_hops=2, max_paths=None):
    """Return paths as lists of steps; target=None returns reachable path prefixes.

    A known start equal to target has one zero-edge path. The optional output cap
    may truncate results; callers must record it rather than claim full coverage.
    This teaching search is not a learned candidate generator or a QA model.
    """
    if max_hops < 0:
        raise ValueError('max_hops must be nonnegative')
    if max_paths is not None and max_paths < 1:
        raise ValueError('max_paths must be positive or None')
    if start not in graph:
        return []

    def visit(node, steps, visited):
        if target is not None and node == target:
            yield steps
            return
        if target is None and steps:
            yield steps
        if len(steps) >= max_hops:
            return
        for edge in graph.get(node, []):
            neighbour = edge['target']
            if neighbour in visited:
                continue
            step = {'source': node, **edge}
            # Each branch receives its own visited set, preserving other routes.
            yield from visit(neighbour, steps + [step], visited | {neighbour})

    paths = []
    for path in visit(start, [], {start}):
        paths.append(path)
        if max_paths is not None and len(paths) >= max_paths:
            break
    return paths


def path_nodes(start, path):
    return [start] + [step['target'] for step in path]


def format_path(start, path):
    parts = [start]
    for step in path:
        relation = step['relation'] + ('^-1' if step['inverse'] else '')
        parts.extend([f' --{relation}--> ', step['target']])
    return ''.join(parts)
