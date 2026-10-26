"""Graph identity must distinguish a changed periodic edge from a reordering."""
import numpy as np
import pytest
from diagnostic import graph_identity, graph_trace


def test_periodic_fingerprint_ignores_order_but_tracks_image_and_multiplicity():
    edges = np.array([[0, 1, 1], [1, 0, 1]])
    images = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    original = graph_identity(edges, images)
    reordered = graph_identity(edges[:, ::-1], images[::-1])
    assert original['canonical'] == reordered['canonical']
    assert original['ordered'] != reordered['ordered']
    changed = images.copy()
    changed[-1, 0] = -1
    assert original['canonical'] != graph_identity(edges, changed)['canonical']
    assert original['canonical'] != graph_identity(edges[:, :2], images[:2])['canonical']


def test_trace_restores_class_method_after_model_failure():
    import torch
    class Gemnet:
        def generate_interaction_graph(self):
            return (torch.tensor([[0], [1]]), *([None]*7), torch.zeros((1, 3)))
    net = Gemnet()
    records = []
    with pytest.raises(RuntimeError):
        with graph_trace(net, records):
            net.generate_interaction_graph()
            raise RuntimeError('model failure')
    assert 'generate_interaction_graph' not in net.__dict__
    assert len(records) == 1 and records[0]['edges'] == 1
