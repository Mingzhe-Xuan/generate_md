"""Scoped dtype-only compatibility for official OCP helpers during F64 diagnostics."""
from contextlib import contextmanager
import hashlib
import inspect
import sys
import textwrap


def clone_with_replacements(function, replacements, globals_update=None):
    source = textwrap.dedent(inspect.getsource(function))
    original = source
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Pinned precision-adapter expression changed: '+old)
        source = source.replace(old, new)
    namespace = dict(function.__globals__)
    namespace.update(globals_update or {})
    exec(compile(source, '<denoised-native-dtype-adapter>', 'exec'), namespace)
    return namespace[function.__name__], dict(
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        adapted_sha256=hashlib.sha256(source.encode()).hexdigest(), replacements=replacements)


@contextmanager
def replace_aliases(replacements):
    changed = []
    try:
        for module in list(sys.modules.values()):
            if module is None or not getattr(module, '__name__', '').startswith('mattergen.'):
                continue
            for name, value in list(vars(module).items()):
                for old, new in replacements:
                    if value is old:
                        changed.append((module, name, old))
                        setattr(module, name, new)
        yield
    finally:
        for module, name, original in reversed(changed):
            setattr(module, name, original)


STABLE_OFFSET_EDITS = [
    ('pos2 = pos2 + pbc_offsets_per_atom', 'relative_positions = pos1 - pos2 - pbc_offsets_per_atom'),
    ('torch.sum((pos1 - pos2) ** 2, dim=1)', 'torch.sum(relative_positions ** 2, dim=1)')]


def helpers(stable_offsets=False):
    from mattergen.common.utils import ocp_graph_utils as ocp, data_utils as data
    neighbor, a = clone_with_replacements(ocp.get_max_neighbors_mask, [
        ('torch.full([num_atoms * max_num_neighbors], np.inf, device=device)',
         'torch.full([num_atoms * max_num_neighbors], np.inf, device=device, dtype=atom_distance_squared.dtype)')])
    radius_edits = [
        ('device=device, dtype=torch.float)', 'device=device, dtype=cell.dtype)'),
        ('cell_offsets.float().view(-1, 1, 3).bmm(cell_repeated.float())',
         'cell_offsets.to(dtype=cell.dtype).view(-1, 1, 3).bmm(cell_repeated)')]
    if stable_offsets:
        radius_edits += STABLE_OFFSET_EDITS
    radius, b = clone_with_replacements(ocp.radius_graph_pbc, radius_edits,
        {'get_max_neighbors_mask': neighbor})
    distances, c = clone_with_replacements(data.get_pbc_distances, [
        ('to_jimages.float()', 'to_jimages.to(dtype=lattice.dtype)')])
    return [(ocp.radius_graph_pbc, radius), (ocp.get_max_neighbors_mask, neighbor),
            (data.get_pbc_distances, distances)], dict(neighbors=a, radius=b, distances=c)
