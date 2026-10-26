"""Scoped dtype compatibility must preserve F32 results and restore aliases."""
import sys
import types
import pytest
import torch
from precision import clone_with_replacements, replace_aliases, STABLE_OFFSET_EDITS


def legacy_offsets(lattice, image):
    return torch.einsum('bi,bij->bj', image.float(), lattice)


def legacy_squared_distance(pos1, pos2, pbc_offsets_per_atom):
    pos2 = pos2 + pbc_offsets_per_atom
    return torch.sum((pos1 - pos2) ** 2, dim=1)


def test_dtype_clone_preserves_f32_and_supports_f64_geometry_derivatives():
    adapted, record = clone_with_replacements(legacy_offsets, [
        ('image.float()', 'image.to(dtype=lattice.dtype)')])
    image = torch.tensor([[1, -2, 1], [0, 1, -1]])
    cell = torch.arange(18., dtype=torch.float32).reshape(2, 3, 3)
    torch.testing.assert_close(adapted(cell, image), legacy_offsets(cell, image), rtol=0, atol=0)
    assert record['original_sha256'] != record['adapted_sha256']
    double = cell.double().requires_grad_()
    with pytest.raises(RuntimeError):
        legacy_offsets(double, image)
    result = adapted(double, image)
    gradient = torch.autograd.grad(result.square().sum(), double)[0]
    d = torch.linspace(-1, 1, 18, dtype=double.dtype).reshape(2, 3, 3)
    epsilon = 1e-6
    fd = (adapted(double+epsilon*d, image).square().sum()-adapted(double-epsilon*d, image).square().sum())/(2*epsilon)
    torch.testing.assert_close(fd, (gradient*d).sum(), rtol=1e-6, atol=1e-5)


def test_clone_rejects_missing_or_ambiguous_source_expression():
    for old in ('not_present', 'image'):
        with pytest.raises(ValueError):
            clone_with_replacements(legacy_offsets, [(old, 'replacement')])


def test_alias_scope_restores_all_imports_even_when_native_check_fails(monkeypatch):
    original = lambda: 'original'
    adapted = lambda: 'adapted'
    a = types.ModuleType('mattergen.precision_fake_a')
    b = types.ModuleType('mattergen.precision_fake_b')
    unrelated = types.ModuleType('unrelated_precision_fake')
    a.function = b.imported = unrelated.function = original
    for module in (a, b, unrelated):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError):
        with replace_aliases([(original, adapted)]):
            assert a.function() == b.imported() == 'adapted'
            assert unrelated.function() == 'original'
            raise RuntimeError('native diagnostic failed')
    assert a.function is b.imported is unrelated.function is original


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_self_image_distance_is_translation_independent_and_exactly_even(dtype):
    adapted, _ = clone_with_replacements(legacy_squared_distance, STABLE_OFFSET_EDITS)
    rng = torch.Generator().manual_seed(32)
    offsets = torch.tensor([[[4.2, -4.2], [1.3, -1.3], [.7, -.7]]], dtype=dtype)
    reference = offsets.square().sum(1)
    observed_legacy_asymmetry = False
    for _ in range(100):
        pos = torch.randn((1, 3, 1), generator=rng, dtype=dtype)*4
        current = adapted(pos, pos, offsets)
        assert torch.equal(current, reference)
        original = legacy_squared_distance(pos, pos, offsets)
        observed_legacy_asymmetry |= not torch.equal(original[:, 0], original[:, 1])
    assert observed_legacy_asymmetry


def test_stable_distance_keeps_ordinary_pair_values_and_geometric_gradients():
    adapted, _ = clone_with_replacements(legacy_squared_distance, STABLE_OFFSET_EDITS)
    a = torch.tensor([[[.2], [.4], [.6]]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[[.7], [.1], [.9]]], dtype=torch.float64)
    offsets = torch.tensor([[[4.2, -4.2], [1.3, -1.3], [.7, -.7]]], dtype=a.dtype)
    torch.testing.assert_close(adapted(a, b, offsets), legacy_squared_distance(a, b, offsets), rtol=1e-14, atol=1e-14)
    gradient = torch.autograd.grad(adapted(a, b, offsets).sum(), a)[0]
    torch.testing.assert_close(gradient, (2*(a-b-offsets)).sum(2, keepdim=True))
