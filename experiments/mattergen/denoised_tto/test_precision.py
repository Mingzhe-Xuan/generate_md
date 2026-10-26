"""Scoped dtype compatibility must preserve F32 results and restore aliases."""
import sys
import types
import pytest
import torch
from precision import clone_with_replacements, replace_aliases


def legacy_offsets(lattice, image):
    return torch.einsum('bi,bij->bj', image.float(), lattice)


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
