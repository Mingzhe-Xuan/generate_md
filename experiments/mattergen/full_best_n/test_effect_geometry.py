"""Separate true geometry magnitudes from periodic bookkeeping and cell scale."""
import importlib.util
from pathlib import Path
import numpy as np
from pymatgen.core import Structure,Lattice
import pytest

spec=importlib.util.spec_from_file_location('effect_metrics',Path(__file__).with_name('effect_geometry.py'))
geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)


def test_identity_translation_images():
    a=Structure(Lattice.from_parameters(4,5,6,65,75,80),['Si','O','O'],[[.1,.2,.3],[.6,.1,.8],[.9,.9,.9]])
    b=Structure(a.lattice,a.species,a.frac_coords+[.37,1.8,-.92])
    assert np.max(np.abs(geometry.internal_vectors(a,b)))<1e-9
    assert geometry.measure(a,b)['species_pair_distance_RMS_A']<1e-9


def test_cell_scale_and_permutation():
    a=Structure(Lattice.cubic(4),['Si','O','O'],[[0,0,0],[.3,.2,.1],[.7,.8,.9]])
    b=Structure(Lattice.cubic(4.4),a.species,a.frac_coords)
    m=geometry.measure(a,b)
    assert m['internal_RMS_A']<1e-10 and m['species_pair_distance_RMS_A']>0
    assert abs(m['volume_change_percent']-33.1)<1e-9
    perm=Structure(a.lattice,[a[i].specie for i in [2,0,1]],[a[i].frac_coords for i in [2,0,1]])
    np.testing.assert_allclose(geometry.distance_signature(a),geometry.distance_signature(perm))
    with pytest.raises(ValueError):geometry.internal_vectors(a,perm)


def test_internal_change():
    a=Structure(Lattice.cubic(5),['Si','O'],[[0,0,0],[.3,.3,.3]])
    b=a.copy();b.translate_sites([1],[.1,0,0],frac_coords=False)
    assert abs(geometry.measure(a,b)['internal_RMS_A']-.05)<1e-9
