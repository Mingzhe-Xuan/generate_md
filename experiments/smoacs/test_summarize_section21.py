import pytest
from summarize_section21 import assess

def row(gap="2.5", energy="-0.6", neutral="True", bond="True"):
    return dict(bandgap_onehot=gap, ef_onehot=energy, is_neutral_cmn=neutral,
                **{"minbond_less_than_0.5": bond})

def test_paper_gap_bounds_and_strict_energy():
    result = assess([row("2.3"), row("2.7"), row(energy="-0.5"), row("2.70001")], 4)
    assert result["counts"] == dict(bandgap=3, formation_energy=3, valid_structure=4, joint=2)

def test_joint_requires_structure_and_missing_candidates_remain_in_denominator():
    result = assess([row(), row(neutral="False"), row(bond="False")], 4)
    assert result["rates"]["joint"] == 0.25
    assert result["missing_candidates"] == 1

def test_nan_or_missing_predictions_fail():
    assert assess([row("nan"), row(energy="")], 2)["counts"]["joint"] == 0

def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError): assess([row(neutral="unknown")], 1)
    with pytest.raises(ValueError): assess([row(), row()], 1)
    with pytest.raises(ValueError): assess([{}], 1)
    with pytest.raises(ValueError): assess([], 0)
