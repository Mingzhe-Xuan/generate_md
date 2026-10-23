"""Exact existential charge balance with the author's unchanged oxidation sets."""
from math import prod


def reachable_zero(options):
    charges={0}
    for values in options:
        charges={charge+oxidation for charge in charges for oxidation in values}
        if not charges:return False
    return 0 in charges


def super_common(key,author):
    elements=[element for element,count in key for _ in range(count)]
    options=[set(author.Element_pmg(element).icsd_oxidation_states)
             &set(author.Element_pmg(element).oxidation_states)
             &set(author.Element(element).oxidation_states)
             &set(author.Element_pmg(element).common_oxidation_states) for element in elements]
    result=reachable_zero(options);assignments=prod(len(value) for value in options)
    checked=assignments<=100000
    if checked:
        expected=bool(author.elec_neutral_check_SUPER_COMMON(0,1,[e for e,c in key],[[c] for e,c in key],False)[0])
        if result!=expected:raise ValueError('Exact neutrality DP disagrees with the pinned author implementation.')
    return result,dict(formula_counts=key,legacy_assignments=assignments,legacy_checked=checked,neutral=result)
