"""Author atomate2-dft relax_bandgap.py maker for the paper-version API."""


def make_paper_maker():
    from atomate2.vasp.flows.mp import MPGGADoubleRelaxStaticMaker
    from atomate2.vasp.jobs.mp import MPGGAStaticMaker
    from pymatgen.io.vasp.sets import MPStaticSet
    static_maker = MPGGAStaticMaker(input_set_generator=MPStaticSet(
        force_gamma=True, auto_metal_kpoints=True, inherit_incar=False,
        user_incar_settings={'NEDOS': 2001}))
    return MPGGADoubleRelaxStaticMaker(static_maker=static_maker)


def leaf_jobs(flow):
    for child in flow.jobs:
        if hasattr(child, 'jobs'):
            yield from leaf_jobs(child)
        else:
            yield child
