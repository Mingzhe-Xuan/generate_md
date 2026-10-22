"""Author atomate2-dft relax_bandgap.py maker for the paper-version API."""


def make_paper_maker():
    from atomate2.vasp.flows.mp import MPGGADoubleRelaxStaticMaker
    from atomate2.vasp.jobs.mp import MPGGAStaticMaker
    from atomate2.vasp.sets.mp import MPGGAStaticSetGenerator
    static_maker = MPGGAStaticMaker(input_set_generator=MPGGAStaticSetGenerator(
        user_incar_settings={'NEDOS': 2001}))
    maker = MPGGADoubleRelaxStaticMaker(static_maker=static_maker)
    # The paper-version generator defaults to PBE despite PMG_DEFAULT_FUNCTIONAL.
    # Select the identified PBE54 datasets consistently in all three stages.
    for stage in stage_makers(maker):
        stage.input_set_generator.potcar_functional = 'PBE_54'
    return maker


def stage_makers(maker):
    return (maker.relax_maker.relax_maker1, maker.relax_maker.relax_maker2, maker.static_maker)


def leaf_jobs(flow):
    for child in flow.jobs:
        if hasattr(child, 'jobs'):
            yield from leaf_jobs(child)
        else:
            yield child
