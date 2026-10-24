"""Create the existing five-variable CF context from immutable MatterGen poses."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import types

from template_policy import neutral_assignments, select_cohort

SMOACS = Path(__file__).resolve().parents[2]/'smoacs'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create(source_root, output, args):
    import numpy as np
    import torch
    from pymatgen.core import Structure
    sys.path.insert(0, str(SMOACS))
    from run_section21 import SOURCE_MEMBER, SOURCE_SHA256
    assets = Path('results/smoacs/assets-199').resolve()
    archive = Path('results/smoacs/downloads/reproducibility.tar.gz')
    if digest(archive) != json.loads((SMOACS/'assets.json').read_text())['reproducibility.tar.gz']['sha256']:
        raise ValueError('Author archive changed.')
    for name, info in json.loads((SMOACS/'checkpoints.json').read_text()).items():
        if digest(assets/'models/crystalformer'/name) != info['sha256']:
            raise ValueError('Frozen predictor changed.')
    with tarfile.open(archive) as handle:
        original = handle.extractfile(SOURCE_MEMBER).read()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError('Author source changed.')
    os.environ.update(SMOACS_ROOT=str(assets), SMOACS_DEVICE='cpu', MPLBACKEND='Agg')
    source = Path('third_party/SMOACS').resolve()
    sys.path[:0] = [str(source), str(source/'src')]
    author = types.ModuleType('mattergen_cf_template_author')
    author.__file__ = str(assets/SOURCE_MEMBER)
    sys.modules[author.__name__] = author
    exec(compile(original, author.__file__, 'exec', dont_inherit=True), author.__dict__)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)

    def states(symbol):
        element = author.Element_pmg(symbol)
        return (set(element.icsd_oxidation_states) & set(element.oxidation_states)
                & set(element.common_oxidation_states) & set(author.Element(symbol).oxidation_states))

    from pymatgen.core import Element
    by_z = [states(Element.from_Z(z).symbol) for z in range(1, 99)]
    inventory = json.loads((source_root/'inventory-249/inventory.json').read_text())
    if inventory['denominator'] != 128 or len(inventory['candidates']) != 128:
        raise ValueError('Require the complete immutable MatterGen cohort.')
    eligibility = []
    structures = {}
    assignments = {}
    for candidate in inventory['candidates']:
        identifier = candidate['candidate_id']
        path = source_root/'structures'/identifier
        if digest(path) != candidate['sha256']:
            raise ValueError('Generated pose changed: '+identifier)
        structure = Structure.from_file(path)
        if not structure.is_ordered or structure.volume <= 1e-6:
            raise ValueError('Invalid input geometry.')
        numbers = list(structure.atomic_numbers)
        if not all(1 <= z <= 98 for z in numbers):
            raise ValueError('Species outside the frozen CF vocabulary.')
        paths = neutral_assignments([by_z[z-1] for z in numbers])
        eligible = bool(paths)
        if eligible != (candidate['original_valid'] == 'True'):
            raise ValueError('Neutral mask eligibility differs from recorded neutral/separated cohort.')
        eligibility.append(dict(candidate_id=identifier, num_atoms=len(structure), eligible=eligible,
                                source_sha256=candidate['sha256'], neutral_assignments=len(paths),
                                reason=None if eligible else 'No SUPER_COMMON neutral assignment preserving original species.'))
        structures[identifier] = structure
        assignments[identifier] = paths
    mode = getattr(args, 'cohort', 'pilot')
    fixed_species = getattr(args, 'fixed_species', False)
    unconstrained = mode == 'unmasked' or fixed_species
    selected = select_cohort(eligibility, mode)
    branch_manifest=None
    parent_structures=None
    if getattr(args,'branch_manifest',None):
        if not fixed_species or mode!='pilot':
            raise ValueError('Branches require fixed original elements and the original pilot parents.')
        branch_path=Path(args.branch_manifest).resolve()
        branch_manifest=json.loads(branch_path.read_text())
        if branch_manifest['source_inventory_sha256']!=digest(source_root/'inventory-249/inventory.json'):
            raise ValueError('Branch manifest uses a different original cohort.')
        parents={r['candidate_id']:r for r in selected}
        if branch_manifest['parent_ids']!=list(parents):
            raise ValueError('Branch parents differ from the property-blind pilot.')
        parent_structures=dict(structures)
        selected=[]
        for proposal in branch_manifest['candidates']:
            parent=proposal['parent_id']
            key=proposal['candidate_id']
            path=Path(proposal['source']).resolve()
            if parent not in parents or not path.is_relative_to(branch_path.parent) or digest(path)!=proposal['sha256']:
                raise ValueError('Branch input provenance failed.')
            structure=Structure.from_file(path)
            if list(structure.atomic_numbers)!=list(parent_structures[parent].atomic_numbers):
                raise ValueError('Branch changed ordered original species.')
            structures[key]=structure
            assignments[key]=assignments[parent]
            selected.append(dict(parents[parent],candidate_id=key,parent_id=parent,
                parent_source_sha256=parents[parent]['source_sha256'],source_sha256=proposal['sha256']))
        if len(selected)!=len({r['candidate_id'] for r in selected}) or not 4<=len(selected)<=128:
            raise ValueError('Invalid number or duplicate branch IDs.')
    (output/'eligibility.json').write_text(json.dumps(dict(denominator=128,
        eligible=sum(r['eligible'] for r in eligibility), candidates=eligibility,
        selection='All paired perturbations and zero controls of four size/ID-selected original parents' if branch_manifest else 'Size-selected pilot' if mode == 'pilot' else 'All source IDs in the '+mode+' strategy, original order; no property selection.',
        strategy=mode,
        selected_ids=[r['candidate_id'] for r in selected]), indent=2)+'\n')
    ids = [r['candidate_id'] for r in selected]
    chosen = [structures[k] for k in ids]
    size = torch.tensor([len(s) for s in chosen], dtype=torch.long)
    n = len(ids)
    batch = torch.repeat_interleave(torch.arange(n), size)
    numbers = torch.tensor([z for s in chosen for z in s.atomic_numbers], dtype=torch.long)
    abc = torch.tensor([s.lattice.abc for s in chosen], dtype=torch.float32)
    angles = torch.tensor([s.lattice.angles for s in chosen], dtype=torch.float32)
    coords = torch.tensor(np.concatenate([s.frac_coords for s in chosen]), dtype=torch.float32)
    total = int(size.sum())
    atomic_mask = torch.zeros(total, 98, 10)
    ox_used = torch.zeros(total, 10)
    ox_weights = torch.zeros(n, 10)
    slices = []
    offset = 0
    for i, key in enumerate(ids):
        count = int(size[i])
        sl = slice(offset, offset+count)
        slices.append(sl)
        paths = assignments[key]
        for channel, path in enumerate(paths):
            for j, charge in enumerate(path):
                atomic_mask[offset+j, :, channel] = torch.tensor([float(charge in q) for q in by_z])
        ox_used[sl, :len(paths)] = 1
        ox_weights[i, :len(paths)] = 1
        offset += count
    atomic = torch.full((total, 98), 1/98)
    atomic[torch.arange(total), numbers-1] += .001
    if fixed_species:
        atomic = torch.nn.functional.one_hot(numbers-1, 98).to(coords.dtype)
        ox_weights.zero_()
    data = dict(size=size, batch=batch, atomic_mask=atomic_mask, ox_states_used_mask=ox_used,
                init_coords=coords.clone(), batch_dir_coords=coords, atomic_distribution=atomic,
                ox_mask_learnable_tensor_per_crystal=ox_weights)
    if fixed_species:
        data['fixed_atomic_distribution'] = atomic.clone()
        from fixed_species import forward as fixed_forward
        author.forward_propagation_Crystalformer = lambda **kw: fixed_forward(author, **kw)
    base = abc.clone()
    lower = torch.minimum(torch.full_like(abc, 2), abc)
    upper = torch.maximum(torch.full_like(abc, 10), abc)
    angle_lower = torch.minimum(torch.full_like(angles, 60), angles)
    angle_upper = torch.maximum(torch.full_like(angles, 120), angles)

    class Lengths:
        def rescale(self, value):
            return value*base

    class Angles:
        def rescale(self, value):
            return value*60+60

    scalers = dict(abc_scaler=Lengths(), angle_scaler=Angles())
    targets = [torch.nn.Parameter(t.clone()) for t in (abc/base, (angles-60)/60, coords, atomic, ox_weights)]
    gap_model, ef_model = author.Load_Pretrained_Crystalformers(device='cpu')
    for model in (gap_model, ef_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    optimizers = [torch.optim.Adam(targets[:2], lr=.002), torch.optim.Adam(targets[3:], lr=.007),
                  torch.optim.Adam([targets[2]], lr=.005)]

    context = dict(args=args, output=output, ids=ids, n=n, targets=targets, data=data, batch=batch,
        size=size, base=base, lower=lower, upper=upper, angle_lower=angle_lower, angle_upper=angle_upper,
        gap_model=gap_model, ef_model=ef_model, optimizers=optimizers, slices=slices, author=author,
        unconstrained_elements=unconstrained, fixed_species=fixed_species, microbatch_size=getattr(args, 'microbatch_size', 4),
        streamed_initialization=n>4)
    if branch_manifest:
        from branch_geometry import install
        original=[parent_structures[r['parent_id']] for r in selected]
        install(context,np.concatenate([s.frac_coords for s in original]),
            np.asarray([s.lattice.matrix for s in original]),np.asarray([s.lattice.abc for s in original]),
            np.asarray([s.lattice.angles for s in original]),max_displacement=branch_manifest.get('max_cf_displacement_A',.4))
        lower,upper,angle_lower,angle_upper=[context[k] for k in ('lower','upper','angle_lower','angle_upper')]
        context.update(branch_manifest=branch_manifest,gradient_denominator=1)

    def forward(temperature):
        if n > 4:
            if torch.is_grad_enabled():
                raise RuntimeError('Use streamed gradients for the full external cohort.')
            from microbatch import Microbatch
            return Microbatch(context, size=context['microbatch_size']).forward(temperature)
        prediction_targets = targets[:4]+[None] if unconstrained else targets
        gap, ef, out = author.forward_propagation_Crystalformer(optimization_targets=prediction_targets,
            fixed_inputs=data, scalers=scalers, temperature=temperature, bandgap_model=gap_model,
            e_form_model=ef_model, adding_noise_scale=None, limit_coords_displacement=None,
            abc_range=(lower, upper), angle_range=(angle_lower, angle_upper))
        gap, ef = gap.reshape(-1), ef.reshape(-1)
        loss = torch.clamp(torch.abs(gap-2.5)-.04, min=0)+ef
        if unconstrained:
            loss = loss+targets[4].sum()*0
        if fixed_species:
            loss = loss+targets[3].sum()*0
        return gap, ef, loss, out

    def hard_predict(out):
        gaps, energies = [], []
        chunk = context['microbatch_size']
        for start in range(0, n, chunk):
            stop = min(n, start+chunk)
            atoms = slice(slices[start].start, slices[stop-1].stop)
            onehot = torch.nn.functional.one_hot(out['normalized_dist'][atoms].argmax(1), 98).to(context['targets'][0].dtype)
            g, e = author.Crystalformer_prediction(gap_model, ef_model, onehot, out['pos'][atoms],
                context['batch'][atoms]-start, out['lattice_vectors'][start:stop], context['size'][start:stop])
            gaps.append(g.reshape(-1))
            energies.append(e.reshape(-1))
        return torch.cat(gaps), torch.cat(energies)

    context.update(forward=forward, hard_predict=hard_predict)

    with torch.no_grad():
        soft_gap, soft_ef, _, initial = forward(.01)
        hard_numbers = initial['normalized_dist'].argmax(1)+1
        if not torch.equal(hard_numbers, numbers):
            raise ValueError('Warm start changed initial ordered elements.')
        gap, ef = hard_predict(initial)
    recorded = {c['candidate_id']: c for c in inventory['candidates']}
    if branch_manifest:
        parent_control=args.branch_parent_control
        if parent_control['source_inventory_sha256']!=digest(source_root/'inventory-249/inventory.json'):
            raise ValueError('Missing independent original-parent control.')
        control_error=max(parent_control['hard_control_error'],float((gap-soft_gap).abs().max()),float((ef-soft_ef).abs().max()))
    else:
        control_error = max(max(abs(float(gap.reshape(-1)[i])-recorded[k]['original_gap']),
                                abs(float(ef.reshape(-1)[i])-recorded[k]['original_eform'])) for i, k in enumerate(ids))
    if control_error > 1e-3:
        raise ValueError('Template hard-prediction control failed: '+str(control_error))
    from solver import gram_check
    for i, structure in enumerate(chosen):
        if not gram_check(initial['lattice_vectors'][i].numpy(), structure.lattice.matrix)['passed']:
            raise ValueError('Input cell was clipped or distorted.')
        delta = initial['normed_batch_dir_coords'][slices[i]].numpy()-structure.frac_coords
        if np.max(np.abs(np.remainder(delta+.5, 1)-.5)) > 1e-6:
            raise ValueError('Input fractional coordinates changed.')

    # Preserve the author's neutral predicate while avoiding exponential enumeration.
    native = types.SimpleNamespace(Element_pmg=author.Element_pmg, Element=author.Element,
        elec_neutral_check_SUPER_COMMON=author.elec_neutral_check_SUPER_COMMON)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from neutrality import super_common

    def neutral_check(_index, _count, elements, stoichs, _all):
        key = tuple(sorted((e, int(c[0])) for e, c in zip(elements, stoichs)))
        return super_common(key, native)[0], [], []

    author.elec_neutral_check_SUPER_COMMON = neutral_check
    template_initialization = dict(source_inventory_sha256=digest(source_root/'inventory-249/inventory.json'),
        source_root=str(source_root), selected=selected, hard_control_error=control_error, strategy=mode,
        element_weights='uniform1/98 plus0.001 at original species; hard identity checked; soft inputs are mixtures',
        initial_hard_gap=gap.reshape(-1).tolist(), initial_soft_gap=soft_gap.tolist(),
        initial_hard_eform=ef.reshape(-1).tolist(), initial_soft_eform=soft_ef.tolist(),
        bounds='Original ranges expanded only to contain input, no clipping.',
        mask='None: actual predictor oxidation target is None; oxidation tensor is an inactive zero-gradient interface placeholder.' if unconstrained else 'Up to10 lexicographic exact neutral SUPER_COMMON assignments in input atom order.')
    template_initialization['species_policy'] = 'fixed_per_site_onehot' if fixed_species else 'optimize_elements'
    if branch_manifest:
        template_initialization.update(branch_manifest_sha256=digest(Path(args.branch_manifest)),
            branch_parent_control=args.branch_parent_control,branch_bounds=context['branch_bounds'],
            hard_control_scope='Original parents checked against frozen native reference; branches checked exact-onehot consistency, CPU/GPU equivalence and later independent frozen pose prediction.',
            gradient_normalization=1)
    if fixed_species:
        template_initialization['element_weights'] = 'Exact original per-site onehot, fixed in every predictor forward; no element or oxidation optimization.'
    (output/'initialization.json').write_text(json.dumps(template_initialization, indent=2)+'\n')
    protocol = dict(ids=ids, initial_control_error=control_error, baseline_npz_sha256=None,
        author_source_sha256=SOURCE_SHA256, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        a_parameters=['scaled lengths', 'scaled angles', 'fractional coordinates', 'element logits', 'oxidation logits'])
    if unconstrained:
        protocol['a_parameters'][-1] = 'inactive oxidation placeholder (no charge mask)'
    if fixed_species:
        protocol['a_parameters'][-2:] = ['fixed original per-site elements', 'inactive oxidation placeholder (no charge mask)']
    context.update(protocol=protocol, template_initialization=template_initialization)

    def initialization_audit(context):
        if n > 4:
            from microbatch import Microbatch
            _, gradients = Microbatch(context, size=context['microbatch_size']).gradients(
                .01, torch.ones(n, dtype=torch.bool, device=context['targets'][0].device))
        else:
            _, _, loss, _ = context['forward'](.01)
            gradients = torch.autograd.grad(loss.mean(), context['targets'])
        checks = [dict(cuda=g.is_cuda, finite=bool(torch.isfinite(g).all()), max_abs=float(g.abs().max()),
                       nonzero=int(torch.count_nonzero(g))) for g in gradients]
        if not all(r['cuda'] and r['finite'] for r in checks) or not any(r['max_abs']>1e-8 for r in checks[:3]):
            raise ValueError('External template CUDA gradients failed.')
        if fixed_species and any(r['nonzero'] for r in checks[3:]):
            raise ValueError('Fixed element or oxidation tensor has a nonzero gradient.')
        (output/'initial-gradient-audit.json').write_text(json.dumps(dict(passed=True, blocks=checks,
            strategy=mode, oxidation_active=not unconstrained, elements_active=not fixed_species,
            max_allocated_bytes=torch.cuda.max_memory_allocated(), max_reserved_bytes=torch.cuda.max_memory_reserved()), indent=2)+'\n')

    context['initialization_audit'] = initialization_audit
    return context
