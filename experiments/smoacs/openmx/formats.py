"""Explicit OpenMX inputs and fail-closed scientific output validation."""
import math
import re
import numpy as np

HARTREE_EV = 27.211386245988
GRADIENT_LIMIT = 1e-4


def input_text(atoms, species, data_path, grid, relax=True, cutoff=200, larger=False, physics=None):
    elements = list(dict.fromkeys(atoms.get_chemical_symbols()))
    hubbard_enabled = bool(physics and any(v['U_eff_eV'] != 0 for v in physics['hubbards'].values()))
    lines = ['System.Name calc', 'System.CurrrentDir ./', f'DATA.PATH {data_path}',
             'level.of.stdout 1', 'level.of.fileout 0', f'Species.Number {len(elements)}',
             '<Definition.of.Atomic.Species']
    for e in elements:
        s = species[e]
        lines.append(f"{e} {s['larger_basis' if larger else 'basis']} {s['potential']}")
    lines += ['Definition.of.Atomic.Species>', f'Atoms.Number {len(atoms)}',
              'Atoms.SpeciesAndCoordinates.Unit Ang', '<Atoms.SpeciesAndCoordinates']
    for i, (e, xyz) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions), 1):
        valence = species[e]['valence']
        moment = physics['initial_moments_muB'][e] if physics else min(valence, 1.0)
        if abs(moment) > valence:
            raise ValueError('Initial moment exceeds available valence electrons.')
        lines.append(f'{i} {e} '+' '.join(f'{v:.14f}' for v in xyz)
                     + f' {(valence+moment)/2:.6f} {(valence-moment)/2:.6f}'
                     # Collinear +U requires an eighth enable token; omission exits0.
                     + (' on' if hubbard_enabled else ''))
    lines += ['Atoms.SpeciesAndCoordinates>', 'Atoms.UnitVectors.Unit Ang', '<Atoms.UnitVectors']
    lines += [' '.join(f'{v:.14f}' for v in row) for row in atoms.cell]
    lines += ['Atoms.UnitVectors>', 'scf.XcType GGA-PBE', 'scf.SpinPolarization on',
              'scf.ElectronicTemperature 300', f'scf.energycutoff {cutoff}',
              'scf.maxIter 300', 'scf.EigenvalueSolver band',
              'scf.Kgrid '+' '.join(map(str, grid)), 'scf.Mixing.Type rmm-diisk',
              'scf.Init.Mixing.Weight 0.01', 'scf.Min.Mixing.Weight 0.001',
              'scf.Max.Mixing.Weight 0.2', 'scf.Mixing.History 20',
              'scf.Mixing.StartPulay 10', 'scf.criterion 1e-8',
              'scf.restart off', 'MD.Type '+('RFC5' if relax else 'nomd'),
              'MD.maxIter '+('200' if relax else '1'), 'MD.Opt.criterion 1e-4',
              'MD.Opt.DIIS.History 7', 'MD.Opt.StartDIIS 5']
    if hubbard_enabled:
        lines += ['scf.Hubbard.U on', 'scf.DFTU.Type 1', 'scf.Hubbard.Occupation dual',
                  '<Hubbard.U.values']
        for e in elements:
            choice = species[e]['larger_basis' if larger else 'basis'].rsplit('-', 1)[1]
            values = []
            for angular, count in re.findall(r'([spdf])(\d+)', choice):
                for radial in range(1, int(count)+1):
                    setting = physics['hubbards'][e]
                    u = setting['U_eff_eV'] if radial == 1 and 'spdf'.index(angular) == setting['angular_momentum'] else 0
                    values += [str(radial)+angular, f'{u:.8f}']
            lines.append(e+' '+' '.join(values))
        lines.append('Hubbard.U.values>')
    return '\n'.join(lines)+'\n'


def validate_log(log, relax):
    if 'The SCF was not fully achieved' in log:
        raise ValueError('An SCF step failed convergence.')
    achieved = re.findall(r'The SCF was achieved at MD=\s*(\d+)', log)
    if not achieved:
        raise ValueError('No explicit successful SCF status.')
    if relax:
        if 'The geometry optimization was achieved.' not in log:
            raise ValueError('No explicit successful geometry status.')
        forces = re.findall(r'\|Maximum (?:gradient|force|dE/da)\|\s*\(Hartree/Bohr\)\s*=\s*([\d.Ee+\-]+)', log)
        if not forces or not math.isfinite(float(forces[-1])) or float(forces[-1]) >= GRADIENT_LIMIT:
            raise ValueError('Variable-cell optimization did not converge.')
    return dict(scf_steps=len(achieved), final_md=int(achieved[-1]),
                generalized_gradient_Ha_Bohr=float(forces[-1]) if relax else None)


def block(text, name):
    matches = re.findall(r'^\s*<'+re.escape(name)+r'\s*\n(.*?)^\s*'+re.escape(name)+r'>', text, re.M | re.S | re.I)
    if not matches:
        raise ValueError('Missing block: '+name)
    return [line.split('#')[0].split() for line in matches[-1].splitlines() if line.split('#')[0].strip()]


def restart_atoms(text):
    from ase import Atoms
    rows = block(text, 'Atoms.SpeciesAndCoordinates')
    cell = np.array(block(text, 'Atoms.UnitVectors'), dtype=float)
    xyz = np.array([row[2:5] for row in rows], dtype=float)
    for keyword in ('Atoms.SpeciesAndCoordinates.Unit', 'Atoms.UnitVectors.Unit'):
        found = re.findall(r'^\s*'+re.escape(keyword)+r'\s+(\S+)', text, re.I | re.M)
        if not found or found[-1].lower() != 'ang':
            raise ValueError('Expected explicit Ang restart units.')
    if not np.isfinite(xyz).all() or not np.isfinite(cell).all() or np.linalg.det(cell) <= 0:
        raise ValueError('Invalid final geometry.')
    return Atoms(symbols=[row[1] for row in rows], positions=xyz, cell=cell, pbc=True)


def parse_output(text):
    if 'Total Computational Time' not in text:
        raise ValueError('Missing normal completion footer.')
    energy = re.findall(r'^\s*Utot\.\s+([\d.Ee+\-]+)', text, re.M)
    mu = re.findall(r'Chemical Potential \(Har?tree\)\s*=\s*([\d.Ee+\-]+)', text)
    if not energy or not mu or not all(math.isfinite(float(v)) for v in (energy[-1], mu[-1])):
        raise ValueError('Missing or nonfinite energy/chemical potential.')
    eigen_section = text.rsplit('Eigenvalues (Hartree) of SCF KS-eq.', 1)[-1]
    if 'Eigenvalues (Hartree) of SCF KS-eq.' not in text:
        raise ValueError('No eigenvalue section.')
    groups = []
    for section in re.split(r'\bkloop\s*=\s*\d+', eigen_section)[1:]:
        bands = re.findall(r'^\s*(\d+)\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)\s*$', section.split('****')[0], re.M)
        if not bands:
            raise ValueError('Incomplete k-point eigenvalues.')
        if [int(row[0]) for row in bands] != list(range(1, len(bands)+1)):
            raise ValueError('Noncontiguous band indices.')
        groups.append([[float(b[1]), float(b[2])] for b in bands])
    if not groups or len({len(g) for g in groups}) != 1:
        raise ValueError('Missing/inconsistent eigenvalue blocks.')
    eigen = np.asarray(groups).transpose(2, 0, 1)
    if not np.isfinite(eigen).all():
        raise ValueError('Nonfinite bands.')
    chemical = float(mu[-1])
    occ = eigen < chemical
    counts = occ.sum(axis=2)
    metal = any(len(set(c)) != 1 for c in counts) or np.min(np.abs(eigen-chemical))*HARTREE_EV < .01
    if not occ.any() or occ.all():
        raise ValueError('Missing occupied or unoccupied bands.')
    vbm, cbm = eigen[occ].max(), eigen[~occ].min()
    return dict(energy_eV=float(energy[-1])*HARTREE_EV,
                chemical_potential_eV=chemical*HARTREE_EV,
                sampled_bandgap_eV=0.0 if metal else float((cbm-vbm)*HARTREE_EV),
                metal_on_sampled_mesh=bool(metal), spin_channels=2,
                sampled_kpoints=int(eigen.shape[1]), bands_per_spin=int(eigen.shape[2]))
