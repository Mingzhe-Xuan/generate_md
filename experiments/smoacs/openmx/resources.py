"""Audit installed PBE19 potentials and matching orbital files; no dataset edits."""
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keyword(text, name):
    found = re.findall(r'^\s*' + re.escape(name) + r'\s+(\S+)', text, re.M | re.I)
    if not found:
        raise ValueError('Missing dataset keyword: ' + name)
    return float(found[-1])


def select_species(element, data):
    data = Path(data)
    # H/S variants change semicore valence; match their orbital family explicitly.
    for variant in ('', 'H', 'S'):
        vps = data/'VPS'/f'{element}_PBE19{variant}.vps'
        if not vps.is_file():
            continue
        choices = []
        for pao in (data/'PAO').glob(f'{element}*.pao'):
            match = re.fullmatch(re.escape(element)+r'(\d+\.\d+)'+variant+r'\.pao', pao.name)
            if match and float(match[1]) >= 6:
                choices.append((float(match[1]), pao))
        for radius, pao in sorted(choices):
            vt = vps.read_text(errors='strict')
            pt = pao.read_text(errors='strict')
            valence = keyword(vt, 'valence.electron')
            if abs(valence-keyword(pt, 'valence.electron')) > 1e-8:
                continue
            if 'GGA' not in vt:
                raise ValueError('PBE19 dataset does not declare GGA.')
            lmax = int(keyword(pt, 'PAO.Lmax'))
            multiplicity = int(keyword(pt, 'PAO.Mul'))
            counts = [3, 3, 2, 1][:min(lmax+1, 4)]
            if multiplicity < max(counts):
                continue
            basis = pao.stem+'-'+''.join(l+str(n) for l,n in zip('spdf', counts))
            larger = pao.stem+'-'+''.join(l+str(n+1) for l,n in zip('spdf', counts))
            if multiplicity < max(counts)+1:
                raise ValueError('No supported larger basis for convergence check.')
            return dict(element=element, vps=str(vps), pao=str(pao),
                        vps_sha256=digest(vps), pao_sha256=digest(pao),
                        potential=vps.stem, basis=basis, larger_basis=larger,
                        valence=valence, radius_au=radius)
    raise FileNotFoundError('No valence-matched PBE19 VPS/PAO for '+element)


def audit(inventory, data):
    species, missing = {}, {}
    for element in sorted({e for c in inventory['candidates'] for e in c['elements']}):
        try:
            species[element] = select_species(element, data)
        except (FileNotFoundError, ValueError) as error:
            missing[element] = str(error)
    coverage = [c['candidate_id'] for c in inventory['candidates']
                if set(c['elements']) <= species.keys()]
    return dict(data_path=str(Path(data).resolve()), species=species, missing=missing,
                element_covered_ids=coverage, denominator=inventory['denominator'])
