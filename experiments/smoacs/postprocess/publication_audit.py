"""Verify publication archives and compare scientific source definitions locally."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def archive_member(path, metadata, suffix, root_only=False):
    data = path.read_bytes()
    if (hashlib.sha256(data).hexdigest() != metadata['sha256']
            or hashlib.md5(data).hexdigest() != metadata['md5']):
        raise ValueError('Publication archive checksum mismatch.')
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)
                   and (not root_only or name.count('/') == 1)]
        if len(matches) != 1:
            raise ValueError('Ambiguous publication archive member.')
        return archive.read(matches[0]).decode('utf-8').replace('\r\n','\n')


def relaxation_ast(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='relax_with_mattersim')
    return ast.dump(node,include_attributes=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives',type=Path,required=True)
    parser.add_argument('--dft-source',type=Path,required=True)
    parser.add_argument('--smoacs-source',type=Path,default=Path('third_party/SMOACS'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sources=json.loads(Path(__file__).with_name('sources.json').read_text(encoding='utf-8'))
    archived_relax=archive_member(args.archives/'zenodo-20149017.zip',sources['publication_archives']['smoacs'],
        '/scripts/calculate_mattersim_convex_hull_energy.py')
    archived_readme=archive_member(args.archives/'zenodo-20149017.zip',sources['publication_archives']['smoacs'],'/README.md',root_only=True)
    archived_dft=archive_member(args.archives/'zenodo-20133844.zip',sources['publication_archives']['dft'],
        '/workspace/template_ws/relax_bandgap.py')
    local_relax=(args.smoacs_source/'scripts/calculate_mattersim_convex_hull_energy.py').read_text(encoding='utf-8')
    local_dft=(args.dft_source/'workspace/template_ws/relax_bandgap.py').read_text(encoding='utf-8')
    result=dict(archives_verified=True,
                relaxed_function_ast_identical=relaxation_ast(archived_relax)==relaxation_ast(local_relax),
                dft_source_lf_identical=archived_dft==local_dft,
                later_readme_pin_present_in_publication=sources['mattersim']['unavailable_readme_commit'] in archived_readme,
                published_relax_ast_sha256=hashlib.sha256(relaxation_ast(archived_relax).encode()).hexdigest(),
                published_dft_source_lf_sha256=hashlib.sha256(archived_dft.encode()).hexdigest(),
                publication_archives=sources['publication_archives'])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['relaxed_function_ast_identical'] or not result['dft_source_lf_identical']:
        raise RuntimeError('Scientific source differs from the publication archive.')


if __name__=='__main__':
    main()
