"""Integrity contracts protect benchmark bytes, splits and existing task data."""
import csv
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from integrity import extract_checked, hashes, verify_asset
from verify import scan_csv


class DatasetIntegrity(unittest.TestCase):
    def archive(self, root, name='reference/file.cif', data=b'original', symlink=False):
        path = root / 'data.tar.gz'
        with tarfile.open(path, 'w:gz') as handle:
            member = tarfile.TarInfo(name)
            if symlink:
                member.type = tarfile.SYMTYPE
                member.linkname = '../outside'
                handle.addfile(member)
            else:
                member.size = len(data)
                handle.addfile(member, io.BytesIO(data))
        return path

    def test_git_blob_hash_and_tampered_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'asset'
            path.write_bytes(b'')
            self.assertEqual(hashes(path)['git_sha1'], 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')
            path.write_bytes(b'abcd')
            spec = dict(size=4, algorithm='md5', checksum=hashes(path)['md5'])
            verify_asset(path, spec)
            path.write_bytes(b'abce')
            with self.assertRaises(ValueError):
                verify_asset(path, spec)

    def test_safe_extraction_is_repeatable_and_preserves_different_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self.archive(root)
            output = root / 'out'
            extract_checked(archive, output)
            extract_checked(archive, output)
            target = output / 'reference/file.cif'
            self.assertEqual(target.read_bytes(), b'original')
            target.write_bytes(b'other result')
            with self.assertRaises(ValueError):
                extract_checked(archive, output)
            self.assertEqual(target.read_bytes(), b'other result')

    def test_rejects_path_escape_and_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('../outside', '/absolute', 'C:/outside', 'dir\\outside'):
                with self.assertRaises(ValueError):
                    extract_checked(self.archive(root, name), root / 'out')
            with self.assertRaises(ValueError):
                extract_checked(self.archive(root, symlink=True), root / 'out')

    def test_csv_multiline_cif_rows_and_duplicate_ids_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['material_id', 'cif', 'pretty_formula'])
                writer.writeheader()
                writer.writerows([dict(material_id='mp-1', cif='data_A\nloop_\n', pretty_formula='Si'),
                                  dict(material_id='mp-1', cif='data_B\n', pretty_formula='Si')])
            summary, ids, examples = scan_csv(path)
            self.assertEqual(summary['rows'], 2)
            self.assertEqual(summary['duplicate_ids'], {'mp-1': 2})
            self.assertEqual(ids, {'mp-1'})
            self.assertEqual(examples[0]['cif'], 'data_A\nloop_\n')
            self.assertEqual(examples[1]['cif'], 'data_B\n')

    def test_rejects_invalid_or_empty_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            for text in ('wrong,header\n1,2\n', 'material_id,cif,pretty_formula\n',
                         'material_id,cif,pretty_formula\nmp-1,,Si\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    scan_csv(path)


if __name__ == '__main__':
    unittest.main()
