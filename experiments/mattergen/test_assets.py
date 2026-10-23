import hashlib
import tempfile
from pathlib import Path
import unittest

from fetch import download


class Response:
    def __init__(self,body,start,end,total,etag='same'):
        self.content=body;self.status_code=206;self.headers={'Content-Range':f'bytes {start}-{end}/{total}','ETag':etag}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):pass


class Session:
    def __init__(self,body,fail=None):self.body=body;self.fail=fail;self.calls=[]
    def get(self,url,headers,timeout):
        start,end=map(int,headers['Range'].split('=')[1].split('-'));self.calls.append(start)
        value=self.body[start:end+1]
        if start==self.fail:value=value[:-1]
        return Response(value,start,end,len(self.body))


class AssetTests(unittest.TestCase):
    def test_truncation_preserves_verified_prefix_and_resume(self):
        body=b'abcdefghijk';sha=hashlib.sha256(body).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'asset.ckpt';session=Session(body,fail=4)
            with self.assertRaisesRegex(ValueError,'Truncated'):download(session,'https://official/asset',target,sha,len(body),chunk_size=4)
            self.assertEqual(target.with_name('asset.ckpt.part').read_bytes(),b'abcd');self.assertFalse(target.exists())
            resumed=Session(body);self.assertEqual(download(resumed,'https://official/asset',target,sha,len(body),chunk_size=4),sha)
            self.assertEqual(resumed.calls,[0,4,8]);self.assertEqual(target.read_bytes(),body)

    def test_changed_identity_and_bad_complete_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'asset.ckpt'
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):download(Session(b'abcd'),'https://official/a',target,'bad',4,chunk_size=2)
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(ValueError,'identity changed'):download(Session(b'efgh'),'https://official/b',target,None,4,chunk_size=2)


if __name__=='__main__':unittest.main()
