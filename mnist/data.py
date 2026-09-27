"""Download and verify the four original MNIST IDX gzip files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np
ROOT=Path(__file__).resolve().parent
def load_mnist(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'config/data_manifest.json').read_text())
    for name,meta in manifest.items():
        path=folder/name
        if not path.exists():
            print('Downloading',name,flush=True)
            with urlopen(meta['url'],timeout=120) as response:data=response.read()
            if hashlib.sha256(data).hexdigest()!=meta['sha256']:raise ValueError('Download checksum mismatch: '+name)
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Checksum mismatch: '+str(path))
    images=[];labels=[]
    for prefix in ['train','t10k']:
        raw=gzip.decompress((folder/f'{prefix}-images-idx3-ubyte.gz').read_bytes())
        header=np.frombuffer(raw,dtype='>u4',count=4);assert header[0]==2051 and tuple(header[2:])==(28,28)
        images.append(np.frombuffer(raw,dtype=np.uint8,offset=16).reshape(int(header[1]),784))
        raw=gzip.decompress((folder/f'{prefix}-labels-idx1-ubyte.gz').read_bytes())
        header=np.frombuffer(raw,dtype='>u4',count=2);assert header[0]==2049
        labels.append(np.frombuffer(raw,dtype=np.uint8,offset=8));assert len(labels[-1])==header[1]
    x=np.concatenate(images);y=np.concatenate(labels);assert x.shape==(70000,784) and y.shape==(70000,)
    return x,y
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    args=parser.parse_args();x,y=load_mnist(args.data_dir);print('Verified',x.shape,len(y))
