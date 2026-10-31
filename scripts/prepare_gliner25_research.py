"""Download a pinned public checkpoint; never commit model weights."""
import json
import shutil
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

from cgraph.common import file_sha256, write_json


def main():
    repo='fastino/gliner2.5-base-v1'
    meta=requests.get(f'https://huggingface.co/api/models/{repo}?blobs=true',timeout=45)
    meta.raise_for_status()
    meta=meta.json()
    lock_path=Path('contracts/ner_v2/gliner25_research_lock.json')
    revision=json.loads(lock_path.read_text(encoding='utf8'))['revision'] if lock_path.exists() else meta['sha']
    if revision!=meta['sha']:
        response=requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',timeout=45)
        response.raise_for_status();meta=response.json()
    names=['README.md','config.json','encoder_config/config.json','model.safetensors','tokenizer.json','tokenizer_config.json']
    needed=sum(x.get('size',0) for x in meta['siblings'] if x['rfilename'] in names)
    free=shutil.disk_usage('.').free
    if free < needed*3+2**30: raise RuntimeError('Insufficient download space')
    root=Path('artifacts/ner_v2/models/gliner25')
    lock=dict(repo_id=repo,revision=revision,disk_free_before_bytes=free,download_size_bytes=needed,files={})
    write_json(lock_path,lock)
    for name in names:
        path=Path(hf_hub_download(repo,name,revision=revision,local_dir=root))
        info=next(x for x in meta['siblings'] if x['rfilename']==name)
        digest=file_sha256(path)
        if path.stat().st_size!=info['size']:raise ValueError('File size mismatch')
        if info.get('lfs',{}).get('sha256') and info['lfs']['sha256']!=digest:raise ValueError('Upstream digest mismatch')
        lock['files'][name]={'bytes':path.stat().st_size,'sha256':digest}
        write_json(lock_path,lock)
        print(name,path.stat().st_size,flush=True)
    # Embedded encoder config is authoritative; eliminate the remote base path.
    config=json.loads((root/'config.json').read_text(encoding='utf8'))
    print('config',json.dumps(config),flush=True)


if __name__=='__main__':main()
