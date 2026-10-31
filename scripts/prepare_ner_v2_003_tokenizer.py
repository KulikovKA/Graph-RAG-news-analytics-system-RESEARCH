"""Fetch Qwen tokenizer assets only; never download model weights or generate."""
from pathlib import Path
import requests
from huggingface_hub import hf_hub_download
from cgraph.common import read_json,write_json,file_sha256


def main():
    repo='Qwen/Qwen3-4B'
    path=Path('reports/ner_v2/research_run2/qwen_tokenizer_lock.json')
    if path.exists():revision=read_json(path)['revision']
    else:
        response=requests.get(f'https://huggingface.co/api/models/{repo}',timeout=40)
        response.raise_for_status();revision=response.json()['sha']
    root=Path('artifacts/ner_v2/qwen3_4b_tokenizer')
    names=['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','config.json']
    lock=dict(repo_id=repo,revision=revision,purpose='TOKENIZATION_ONLY_NO_MODEL_LOAD',files={})
    for name in names:
        file=Path(hf_hub_download(repo,name,revision=revision,local_dir=root))
        lock['files'][name]=dict(bytes=file.stat().st_size,sha256=file_sha256(file))
        write_json(path,lock);print(name,file.stat().st_size,flush=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True)
    ids=tokenizer.apply_chat_template([{'role':'user','content':'Classify the scientific mentions.'}],
                                     tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    print('tokenizer-only smoke tokens',len(ids),flush=True)


if __name__=='__main__':main()
