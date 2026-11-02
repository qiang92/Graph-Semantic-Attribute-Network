"""Optionally regenerate TEP language inputs through a local Ollama API."""
import argparse
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

import numpy as np


def post(base, endpoint, payload):
    request=Request(base.rstrip('/')+endpoint,data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=900) as response:
        return json.load(response)


def clean(text):
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
    return text.strip('"\'').strip()


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--base-url',default='http://localhost:11434')
    cli.add_argument('--generator',default='qwen3:14b')
    cli.add_argument('--encoder',default='qwen3-embedding:0.6b-q8_0')
    cli.add_argument('--prompt',choices=['structured','concise'],default='structured')
    cli.add_argument('--reuse-primary-texts',action='store_true',help='Encoder comparison: keep the exact primary descriptions.')
    cli.add_argument('--output',required=True,type=Path)
    args=cli.parse_args()
    root=Path(__file__).resolve().parent
    primary=json.loads((root/'data/semantic_cache/tep_fault_semantics_qwen3_embedding.json').read_text(encoding='utf-8'))
    template=(root/f'data/language/{args.prompt}_prompt.txt').read_text(encoding='utf-8')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new directory to retain the published frozen inputs.')
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for original in primary['records']:
        row=dict(original)
        if not args.reuse_primary_texts:
            payload={'model':args.generator,'prompt':template.format(**row).strip(),'stream':False,
                     'options':{'temperature':.2,'top_p':.9,'num_predict':700}}
            response=post(args.base_url,'/api/generate',payload)
            text=clean(response['response'])
            row['semantic_text']=text if len(text.split())>=30 else row['base_description']
        records.append(row)
    response=post(args.base_url,'/api/embed',{'model':args.encoder,'input':[r['semantic_text'] for r in records]})
    vectors=np.asarray(response['embeddings'],dtype=np.float32)
    np.save(args.output/'embeddings.npy',vectors)
    (args.output/'descriptions.json').write_text(json.dumps({
        'generate_model':args.generator,'embed_model':args.encoder,'prompt':args.prompt,
        'class_order':primary['class_order'],'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'records':len(records),'embedding_shape':list(vectors.shape)}))


if __name__=='__main__':
    main()
