"""Prompt/schema generation from the already frozen label-free ontology."""
import json
from pathlib import Path


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def schema(ontology):
    string = dict(type='string')
    integer = dict(type='integer', minimum=0)
    anchored = dict(s=dict(type='string', minLength=1), n=integer, o=dict(type='integer', minimum=1))
    qualifier = obj(dict(anchored, k=dict(enum=list(ontology['qualifiers'])), v=string, u=string))
    return obj(dict(
        entities=dict(type='array', items=obj(dict(anchored,
            k=dict(enum=list(ontology['entities'])),
            r=dict(enum=['TARGET', 'COMPONENT', 'BEARER', 'METHOD', 'CONDITION', 'UNKNOWN'])))),
        relations=dict(type='array', items=obj(dict(
            h=integer, t=integer, r=dict(enum=list(ontology['relations'])),
            e=dict(type='array', items=integer, minItems=1, uniqueItems=True),
            p=dict(enum=['POSITIVE', 'NEGATIVE', 'UNKNOWN']),
            m=dict(enum=['ASSERTED', 'PROPOSED', 'HYPOTHETICAL', 'COMPARED', 'UNKNOWN']),
            q=dict(type='array', items=qualifier), d=dict(enum=['ASSERT', 'ABSTAIN'])))),
        unresolved=dict(type='array', items=obj(dict(anchored, reason=dict(enum=['UNKNOWN', 'UNSUPPORTED']))))))


def system_prompt(cfg, ontology):
    lines = [f"{k}: {v['definition']}" for k, v in ontology['entities'].items()]
    for k, v in ontology['relations'].items():
        lines.append(f"{k} ({','.join(v['head_kinds'])} -> {','.join(v['tail_kinds'])}): "
                     f"{v['definition']} Required qualifiers: {json.dumps(v['required_qualifiers'])}. "
                     f"Optional: {','.join(v['optional_qualifiers'])}.")
    return Path(cfg['prompt']).read_text(encoding='utf-8').replace('{ontology}', '\n'.join(lines))
