"""Deterministic source mapping and ontology/qualifier gates; no semantic oracle."""
import json
import re
from cgraph.rules import evidence, link
from cgraph.validation import validate_chunk
from cgraph.zeroshot_re import candidate_pairs as c_candidate_pairs
from cgraph.zeroshot_rules import linguistic_negations
from .windows import anchor


def process(doc, chunk, responses, ontology):
    entities, by_key, raw, accepted, validated, rejected, unresolved = [], {}, [], [], [], [], []
    candidate_keys = set()
    raw_keys, accepted_keys = set(), set()
    rows_by_id = {}
    for rows, response in responses:
        rows_by_id.update({r['id']: r for r in rows})
        local = []
        for index, ent in enumerate(response['entities']):
            try:
                a, b = anchor(ent, rows)
                key = (a, b, ent['k'])
                if key not in by_key:
                    identifier = f"{chunk['chunk_id']}:m:{a}:{b}:{ent['k']}"
                    mention = dict(mention_id=identifier, start=a, end=b, surface_text=chunk['text'][a:b],
                                   kind=ent['k'], contextual_role=ent['r'], score=None,
                                   source_spans=[evidence(doc, chunk, a, b)])
                    entities.append(mention)
                    by_key[key] = mention
                local.append(by_key[key])
            except (ValueError, KeyError, TypeError) as exc:
                local.append(None)
                rejected.append(dict(stage='entity_mapping', index=index, prediction=ent, reason=str(exc)))
        for item in response['unresolved']:
            try:
                a, b = anchor(item, rows)
                unresolved.append(dict(reason=item['reason'], evidence=evidence(doc, chunk, a, b)))
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append(dict(stage='unresolved_mapping', prediction=item, reason=str(exc)))
        for relation_index, rel in enumerate(response['relations']):
            try:
                def endpoint(spec):
                    a, b = anchor(dict(s=spec['s'], n=rows[0]['id'], o=spec['o']), rows)
                    key = (a, b, spec['k'])
                    if key not in by_key:
                        identifier = f"{chunk['chunk_id']}:m:{a}:{b}:{spec['k']}"
                        mention = dict(mention_id=identifier, start=a, end=b, surface_text=chunk['text'][a:b],
                                       kind=spec['k'], contextual_role='UNKNOWN', score=None,
                                       source_spans=[evidence(doc, chunk, a, b)])
                        entities.append(mention)
                        by_key[key] = mention
                        local.append(mention)
                    return by_key[key]
                h, t = endpoint(rel['h']), endpoint(rel['t'])
                ids = set(rel['e'])
                selected = [r for r in rows if r['id'] in ids]
                if len(selected) != len(ids) or not selected:
                    raise ValueError('INVALID_EVIDENCE_SENTENCE')
                start, end = min(r['start'] for r in selected), max(r['end'] for r in selected)
                if not (start <= min(h['start'], t['start']) and max(h['end'], t['end']) <= end):
                    raise ValueError('EVIDENCE_DOES_NOT_COVER_ENDPOINTS')
                if ids != set(range(min(ids), max(ids) + 1)):
                    raise ValueError('NONCONTIGUOUS_EVIDENCE')
                ev = evidence(doc, chunk, start, end)
                key = (h['mention_id'], rel['r'], t['mention_id'], rel['p'], rel['m'],
                       json.dumps(rel['q'], sort_keys=True, ensure_ascii=False))
                identifier = f"{chunk['chunk_id']}:r:{rows[0]['id']}:{relation_index}"
                base = dict(assertion_id=identifier, head_mention_id=h['mention_id'],
                            tail_mention_id=t['mention_id'], relation_type=rel['r'], evidence=[ev])
                # Raw metric: one directed typed edge, with repeated overlapping windows deduplicated.
                if key not in raw_keys:
                    raw.append(base)
                    raw_keys.add(key)
                rule = ontology['relations'][rel['r']]
                reasons = []
                if h['kind'] not in rule['head_kinds'] or t['kind'] not in rule['tail_kinds']:
                    reasons.append('ONTOLOGY_DOMAIN_RANGE')
                if rel['d'] == 'ABSTAIN':
                    reasons.append('MODEL_ABSTAIN')
                if rel['p'] not in rule['supported_polarities'] or rel['m'] not in rule['supported_modalities']:
                    reasons.append('UNKNOWN_OR_UNSUPPORTED_POLARITY_MODALITY')
                # Detect direct scope contradictions without treating the chemical NO as English "no".
                left, right = sorted((h, t), key=lambda x: x['start'])
                between = chunk['text'][left['end']:right['start']] if left['end'] <= right['start'] else ''
                if rel['p'] == 'POSITIVE' and linguistic_negations(between):
                    reasons.append('EXPLICIT_NEGATION_CONFLICT')
                if rel['p'] == 'NEGATIVE' and not linguistic_negations(ev['text']):
                    reasons.append('NEGATION_NOT_IN_EVIDENCE')
                qout, scope = [], []
                allowed = set(rule['required_qualifiers']) | set(rule['optional_qualifiers'])
                for q in rel['q']:
                    try:
                        qa, qb = anchor(q, rows)
                        if q['n'] not in ids or not start <= qa < qb <= end:
                            raise ValueError('QUALIFIER_OUTSIDE_EVIDENCE')
                        if q['k'] not in allowed:
                            raise ValueError('QUALIFIER_NOT_ALLOWED')
                        enum = ontology['qualifiers'][q['k']].get('enum')
                        if enum:
                            if q['v'] not in enum:
                                raise ValueError('QUALIFIER_ENUM_INVALID')
                            word = q['v'].lower()
                            literal = q['s'].lower()
                            stems = {'increase': ('increas',), 'decrease': ('decreas', 'reduc'),
                                     'unchanged': ('unchanged',), 'unspecified': ('unspecified',)}
                            if not any(x in literal for x in stems.get(word, (word,))):
                                raise ValueError('QUALIFIER_ENUM_NOT_GROUNDED')
                        elif not q['v'] or q['v'] not in q['s']:
                            raise ValueError('QUALIFIER_VALUE_NOT_EXACT')
                        if q['u'] and q['u'] not in q['s']:
                            raise ValueError('QUALIFIER_UNIT_NOT_EXACT')
                        types = ontology['qualifiers'][q['k']]['raw_value_types']
                        raw_value = q['v']
                        if 'string' not in types:
                            if 'number' not in types or not re.fullmatch(r'[+-]?\d+(?:\.\d+)?', q['v']):
                                raise ValueError('QUALIFIER_VALUE_TYPE_UNSUPPORTED')
                            raw_value = float(q['v']) if '.' in q['v'] else int(q['v'])
                        span = evidence(doc, chunk, qa, qb)
                        scope.append(span)
                        qout.append(dict(qualifier_type=q['k'], raw_value=raw_value, normalized_value=None,
                                         raw_text=q['s'], unit=q['u'] or None, evidence_refs=[span['span_id']],
                                         applies_to_assertion_id=identifier))
                    except (ValueError, KeyError, TypeError) as exc:
                        # Retain uncertain attachment separately, never invent an assignment.
                        unresolved.append(dict(reason='UNSUPPORTED', source_prediction=q, detail=str(exc)))
                if not set(rule['required_qualifiers']) <= {q['qualifier_type'] for q in qout}:
                    reasons.append('REQUIRED_QUALIFIER_UNSUPPORTED')
                record = dict(base, polarity=rel['p'], modality=rel['m'], qualifiers=qout,
                              qualifier_status='UNKNOWN', scope_evidence=scope,
                              semantic_verification='NOT_INDEPENDENTLY_VERIFIED',
                              decision_status='ACCEPTED_PROVISIONAL' if not reasons else 'ABSTAIN',
                              reasons=reasons)
                validated.append(record)
                if reasons:
                    rejected.append(dict(stage='ontology_scope', prediction=record, reason=';'.join(reasons)))
                elif key not in accepted_keys:
                    accepted.append(record)
                    accepted_keys.add(key)
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append(dict(stage='relation_mapping', prediction=rel, reason=str(exc)))
        # Pair coverage includes exact surfaces first proposed by relations.
        for h in local:
            for t in local:
                if h and t and h['mention_id'] != t['mention_id']:
                    candidate_keys.add((h['mention_id'], t['mention_id']))
    window_pairs = [dict(head_mention_id=h, tail_mention_id=t, model_served=True,
                  decision_status='ACCEPTED_PROVISIONAL' if any(a['head_mention_id']==h and a['tail_mention_id']==t for a in accepted) else 'ABSTAIN',
                  reasons=[], unsupported_labels=[])
             for h, t in sorted(candidate_keys)]
    # C's exact single-sentence candidate definition, for a directly comparable
    # diagnostic. B itself serves wider windows; it does not prune on this list.
    pairs = [dict(head_mention_id=p['head']['mention_id'], tail_mention_id=p['tail']['mention_id'],
                  model_served=(p['head']['mention_id'], p['tail']['mention_id']) in candidate_keys,
                  decision_status='ACCEPTED_PROVISIONAL' if any(a['head_mention_id']==p['head']['mention_id'] and a['tail_mention_id']==p['tail']['mention_id'] for a in accepted) else 'ABSTAIN',
                  reasons=[], unsupported_labels=[]) for p in c_candidate_pairs(chunk, entities, 1)]
    out = dict(chunk_id=chunk['chunk_id'], source_text_sha256=chunk['text_sha256'], section=chunk['section'],
               language=chunk['language'], section_start=chunk['section_start'], section_end=chunk['section_end'],
               source_text=chunk['text'], entities=entities, candidate_pairs=pairs, raw_proposals=raw,
               structurally_validated=validated, assertions=accepted, rejected=rejected,
               unresolved=unresolved, window_pairs=window_pairs, stage_ms={})
    validate_chunk(doc, chunk, out, ontology)
    return out


def finish_document(doc, chunks, fingerprint, execution):
    concepts, identities = link(doc, chunks)
    return dict(schema_version='bgraph-v1', architecture_id='B-001', mode='end_to_end',
                document_code=doc['document_code'], document_id=doc['document_id'], revision_id=doc['revision_id'],
                split='DEV', ontology_version='1.0', config_sha256=fingerprint, chunks=chunks,
                concepts=concepts, identity_decisions=identities, execution=execution,
                extraction_status='PARTIAL' if any(c['assertions'] for c in chunks) else 'ABSTAINED',
                limitations=['Exact evidence is not semantic verification', 'Qualifier completeness UNKNOWN'])
