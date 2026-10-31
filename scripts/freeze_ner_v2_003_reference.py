"""Assistant reference authored from six frozen abstracts BEFORE NER inference.

No predictions are opened by this script. This is not independent human Gold.
"""
import re
from pathlib import Path
from cgraph.common import file_sha256, read_json, write_json
from cgraph.ner_v2.research import jsonl
from cgraph.ner_v2.io import write_jsonl

ROOT=Path('reports/ner_v2/research_run2')
TERMS={
    'W4230116451': {
        'OBJECT':['advanced computing systems','graphics processing units','GPUs','distributed cloud e-health system'],
        'DATA_RESOURCE':['biomedical data'],
        'METHOD':['magnetic resonance imaging','MRI','GPU parallel algorithm','optimised blockwise non-local means','OB-NLM'],
        'PROCESS':['denoising'],
        'TASK':['real-time diagnosis'],
        'ATTRIBUTE':['execution time'],
        'RESULT':['significant improvement of healthcare processing practice'],
    },
    'W2791492120': {
        'CONCEPT':['chronic rhinosinusitis with nasal polyposis','CRSwNP','autoimmune disorders','systemic lupus erythematosus'],
        'OBJECT':['B cells','Breg','T cell','Polypoid tissue','blood samples','lymphocytes','nasal polyps',
                  'peripheral blood mononuclear cells','PBMC','Mature resting B cells','memory B cells','mature B cells'],
        'PROCESS':['suppression of T cell activity','paranasal sinus surgery'],
        'METHOD':['multicolor flow cytometry'],
        'RESULT':['significantly lower frequency of B cells in nasal polyps'],
        'TASK':['characterize subpopulations of peripheral and edaphic B cells'],
        'ATTRIBUTE':['frequency of B cells'],
        'CONDITION':['10 patients'],
    },
    'W2317432785': {
        'ATTRIBUTE':['electronic structure','magnetic properties','thicknesses of sublayers','repetition periods','sublayer thicknesses'],
        'OBJECT':['multilayers','Multilayers','sublayers','silicides','grains','a-Si layer'],
        'METHOD':['macroscopic magneto-optical transversal Kerr effect','TKE','low-angle X-ray reflectivity method',
                  'X-ray absorption spectroscopy','XANES','X-ray magnetic circular dichroism','XMCD'],
        'PROCESS':['grain formation'],
        'RESULT':['magneto-optical TKE effect enhancement','huge reduction'],
        'CONCEPT':['K absorption edges'],
    },
    'W2076293586': {
        'OBJECT':['inbred mouse models','CD-1 outbred population','animals','mouse lines','genomic loci'],
        'CONCEPT':['psychiatric diseases','depression','anxiety disorders','pathological stress reactivity','endophenotypes'],
        'PROCESS':['gene expression'],
        'METHOD':['inbreeding protocol','Two array approaches','aCGH','SNP-based invariant genomic probes','association studies'],
        'ACTOR':['NimbleGen','Jackson Laboratories'],
        'DATA_RESOURCE':['CNV data sets'],
        'ATTRIBUTE':['copy number variants','CNVs','gene dosage','copy numbers'],
        'RESULT':['approximately 100 CNV regions ranging from around 10 kb to 7 Mb'],
        'TASK':['discovery of copy number variants'],
    },
    'W2906067930': {
        'CONCEPT':['Harvest Paradox','supply and demand theorem','non-grain price factors'],
        'ACTOR':['China','Samuelson','Nordhaus','Chinese government'],
        'PROCESS':['grain production','grain purchase and sales'],
        'ATTRIBUTE':['domestic residents’ income','food demand',"farmers' total income",'grain price'],
        'CONDITION':['from 2003 to 2014'],
        'METHOD':['methods of Samuelson and Nordhaus'],
        'MECHANISM':['demand line moved in excess of the supply line'],
        'RESULT':['grain price rose'],
        'TASK':['solving food problems of China'],
    },
    'W4312422299': {
        'DATA_RESOURCE':['The Book of Mormon'],
        'ACTOR':['Nephite people','Lehi','God','Nephite race','Nephites'],
        'OBJECT':['biblical Jerusalem','Jerusalem'],
        'PROCESS':['movement and escape','reproduction','avoiding annihilation'],
        'CONCEPT':['destruction'],
    },
}


def main():
    manifest=read_json(ROOT/'corpus_manifest.json')
    assert set(TERMS)==set(manifest['audit_document_ids'])
    assert not any(ROOT.glob('E*/documents/*.json')),'Freeze reference before any model predictions'
    docs={d['document_id']:d for d in jsonl(ROOT/'audit_abstracts_before_predictions.jsonl')}
    result=[]
    for doc_id,groups in TERMS.items():
        doc=docs[doc_id];seen=set()
        for label,terms in groups.items():
            for term in terms:
                matches=list(re.finditer(re.escape(term),doc['text']))
                assert matches,(doc_id,term)
                for match in matches:
                    a,b=match.span()
                    # Avoid inner-word matches, preserve intended nested phrases.
                    if (a and doc['text'][a-1].isalnum()) or (b<len(doc['text']) and doc['text'][b].isalnum()):continue
                    if (a,b) in seen:continue
                    seen.add((a,b))
                    result.append(dict(reference_id=f'{doc_id}:{a}:{b}',document_id=doc_id,
                         discipline=doc['discipline'],source_url=doc['source_url'],text_sha256=doc['text_sha256'],
                         start=a,end=b,surface=doc['text'][a:b],suggested_type=label,
                         significance='SCIENTIFICALLY_RELEVANT_DRAFT',status='ASSISTED_REFERENCE_DRAFT',
                         review_status='UNREVIEWED',origin='assistant_authored_from_source_before_NER',
                         context=doc['text'][max(0,a-70):min(len(doc['text']),b+70)],
                         human_review={'span_category':None,'type_category':None,'reviewer':None,'notes':None}))
    write_jsonl(ROOT/'assisted_reference_frozen.jsonl',sorted(result,key=lambda x:(x['document_id'],x['start'],x['end'])))
    write_json(ROOT/'reference_lock.json',dict(status='FROZEN_BEFORE_NER',reference_sha256=file_sha256(ROOT/'assisted_reference_frozen.jsonl'),
               corpus_sha256=manifest['input_sha256'],audit_document_ids=manifest['audit_document_ids'],
               references=len(result),by_discipline={d:sum(r['discipline']==d for r in result) for d in sorted({x['discipline'] for x in result})},
               policy='Source-authored selected expressions, all exact occurrences, nested spans allowed; not exhaustive; draft types separate from boundary matching',
               caveats=['No independent human annotator','Not Gold','Humanities include narrative actors and textual sources',
                        'Some source formula strings contain U+FFFD; original text is preserved']))
    print('frozen reference spans',len(result),read_json(ROOT/'reference_lock.json')['by_discipline'])


if __name__=='__main__':main()
