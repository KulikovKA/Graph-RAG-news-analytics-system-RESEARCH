"""Freeze technical expectations before experiment execution (no model outputs read)."""
import json
from pathlib import Path

from cgraph.common import write_json, file_sha256
from cgraph.ner_v2.io import read_documents


def main():
    docs = read_documents('reports/ner_v2/benchmark_run1/input_abstracts.jsonl')
    groups = {
        'Computer Science': [
            ('cars', 'car'), ("car's", 'car'), ('children', 'child'),
            ('neural networks', 'neural network'), ('convolutional neural networks', 'convolutional neural network'),
            ('BERT', 'BERT'), ('support vector machines', 'support vector machine'),
            ('random forests', 'random forest'), ('feature extraction', 'feature extraction'),
            ('graph-based models', 'graph-based model'), ('GPT-4', 'GPT-4')],
        'Medicine': [
            ('odds ratio', 'odds ratio'), ('COVID-19', 'COVID-19'), ('clinical trials', 'clinical trial'),
            ('blood samples', 'blood sample'), ('randomized controlled trials', 'randomized controlled trial'),
            ('magnetic resonance imaging', 'magnetic resonance imaging'), ('95% CI', '95% CI'),
            ('10 mg', '10 mg'), ('diabetes mellitus', 'diabetes mellitus'),
            ("patients'", 'patient'), ('reduced mortality', 'reduced mortality')],
        'Biology': [
            ('pH', 'pH'), ('ph', 'ph'), ('DNA', 'DNA'), ('gene expressions', 'gene expression'),
            ('stem cells', 'stem cell'), ('Homo sapiens', 'Homo sapiens'), ('NH₃', 'NH₃'), ('NH3', 'NH3'),
            ('protein complexes', 'protein complex'), ('β-catenin', 'β-catenin'), ('RNA-seq', 'RNA-seq')],
        'Physics': [
            ('weakly coupled plasmas', 'weakly coupled plasma'), ('graphene oxide', 'graphene oxide'),
            ('reduced graphene oxide', 'reduced graphene oxide'), ('density functional theory', 'density functional theory'),
            ('50 ℃', '50 ℃'), ('E = mc²', 'E = mc²'), ('10⁻³ mol/L', '10⁻³ mol/L'),
            ('quantum dots', 'quantum dot'), ('nonlinear waves', 'nonlinear wave'),
            ('Fe2O3', 'Fe2O3'), ('coupled plasma', 'coupled plasma')],
        'Economics': [
            ('interest rates', 'interest rate'), ('economic crises', 'economic crisis'),
            ('business cycles', 'business cycle'), ('real GNP', 'real GNP'), ('GDP', 'GDP'),
            ('supply chains', 'supply chain'), ('fixed effects', 'fixed effect'),
            ('capital markets', 'capital market'), ('increased taxation', 'increased taxation'),
            ('5%', '5%'), ('odd ratio', 'odd ratio')],
        'History': [
            ('Bronze Age', 'Bronze Age'), ('Roman Empire', 'Roman Empire'), ('trade routes', 'trade route'),
            ('archaeological sites', 'archaeological site'), ('oral histories', 'oral history'),
            ('cafe\u0301', 'café'), ('  lunar\tcalendar\n', 'lunar calendar'),
            ('high-quality objects', 'high-quality object'), ('social practices', 'social practice'),
            ('natural childbirth', 'natural childbirth'), ("'quoted terms'", "'quoted term'")],
    }
    cases = []
    for discipline, pairs in groups.items():
        for surface, expected in pairs:
            # A sentence supplies POS context. Possessive constructions have an explicit head.
            if surface == "car's":
                text = "The car's engine failed during the experiment."
            elif surface == "patients'":
                text = "The patients' responses were recorded."
            else:
                text = 'We investigated ' + surface + ' in this study.'
            start = text.index(surface)
            cases.append(dict(case_id=f'T{len(cases)+1:03}', kind='technical_control',
                              status='SPECIFICATION_NOT_GOLD', discipline=discipline,
                              text=text, start=start, end=start+len(surface), surface=surface,
                              expected=expected, rationale='Predefined conservative search-form specification'))
    real = [
        ['convolutional neural network', 'feature extraction'],
        ['DNA storage', 'parameters'], ['odds ratio', 'oocytes'], ['support groups', 'patients'],
        ['Laplacian regularisation', 'graphs'], ['weakly coupled plasmas', 'heating rates'],
        ['anthocyanins', 'temperature'], ['pH', 'enzymatic hydrolysis reactions'],
        ['trade routes', 'increased taxation'], ['recessions', 'permanent component'],
        ['animal heads', 'lunar calendar'], ['natural childbirth', 'British pioneers'],
    ]
    expected_real = [
        ['convolutional neural network','feature extraction'],['DNA storage','parameter'],
        ['odds ratio','oocyte'],['support group','patient'],['Laplacian regularisation','graph'],
        ['weakly coupled plasma','heating rate'],['anthocyanin','temperature'],
        ['pH','enzymatic hydrolysis reaction'],['trade route','increased taxation'],
        ['recession','permanent component'],['animal head','lunar calendar'],
        ['natural childbirth','British pioneer'],
    ]
    for doc, terms, expected in zip(docs, real, expected_real):
        for term, form in zip(terms, expected):
            start=doc['text'].index(term)
            cases.append(dict(case_id=f'R{len(cases)-65:03}', kind='publication_assisted',
                              status='UNREVIEWED', discipline=doc['discipline'],
                              document_id=doc['document_id'], source_url=doc['source_url'],
                              text=doc['text'], start=start,end=start+len(term),surface=term,
                              expected=form,rationale='Assistant-authored expectation; needs independent human review'))
    abbreviations = []
    for long, short in [('convolutional neural network','CNN'),('surface electromyography','sEMG'),
                        ('principal component analysis','PCA'),('density functional theory','DFT'),
                        ('magnetic resonance imaging','MRI'),('molecular dynamic','MD'),
                        ('inverse bremsstrahlung','IB'),('confidence intervals','CIs'),
                        ('polymerase chain reaction','PCR'),('gross domestic product','GDP')]:
        abbreviations.append({'text': f'We used {long} ({short}) in this study.',
                              'pairs': [[short,long]], 'status':'SPECIFICATION_NOT_GOLD'})
    abbreviations += [
        {'text':'PCA (principal component analysis) was applied.', 'pairs':[['PCA','principal component analysis']], 'status':'SPECIFICATION_NOT_GOLD'},
        {'text':'principal component analysis (PCA) differs from patient-controlled analgesia (PCA).',
         'pairs':[['PCA','principal component analysis'],['PCA','patient-controlled analgesia']], 'status':'SPECIFICATION_NOT_GOLD'},
        {'text':'BERT and CNN were used without a local definition.', 'pairs':[], 'status':'SPECIFICATION_NOT_GOLD'},
        {'text':'The sample (n=50) and dose (10 mg) were recorded.', 'pairs':[], 'status':'SPECIFICATION_NOT_GOLD'}]
    write_json('contracts/ner_v2/research_cases.json', {'seed':42, 'created_before_inference':True,
               'cases':cases, 'abbreviations':abbreviations,
               'distinct_pairs':[['pH','ph'],['NH₃','NH3'],['graphene oxide','reduced graphene oxide'],
                                 ['odds ratio','odd ratio'],['weakly coupled plasmas','coupled plasma']],
               'equivalent_pairs':[['cars','car'],['children','child'],['neural networks','neural network'],
                                   ['weakly coupled plasmas','weakly coupled plasma'],
                                   ['convolutional neural networks','convolutional neural network']]})
    write_json('contracts/ner_v2/research_input_lock.json', {'seed':42,'input':'reports/ner_v2/benchmark_run1/input_abstracts.jsonl',
               'input_sha256':file_sha256('reports/ner_v2/benchmark_run1/input_abstracts.jsonl'),
               'cases_sha256':file_sha256('contracts/ner_v2/research_cases.json'),
               'documents':[{k:d[k] for k in ['document_id','title','year','doi','discipline','text_sha256']} for d in docs]})
    print(len(cases),'normalization cases frozen')


if __name__ == '__main__':
    main()
