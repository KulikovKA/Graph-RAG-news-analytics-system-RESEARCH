"""Descriptive metrics, UNREVIEWED diagnostics and exportable scientific plots."""
import csv
import itertools
from collections import Counter
from pathlib import Path

from cgraph.common import read_json, write_json, file_sha256
from cgraph.ner_v2.io import read_documents, write_jsonl, validate_span
from cgraph.ner_v2.research import ROOT, HIST, SHORT, jsonl


def csv_file(path, rows):
    rows=list(rows)
    if not rows:return
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def spans(rows):
    return {(m['document_id'],m['start'],m['end']) for m in rows}


def describe(rows,docs):
    unique=spans(rows)
    overlap=nested=crossing=0
    for (doc,a,b),(other,c,d) in itertools.combinations(sorted(unique),2):
        if doc==other and max(a,c)<min(b,d):
            overlap+=1
            if (a<=c and d<=b) or (c<=a and b<=d):nested+=1
            else:crossing+=1
    return dict(mentions=len(rows),unique_spans=len(unique),documents=len({x[0] for x in unique}),
                overlap_pairs=overlap,nested_pairs=nested,crossing_pairs=crossing,
                same_span_extra_hints=len(rows)-len(unique),
                normalization_changes=sum(m['normalized_form']!=m['surface_text'] for m in rows))


def main():
    global ROOT
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',default=str(ROOT))
    ROOT=Path(parser.parse_args().output_root)
    docs=read_documents(HIST/'input_abstracts.jsonl');by_id={d['document_id']:d for d in docs}
    hist=read_json(HIST/'manifest.json')
    ROOT.mkdir(parents=True,exist_ok=True)
    metrics=[];outputs={};discipline_rows=[];label_rows=[];validations=[]
    for name in ['E0','E1','E2','E3','E4']:
        for threshold in ([.35] if name=='E0' else [.15,.25,.35]):
            path=HIST/'large.jsonl' if name=='E0' else ROOT/name/f'mentions_{int(threshold*100):03}.jsonl'
            if not path.exists():continue
            rows=jsonl(path);outputs[(name,threshold)]=rows
            for m in rows:
                validate_span(by_id[m['document_id']]['text'],m['start'],m['end'],m['surface_text'])
            if len({(m['document_id'],m['start'],m['end'],m['ner_label_hint']) for m in rows})!=len(rows):
                raise ValueError('Duplicate exact span and hint')
            validations.append(dict(configuration=name,threshold=threshold,records=len(rows),invalid_offsets=0,
                                    duplicate_span_hint=0,input_sha256=file_sha256(HIST/'input_abstracts.jsonl')))
            base=describe(rows,docs)
            performance=hist['models']['large'] if name=='E0' else read_json(ROOT/name/'performance.json')
            metrics.append(dict(configuration=name,threshold=threshold,**base,
                                processing_seconds=performance.get('processing_wall_seconds',performance.get('detection_and_normalization_seconds')),
                                inference_seconds=performance.get('inference_seconds'),
                                normalization_seconds=performance.get('normalization_seconds'),
                                load_seconds=performance.get('model_load_seconds',performance.get('load_seconds')),
                                peak_rss_bytes=performance.get('peak_rss_bytes'),
                                encoder_calls=performance.get('encoder_calls'),
                                timing_inference_threshold=.35 if name=='E0' else .15))
            for discipline in sorted({d['discipline'] for d in docs}):
                subset=[m for m in rows if by_id[m['document_id']]['discipline']==discipline]
                discipline_rows.append(dict(configuration=name,threshold=threshold,discipline=discipline,
                                            mentions=len(subset),unique_spans=len(spans(subset)),
                                            source_documents=sum(d['discipline']==discipline for d in docs)))
            counts=Counter(m['ner_label_hint'] for m in rows)
            for label in (SHORT if name!='E4' else counts):
                label_rows.append(dict(configuration=name,threshold=threshold,label=label,mentions=counts[label]))
    agreement=[]
    for left,right in itertools.combinations([n for n,t in outputs if t==.35],2):
        a,b=spans(outputs[(left,.35)]),spans(outputs[(right,.35)])
        agreement.append(dict(left=left,right=right,intersection=len(a&b),union=len(a|b),
                              jaccard=len(a&b)/len(a|b) if a|b else None,
                              left_only=len(a-b),right_only=len(b-a)))
    csv_file(ROOT/'ner_metrics.csv',metrics);write_json(ROOT/'ner_metrics.json',metrics)
    csv_file(ROOT/'discipline_metrics.csv',discipline_rows)
    csv_file(ROOT/'label_metrics.csv',label_rows)
    write_json(ROOT/'agreement.json',agreement)
    write_json(ROOT/'offset_validation.json',validations)
    # Predictions added/lost relative to E0; neither category implies correctness.
    for name in ['E1','E2','E3','E4']:
        if (name,.35) not in outputs:continue
        baseline=spans(outputs[('E0',.35)]);current=spans(outputs[(name,.35)])
        write_jsonl(ROOT/name/'additional_vs_E0.jsonl',[m for m in outputs[(name,.35)] if (m['document_id'],m['start'],m['end']) not in baseline])
        write_jsonl(ROOT/name/'lost_vs_E0.jsonl',[m for m in outputs[('E0',.35)] if (m['document_id'],m['start'],m['end']) not in current])
    # Three source expressions per document, chosen for scientific content. This
    # assistant-authored diagnostic is explicitly not a completeness annotation.
    terms=[
        [('CNN','METHOD'),('sEMG','OBJECT'),('feature extraction','PROCESS')],
        [('DNA storage','CONCEPT'),('FLL','ATTRIBUTE'),('anticode','CONCEPT')],
        [('odds ratio','ATTRIBUTE'),('paired t -test','METHOD'),('medroxyprogesterone acetate','OBJECT')],
        [('support groups','OBJECT'),('cancer','CONCEPT'),('vocational rehabilitation','PROCESS')],
        [('Laplacian regularisation','METHOD'),('diffusion process','PROCESS'),('steady‐state spread','ATTRIBUTE')],
        [('weakly coupled plasmas','OBJECT'),('inverse bremsstrahlung','PROCESS'),('Coulomb logarithm','CONCEPT')],
        [('temperature','ATTRIBUTE'),('anthocyanins','OBJECT'),('nitrogen resorption','PROCESS')],
        [('pH','ATTRIBUTE'),('temperature','ATTRIBUTE'),('orthogonal experiment','METHOD')],
        [('India','ACTOR'),('Kalinga War','PROCESS'),('inflationary pressures','CONCEPT')],
        [('real GNP','ATTRIBUTE'),('recessions','PROCESS'),('stochastic trend representation','METHOD')],
        [('animal heads','OBJECT'),('lunar calendar','OBJECT'),('Bronze Age','CONCEPT')],
        [('natural childbirth','METHOD'),('Britain','ACTOR'),('maternity care','PROCESS')],
    ]
    diagnostics=[]
    for d,entries in zip(docs,terms):
        for term,label in entries:
            a=d['text'].index(term);b=a+len(term)
            diagnostics.append(diagnostic(d,a,b,label,outputs,'scientific_expression'))
    seen=set()
    for key,rows in outputs.items():
        if key[1]!=.35:continue
        for m in rows:
            if m['surface_text'] in ['we','We','ABSTRACT','scenario','paper','study','results','Results']:
                identity=(m['document_id'],m['start'],m['end'])
                if identity not in seen and len(seen)<12:
                    seen.add(identity)
                    diagnostics.append(diagnostic(by_id[m['document_id']],m['start'],m['end'],None,outputs,'suspected_non_entity'))
    write_jsonl(ROOT/'diagnostics_UNREVIEWED.jsonl',diagnostics)
    watchlist={'we','We','ABSTRACT','scenario','paper','study','results','Results'}
    suspected=[]
    for (name,threshold),rows in outputs.items():
        if threshold==.35:
            subset=[m for m in rows if m['surface_text'] in watchlist]
            suspected.append(dict(configuration=name,status='POSTHOC_UNREVIEWED_WATCHLIST',
                                  matching_mentions=len(subset),matching_unique_spans=len(spans(subset)),
                                  surfaces='; '.join(sorted({m['surface_text'] for m in subset}))))
    csv_file(ROOT/'suspected_noise_watchlist.csv',suspected)
    norm=read_json(ROOT/'normalization/metrics.json')
    csv_file(ROOT/'normalization_metrics.csv',[{k:v for k,v in m.items() if k!='by_kind'} for m in norm])
    plot(metrics,discipline_rows,norm)
    print('metrics rows',len(metrics),'diagnostics',len(diagnostics))


def diagnostic(doc,a,b,label,outputs,category):
    predictions={}
    for (name,threshold),rows in outputs.items():
        if threshold!=.35:continue
        exact=[m for m in rows if m['document_id']==doc['document_id'] and m['start']==a and m['end']==b]
        intersect=[m for m in rows if m['document_id']==doc['document_id'] and max(a,m['start'])<min(b,m['end'])]
        predictions[name]=dict(exact=[dict(label=m['ner_label_hint'],score=m['score']) for m in exact],
                               overlapping=[dict(surface=m['surface_text'],label=m['ner_label_hint'],score=m['score'],start=m['start'],end=m['end']) for m in intersect],
                               observation='exact_present' if exact else ('only_overlapping' if intersect else 'absent'))
    return dict(status='UNREVIEWED',annotation_origin='assistant_semantic_hypothesis',
                independent_human_review=False,document_id=doc['document_id'],discipline=doc['discipline'],
                source_url=doc['source_url'],start=a,end=b,surface=doc['text'][a:b],
                context=doc['text'][max(0,a-90):min(len(doc['text']),b+90)],
                suggested_label=label,category=category,predictions=predictions,
                interpretation='Предположительно научное выражение; границы и роль требуют ручной проверки.' if label else
                               'Предположительно дискурсивное слово или заголовок; проверить контекст вручную.')


def plot(metrics,disciplines,norm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    main=[m for m in metrics if m['threshold']==.35]
    names=[m['configuration'] for m in main]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].bar(names,[m['unique_spans'] for m in main],color='#3c7499')
    axes[0].set(title='Уникальные source spans, threshold 0,35',ylabel='Число spans')
    axes[1].bar(names,[m['processing_seconds'] for m in main],color='#d8934c')
    axes[1].set(title='Обработка 12 аннотаций, CPU / 4 потока',ylabel='Wall time, с')
    axes[1].text(.02,.98,'E0: inference при 0,35; E1–E4: при 0,15',transform=axes[1].transAxes,va='top',fontsize=8)
    fig.savefig(figures/'ner_counts_time.png',dpi=180);fig.savefig(figures/'ner_counts_time.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].bar([m['variant'] for m in norm],[m['correct'] for m in norm],color='#3c7499')
    axes[0].set(title='Совпадение с заданной формой (90 случаев)',ylabel='Число совпадений',ylim=(0,100))
    axes[1].bar([m['variant'] for m in norm],[m['incorrect_transformations'] for m in norm],color='#c66b5c')
    axes[1].set(title='Неверные преобразования (из 90 случаев)',ylabel='Число случаев')
    fig.savefig(figures/'normalization.png',dpi=180);fig.savefig(figures/'normalization.svg');plt.close(fig)
    fields=sorted({r['discipline'] for r in disciplines})
    matrix=[[next(r['unique_spans'] for r in disciplines if r['configuration']==n and r['threshold']==.35 and r['discipline']==f) for f in fields] for n in names]
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    im=ax.imshow(matrix,aspect='auto',cmap='Blues')
    ax.set_xticks(range(len(fields)),[f.replace(' / Machine Learning','') for f in fields],rotation=20,ha='right')
    ax.set_yticks(range(len(names)),names)
    ax.set_title('Source spans по дисциплинам; по 2 документа, threshold 0,35')
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>max(max(r) for r in matrix)*.55 else 'black')
    fig.colorbar(im,ax=ax,label='Число spans')
    fig.savefig(figures/'disciplines.png',dpi=180);fig.savefig(figures/'disciplines.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for name in names:
        subset=[r for r in metrics if r['configuration']==name]
        ax.plot([r['threshold'] for r in subset],[r['unique_spans'] for r in subset],marker='o',label=name)
    ax.set(title='Исследовательский анализ порогов',xlabel='Score threshold',ylabel='Уникальные source spans',xticks=[.15,.25,.35])
    ax.legend();fig.savefig(figures/'thresholds.png',dpi=180);fig.savefig(figures/'thresholds.svg');plt.close(fig)
    # Matplotlib paths contain trailing spaces; keep committed SVG text clean and
    # preserve identical SHA-256 bytes after Git's LF checkout normalization.
    for path in figures.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n',
                        encoding='utf8',newline='\n')


if __name__=='__main__':main()
