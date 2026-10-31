"""Record the assistant's contextual review of the fixed 120-candidate sample.

These are UNREVIEWED draft judgements, not a human audit or a Gold standard.
The indices are guarded by candidate IDs; no re-sampling or corpus changes occur.
"""
from cgraph.common import write_json,fingerprint
from cgraph.ner_v2.research import jsonl
from cgraph.ner_v2.quality_audit import ROOT

# category, plausible ontology hints, contextual reason; all remain hypotheses.
JUDGEMENTS={
1:('VALID_SPAN','PROCESS','Biological inflammatory process with explicit modality in context.'),
2:('VALID_SPAN','OBJECT','Named allele; biological object, not an acting participant.'),
3:('VALID_SPAN','OBJECT','Yeast species is a biological object; ACTOR hint is suspect.'),
4:('PARTIAL_SPAN','ATTRIBUTE CONCEPT','Order omits the informative gene-order modifier.'),
5:('VALID_SPAN','ATTRIBUTE','Cell permeability is a measured property, not its change process.'),
6:('WRONG_BOUNDARY','METHOD','Prepositional framing overextends the analysis-method phrase.'),
7:('IRRELEVANT','','Unspecified previous studies are a discourse reference.'),
8:('VALID_SPAN','ATTRIBUTE','Product titer is the metric; the 8-fold result lies outside this span.'),
9:('VALID_SPAN','METHOD','Computational design is explicitly used to screen candidates.'),
10:('VALID_SPAN','METHOD CONCEPT','Scientific engineering field/method; not a described causal mechanism.'),
11:('PARTIAL_SPAN','PROCESS','Loss omits loss of photosynthetic genes, the informative process object.'),
12:('WRONG_BOUNDARY','PROCESS','Full subject-predicate assertion is overextended for a mention.'),
13:('VALID_SPAN','ACTOR','Human population, not a dataset of observations.'),
14:('VALID_SPAN','CONDITION','Explicit timing condition relative to FGF-1 application.'),
15:('VALID_SPAN','OBJECT','Biological taxon; not an organization acting as a subject.'),
16:('VALID_SPAN','CONCEPT','Specific evolutionary concept in a phylogenetic argument.'),
17:('VALID_SPAN','ATTRIBUTE','Energy consumption is a process property; modifier is meaningful.'),
18:('PARTIAL_SPAN','MECHANISM','Generic underlying mechanism lacks the limonene-cytotoxicity qualifier.'),
19:('IRRELEVANT','','This study is a discourse self-reference.'),
20:('VALID_SPAN','ATTRIBUTE','Binding affinity is a property; reduction is outside this span.'),
21:('VALID_SPAN','OBJECT','Detection targets in UAV imagery.'),
22:('VALID_SPAN','OBJECT','Digital questionnaire page, not the research task.'),
23:('VALID_SPAN','ATTRIBUTE','Network life cycle is a characteristic; 44% result is outside the span.'),
24:('AMBIGUOUS','METHOD','Generic universal method refers to an actual proposal but is weakly informative.'),
25:('WRONG_BOUNDARY','ACTOR','Lowercase pnm is inside a URL, not a clean institution mention.'),
26:('VALID_SPAN','PROCESS','Disposal is an explicit composite lifecycle stage.'),
27:('VALID_SPAN','ATTRIBUTE','Packet-transfer count is a performance measure.'),
28:('VALID_SPAN','DATA_RESOURCE OBJECT','UAV images can be viewed as research inputs or digital objects.'),
29:('VALID_SPAN','CONDITION','Access to an encryption system conditions the stated coding capability.'),
30:('VALID_SPAN','METHOD','Protocols are technical procedures; mechanism hint is unsupported.'),
31:('VALID_SPAN','CONCEPT','Conceptual inspiration from crowd counting.'),
32:('AMBIGUOUS','PROCESS','Generic changes are informative only with the omitted design context.'),
33:('VALID_SPAN','OBJECT','Proposal boxes are digital spatial objects, not a task or scalar metric.'),
34:('VALID_SPAN','OBJECT','Network is the investigated technical system, despite being a nested modifier.'),
35:('VALID_SPAN','ACTOR','Students are actual users of the evaluation system.'),
36:('VALID_SPAN','OBJECT','Proposal-box objects; their quality is the attribute outside the span.'),
37:('VALID_SPAN','METHOD','Explicit coding procedure, not an experimental condition.'),
38:('AMBIGUOUS','METHOD OBJECT','ARPN occurs in a code repository URL; method versus artifact is uncertain.'),
39:('PARTIAL_SPAN','RESULT DATA_RESOURCE','Result alone omits the quality-control referent.'),
40:('VALID_SPAN','CONCEPT METHOD OBJECT','Named assurance system/framework; ontology role needs adjudication.'),
41:('VALID_SPAN','ATTRIBUTE','Stochastic nature is a property of the data, not the dataset itself.'),
42:('VALID_SPAN','OBJECT CONCEPT','Financial assets are relevant; intangible-object ontology fit is uncertain.'),
43:('VALID_SPAN','ACTOR','Rail sector acts as a cooperating institutional participant.'),
44:('PARTIAL_SPAN','PROCESS','Implementation requires the legislation qualifier.'),
45:('VALID_SPAN','METHOD CONCEPT','Statistical formulation used for testing; method/construct ambiguity.'),
46:('VALID_SPAN','TASK','Fare optimization is the stated research objective.'),
47:('VALID_SPAN','CONCEPT ATTRIBUTE','Risk concept/property, not an explicit experimental condition.'),
48:('VALID_SPAN','CONCEPT','Convergence is explicitly defined and statistically tested.'),
49:('VALID_SPAN','DATA_RESOURCE','Named scholarly journal serves as the cited information source.'),
50:('PARTIAL_SPAN','ATTRIBUTE','Adjective network-organized omits the banking-system head.'),
51:('IRRELEVANT','','Results is a generic reporting word without an observed outcome in its span.'),
52:('IRRELEVANT','','Unresolved author pronoun.'),
53:('VALID_SPAN','OBJECT CONCEPT','Relevant financial instruments; intangible-object typing is uncertain.'),
54:('VALID_SPAN','PROCESS CONCEPT','Convergence may be the occurring process or the tested concept.'),
55:('VALID_SPAN','CONCEPT','Named mathematical equilibrium model; not a source-described mechanism.'),
56:('AMBIGUOUS','ATTRIBUTE','Generic features need the financial-instrument qualifier.'),
57:('AMBIGUOUS','METHOD','Coordinated algorithms may require splitting into two individual mentions.'),
58:('VALID_SPAN','RESULT ATTRIBUTE','Observed earnings losses may be an outcome or a measured quantity.'),
59:('PARTIAL_SPAN','PROCESS TASK','Testing omits convergence, the object of the statistical task.'),
60:('WRONG_BOUNDARY','DATA_RESOURCE','On the basis of overextends the named Displaced Worker Surveys span.'),
61:('VALID_SPAN','OBJECT','Historical medals are the central investigated artifacts, despite low score.'),
62:('VALID_SPAN','ACTOR','An interviewer is a participant in oral-history data creation.'),
63:('VALID_SPAN','PROCESS','Explicit development of emotional skills.'),
64:('VALID_SPAN','ATTRIBUTE','Emotional skills are capacities of the human participants.'),
65:('WRONG_BOUNDARY','DATA_RESOURCE','Pronoun plus predicative complement is not a clean source mention.'),
66:('IRRELEVANT','','Innovative way contains novelty framing without a specified technique.'),
67:('VALID_SPAN','PROCESS','Preservation is an archival activity, not an experimental condition.'),
68:('VALID_SPAN','ACTOR','Historical social and political groups are the discussed actors.'),
69:('VALID_SPAN','PROCESS ACTOR','Synod may denote the event or its collective institution.'),
70:('VALID_SPAN','CONCEPT','Named sociological/religious concept.'),
71:('AMBIGUOUS','DATA_RESOURCE','Generic valuable sources refers to medals, but is weak as an entity span.'),
72:('WRONG_BOUNDARY','METHOD','Long way-of-dealing phrase overextends an IFS-method mention; not a person.'),
73:('VALID_SPAN','PROCESS','Earthquakes are events; no mechanism is described by the word alone.'),
74:('PARTIAL_SPAN','TASK','Generic infinitive lacks the studied historical phenomena.'),
75:('VALID_SPAN','PROCESS','Named historical event.'),
76:('VALID_SPAN','PROCESS','Archival conservation activity; no particular technique is specified.'),
77:('VALID_SPAN','CONDITION CONCEPT','Geographic/cultural context is relevant; ontology fit is uncertain.'),
78:('IRRELEVANT','','First-person author pronoun.'),
79:('IRRELEVANT','','Unspecified achievement has no identifiable historical outcome in its span.'),
80:('VALID_SPAN','PROCESS','Historical production of annual medals.'),
81:('IRRELEVANT','','Structured-abstract header.'),
82:('VALID_SPAN','METHOD ATTRIBUTE','Semiautomated measurement tool versus measured index is uncertain.'),
83:('IRRELEVANT','','Structured-abstract header.'),
84:('VALID_SPAN','ATTRIBUTE','Photon energy is a physical parameter.'),
85:('VALID_SPAN','ACTOR','Enrolled human participants.'),
86:('VALID_SPAN','OBJECT','Biological graft artifacts.'),
87:('IRRELEVANT','','Under consideration is generic decision phrasing, not a scientific task entity.'),
88:('VALID_SPAN','METHOD CONCEPT','Statistical modeling method/construct; not the dataset used to fit it.'),
89:('VALID_SPAN','DATA_RESOURCE','Patient medical records are the retrospectively analyzed data source.'),
90:('VALID_SPAN','CONCEPT PROCESS','Infection is a relevant disease/process even though negated in context.'),
91:('VALID_SPAN','ATTRIBUTE RESULT','Measurement metric versus obtained measurements needs adjudication.'),
92:('VALID_SPAN','ACTOR','Explicit human participant group; count is preserved.'),
93:('VALID_SPAN','METHOD PROCESS','Surgical procedure, not a mechanism explaining an effect.'),
94:('VALID_SPAN','PROCESS CONDITION','Follow-up can denote observation or its contextual timepoint.'),
95:('VALID_SPAN','ATTRIBUTE','Device lead impedance is a measured parameter.'),
96:('PARTIAL_SPAN','ATTRIBUTE','The named index begins with Western Ontario and, omitted by the candidate.'),
97:('VALID_SPAN','DATA_RESOURCE','Dataset partition for model fitting.'),
98:('IRRELEVANT','','Structured-abstract header.'),
99:('VALID_SPAN','PROCESS RESULT','Device malfunction is an investigated event/outcome, not its causal mechanism.'),
100:('VALID_SPAN','ATTRIBUTE RESULT','Measurements obtained by Cumulus; method/data-value role is uncertain.'),
101:('VALID_SPAN','ATTRIBUTE PROCESS','Particle flux denotes a physical quantity/transport, not a described mechanism.'),
102:('VALID_SPAN','ATTRIBUTE','Temperature property; the 257 K result is outside the candidate.'),
103:('VALID_SPAN','CONCEPT','Differential equations are mathematical constructs; method label is debatable.'),
104:('AMBIGUOUS','OBJECT PROCESS','Awkward source wording merges a motor, lightning, and a location.'),
105:('VALID_SPAN','RESULT CONCEPT','Computed mathematical solutions are meaningful in context.'),
106:('VALID_SPAN','CONCEPT RESULT','Wave solutions are mathematical constructs or obtained outcomes.'),
107:('VALID_SPAN','OBJECT','Named simulation software is a digital artifact, not a data corpus.'),
108:('VALID_SPAN','PROCESS','Emission fluctuation is the investigated plasma process.'),
109:('VALID_SPAN','CONDITION','Plasma configuration constrains visibility of the x-point.'),
110:('VALID_SPAN','ATTRIBUTE','Regolith properties are the study objects; the task is outside this phrase.'),
111:('VALID_SPAN','OBJECT','Technical power-supply system, not an acting human institution.'),
112:('VALID_SPAN','ATTRIBUTE','Brightness temperature is the metric; 233 K is outside the span.'),
113:('VALID_SPAN','OBJECT','Power station is a physical technical installation.'),
114:('WRONG_BOUNDARY','OBJECT','Two named software codes are merged into one coordinated candidate.'),
115:('VALID_SPAN','ATTRIBUTE','Degree of polarization is a metric; 6% is outside this span.'),
116:('VALID_SPAN','PROCESS CONCEPT','Turbulence is a process/phenomenon; causal mechanism is not specified by the word.'),
117:('AMBIGUOUS','METHOD','Generic procedures refer back to named approaches; informativeness is uncertain.'),
118:('VALID_SPAN','CONCEPT','Specific mathematical class of wave solutions.'),
119:('PARTIAL_SPAN','TASK PROCESS','Here study is a verb missing its regolith-properties object, not an abstract header.'),
120:('IRRELEVANT','','First-person author pronoun.'),
}


def main():
    rows=jsonl(ROOT/'audit_stratified_120.jsonl')
    assert len(rows)==len(JUDGEMENTS)==120
    # Pin ordering with boundary candidate IDs from the exported, seeded sample.
    assert rows[0]['candidate_id']=='c:d19d1535fbfbc85ee3ad'
    assert rows[-1]['candidate_id']=='c:41e808de9ac07def7d74'
    assert fingerprint([[r['candidate_id'],r['document_id'],r['start'],r['end']] for r in rows])=='560f2b484fa4dec760846e5f2299c562fb584fb46edebec9bd01b77b369c83d3'
    amendments={}
    for index,row in enumerate(rows,1):
        category,types,reason=JUDGEMENTS[index]
        amendments[row['candidate_id']]=dict(boundary_category=category,suggested_types=types.split(),reason=reason,
             scientific_significance='LOW_DRAFT' if category=='IRRELEVANT' else ('HIGH_DRAFT' if category=='VALID_SPAN' else 'POSSIBLY_RELEVANT'),
             context_review='Assistant read the source context and all model hint alternatives; no human review')
    write_json(ROOT/'assistant_audit_amendments.json',amendments)
    print('assistant contextual amendments',len(amendments),'all UNREVIEWED')


if __name__=='__main__':main()
