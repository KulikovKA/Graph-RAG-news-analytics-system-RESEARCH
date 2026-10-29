# Runtime audit and final inference protocol (before Gold scoring)

The original GLiREL experiment and the initial protocol are preserved. A guarded
12-DEV run at 0.5 produced no raw relations. The published model-card control then
failed: characters scores around 0.001–0.002, versus about 0.99 in the card.
Strict checkpoint loading and its SHA256 succeeded. The same discrepancy occurs
with GLiREL 1.0.1/Flair 0.14.0, Transformers 4.44.2 and 4.51.3, fast/slow
tokenizers, and the official GLiREL loader with the identical backbone weights.
Thus it is not established that our custom loader caused this discrepancy.
The cause is unresolved; the initial run is a technical diagnostic, not evidence
that zero-shot scientific RE necessarily fails. Do not change thresholds to hide it.

Separately, Transformers 5.1.0's conversion of the local DeBERTa tokenizer omitted
sequence CLS/SEP tokens. The final runtime pins Transformers 4.51.3 and checks
model special tokens. The resulting NER runtime differs from historical C-001,
so the comparison cannot isolate the effect of replacing only the RE model.

Final backend, fixed before opening labels to score any new predictions:
`cross-encoder/nli-deberta-v3-xsmall`, revision
`a150876415327c80daeff35ca6f68f5ed8cf5c24`, Apache-2.0; 70.8M parameters.
This is a ready three-class entailment classifier trained externally on SNLI and
MultiNLI, not a base encoder with random relation heads. No training, adaptation,
optimizer, fit, or generated LLM output. No downstream reranking: the checkpoint
is the relation extraction component itself. The research basis is Sainz et al.,
EMNLP 2021, https://aclanthology.org/2021.emnlp-main.92/.

One ontology-derived hypothesis per directed relation in config; 22 experimental
verbalizations, MEETS_REQUIREMENT unsupported. Scientific correctness of these
verbalizations remains unvalidated. For sentences with linguistic negation, also
score the explicit negated hypothesis. Require entailment softmax probability
strictly above 0.5, then check agreement with deterministic polarity scope.
No Gold-based threshold, template, model, or NER label selection. Do not tune
after scoring. Do not treat scores as calibrated scientific probabilities.

NER remains the pinned pretrained GLiNER-small with the original 0.35 threshold
and 12 labels. Final NLI RE uses **single sentences**, four CPU threads, float32,
batch size 32, exact untruncated premise/hypothesis pairs, max 512 subtokens.
Single-sentence restriction bounds CPU cost and ambiguous scope; cross-sentence
claims/coreference are unsupported. Unlike C-001, no gap or pair-budget pruning.
Enumerate candidates before domain filtering, then evaluate compatible relation
hypotheses only. Record every real NLI query/three-class score locally, and trace
candidate/model coverage in the offline evaluator. No fictitious 23-class head.
All other evidence, scope, qualifiers and matching rules from the initial
protocol apply. UNKNOWN qualifier completeness gets zero strict qualified credit.

Before final DEV inference, require the published NLI model-card control to
return entailment for the pizza example and contradiction for the race-car
example. This verifies checkpoint/runtime integration, not scientific relation
correctness. Controls are excluded from DEV quality/performance metrics.
