# Welcome to TaxoPTM: Cross-Species Phosphorylation Site Prediction with Candidate-Centered and Taxonomy-Conditioned Protein Language Models

Accurate phosphorylation-site prediction across species remains
challenging because experimentally validated sites are unevenly
distributed among taxonomic groups. Existing protein language
model-based methods often represent local sequence windows as
generic residue features, without explicitly modeling taxonomic
context or the central role of the candidate residue. We propose
TaxoPTM, a taxonomy-aware and candidate-centered framework for
cross-species phosphorylation-site prediction. TaxoPTM fine-tunes
a pretrained protein language model on sequence windows centered
on candidate serine, threonine, or tyrosine residues. Learnable
taxonomic prompts provide group-level context for observed taxa,
while an unknown prompt serves as a fallback for groups not
encountered during training. A site-relative encoder represents
surrounding residues according to their positions relative to the
candidate site, and center-conditioned attention aggregates
informative local context from the perspective of the central
residue. Motif prototypes and evidence spans further provide
structured local signals associated with individual predictions.
Across balanced, imbalanced, and taxonomic out-of-distribution
test sets, TaxoPTM achieves the strongest overall MCC and AUC
among the compared methods. Backbone-controlled comparisons
and component ablations support the contribution of its
candidate-centered prediction head. Masking analysis further
shows that high-weight evidence positions materially influence
model outputs, while their enrichment in known phosphorylation
motifs supports the biological plausibility of the learned
evidence.

![The workflow of this study](https://github.com/SamHe6/PhaseNet/blob/main/workflow.png)

# Requirements<bar>
```
torch==2.12.0
transformers==5.9.0
numpy==2.2.6
pandas==2.3.3
scikit-learn==1.7.2
tqdm==4.67.3
```

# Datasets
We provided our dataset and you can find them [Dataset](https://github.com/SamHe6/TaxoPTM/tree/main/data), processed 41-residue sequence windows and the training,
  validation, balanced-test, imbalanced-test, and taxonomic OOD splits.

Each sequence window is centered on a candidate serine, threonine, or
tyrosine residue. Taxonomic-group annotations are provided in
`data/all_samples_with_taxon_group.tsv`.

# Code
We provide the source code and you can find them [Code](https://github.com/SamHe6/TaxoPTM/tree/main/code)
