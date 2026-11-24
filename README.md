# Go Player Identification and Rank Prediction

**STAT5011 Statistical Deep Learning — Final Project**

**James Christian · R13921031**

Identify a Go player from a small collection of game records by ranking candidate reference players. This project combines move-sequence Transformer embeddings with frozen KataGo HumanSL decision features, a learned identity metric and variance-aware matching.

## Deliverables

- [Presentation — 30 editable slides, Tasks 1 and 2](slides/STAT5011_Final_Presentation.pptx)
- [Final report — PDF](report/STAT5011_Final_Report.pdf)
- [Final report — editable Word](report/STAT5011_Final_Report.docx)
- [Report authoring text](report/STAT5011_Final_Report.md)

The report has **8 pages of main text** (Task 2 is Section 8), followed by **1 reference page, 2 appendix pages (A: Task 1 evidence, B: Task 2 evidence) and 1 member-contribution page**. The separate final contribution page names James Christian and discloses AI assistance. The slides and report cover motivation, problem definition, data, literature, model comparison, methods, experimental results and conclusions.

## Results snapshot

| Item | Recorded result |
|---|---:|
| Highest achieved public score | **0.8721950** |
| Team | TEAM_10990 |
| Team rank at snapshot | **20 / 131** |
| Task 1 snapshot | November 24, 2026, 01:42:56 JST |
| Leader score | 0.9313515 |
| Best prediction | `g23_greplace.csv` (restored as `best_872195.csv`) |

The headline is the highest achieved public score. The competition submission period has ended. Task 1 standing is the dated November 24 upload snapshot; a verified final private result is not included here. This is not a medal claim.

## Visual overview

![System pipeline](figures/method_pipeline.png)

![Training data distribution](figures/data_distribution.png)

![Final Task 1 experiment comparison](figures/public_ablations.png)

## Task 2 snapshot (rank prediction)

| Item | Recorded result |
|---|---:|
| Highest achieved public score | **0.7692** (`results/task2_best_prediction.csv`) |
| Rank at snapshot | 19 / 175 (November 25, 2026 JST; public) |
| Method | 6 bag-level SORD language models + 8 HumanSL sequence models (4 GRU, 4 Transformer, 5 game sources), blend 1:2, expected-score decoding |

Scores of all variants: `results/task2_scores.csv`. Code, seeds, saved per-model outputs and the byte-identical CPU regeneration path: [`task2/REPRODUCE_TASK2_BEST.md`](task2/REPRODUCE_TASK2_BEST.md). Figures: `figures/task2_*.png`. Builders: `scripts/build_task2_figures.py`, `scripts/build_task2_slides.py`.

## Data and methods

The official processed training collection contains **1,000,000 games and 8,007 players**. The updated evaluation contains original pools of 400 players with 100 reference games each and new pools of 600 players with ten reference games each. New-family queries contain 1, 2, 3 or 5 games. The submission has 1,600 rows, split across public/private files and both test families.

The accepted score is `z(coordinate fusion with LM4) + 0.25*z(policy MLP cosine) + 0.10*z(expanded LDA Gaussian matching)`. Coordinate encoders use angular-margin identity training. HumanSL supplies 384 contextual and 48 policy-head residual channels per sampled move. The project evaluates repeated-player retrieval; it does not impose a one-to-one answer assignment.

The scaling study fits a projection on **90,000 games from 6,000 training players**, disjoint from the fixed holdout identities. Expanded cosine improved the local audit proxy from 0.7104 to 0.7191 but regressed on the website (0.8672784). Using that projection for Gaussian matching at weight0.10 instead produced the **0.8721950** best. The three final experimental variants did not improve it; their feedback is recorded in results/submissions.csv.

## Reproduce

- [Official data placement and model inference](docs/REPRODUCTION.md)
- [Build the report, presentation and figures](docs/BUILD.md)
- [Sources, attribution and limitations](docs/ATTRIBUTION.md)
- [Machine-readable evidence](results/)

Saved-model inference from frozen intermediate coordinate scores and extracted HumanSL features is verified against the exact scored prediction hash. A legacy base-view raw-input reconstruction discrepancy is retained explicitly in the report and reproduction guide. Raw game records, account credentials and large model checkpoints are not committed here. Frozen model assets remain in the authorized private competition repository.

## Repository map

```text
slides/        editable presentation and structured slide snapshot
report/        8-page main report, excluded sections, and authoring source
figures/       measured charts and schematic diagrams, PNG and editable SVG
results/       aggregate data summary, sanitized scores, validation evidence
model_src/     frozen model and inference source with a hash manifest
scripts/       document, chart and presentation builders
 docs/         reproduction, build, attribution and sources
```

All course-specific material is private while the competition is active. Share access only with the intended authorized course audience. The official data and external model licenses retain their own terms. Generative-AI assistance is disclosed on the report's separate contribution page.

## Task 1 review after submission closing

Reviewed November25,2026. The final accepted Task 1 prediction and clean-workspace replay evidence are included. See [Task 1 review notes](docs/TASK1_REVIEW_20261125.md). Task 2 content is preserved from the integrated version.
