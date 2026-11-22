# Sources and attribution

## Project author

James Christian, R13921031. Course: STAT5011 Statistical Deep Learning.

The project used ChatGPT/Codex, Claude and Gemini for development support, research synthesis, experiment suggestions, figures and drafting. The separate contribution page in the report discloses this assistance. This is a course project built from the user's competition experiments; no unaided-authorship claim is made.

## External resources

1. Official AI CUP Task1 data, rules and updated-test notice: https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688
2. Official tutorial: https://github.com/AILAB-NDHU/AICup-2026-Tutorial
3. Behavioral stylometry in chess: https://arxiv.org/abs/2208.01366
4. Transformer architecture: https://arxiv.org/abs/1706.03762
5. ArcFace: https://arxiv.org/abs/1801.07698
6. KataGo HumanSL: https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md#human-sl-analysis-guide
7. CSLS retrieval motivation: https://arxiv.org/abs/1710.04087

HumanSL source is pinned to commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1`. HumanSL checkpoint SHA-256 is `84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27`. Preserve the original network copyright notice in KATAGO_NETWORK_LICENSE.txt; the source/model are not relicensed by this project. The cited papers provide motivation, not directly comparable Go leaderboard results.

The model source originates in the private competition repository at release `go-best-20261122` (source commit `32a0e9f`). `source_manifest.json` records hashes. Operational upload/authentication/backup helpers have been excluded from the course model-source copy. Authorized raw inputs and large checkpoints remain external.

## Figures and evidence

All figures were created for this project. PNGs are display copies; SVGs retain editable vector content. Dataset and score charts use `results/*.json` and `results/submissions.csv`. Schematic pipeline diagrams do not depict hidden labels. Charts use truncated axes where stated, and local scores are distinguished from official public evaluations.

The headline reports the highest achieved public score. No separate latest-upload score is used as a headline metric. Rank 20/125 is a dated team-standing snapshot. No final private result, medal, leading-team architecture, or statistical significance is claimed.
