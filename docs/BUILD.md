# Build the course materials

## Python report and figures

Use Python3.12 and an isolated environment. The report builder uses python-docx, matplotlib, Pillow and NumPy; PDF inspection uses pypdf. The pinned build versions are in requirements-docs.txt.

```bash
python -m pip install -r requirements-docs.txt
MPLCONFIGDIR=.build/matplotlib python scripts/build_figures.py
python scripts/build_report.py
MPLCONFIGDIR=.build/matplotlib python scripts/build_task2_figures.py   # run before build_report.py
```

`report/content.py` holds the eight main pages and references. `scripts/build_report.py` constructs Word tables, places figures within the text margins, and adds the excluded references, appendix and separate contribution page. It uses explicit page breaks to preserve the main-page budget. Rendering was performed with LibreOffice through the Codex documents renderer, and the resulting twelve-page PDF (8 main, references, Appendices A and B, contribution) was visually inspected.

For an independent PDF export, open the DOCX in Word/LibreOffice and export to PDF, then check that pages 1-8 are the main report, page 9 references, pages 10-11 appendices and page 12 the contribution statement. Layout may vary across office versions.

## Editable presentation

`slides/STAT5011_Final_Presentation.pptx` contains 30 slides (22 Task 1 + 8 Task 2), native tables/charts, embedded chart workbooks and editable diagrams. It can be edited directly in PowerPoint without running a builder. Speaker notes contain source links. `slides/deck.snapshot.json` records the updated integrated 30-slide deck. The historical `scripts/build_slides.mjs` describes the original 22-slide Task 1 deck; current integrated edits use the November25 workflow below.

The current editor uses the Codex bundled `@oai/artifact-tool` runtime and preserves the supplied integrated deck. Use the current workflow below; legacy builders are retained as authoring history. All eight native charts have embedded workbooks, and the literal values are checked against their worksheet ranges.

## Updating results

Change `results/leaderboard_snapshot.json` only with verified evidence. Retain prior score receipts and model hashes. Update report/content.py and the slide result text together; rerun the figure and document builders, render every page/slide, and inspect the affected outputs. Do not replace the best achieved score with a lower exploratory upload.

## Task 2 slides

`scripts/build_task2_slides.py` (python-pptx) inserts the eight Task 2 slides (after slide 18) into the 22-slide Task 1 deck and updates the title, conclusion and reference slides, giving 30 slides. It refuses to run on an already-extended deck. This is historical code supplied with the Task2 integration. Do not run it for the current Task1-only review; use the preserved integrated source and update workflow below.

## Current Task 1 update workflow (November25 review)

The integrated deck has30slides. `slides/source/CLAUDE_INTEGRATED_20261122.pptx` freezes the supplied integrated deck; Task2 is displayed on slides19–26. The older `build_slides.mjs` / `build_task2_slides.py` are historical builders. Do not use them to replace the current integrated deck with an older Task1 score.

Use the bundled Node/Python executables and package directories; link `scripts/node_modules` to `$RUNTIME_NODE_MODULES`. The current slide rebuild is:

```bash
"$RUNTIME_NODE" scripts/update_task1_slides_20261125.mjs
"$RUNTIME_PYTHON" scripts/preserve_task2_parts.py
"$RUNTIME_PYTHON" scripts/repair_task1_chart_workbooks.py
"$RUNTIME_PYTHON" scripts/align_chart_cache_decimals.py
RUNTIME_NODE_MODULES="$RUNTIME_NODE_MODULES" \
  PRESENTATIONS_SKILL_DIR="$PRESENTATIONS_SKILL_DIR" \
  RUNTIME_PYTHON="$RUNTIME_PYTHON" \
  "$RUNTIME_NODE" scripts/finalize_task1_review.mjs
```

The finalizer writes a new file under `.build/review/final`; it protects an existing output rather than overwriting it. It checks30slides, six native tables, all eight native charts/workbooks, package integrity, fonts and first-party import. Copy the validated PPTX to `slides/` after reviewing its renders. Task2 text, shapes, workbooks and data are preserved; two inherited floating-point chart-cache strings are aligned to their existing workbook values (differences below1e-11, no displayed-score change).

After changing report/content.py or Task1 evidence, rebuild figures and report, render the DOCX and inspect it. Current pagination is8main pages, page9references, page10Task1appendix, page11Task2appendix, and page12contribution statement. `python scripts/verify_results.py` checks the final evidence and pagination. No website calls or submissions occur in course-document builds.
