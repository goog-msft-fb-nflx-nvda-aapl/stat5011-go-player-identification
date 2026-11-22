# Build the course materials

## Python report and figures

Use Python3.12 and an isolated environment. The report builder uses python-docx, matplotlib, Pillow and NumPy; PDF inspection uses PyMuPDF. The pinned build versions are in requirements-docs.txt.

```bash
python -m pip install -r requirements-docs.txt
MPLCONFIGDIR=.build/matplotlib python scripts/build_figures.py
python scripts/build_report.py
```

`report/content.py` holds the seven main pages and references. `scripts/build_report.py` constructs Word tables, places figures within the text margins, and adds the excluded references, appendix and separate contribution page. It uses explicit page breaks to preserve the main-page budget. Rendering was performed with LibreOffice through the Codex documents renderer, and the resulting ten-page PDF was visually inspected.

For an independent PDF export, open the DOCX in Word/LibreOffice and export to PDF, then check that pages1-7 are the main report, page8 references, page9 appendix and page10 the contribution statement. Layout may vary across office versions.

## Editable presentation

`slides/STAT5011_Final_Presentation.pptx` contains 22 slides, native tables/charts, embedded chart workbooks and editable diagrams. It can be edited directly in PowerPoint without running a builder. Speaker notes contain source links. `slides/deck.snapshot.json` preserves its structured authoring snapshot, while `scripts/build_slides.mjs` is the JavaScript source.

The builder uses the Codex bundled `@oai/artifact-tool` runtime. In that runtime:

```bash
ln -s "$RUNTIME_NODE_MODULES" node_modules
node scripts/build_slides.mjs
```

The draft is `.build/candidate.pptx`; finalization supplies embedded workbooks for the literal chart data and validates the editable chart/table structure. Exact numeric evidence remains in results; chart values with longer precision are displayed/embedded to12 significant digits, beyond their visible precision. The final validated file is in slides/.

## Updating results

Change `results/leaderboard_snapshot.json` only with verified evidence. Retain prior score receipts and model hashes. Update report/content.py and the slide result text together; rerun the figure and document builders, render every page/slide, and inspect the affected outputs. Do not replace the best achieved score with a lower exploratory upload.
