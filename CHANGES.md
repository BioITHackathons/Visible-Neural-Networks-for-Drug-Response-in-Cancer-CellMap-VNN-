# NeST-VNN BioIT Hackathon — Changes Summary

## Dashboard (`visualization/BioiThackCellMAP/static/dashboard.html`)

### Title & Header
- Title: **"NeST-VNN: Interpretable Multimodal Cancer Prognostics"**
- Subtitle: "Enhanced ontology-guided neural network with multimodal data, clinical features, and explainable AI interpretability"
- Study pill: `breast_msk_2025 · Memorial Sloan Kettering 2025`

### "What We Built" Section (5 innovation cards)
| Card | Content |
|---|---|
| Multimodal Data Integration | Mutations + CNV amplifications/deletions + gene fusions |
| CFDE Dataset Integration | CFDE IDG Druggable Genome (Pharos/TCRD) cross-reference |
| Custom Model Architecture | Per-gene multi-omic feature layers, improved auxiliary supervision |
| Clinical Feature Integration | Sample type, mutation burden, fraction genome altered |
| LLM Descriptive Interpretations | Per-patient pathway explanations via Claude (AWS Bedrock) |

### Study Snapshot (5 stat cards)
- Total Patients, Predicted High Risk, Predicted Low Risk, Model Genes (407), Biological Systems (103)
- Data from `/api/patients` and `/api/interp/patients`

### Risk Prediction & Clinical Features
- Donut chart (Chart.js) for high/low risk distribution
- Comparison bars: avg FGA, mutation count, % metastatic — high vs low risk groups

### Top Biological Systems (NeSTs)
- Top 10 NeSTs by population frequency
- Inline frequency bars, patient count, % of cohort, RLIPP badge (green ≥1.1, orange ≥0.9)

### Top Recurrently Altered Genes
- Color-coded by alteration type: mutation (red), amplification (orange), deletion (purple)
- 💊 badge for CFDE Tclin genes

### Druggable Targets (CFDE IDG)
- Lists CFDE Tclin genes present in cohort with breast-cancer-filtered drug names and patient counts

### "Powered By" Footer
- Centered footer with logos: **CFDE IDG**, **CM4AI**, **Bridge2AI**
- Logos in `static/logos/` (cfde_idg.png, cm4ai.png, bridge2ai.png)
- Bridge2AI logo CSS-inverted (`filter: invert(1)`)

---

## LLM Interpretation Page (`static/interpretation.html`)

- Page title changed: **"Descriptive Interpretations"** (was "Patient Explorer")
- Patient table removed → replaced with **dropdown** (`<select id="f-patient">`)
- Label "Patient ID" → **"Sample ID"** throughout
- First patient auto-loaded on page load; URL `?patient=PID` param restores selection on tab switch
- `sessionStorage.setItem('syncPatientId', pid)` written on every patient selection
- **Summary accordion always open** by default (`class="accordion open"`)
- "Full LLM Reasoning" label → **`<strong style="font-size:16px">Summary</strong>`**
- AI disclaimer footer: *"The interpretations are generative AI-assisted and may contain inaccuracies. All biological and clinical findings should be independently verified by a qualified expert before clinical use."*

---

## Tab Shell (`static/index.html`)

- **"NeST-VNN" logo** has animated gradient text (blue → purple → red, loops every 4s)
- Cross-tab patient sync: on tab switch, appends `?patient=PID` (interpretation) or `&patient=PID` (explorer) to iframe src

---

## NeST-STRING Explorer (`static/explorer.html`)

- Boot sequence uses `Promise.all([loadNetwork(), loadPatients()])` to fix race condition where patient mode was applied before Cytoscape was ready
- Removed BroadcastChannel (replaced by URL param approach)

---

## CFDE Drug Filter (`bioitworld_nest_vnn/interpretation/cfde_drugs.py`)

Added `_BREAST_CANCER_DRUGS` frozenset and `filter_breast_cancer(drugs)` function.

Breast cancer drugs included:
- **CDK4/6 inhibitors**: abemaciclib, palbociclib, ribociclib
- **HER2-targeted**: trastuzumab, pertuzumab, trastuzumab emtansine, trastuzumab deruxtecan, lapatinib, neratinib, tucatinib, margetuximab
- **Endocrine**: tamoxifen, toremifene, raloxifene, fulvestrant, elacestrant, letrozole, anastrozole, exemestane
- **PARP inhibitors**: olaparib, talazoparib
- **PI3K/AKT/mTOR**: alpelisib, inavolisib, capivasertib, everolimus
- **Immunotherapy (TNBC)**: pembrolizumab, atezolizumab
- **ADCs**: sacituzumab govitecan, datopotamab deruxtecan
- **Other**: bevacizumab, eribulin, capecitabine, ixabepilone

Filter applied in:
- `prompts.py` — drugs passed to LLM for future pipeline runs
- `server.py` `/api/cfde/drugs` — dashboard druggable targets panel
- `server.py` `/api/patient/{id}/gene-interpretation/{nest_id}` — explorer gene cards
- `server.py` `/api/interp/patient/{id}/genes` — interpretation page gene table

---

## LLM Output Format (`bioitworld_nest_vnn/interpretation/prompts.py` + `pipeline.py`)

New prose-based markdown format replacing the old table format:

```
# Patient {ID}
[clinical context paragraph]

## NEST 1: NEST:XXXX
### Summary
[prose: pathway, direction, clinical reasoning with Reactome link]

### Top Genes
#### GENE_NAME
[prose: alteration, direction, biological role, drugs]

## Overall Patient Interpretation
[overall risk summary]
```

`_parse_llm_response` in `pipeline.py` fully rewritten with regex parsers for the new format.

---

## Server (`visualization/BioiThackCellMAP/server.py`)

- Added `/api/cfde/drugs` endpoint serving breast-cancer-filtered drug lookup JSON
- Imported `filter_breast_cancer` from `interpretation.cfde_drugs`
- Applied filter in gene interpretation endpoints

---

## How to Re-run the LLM Pipeline

```powershell
# Set credentials in your terminal session first
$env:AWS_ACCESS_KEY_ID = "your-key"
$env:AWS_SECRET_ACCESS_KEY = "your-secret"
$env:AWS_DEFAULT_REGION = "us-east-1"

# Delete old DB (make sure server is stopped first)
Remove-Item "C:\Users\prath\OneDrive\Desktop\cellmapvnn\bioitworld_nest_vnn\interpretation\nest_vnn_interpretations.db"

# Run pipeline (fresh start)
cd "C:\Users\prath\OneDrive\Desktop\cellmapvnn\bioitworld_nest_vnn"
python interpretation/run_pipeline.py breast_msk_2025 OS_MONTHS_binary --batch_size 5

# Restart server
cd "C:\Users\prath\OneDrive\Desktop\cellmapvnn\visualization\BioiThackCellMAP"
python server.py
```
