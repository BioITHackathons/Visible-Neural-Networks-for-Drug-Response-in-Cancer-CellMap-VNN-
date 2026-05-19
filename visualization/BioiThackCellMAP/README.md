# NeST-STRING Explorer

Interactive web tool for exploring NeST (Nested Systems in Tumors) pathway maps and STRING protein-protein interaction networks, with per-patient genomic context from NeST-VNN model outputs.

## Features

- **NeST Map** — full Cytoscape.js visualization of the NeST hierarchy loaded from NDEx
- **STRING integration** — click any NeST complex to fetch its STRING PPI network image and protein descriptions
- **Patient mode** — select a patient to highlight their top 3 directionally-significant NeST systems (green = higher outcome direction, red = lower), with each node's depth-level color shown as a border
- **Gene alteration badges** — per-patient mutation (MUT), copy-number amplification (AMP), deletion (DEL), and fusion (FUS) status shown on protein cards
- **Full gene descriptions** — click any protein card or gene chip to open a modal with the complete gene function summary from MyGene.info
- **RLIPP interpretation** — patient-level and cohort-level RLIPP scores with plain-English explanation of pathway vs. gene-level contribution

## Setup

```bash
pip install flask flask-cors requests
python server.py
```

Open http://localhost:5001

## Data requirements

Patient mode requires a `patient_viz.html` output file from [NeST-VNN](https://github.com/CM4AI/bioitworld_nest_vnn). Update `SEARCH_PATHS` in `patient_loader.py` to point to your file.

Alteration badges require the corresponding `nest_vnn_input/` directory (cell2mutation.txt, cell2cnamplification.txt, cell2cndeletion.txt, cell2fusion.txt) to be present at the study level.

## Files

| File | Description |
|------|-------------|
| `server.py` | Flask proxy — NDEx, STRING, MyGene.info, and patient data endpoints |
| `patient_loader.py` | Parses patient_viz.html and loads binary alteration matrices at startup |
| `static/index.html` | Single-page frontend (Cytoscape.js, vanilla JS) |
| `nest_cli.py` | CLI utilities |
