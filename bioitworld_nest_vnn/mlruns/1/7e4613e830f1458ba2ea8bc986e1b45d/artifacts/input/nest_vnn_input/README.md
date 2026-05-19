# NeST-VNN Input Data

Generated: 2026-05-18 18:37:36

- **Samples:** 3784
- **Gene panel:** 407 genes

## Data Sources

- **cBioPortal study:** [breast_msk_2025](https://www.cbioportal.org/study/summary?id=breast_msk_2025)
- **Ontology (NDEx):** [9a8f5326-aa6e-11ea-aaef-0ac135e8bacf](https://www.ndexbio.org/viewer/networks/9a8f5326-aa6e-11ea-aaef-0ac135e8bacf)
- **Min genes per assembly:** 5 (paper default: 5)
- **Gene selection:** frequency-filtered — genes altered in ≥ 0.5% of 3784 samples (407 genes retained)

## Feature Matrices

| Feature | Events | Genes hit | Density |
|---|---|---|---|
| Mutations | 21004 | 406/407 | 1.36% |
| CN Deletions | 2835 | 354/407 | 0.18% |
| CN Amplifications | 15036 | 384/407 | 0.98% |
| Fusions | 1133 | 292/407 | 0.07% |
| **Any alteration** | 39484 | 407/407 | samples=3783/3784, 2.56% |

## Endpoints

| Label | Mode | N | Summary |
|---|---|---|---|
| `binary_os_months` | binary | 3777 | event(0)=2071, no-event(1)=1706 |

## Endpoint Configuration

```json
[
  {
    "label_name": "binary_os_months",
    "clinical_col": "OS_MONTHS",
    "mode": "binary_threshold",
    "mapping": {
      "threshold": 46.0
    }
  }
]
```

## Usage

```bash
python src/train.py \
  -train data\output\breast_msk_2025\nest_vnn_input/training_data.txt \
  -label binary_os_months -task binary \
  -mutations data\output\breast_msk_2025\nest_vnn_input/cell2mutation.txt \
  -cn_deletions data\output\breast_msk_2025\nest_vnn_input/cell2cndeletion.txt \
  -cn_amplifications data\output\breast_msk_2025\nest_vnn_input/cell2cnamplification.txt \
  -fusions data\output\breast_msk_2025\nest_vnn_input/cell2fusion.txt \
  -onto data\output\breast_msk_2025\nest_vnn_input/ontology.txt \
  -gene2id data\output\breast_msk_2025\nest_vnn_input/gene2ind.txt \
  -cell2id data\output\breast_msk_2025\nest_vnn_input/cell2ind.txt \
  -std MODEL/std.txt -model MODEL/ \
  -cuda 0 -epoch 300 -batchsize 64
```
