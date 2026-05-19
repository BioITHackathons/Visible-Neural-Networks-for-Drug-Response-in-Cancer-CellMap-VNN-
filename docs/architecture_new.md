# CellMap-VNN New Architecture

Genotype + Clinical Covariates

```mermaid
flowchart TD
    subgraph GENOMIC["Genomic Pathway"]
        direction TB
        MUT["Mutations\n(S × genes)"]
        CND["CN Deletions\n(S × genes)"]
        CNA["CN Amplifications\n(S × genes)"]
        DSTACK["np.dstack → Cell Features\n(S × genes × 3)"]
        PERGENE["Per-Gene Feature Layer\nLinear(feature_dim → 1) → tanh → BatchNorm1d(1)\n<i>one per gene</i>"]
        GENEEMB["Gene Embeddings\nscalar per gene → cat → gene_input"]
        DIRECT["Direct Gene Layers\nLinear(num_genes → |annotated_genes|) per term\n<i>weights masked by ontology</i>"]
        ONTOLOGY["Ontology Term Layers (leaf → root)\nFor each term, bottom-up:\ncat(children_hidden + direct_genes)\n→ Dropout → Linear(in, h) → tanh → BatchNorm(h)"]
        AUX["Aux Head (per term)\nLinear(h → 1) → tanh → Linear(1 → 1)\n<i>supervised against same label</i>"]
        ROOT["root_hidden\n(h)"]

        MUT --> DSTACK
        CND --> DSTACK
        CNA --> DSTACK
        DSTACK --> PERGENE
        PERGENE --> GENEEMB
        GENEEMB --> DIRECT
        DIRECT --> ONTOLOGY
        ONTOLOGY -.->|each term| AUX
        ONTOLOGY --> ROOT
    end

    subgraph CLINICAL["Clinical Pathway"]
        direction TB
        CLINFEAT["Clinical Features\ncov_* columns from training_data.txt\nMutation count | z-score | FGA\nSample type | Met site | ...\n(S × clin_dim)"]
        CLINENC["Clinical Encoder\nLinear(clin_dim → h) → tanh → BatchNorm(h)\n<i>projects to same dim as root_hidden</i>"]
        CLINH["clinical_h\n(h)"]

        CLINFEAT --> CLINENC
        CLINENC --> CLINH
    end

    subgraph MERGED["Merged Pathway"]
        direction TB
        CONCAT["torch.cat dim=1\n(2h)"]
        FINAL1["Linear(2h → 1) → tanh\n<i>final_aux_linear_layer</i>"]
        FINAL2["Linear(1 → 1)\n<i>final_linear_layer_output</i>"]
        PRED(["Prediction\nlogits or z-score"])

        CONCAT --> FINAL1
        FINAL1 --> FINAL2
        FINAL2 --> PRED
    end

    ROOT --> CONCAT
    CLINH --> CONCAT

    style GENOMIC fill:#2a2050,stroke:#6b5bce,color:#e0d8ff
    style CLINICAL fill:#0f3a34,stroke:#2d8a7a,color:#c8f0e8
    style MERGED fill:#4a2a0a,stroke:#c0783a,color:#ffe0c0

    style MUT fill:#4a3f8a,stroke:#7c6fca,color:#e0d8ff
    style CND fill:#4a3f8a,stroke:#7c6fca,color:#e0d8ff
    style CNA fill:#4a3f8a,stroke:#7c6fca,color:#e0d8ff
    style DSTACK fill:#4a3f8a,stroke:#7c6fca,color:#e0d8ff
    style PERGENE fill:#3d2e9e,stroke:#6b5bce,color:#fff
    style GENEEMB fill:#1a5c3a,stroke:#10b981,color:#c8f0e8
    style DIRECT fill:#3d2e9e,stroke:#6b5bce,color:#fff
    style ONTOLOGY fill:#3d2e9e,stroke:#6b5bce,color:#fff
    style AUX fill:#5a4a8a,stroke:#8b5cf6,color:#e0d8ff
    style ROOT fill:#3a3a4e,stroke:#6a6a80,color:#ccc

    style CLINFEAT fill:#1a5c52,stroke:#2d8a7a,color:#c8f0e8
    style CLINENC fill:#145c52,stroke:#2d8a7a,color:#c8f0e8
    style CLINH fill:#3a3a4e,stroke:#6a6a80,color:#ccc

    style CONCAT fill:#7a4a1a,stroke:#c0783a,color:#ffe0c0
    style FINAL1 fill:#7a4a1a,stroke:#c0783a,color:#ffe0c0
    style FINAL2 fill:#7a4a1a,stroke:#c0783a,color:#ffe0c0
    style PRED fill:#7a4a1a,stroke:#c0783a,color:#ffe0c0
```

## Forward pass summary

| Step | Layer | Input shape | Output shape |
|------|-------|-------------|--------------|
| 1 | `np.dstack` | 3 matrices (S × genes) | S × genes × feature_dim |
| 2 | Per-gene `Linear → tanh → BatchNorm` | S × feature_dim (per gene) | S × 1 (per gene) |
| 3 | `torch.cat` gene embeddings | S × 1 × num_genes | S × num_genes |
| 4 | Direct gene layer (per term) | S × num_genes | S × \|annotated_genes\| |
| 5 | Ontology term layers (leaf → root) | S × (children_h + direct_genes) | S × h |
| 5a | Aux head (per term) | S × h | S × 1 (loss only) |
| 6 | Clinical encoder | S × clin_dim | S × h |
| 7 | `torch.cat([root_hidden, clinical_h])` | S × h, S × h | S × 2h |
| 8 | `Linear(2h, 1) → tanh` | S × 2h | S × 1 |
| 9 | `Linear(1, 1)` | S × 1 | S × 1 |

Where **h** = `num_hiddens_genotype` (default 4), **S** = batch size.
