from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = ROOT / "data/processed/pyg/network_2472.pt"
OUTPUT_DIR = ROOT / "data/results/shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def link_score(z_src, z_dst):
    return (z_src * z_dst).sum(dim=1)


def main():

    print("Loading graph...")

    data = torch.load(
        GRAPH_PATH,
        weights_only=False
    )

    embeddings = data.x.float()

    print(
        f"Network: 2472 | "
        f"nodes={data.num_nodes} | "
        f"edges={data.num_edges}"
    )

    # Use real positive Mangal interactions.
    edge_index = data.edge_index
    sample_size = min(100, edge_index.shape[1])
    edge_index = edge_index[:, :sample_size]

    src = edge_index[0]
    dst = edge_index[1]

    original_scores = link_score(
        embeddings[src],
        embeddings[dst]
    )

    # Permutation importance:
    # measure how much the link score changes when each
    # node-feature dimension is randomly permuted.
    num_features = embeddings.shape[1]

    rng = np.random.default_rng(42)

    importance = []

    for feature_idx in range(num_features):

        perturbed = embeddings.clone()

        permutation = torch.tensor(
            rng.permutation(data.num_nodes),
            dtype=torch.long
        )

        perturbed[:, feature_idx] = (
            perturbed[permutation, feature_idx]
        )

        perturbed_scores = link_score(
            perturbed[src],
            perturbed[dst]
        )

        score_change = torch.mean(
            torch.abs(
                original_scores - perturbed_scores
            )
        ).item()

        importance.append(score_change)

    importance_df = pd.DataFrame(
        {
            "feature_index": np.arange(num_features),
            "mean_absolute_permutation_importance": importance
        }
    )

    importance_df = importance_df.sort_values(
        "mean_absolute_permutation_importance",
        ascending=False
    )

    output_csv = (
        OUTPUT_DIR /
        "feature_permutation_importance.csv"
    )

    importance_df.to_csv(
        output_csv,
        index=False
    )

    print("\nExplainability analysis completed.")
    print("\nTop 10 features:")

    print(
        importance_df.head(10).to_string(index=False)
    )

    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()