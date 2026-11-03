from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mangal"
    / "marine_network_candidates.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mangal"
    / "filtered_marine_networks.csv"
)


MARINE_TERMS = [
    "marine",
    "ocean",
    "oceanic",
    "sea",
    "coastal",
    "coast",
    "estuary",
    "estuarine",
    "reef",
    "coral",
    "kelp",
    "pelagic",
    "benthic",
    "intertidal",
    "tidal",
    "saltwater",
    "seagrass",
    "continental shelf",
    "bay",
    "gulf",
    "antarctic",
    "arctic",
]


FRESHWATER_TERMS = [
    "lake",
    "lakes",
    "pond",
    "ponds",
    "river",
    "rivers",
    "stream",
    "streams",
    "freshwater",
    "fresh water",
    "reservoir",
]


def classify_network(name, description):

    text = (
        f"{name or ''} "
        f"{description or ''}"
    ).lower()

    has_marine = any(
        term in text
        for term in MARINE_TERMS
    )

    has_freshwater = any(
        term in text
        for term in FRESHWATER_TERMS
    )

    if has_marine and not has_freshwater:
        return "marine"

    if has_marine and has_freshwater:
        return "mixed_or_uncertain"

    return "non_marine"


print("=" * 70)
print("CAMT-GNN | MARINE NETWORK FILTER")
print("=" * 70)


df = pd.read_csv(INPUT_FILE)


df["classification"] = df.apply(
    lambda row: classify_network(
        row["name"],
        row["description"],
    ),
    axis=1,
)


filtered_df = df[
    df["classification"] == "marine"
].copy()


filtered_df = filtered_df.sort_values(
    by=["date", "id"],
    na_position="last",
)


filtered_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(f"\nOriginal candidates : {len(df)}")
print(f"Marine candidates   : {len(filtered_df)}")

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)


print("\nMarine candidates:\n")

print(
    filtered_df[
        [
            "id",
            "name",
            "date",
            "description",
            "all_interactions",
        ]
    ].to_string(index=False)
)


print("\n✅ Filtering completed.")