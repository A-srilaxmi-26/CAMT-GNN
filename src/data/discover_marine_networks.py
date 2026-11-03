from pathlib import Path

import pandas as pd

from mangal_api import get_all_records


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_MANGAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mangal"
)

RAW_MANGAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MARINE KEYWORDS
# ============================================================

MARINE_KEYWORDS = [
    "marine",
    "ocean",
    "sea",
    "reef",
    "estuary",
    "coastal",
    "pelagic",
    "benthic",
    "kelp",
    "lagoon",
    "antarctic",
    "arctic",
    "ross sea",
    "bay",
    "gulf",
]


# ============================================================
# MARINE NETWORK DETECTION
# ============================================================

def is_marine_network(description, name):

    text = (
        f"{description or ''} "
        f"{name or ''}"
    ).lower()

    return any(
        keyword in text
        for keyword in MARINE_KEYWORDS
    )


# ============================================================
# DOWNLOAD NETWORK CATALOGUE
# ============================================================

print("=" * 70)
print("CAMT-GNN | MANGAL MARINE NETWORK DISCOVERY")
print("=" * 70)

print("\nDownloading Mangal network catalogue...\n")


networks = get_all_records(
    "network"
)


print(
    f"\nTotal networks retrieved: "
    f"{len(networks)}"
)


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

rows = []

for network in networks:

    rows.append(
        {
            "id": network.get("id"),
            "name": network.get("name"),
            "date": network.get("date"),
            "description": network.get("description"),
            "public": network.get("public"),
            "all_interactions": network.get(
                "all_interactions"
            ),
            "dataset_id": network.get(
                "dataset_id"
            ),
            "geom": network.get("geom"),
        }
    )


df = pd.DataFrame(rows)


# ============================================================
# IDENTIFY MARINE CANDIDATES
# ============================================================

marine_mask = df.apply(
    lambda row: is_marine_network(
        row["description"],
        row["name"],
    ),
    axis=1,
)


marine_df = df[marine_mask].copy()


# ============================================================
# SAVE FULL CATALOGUE
# ============================================================

catalogue_path = (
    RAW_MANGAL_DIR
    / "networks_catalog.csv"
)

df.to_csv(
    catalogue_path,
    index=False,
)


# ============================================================
# SAVE MARINE CANDIDATES
# ============================================================

marine_output_path = (
    RAW_MANGAL_DIR
    / "marine_network_candidates.csv"
)

marine_df.to_csv(
    marine_output_path,
    index=False,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)

print(
    f"Total networks retrieved : {len(df)}"
)

print(
    f"Marine candidates found  : {len(marine_df)}"
)

print("=" * 70)


print(
    f"\nFull catalogue saved to:\n"
    f"{catalogue_path}"
)

print(
    f"\nMarine candidates saved to:\n"
    f"{marine_output_path}"
)


# ============================================================
# SHOW SAMPLE
# ============================================================

print("\nSample marine candidates:\n")


if len(marine_df) > 0:

    print(
        marine_df[
            [
                "id",
                "name",
                "date",
                "description",
                "public",
                "all_interactions",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

else:

    print("No marine candidates found.")


print("\n✅ Marine network discovery completed.")