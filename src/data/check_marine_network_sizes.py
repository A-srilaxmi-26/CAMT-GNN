from pathlib import Path
import pandas as pd

from mangal_api import get_all_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mangal"
    / "filtered_marine_networks.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mangal"
    / "marine_network_sizes.csv"
)


df = pd.read_csv(INPUT_FILE)

results = []

print("=" * 70)
print("CAMT-GNN | CHECKING REAL MARINE NETWORK SIZES")
print("=" * 70)

for i, row in df.iterrows():

    network_id = int(row["id"])

    print(
        f"[{i + 1}/{len(df)}] "
        f"Network {network_id} | {row['name']}"
    )

    try:

        interactions = get_all_records(
            "interaction",
            params={"network_id": network_id}
        )

        results.append(
            {
                "id": network_id,
                "name": row["name"],
                "date": row["date"],
                "description": row["description"],
                "interaction_count": len(interactions),
            }
        )

        print(
            f"    Interactions: {len(interactions)}"
        )

    except Exception as e:

        print(f"    ERROR: {e}")

        results.append(
            {
                "id": network_id,
                "name": row["name"],
                "date": row["date"],
                "description": row["description"],
                "interaction_count": None,
            }
        )


result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    "interaction_count",
    ascending=False,
    na_position="last"
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("\nLargest networks:")

print(
    result_df[
        [
            "id",
            "name",
            "date",
            "interaction_count"
        ]
    ].head(20).to_string(index=False)
)