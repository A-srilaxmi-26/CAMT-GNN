from pathlib import Path
from mangal_api import get_network_interactions, save_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NETWORK_IDS = [2472, 2474, 5028, 5027, 5029]

for network_id in NETWORK_IDS:

    print("=" * 60)
    print(f"Refreshing Network {network_id}")

    interactions = get_network_interactions(network_id)

    output = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "mangal"
        / str(network_id)
        / "interactions.json"
    )

    save_json(interactions, output)

    print(f"Saved {len(interactions)} interactions")
    print(f"Unique IDs: {len({x['id'] for x in interactions})}")

print("\n✅ All interaction files refreshed.")