import json
import time
from pathlib import Path

import requests


BASE_URL = "https://mangal.io/api/v2"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_MANGAL_DIR = PROJECT_ROOT / "data" / "raw" / "mangal"

RAW_MANGAL_DIR.mkdir(parents=True, exist_ok=True)


SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": "CAMT-GNN-Research-Project/1.0"
    }
)


def get_json(endpoint, params=None, timeout=180):
    """
    Send a GET request to the Mangal API and return JSON.
    """

    url = f"{BASE_URL}/{endpoint.lstrip('/')}"

    response = SESSION.get(
        url,
        params=params,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


def get_all_records(
    endpoint,
    params=None,
    count=1000,
    delay=0.2,
):
    """
    Retrieve all records from a paginated Mangal endpoint.

    Mangal supports up to 1000 records per request.
    We continue requesting pages until the returned page
    contains fewer records than the requested count.
    """

    if params is None:
        params = {}

    params = params.copy()

    all_records = []
    page = 0

    while True:

        request_params = params.copy()

        request_params["page"] = page
        request_params["count"] = count

        print(
            f"Downloading {endpoint} | "
            f"page={page} | "
            f"records={len(all_records)}"
        )

        records = get_json(
            endpoint,
            params=request_params,
        )

        if not isinstance(records, list):
            raise ValueError(
                f"Expected a list from /{endpoint}, "
                f"received {type(records)}"
            )

        all_records.extend(records)

        print(
            f"  Received {len(records)} records"
        )

        if len(records) < count:
            break

        page += 1

        time.sleep(delay)

    print(
        f"Completed /{endpoint}: "
        f"{len(all_records)} total records"
    )

    return all_records


def get_network(network_id):
    """
    Retrieve one network's metadata.
    """

    return get_json(
        f"network/{network_id}"
    )


def get_network_nodes(network_id):
    """
    Retrieve all nodes belonging to a network.
    """

    return get_all_records(
        "node",
        params={
            "network_id": network_id
        },
    )


def get_network_interactions(network_id):
    return get_all_records(
        "interaction",
        params={
            "network_id": network_id,
            "sort": "id"
        }
    )


def save_json(data, filepath):
    """
    Save data as formatted JSON.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def download_network(network_id):
    """
    Download metadata, nodes and interactions
    for one complete Mangal network.
    """

    network_dir = RAW_MANGAL_DIR / str(network_id)

    network_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print(f"NETWORK {network_id}")
    print("=" * 70)

    print("\n[1/3] Network metadata")

    network = get_network(network_id)

    print(
        "Name:",
        network.get("name")
    )

    print(
        "Date:",
        network.get("date")
    )

    print(
        "Description:",
        network.get("description")
    )

    print(
        "All interactions:",
        network.get("all_interactions")
    )

    save_json(
        network,
        network_dir / "network.json"
    )

    print("\n[2/3] Nodes")

    nodes = get_network_nodes(
        network_id
    )

    print(
        f"Total nodes: {len(nodes)}"
    )

    save_json(
        nodes,
        network_dir / "nodes.json"
    )

    print("\n[3/3] Interactions")

    interactions = get_network_interactions(
        network_id
    )

    print(
        f"Total interactions: "
        f"{len(interactions)}"
    )

    save_json(
        interactions,
        network_dir / "interactions.json"
    )

    print("\n✅ Network downloaded successfully.")

    return {
        "network": network,
        "nodes": nodes,
        "interactions": interactions,
    }