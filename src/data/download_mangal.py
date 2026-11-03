from mangal_api import download_network


if __name__ == "__main__":

    print("CAMT-GNN Mangal Data Downloader")

    network_ids = [
        2472,
        2474,
    ]

    for network_id in network_ids:

        download_network(
            network_id
        )