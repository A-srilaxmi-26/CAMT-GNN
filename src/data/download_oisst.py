from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]

output_dir = ROOT / "data/raw/oisst"
output_dir.mkdir(parents=True, exist_ok=True)

url = (
    "https://erddap.aoml.noaa.gov/hdb/erddap/griddap/"
    "SST_OI_DAILY_1981_PRESENT_T_V1.csv?"
    "sst[(2005-06-01T00:00:00Z)]"
    "[(17.75)]"
    "[(-76.75)]"
)

output_file = output_dir / "oisst_20050601.csv"

response = requests.get(url, timeout=180)
response.raise_for_status()

output_file.write_bytes(response.content)

print(f"Downloaded: {output_file}")
print(f"Size: {output_file.stat().st_size / 1024:.2f} KB")
print(response.text)