"""Download the FinQA and TAT-QA dev and test files into data/raw/."""
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URLS = {
    "finqa_dev.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "finqa_test.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
    "tatqa_dev.json": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json",
    "tatqa_test_gold.json": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_test_gold.json",
}


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        target = RAW / name
        if target.exists():
            print(f"have  {name}")
            continue
        print(f"fetch {name}")
        urllib.request.urlretrieve(url, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
