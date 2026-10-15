"""Download the FinQA and TAT-QA dev and test files into data/raw/."""
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URLS = {
    "finqa_dev.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "finqa_test.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
    "tatqa_dev.json": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json",
    "tatqa_test_gold.json": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_test_gold.json",
}
# Some Python installs (e.g. Homebrew) ship without a certificate list.
_TLS = ssl.create_default_context(cafile=certifi.where())


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        target = RAW / name
        if target.exists():
            print(f"have  {name}")
            continue
        print(f"fetch {name}")
        with urllib.request.urlopen(url, context=_TLS) as response, open(target, "wb") as out:
            shutil.copyfileobj(response, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
