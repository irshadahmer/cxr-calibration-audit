import os
import urllib.request
import tarfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    print(f"Downloading: {url} -> {output_path}")
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.info().get('Content-Length', 0))
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path), total=total_size) as t:
            with open(output_path, 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    t.update(len(chunk))

def main():
    os.makedirs("data/nih", exist_ok=True)
    os.makedirs("data/nih/images", exist_ok=True)

    meta_path = "data/nih/Data_Entry_2017.csv"

    # Official NIH clinical metadata mirrors
    # 1. Official Box direct link
    # 2. Permanent raw GitHub mirror from academic repository
    csv_mirrors = [
        "https://raw.githubusercontent.com/zoho/ChestX-ray14/master/Data_Entry_2017.csv",
        "https://nihcc.box.com/shared/static/d60241hs5asrnwwm7sclj4uvqww7z4ec.csv",
        "https://raw.githubusercontent.com/marshald/ChestX-ray14/master/Data_Entry_2017.csv"
    ]

    if not os.path.exists(meta_path) or os.path.getsize(meta_path) < 1000:
        downloaded = False
        for url in csv_mirrors:
            try:
                download_file(url, meta_path)
                if os.path.exists(meta_path) and os.path.getsize(meta_path) > 1000:
                    print("Metadata CSV successfully acquired.")
                    downloaded = True
                    break
            except Exception as e:
                print(f"Mirror failed ({url}): {e}")
        if not downloaded:
            print("Failed to download metadata from mirrors.")
            return
    else:
        print(f"Valid metadata CSV already present at {meta_path}")

    # Test reading the metadata
    import pandas as pd
    df = pd.read_csv(meta_path)
    print("\nNIH ChestX-ray14 Metadata Loaded Successfully:")
    print(f"Total scans listed: {len(df):,}")
    print("Columns available:", list(df.columns))
    print(df[['Image Index', 'Finding Labels', 'Patient Age', 'Patient Gender']].head(3))

if __name__ == "__main__":
    main()
