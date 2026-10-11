import os
import urllib.request
import zipfile
import pandas as pd
import torchxrayvision as xrv

os.makedirs("data/nih/images", exist_ok=True)

# Test accessing NIH dataset instance
ds = xrv.datasets.NIH_Dataset(imgpath="data/nih/images")
print(f"Dataset successfully initialized. Total cohort records: {len(ds.csv)}")
print(f"Demographics available: Age, Gender ({ds.csv['Patient Gender'].value_counts().to_dict()})")

# Test URL: NIH Clinical Center public direct batch
# https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorzs5.gz is batch 1
batch1_url = "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorzs5.gz"
print(f"Checking official NIH archive URL connection...")

req = urllib.request.Request(batch1_url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print("HTTP Status:", resp.status)
        print("Archive Content-Length:", resp.headers.get('Content-Length'), "bytes")
except Exception as e:
    print("Box link status:", e)
