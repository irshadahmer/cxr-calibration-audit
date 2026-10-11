import os
import pandas as pd
import torchxrayvision as xrv

csv_path = "data/external_cxr/metadata.csv"
img_dir = "data/external_cxr/images"

df = pd.read_csv(csv_path)
print(f"Total clinical records: {len(df)}")
print("Available columns:", [c for c in df.columns if c in ['patientid', 'age', 'sex', 'finding', 'view', 'modality']])

# Filter to frontal chest X-rays (PA / AP)
frontal = df[df['view'].isin(['PA', 'AP', 'AP Supine'])]
print(f"Total frontal CXR records: {len(frontal)}")
print(f"Sex distribution:\n{frontal['sex'].value_counts(dropna=False)}")
print(f"Age range: {frontal['age'].dropna().astype(float).min()} to {frontal['age'].dropna().astype(float).max()} (mean {frontal['age'].dropna().astype(float).mean():.1f})")

# Test torchxrayvision dataset loader on these real images
ds = xrv.datasets.COVID19_Dataset(imgpath=img_dir, csvpath=csv_path, views=["PA", "AP", "AP Supine"])
print(f"\nTorchXRayVision dataset initialized with {len(ds)} real clinical scans.")
print("Pathology targets available in this cohort:", ds.pathologies)

sample = ds[0]
print(f"Sample 0 image shape: {sample['img'].shape}, label vector sum: {sample['lab'].sum()}")
