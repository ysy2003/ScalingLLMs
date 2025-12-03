"""
Merge results from parallel execution on two servers
Server 1: samples 100-299
Server 2: samples 300-483
"""
import pandas as pd
import os
import shutil

print("="*60)
print("MERGING PARALLEL RESULTS")
print("="*60)

# Configuration
SERVER1_DIR = "results_Qwen"  # Server 1's results (0-380)
SERVER2_DIR = "results_Qwen(300-484)"  # Server 2's results (300-484)
MERGED_DIR = "results_Qwen_merged"

# Create merged directory
os.makedirs(f"{MERGED_DIR}/predictions", exist_ok=True)
os.makedirs(f"{MERGED_DIR}/results", exist_ok=True)

# Step 1: Load checkpoint_100.xlsx (samples 0-99)
print("\n1. Loading checkpoint_100.xlsx (samples 0-99)...")
df_0_99 = pd.read_excel(f"{SERVER1_DIR}/results/checkpoint_100.xlsx")
print(f"   ✓ Loaded {len(df_0_99)} samples")

# Step 2: Load Server 1 checkpoint_300 (contains 100-299)
print("\n2. Loading Server 1 results (samples 100-299)...")
df_server1 = pd.read_excel(f"{SERVER1_DIR}/results/checkpoint_300.xlsx")
df_100_299 = df_server1[(df_server1['number'] >= 100) & (df_server1['number'] < 300)]
print(f"   ✓ Loaded {len(df_100_299)} samples (100-299)")

# Step 3: Load Server 2 detailed results (contains 300-483)
print("\n3. Loading Server 2 results (samples 300-483)...")
df_server2 = pd.read_excel(f"{SERVER2_DIR}/results/detailed_results.xlsx")
df_300_483 = df_server2[df_server2['number'] >= 300]
print(f"   ✓ Loaded {len(df_300_483)} samples")

# Step 4: Combine all results
print("\n4. Combining all results...")
df_merged = pd.concat([df_0_99, df_100_299, df_300_483], ignore_index=True)
df_merged = df_merged.sort_values('number').reset_index(drop=True)
print(f"   ✓ Total samples: {len(df_merged)}")

# Step 5: Save merged results
df_merged.to_excel(f"{MERGED_DIR}/results/all_results.xlsx", index=False)
print(f"\n✓ Saved merged results to {MERGED_DIR}/results/all_results.xlsx")

# Step 6: Copy HTML files
print("\n5. Copying HTML files...")
for idx in range(484):
    # Determine source
    if idx < 100:
        src = f"{SERVER1_DIR}/predictions/{idx}.html"
    elif idx < 300:
        src = f"{SERVER1_DIR}/predictions/{idx}.html"
    else:
        src = f"{SERVER2_DIR}/predictions/{idx}.html"
    
    dst = f"{MERGED_DIR}/predictions/{idx}.html"
    
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"   ⚠ Missing: {idx}.html")

print(f"✓ Copied HTML files to {MERGED_DIR}/predictions/")

print("\n" + "="*60)
print("MERGE COMPLETE!")
print("="*60)
print(f"\nMerged data location: {MERGED_DIR}/")
print(f"  Total samples: {len(df_merged)}")
print(f"  HTML files: {len(os.listdir(f'{MERGED_DIR}/predictions'))}")
