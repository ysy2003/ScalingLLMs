#!/bin/bash
# Alternative: Server 2 doesn't need checkpoint_300.xlsx
# It will save ONLY samples 300-483, merge later manually

echo "=================================================="
echo "PREPARING SERVER 2 (NO CHECKPOINT NEEDED)"
echo "=================================================="

# Modify skip condition
sed -i 's/if idx < 100:/if idx < 300:/' test_qwen_metrics.py

# REMOVE the checkpoint loading section (make it optional)
sed -i '/Load checkpoint data for samples/,/Total samples:/c\
# Server 2: Save only new results (300-483)\
# Merge with Server 1 results later\
new_results_df = pd.DataFrame(results)\
df = new_results_df\
df.to_excel(f"{RESULTS_DIR}/detailed_results.xlsx", index=False)\
print(f"\\n✓ Saved detailed results to {RESULTS_DIR}/detailed_results.xlsx")\
print(f"  Total samples: {len(df)}")' test_qwen_metrics.py

echo "✓ Modified skip condition to: if idx < 300"
echo "✓ Removed checkpoint dependency"
echo ""
echo "Server 2 will save ONLY samples 300-483"
echo "Merge with Server 1 results after completion using merge_parallel_results.py"
echo ""
echo "Ready to run: python test_qwen_metrics.py"
