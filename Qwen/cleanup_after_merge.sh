#!/bin/bash
# Cleanup script after successful merge
# Run this ONLY after verifying results_Qwen_merged is complete

echo "============================================================"
echo "CLEANUP AFTER MERGE"
echo "============================================================"
echo ""
echo "This will clean up the following:"
echo "  1. results_Qwen (16M) - Source Server 1 data (samples 0-299)"
echo "  2. results_Qwen(300-484) (7.9M) - Source Server 2 data (samples 300-483)"
echo "  3. Temporary merge scripts and instructions"
echo ""
echo "KEEPING:"
echo "  ✓ results_Qwen_merged (19M) - Your complete merged dataset"
echo ""
read -p "Are you sure you want to proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Step 1: Creating backup archive (optional)..."
read -p "Create backup archive before deletion? (yes/no): " backup

if [ "$backup" = "yes" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    tar -czf "backup_original_results_${timestamp}.tar.gz" results_Qwen results_Qwen\(300-484\)
    echo "✓ Backup created: backup_original_results_${timestamp}.tar.gz"
fi

echo ""
echo "Step 2: Removing source result directories..."
rm -rf results_Qwen
echo "✓ Removed results_Qwen"
rm -rf "results_Qwen(300-484)"
echo "✓ Removed results_Qwen(300-484)"

echo ""
echo "Step 3: Removing temporary merge files..."
rm -f MERGE_INSTRUCTIONS.txt
rm -f README_PARALLEL.txt
rm -f prepare_server2_no_checkpoint.sh
rm -f restore_and_clean_from_checkpoint.py
echo "✓ Removed temporary merge-related files"

echo ""
echo "Step 4: Cleanup results_WebSight (empty directory)..."
rmdir results_WebSight 2>/dev/null && echo "✓ Removed empty results_WebSight" || echo "⚠ results_WebSight not empty or doesn't exist"

echo ""
echo "============================================================"
echo "CLEANUP COMPLETE!"
echo "============================================================"
echo ""
echo "Remaining files:"
du -sh results_* 2>/dev/null
echo ""
echo "Space saved: ~24M"
