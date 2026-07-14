# 1. Prepare data (once).
python benchmark/scripts/prepare_data.py

# 2. Run the sweep (starts workers, trains, evaluates, stops workers).
bash benchmark/scripts/run_sweep.sh

# 3. Collect results into CSV.
python benchmark/scripts/collect_results.py

# 4. Generate plots.
python benchmark/scripts/plot.py

# 5. Confirm outputs exist.
ls -lh benchmark/results/accuracy_vs_epsilon.png benchmark/results/privacy_cost.png

