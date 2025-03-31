
import os

print("\n===== Running All Evaluation Scripts =====\n")

scripts = [
    "evaluate_embeddings.py",
    "evaluate_clustering.py",
    "evaluate_queries.py",
    "dump_summaries.py"
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    os.system(f"python {script}")
