import os
import time
import psutil
import tracemalloc
import pandas as pd

import pynvml

from vectorization import add_documents_to_collection
from database_connection import connect_to_chromadb

BATCH_DIRS = ["documents/10", "documents/100", "documents/1000", "documents/10000"]

class Timer:
    def __enter__(self):
        self.start = time.time()
        tracemalloc.start()
        self.process = psutil.Process(os.getpid())

        self.cpu_percent = []
        self.running = True

        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_util = []
            self.gpu_mem = []
        except:
            self.gpu_handle = None

        import threading
        def track():
            while self.running:
                self.cpu_percent.append(self.process.cpu_percent(interval=0.1))
                if self.gpu_handle:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.gpu_util.append(util.gpu)
                    self.gpu_mem.append(mem.used / 1024**2)  # MB

        self.tracker = threading.Thread(target=track)
        self.tracker.start()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
        current, peak = tracemalloc.get_traced_memory()
        self.peak_mem = peak / 1024 / 1024  # MB
        tracemalloc.stop()

        self.running = False
        self.tracker.join()

        self.cpu_peak = max(self.cpu_percent) if self.cpu_percent else 0
        self.cpu_mean = sum(self.cpu_percent) / len(self.cpu_percent) if self.cpu_percent else 0

        self.gpu_peak_mem = max(self.gpu_mem) if self.gpu_mem else 0
        self.gpu_mean_util = sum(self.gpu_util) / len(self.gpu_util) if self.gpu_util else 0

results = []

chroma_client, collection = connect_to_chromadb()

for folder in BATCH_DIRS:
    print(f"\\n--- Benchmarking folder: {folder} ---")
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}, skipping...")
        continue

    with Timer() as t_upload:
        count = add_documents_to_collection(folder, collection)

    results.append({
        "Documents": count,
        "Upload Time (s)": t_upload.elapsed,
        "Upload Mem (MB)": t_upload.peak_mem,
        "Peak CPU (%)": t_upload.cpu_peak,
        "Mean CPU (%)": t_upload.cpu_mean,
        "Peak GPU Mem (MB)": t_upload.gpu_peak_mem,
        "Mean GPU Util (%)": t_upload.gpu_mean_util
    })

results_df = pd.DataFrame(results)
results_df.to_csv("scalability_metrics.csv", index=False)
print("\\n✅ Benchmark complete. Saved as scalability_metrics.csv")