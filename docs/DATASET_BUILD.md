# Dataset Build Notes

The repository includes the processed benchmark used by the reported experiments:

```text
datasets/alibaba_pcea/processed_benchmark/
```

The original Alibaba trace CSV files are not redistributed. To rebuild derived data from the raw traces, place the raw CSV files locally under:

```text
datasets/alibaba_raw/
```

Expected raw files:

- `batch_task.csv`
- `batch_instance.csv`

The trace-conversion script is:

```powershell
$env:PYTHONPATH="src"
python scripts/build_alibaba_pcea_dataset.py
```

The uploaded benchmark is a processed and curated split intended for reproducible evaluation of the scheduling methods in this repository.