# Benchmark Dataset Notes

The benchmark data root is:

```text
datasets/alibaba_pcea/processed_benchmark/
```

The directory contains the workflow JSON files and split manifests used by the evaluation scripts:

- `train_manifest.csv`
- `val_manifest.csv`
- `test_manifest.csv`
- `benchmark_manifest.csv`
- `manifest.csv`

The `benchmark` split contains 136 workflows. The group-size-5 experiment forms 27 complete groups from 135 workflows; the remaining workflow is not used in that grouped run.

The benchmark is derived from Alibaba batch traces with additional CPU-GPU workflow attributes required by the scheduling model. The raw trace CSV files should be obtained from the original data source when rebuilding the processed data locally.