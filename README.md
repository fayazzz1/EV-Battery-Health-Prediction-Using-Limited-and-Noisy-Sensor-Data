# EV Battery Health Prediction Using Limited and Noisy Sensor Data

The project estimates battery **state of health (SOH)** and **remaining useful life (RUL)** while sensor readings are noisy, sparse, missing or limited to part of a charge. The final evaluation uses 633 paired cycles across four NASA laboratory cells: B0005, B0006, B0007 and B0018. Models are trained on three cells and tested on the fourth, rotating the held-out cell.

[Open the interactive prototype](https://ev-battery-health-lab.shaik-fayaz2467.chatgpt.site/)

## Run the included demonstration

**From the submission ZIP:** Extract the entire source folder and install Python 3.12. On Windows, double-click `Start_Browser_Demo.bat`. On any platform, run `python start_browser.py` from the source folder. Open http://127.0.0.1:8765/ if the browser does not open automatically. Keep the terminal running; press Ctrl+C to stop. This mode uses the included browser assets and does not install packages or need internet access. If the port is busy, run `python start_browser.py --port 8766`.

**From GitHub:** Use the live link above for immediate access, or follow the setup and rebuild steps below to generate the browser's data and models locally. The repository contains the implementation and dataset links. The submission ZIP additionally contains raw MAT files, generated browser assets, and measured outputs for offline use.

## Run the Python application

Use Python 3.12. From the project root, create a virtual environment and activate it:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate`. Then install and run:

```sh
python -m pip install -r python_app/requirements.txt
python -m streamlit run python_app/app.py
```

The app opens at http://localhost:8501/. Windows users can also run `Start_Python_App.bat` to create the environment and launch it. Internet access is needed for the first dependency installation. When using the repository rather than the ZIP, install the NASA records first.

## Dataset and provenance

- [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [Battery Data Set ZIP](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip), B. Saha and K. Goebel, NASA Ames Prognostics Data Repository.
- `data_source.json` records the source and SHA-256 checksums for the exact four MAT files.
- The offline submission already contains the data in `python_app/data/raw/`. For a GitHub checkout, download the official ZIP above and run `python research_tools/prepare_nasa_data.py "path/to/Battery Data Set.zip"`. The script checks the four files against their recorded hashes before copying them.

## Reproduce the final experiment

Run from the project root after installing the dependencies and data. The full repeated experiment takes longer than starting the demo.

```sh
python python_app/precompute_benchmarks.py --source nasa --data-dir python_app/data/raw --output-dir results --repeats 3
python research_tools/export_browser_models.py --data-dir python_app/data/raw --results-dir results
python start_browser.py
```

To see the recomputed benchmark tables in the Python dashboard, also copy `results/benchmark_summary.json` to `python_app/data/benchmark_summary.json`. To regenerate supplementary analysis and plots, install Matplotlib and run `python research_tools/analyse_results.py`. The supplied numerical and experiment outputs are in `results/` in the submission ZIP.

## Model choices and demo steps

1. Open **Live prediction** and choose battery B0005, cycle 84, and Random Forest.
2. Compare predicted SOH and RUL with the observed values. Change to Ridge Regression or Gradient Boosting to compare predictions; the Cycle Count Baseline only uses cycle index.
3. Apply a noise or partial-charge preset and watch the input trace and predictions change. **Clean + degraded** changes the training variant.
4. Open **Cross-battery results** for held-out scores or **Robustness** for the measured sweep results.

Ridge has the lowest clean SOH MAE (2.191 percentage points). Random Forest has the lowest all-cycle RUL MAE (9.392 cycles). Gradient Boosting has the lowest pre-retirement RUL MAE (17.330 cycles). The experiments are on four lab cells; they do not establish pack or vehicle accuracy.

## Source layout and checks

| Path | Purpose |
| --- | --- |
| `python_app/` | Streamlit interface, data import, features, perturbations, training and benchmarks |
| `browser_app/` | HTML, CSS, JavaScript prediction engine and generated browser assets |
| `research_tools/` | Dataset preparation, browser export and supplementary analysis |
| `results/` | Benchmark results, audit, predictions and verification records (full set in submission ZIP) |
| `browser_tests/` | Python/browser parity fixtures (submission ZIP) |

Run `python -m pytest python_app/test_pipeline.py -q`. From the submission source folder, run `node browser_tests/browser_numerics.mjs` with Node.js for browser numerical checks. The report explains the methodology and limitations.
