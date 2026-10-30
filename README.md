# DeepfakeCheck

DeepfakeCheck is a research demo for comparing two image classifiers: ResNet-50 and DeiT-S. It provides image inference when a matching trained checkpoint is available and displays the experiment metrics recorded in `results/metrics.json`.

Model outputs are predictions from the selected checkpoint, not forensic proof. The Results page reports measured experiment values; the app does not invent or recalculate those results.

## Current experiment files

The project includes the supplied ResNet-50 and DeiT-S checkpoints for training size 2,000 and seed 42, plus eight measured runs in `results/metrics.json` (two models evaluated on BigGAN, SD 1.4, ADM, and Midjourney test sets). These are committed as ordinary Git files so a deployment can download them with the repository.

The GitHub repository is public, so the committed metrics and model checkpoints are publicly downloadable. Do not commit private data or images.

## Run locally

Requirements: Python 3.11 and an internet connection for package installation. The pinned PyTorch dependencies use CPU builds.

On Windows, open PowerShell in this folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The results API is at <http://127.0.0.1:8000/api/results> and the health check is at <http://127.0.0.1:8000/api/health>.

## Run tests

```bash
pytest
```

The tests cover the health endpoint, model inventory, missing-checkpoint behavior, invalid and oversized uploads, page loading, and the missing-results response.

## Deploy for a teammate

`render.yaml` configures a Render Docker web service named `deepfakecheck`, with `/api/health` as its health check. Connect the public GitHub repository to Render and deploy from `main`. The configured instance is 1 CPU / 2 GB RAM because the service loads PyTorch and the selected checkpoint during inference. Render's paid plan charges apply; check the current plan price in Render before creating the service.

After deployment, Render provides a public `*.onrender.com` URL that can be shared with a teammate. The free service tier may not have enough memory for model inference and can sleep while idle, so it is not the configured option.

## Upload privacy and behavior

The app accepts JPEG, PNG, and WebP images up to 10 MB. Request bytes and decoded images are kept in memory for the request and discarded afterward. The app does not write uploaded images or filenames to disk or log them.

The prediction endpoint uses only a checkpoint matching the selected model, training size, and seed. If a matching checkpoint is unavailable, it responds with “Model not trained yet”; it never substitutes a fabricated prediction.
