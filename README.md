# DeepfakeCheck

DeepfakeCheck is a small demonstration and results viewer for a university research project. It compares a ResNet-50 and a DeiT-S classifier fine-tuned on the GenImage BigGAN subset. The models are intended to be evaluated on diffusion-generated images.

This app displays model outputs and experiment metrics; it does not run the evaluation itself. A model prediction is not forensic proof.

## Requirements

- Python 3.11
- Windows, macOS, or Linux
- An internet connection for the first package installation

The pinned PyTorch packages use CPU wheels by default, so this setup runs on CPU. For GPU inference, install matching CUDA builds of both `torch` and `torchvision` for your machine; the app will select CUDA automatically when it is available.

## Set up and start

Open a terminal in this `deepfakecheck` folder and create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, use these activation commands instead:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the server from this folder:

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The API health check is at <http://127.0.0.1:8000/api/health>.

## Add trained checkpoints

Place each student-produced PyTorch `state_dict` in `checkpoints/`, using this exact filename pattern:

```text
{model}_biggan_n{size}_seed{seed}.pt
```

For example, a run for the ResNet-50 model with training size 20,000 and seed 42 is named `resnet50_biggan_n20000_seed42.pt`. Model names are `resnet50` and `deit_small`; supported training sizes are 20000, 10000, 5000, and 2000. Restart the server after adding or replacing checkpoints. The startup log reports the number of available runs for each model.

The checkpoints must contain a `state_dict` for the matching `timm` model, created as `resnet50` or `deit_small_patch16_224` with `num_classes=1`. No pretrained or sample checkpoints are included.

If the selected checkpoint is missing, the prediction endpoint returns HTTP 503 with a “Model not trained yet” message. It never substitutes a made-up prediction.

## Add experiment metrics

Save the students' metrics JSON at `results/metrics.json`. The Results page reads the file when opened. Its expected structure is:

```json
{
  "runs": [
    {
      "model": "resnet50",
      "train_size": 20000,
      "seed": 42,
      "test_set": "biggan",
      "auc": 0.0,
      "accuracy": 0.0,
      "f1": 0.0
    }
  ]
}
```

The values above describe the file format only; replace them with measured experiment results. No metrics file is included. If it is missing, the Results page says “No experiment results yet”.

## Privacy and image validation

Only JPEG, PNG, and WebP uploads up to 10 MB are accepted. The request body and multipart file remain in memory, are decoded for the prediction, and are discarded when the request finishes. Uploaded images and their filenames are not written to disk or logged.

## Run the tests

With the virtual environment activated, run:

```bash
pytest
```

The tests check the health endpoint, missing-checkpoint behavior, inference from a temporary randomly initialized test checkpoint, invalid and oversized uploads, and the missing-results response.

