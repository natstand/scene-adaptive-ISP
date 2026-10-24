# scene-adaptive-ISP

## Environment

Python 3.10+, PyTorch 2.x, NumPy, Pillow, OpenCV, and `romatch` are required.
Install the dependencies and use a CUDA GPU when available:

```bash
pip install numpy Pillow opencv-python romatch
```

The alignment weights download automatically on the first run when they are not
already cached, so internet access is needed for that first run.

## Run

From this directory:

```bash
# Default video3 example: 300 adaptation steps.
python run_adaptation.py --video3

# Transfer the bundled video3 ISP checkpoint to video30: 100 steps, lr=5e-3.
python run_adaptation.py --video30-from-video3

# Train video30 from the default initialization: 300 steps.
python run_adaptation.py --video30
```

Use `--device cuda:1` to select another GPU or `--device cpu` for a CPU run.
The video30 example contains pre-cropped RAW and built-in frames, so no crop
argument is needed.

## Results

Results are written under `example/output/`:

- `video3/` for the default run;
- `video30_from_video3_100steps/` for the transfer run;
- `video30_from_scratch_300steps/` for the video30 from-scratch run.

Each result directory contains `built-in_vs_ours.png`, `phone.png`,
`isp_checkpoint.pth`, `timing.json`, and `render_selection.json`.
`timing.json` reports `adaptation_seconds`; render timing excludes image I/O.
