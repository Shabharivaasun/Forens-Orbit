# ForensOrbit — native Raspberry Pi evidence pipeline

This repository retains its original ROS 2 package files for demonstration
purposes. Deployment on the Raspberry Pi 4 does **not** require ROS 2, Docker,
PyTorch, CUDA, or NVIDIA packages.

`forensorbit_detection/native_pipeline.py` is the deployment entry point:

```text
USB webcam -> untouched MP4 -> sparse frame samples -> NCNN CPU inference
                                  -> annotated positives + report.json
```

The laptop performs dense extraction, reconstruction, Gaussian Splatting, and
VR viewing. The Pi retains/transfers only the raw video and its report folder.

## Project location

All commands below use this restored location:

```bash
cd /home/shabhari/ros2_ws
```

## 1. USB webcam prerequisite

The deployment camera is the USB webcam, detected on this Pi as `/dev/video1`
(`ZEB LIVE PRO`). Confirm its capture interface before recording:

```bash
v4l2-ctl --device=/dev/video1 --all
```

It must report `Video Capture` and `Streaming`. The CSI Pi Camera does not
need to be removed; it is not used by the USB capture path.

Record a smoke test only after the detection test succeeds:

```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 \
  -i /dev/video1 -t 5 -c:v copy videos/webcam_smoketest.mp4
```

## 2. Network prerequisite

Package installation requires DNS and Internet access. Verify it before using
pip or apt:

```bash
getent hosts pypi.org
getent hosts deb.debian.org
```

If either command prints nothing, connect the Pi to the network and repair its
DNS configuration first. Repeated pip retries cannot solve a DNS failure.

## 3. Install only CPU dependencies

Use one virtual environment for the native pipeline:

```bash
cd /home/shabhari/ros2_ws
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r src/forensorbit_detection/requirements-native.txt
```

On Raspberry Pi OS/Debian, `python3-opencv` is a valid alternative when an
OpenCV wheel is not published for the installed Python release. NCNN must be a
Python binding compatible with the Pi's `aarch64` architecture and current
Python version. Do not install `torch`, `ultralytics`, CUDA, Docker, ROS 2, or
any NVIDIA package on the Pi for inference.

## 4. Local mechanical test (no model)

This validates capture, sparse sampling, and the JSON report without trying to
load a model:

```bash
.venv/bin/python src/forensorbit_detection/forensorbit_detection/native_pipeline.py \
  --record-seconds 10 --camera-source usb --camera-device /dev/video1 \
  --sample-period 1.5 --no-detect
```

It creates a raw MP4 under `src/forensorbit_detection/output/videos/` and a
matching report folder at `src/forensorbit_detection/output/reports/<video-name>/`:

```text
report.json          machine-readable detections, timestamps, and XYXY boxes
forensic_report.pdf  investigator-readable evidence dossier
annotated/           annotated positive frames (only when detections exist)
```

The raw MP4 is never altered.

## 5. Detection test

Place the exported NCNN `.param` and `.bin` files plus the class-label list in
`weights/`. A forensic model must be trained for the intended evidence classes
(for example weapon, shell casing, drug paraphernalia, and blood-stain-related
classes). A stock COCO model is only suitable for validating mechanics; it is
not forensic evidence detection.

Then run the same command without `--no-detect` and provide the model path.
`report.json` records each detection's timestamp, label, confidence, and XYXY
bounding box. Annotated images are generated only for positive sampled frames.

## 6. Transfer (after local tests pass)

Use resumable `rsync` over SSH to transfer only raw videos and report folders:

```bash
rsync -avP src/forensorbit_detection/output/videos/ USER@LAPTOP:/path/to/forensorbit/videos/
rsync -avP src/forensorbit_detection/output/reports/ USER@LAPTOP:/path/to/forensorbit/reports/
```
