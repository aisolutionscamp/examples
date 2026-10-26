# Anonymous people and vehicle counting (YOLO + ONNX Runtime)

Counts people, bicycles, motorcycles, cars, buses and trucks in a camera stream and outputs only aggregate numbers
per interval: line crossings per direction and zone occupancy (maximum and mean).

## Privacy by design

- **No face detection, no face recognition, no identification of individuals.** The detector only answers
  "there is a person/car in this box".
- **Short-lived, anonymous track ids.** Boxes are linked frame to frame by overlap only (no appearance
  features). An id exists in memory while the object is in view and is never output. The same person gets a new
  id after leaving the picture or on another camera.
- **No storage of frames.** Frames are processed in memory and dropped; the tool writes no images or video.
- **Aggregation only.** The only output is one JSON record per interval, for example
  `{"camera_id": "gate-1", "start": ..., "end": ..., "crossings": {"person_in": 41, "person_out": 37},
  "occupancy_max": {"person": 9}, "occupancy_mean": {"person": 3.4}, "frames": 900}`.
- **Pixelated preview.** The optional preview (for setting up the line and zone) pixelates every person box.

Technical measures do not replace the legal ones: define purpose and retention, check the legal basis and run a
data protection impact assessment (GDPR Art. 35) before installing cameras in public or work areas, and inform
people with signage. Real-time remote biometric identification in publicly accessible spaces is prohibited for
almost all purposes under the EU AI Act; this tool does not do it and must not be extended to do it.

## Architecture

```
camera (RTSP / file) ──► every n-th frame ──► letterbox 640x640 ──► YOLO11 ONNX (ONNX Runtime, CPU)
   ──► keep person/vehicle classes + NMS ──► IoU tracker (temporary ids)
   ──► line counter (in/out per class) + zone occupancy ──► interval aggregate (JSON lines)
                                         └─► optional preview with pixelated people (not stored)
```

- `detector.py`: letterbox, YOLOv8/YOLO11 output parsing (`(1, 84, N)`), class filter, NMS, ONNX Runtime session.
- `tracking.py`: `IouTracker`, `LineCounter`, `ZoneOccupancy`.
- `pipeline.py`: the frame loop, aggregation and pixelation.

## Setup and run

The model is exported once with Ultralytics (needs PyTorch, only for the export, can run on another machine):

```bash
pip install ultralytics
yolo export model=yolo11n.pt format=onnx imgsz=640      # downloads yolo11n.pt, writes yolo11n.onnx
mkdir -p models && mv yolo11n.onnx models/
```

Then, with only the runtime dependencies:

```bash
pip install -r requirements.txt
cd src
python -m counting --model ../models/yolo11n.onnx --source entrance.mp4 --camera-id entrance \
    --interval 60 --line 0,360,1280,360 --zone 100,100,1180,100,1180,700,100,700
```

`--line x1,y1,x2,y2` and `--zone x1,y1,...` are pixel coordinates in the camera image. For a line drawn top to
bottom, movement from left to right counts as `in`; swap the two points to swap directions.

Tests (no model file, no download: the ONNX session and the detector are replaced by fakes): `pytest`.

## Configuration

| Option | Default | Meaning |
|---|---|---|
| `--model` / `YOLO_ONNX_MODEL` | `models/yolo11n.onnx` | exported model; larger variants (`yolo11s`, `yolo11m`) are more accurate and slower |
| `--interval` | 60 | seconds per aggregate record |
| `--every-nth` | 2 | process every n-th frame |
| `--conf` | 0.35 | minimum detection confidence |
| `COUNTED_CLASSES` | person, bicycle, car, motorcycle, bus, truck | COCO class ids to keep |

## Limitations

- The parser expects the YOLOv8/YOLO11 ONNX layout `(1, 4 + classes, N)` without built-in NMS. Models exported
  in another layout need a different `postprocess`.
- The IoU tracker loses objects in dense crowds and long occlusions; counts can be too high or too low. For busy
  scenes use a stronger motion-only tracker (for example ByteTrack) without appearance embeddings.
- Accuracy depends on camera angle, resolution, light and weather. Validate every camera against manual counts.
- Ultralytics YOLO models and code are licensed under AGPL-3.0 (or an Ultralytics enterprise license); check
  the license terms for your use.

## For production

- Run one process per camera (or batch frames on a GPU), and send the interval records to a time-series database.
- Suppress very small counts in published reports if they could single out individuals in quiet areas.
- Log only aggregates and health data (fps, dropped frames). Do not add frame recording to this pipeline.
- Review the data protection documentation whenever cameras, purpose or retention change.
