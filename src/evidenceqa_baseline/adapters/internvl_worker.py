"""JSONL worker for running InternVL with its official Transformers version."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from evidenceqa_baseline.adapters.internvl import InternVLAdapter, InternVLConfig
from evidenceqa_baseline.dataset import (
    BoxTrackItem,
    DatasetSample,
    FrameRef,
    PointTrackItem,
    SpatialSample,
)


def main() -> None:
    adapter: InternVLAdapter | None = None
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            op = request.get("op")
            if op == "init":
                adapter = InternVLAdapter(_config_from_payload(request["config"]))
                _send({"ok": True})
            elif op == "predict":
                if adapter is None:
                    raise RuntimeError("InternVL worker is not initialized")
                sample = DatasetSample(**request["sample"])
                result = adapter.predict(sample, Path(request["media_path"]))
                _send({"ok": True, "result": result})
            elif op == "predict_spatial":
                if adapter is None:
                    raise RuntimeError("InternVL worker is not initialized")
                sample = _spatial_sample_from_payload(request["sample"])
                frame_paths = [
                    (int(frame_index), Path(path))
                    for frame_index, path in request["frame_paths"]
                ]
                result = adapter.predict_spatial(sample, frame_paths)
                _send({"ok": True, "result": result})
            elif op == "close":
                _send({"ok": True})
                return
            else:
                raise RuntimeError(f"unknown InternVL worker op: {op}")
        except Exception:
            _send({"ok": False, "error": traceback.format_exc()})


def _send(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _config_from_payload(payload: dict[str, Any]) -> InternVLConfig:
    values = dict(payload)
    if values.get("model_cache_dir") is not None:
        values["model_cache_dir"] = Path(values["model_cache_dir"])
    return InternVLConfig(**values)


def _spatial_sample_from_payload(payload: dict[str, Any]) -> SpatialSample:
    values = dict(payload)
    values["frames"] = [FrameRef(**item) for item in values.get("frames", [])]
    values["gt_box_track"] = [
        BoxTrackItem(**item) for item in values.get("gt_box_track", [])
    ]
    values["gt_point_track"] = [
        PointTrackItem(**item) for item in values.get("gt_point_track", [])
    ]
    return SpatialSample(**values)


if __name__ == "__main__":
    main()
