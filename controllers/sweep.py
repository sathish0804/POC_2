import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, List, Tuple


def run_single(config: Dict[str, Any], main_py: str, base_env: Dict[str, str]) -> Tuple[int, str]:
    """Run one configuration by invoking main.py via subprocess.

    Returns (returncode, out_events_path)
    """
    # Build output directory and file
    tag_parts = [f"{k}-{v}" for k, v in sorted(config.items())]
    tag = "__".join(tag_parts)
    out_dir = os.path.join(config.get("out_root", "./output"), "sweeps", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"events__{tag}.json")

    # Compose args
    cmd = [sys.executable, main_py,
           "--video", config["video"],
           "--model", config.get("model", "yolov8l.pt"),
           "--fps", str(config.get("fps", 1)),
           "--out", out_file,
           "--max_frames", str(config.get("max_frames", 0)),
           "--sleep_eye_thresh", str(config.get("sleep_eye_thresh", 0.18)),
           "--sleep_headdown_deg", str(config.get("sleep_headdown_deg", 100.0)),
           "--sleep_micro_max_min", str(config.get("sleep_micro_max_min", 15.0)),
           "--sleep_min_duration", str(config.get("sleep_min_duration", 10.0)),
           ]
    if config.get("disable_ocr", True):
        cmd.append("--disable_ocr")
    if config.get("verbose", False):
        cmd.append("--verbose")

    # Optional YOLO conf/iou via env
    env = dict(base_env)
    if "yolo_conf" in config:
        env["YOLO_CONF"] = str(config["yolo_conf"])
    if "yolo_iou" in config:
        env["YOLO_IOU"] = str(config["yolo_iou"])

    rc = subprocess.call(cmd, env=env)
    return rc, out_file


def summarize_events(path: str) -> Dict[str, Any]:
    """Compute simple metrics from events file: counts per objectType and per activityType."""
    if not os.path.exists(path):
        return {"error": "missing_output", "path": path}
    try:
        with open(path, "r", encoding="utf-8") as f:
            events = json.load(f)
    except Exception as e:
        return {"error": f"json_error: {e}", "path": path}

    counts_by_object: Dict[str, int] = {}
    counts_by_type: Dict[int, int] = {}
    for ev in events:
        obj = (ev.get("objectType") or "").lower()
        if obj:
            counts_by_object[obj] = counts_by_object.get(obj, 0) + 1
        at = int(ev.get("activityType", 0))
        counts_by_type[at] = counts_by_type.get(at, 0) + 1

    return {
        "num_events": len(events),
        "counts_by_object": counts_by_object,
        "counts_by_activityType": counts_by_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sweep runner for ActivityPipeline")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8l.pt")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--out_root", type=str, default="/Users/satishvanga/Documents/Vanga/POC_2/output")
    parser.add_argument("--main_py", type=str, default="/Users/satishvanga/Documents/Vanga/POC_2/main.py")
    parser.add_argument("--disable_ocr", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    # Grids
    parser.add_argument("--sleep_eye_thresh_grid", type=str, default="0.16,0.18,0.20")
    parser.add_argument("--sleep_headdown_deg_grid", type=str, default="90,100,110")
    parser.add_argument("--sleep_micro_max_min_grid", type=str, default="10,15,20")
    parser.add_argument("--sleep_min_duration_grid", type=str, default="8,10,12")
    # Antenna parameters removed
    parser.add_argument("--yolo_conf_grid", type=str, default="0.20,0.25,0.30")
    parser.add_argument("--yolo_iou_grid", type=str, default="0.40,0.45,0.50")
    parser.add_argument("--limit", type=int, default=0, help="Limit total combinations (0 = all)")

    args = parser.parse_args()

    # Prepare grids
    def parse_grid(s: str, cast):
        return [cast(x.strip()) for x in s.split(",") if x.strip()]

    grid = {
        "sleep_eye_thresh": parse_grid(args.sleep_eye_thresh_grid, float),
        "sleep_headdown_deg": parse_grid(args.sleep_headdown_deg_grid, float),
        "sleep_micro_max_min": parse_grid(args.sleep_micro_max_min_grid, float),
        "sleep_min_duration": parse_grid(args.sleep_min_duration_grid, float),
        # Antenna parameters removed
        "yolo_conf": parse_grid(args.yolo_conf_grid, float),
        "yolo_iou": parse_grid(args.yolo_iou_grid, float),
    }

    keys = list(grid.keys())
    values_lists = [grid[k] for k in keys]
    combos = list(itertools.product(*values_lists))
    if args.limit > 0:
        combos = combos[: args.limit]

    base_env = dict(os.environ)

    results: List[Dict[str, Any]] = []
    for combo in combos:
        cfg = {k: v for k, v in zip(keys, combo)}
        cfg.update({
            "video": args.video,
            "model": args.model,
            "fps": args.fps,
            "max_frames": args.max_frames,
            "out_root": args.out_root,
            "disable_ocr": args.disable_ocr,
            "verbose": args.verbose,
        })
        rc, out_path = run_single(cfg, args.main_py, base_env)
        summary = summarize_events(out_path)
        summary.update({
            "returncode": rc,
            "out_path": out_path,
            "config": cfg,
        })
        results.append(summary)
        # Stream simple progress
        print(json.dumps({"out": out_path, "rc": rc, "metrics": summary}, ensure_ascii=False))

    # Write aggregated summary
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_path = os.path.join(args.out_root, "sweeps", f"summary_{stamp}.json")
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote sweep summary to {agg_path}")


if __name__ == "__main__":
    main()


