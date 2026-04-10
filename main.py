"""
Real-time video vision pipeline:
  - YOLO for object segmentation (fast, real-time)
  - Gemma 4 (via LM Studio) for scene captioning

Usage:
  python main.py                    # webcam
  python main.py --source video.mp4 # video file
  python main.py --help             # all options
"""

import argparse
import base64
import sys
import threading
import time

import cv2
import numpy as np
from openai import OpenAI
from ultralytics import YOLO


class SceneCaptioner:
    """Calls Gemma 4 via LM Studio's OpenAI-compatible API to describe frames."""

    def __init__(self, base_url: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model
        self.current_caption = "Waiting for first caption..."
        self._lock = threading.Lock()
        self._busy = False

    def _encode_frame(self, frame: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode()

    def request_caption(self, frame: np.ndarray) -> None:
        """Fire-and-forget: starts a background thread to caption the frame."""
        with self._lock:
            if self._busy:
                return
            self._busy = True

        t = threading.Thread(target=self._worker, args=(frame.copy(),), daemon=True)
        t.start()

    def _worker(self, frame: np.ndarray) -> None:
        try:
            b64 = self._encode_frame(frame)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Describe this scene in one concise sentence. "
                                    "Focus on the main subjects, their actions, "
                                    "and the environment. "
                                    "Do NOT use any thinking or reasoning tags. "
                                    "Reply with ONLY the description."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.3,
            )
            raw = resp.model_dump()
            text = resp.choices[0].message.content or ""
            if not text.strip():
                reasoning = raw["choices"][0]["message"].get("reasoning_content", "")
                if reasoning:
                    lines = [
                        ln.strip() for ln in reasoning.strip().splitlines()
                        if ln.strip() and not ln.strip().startswith("*")
                    ]
                    text = lines[-1] if lines else "[thinking...]"
            text = text.strip()
            with self._lock:
                self.current_caption = text
        except Exception as e:
            with self._lock:
                self.current_caption = f"[caption error: {e}]"
        finally:
            with self._lock:
                self._busy = False


def draw_caption_bar(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a semi-transparent caption bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    max_width = w - 40

    words = text.split()
    lines, line = [], ""
    for word in words:
        test = f"{line} {word}".strip()
        if cv2.getTextSize(test, font, font_scale, thickness)[0][0] > max_width:
            lines.append(line)
            line = word
        else:
            line = test
    if line:
        lines.append(line)

    lines = lines[:2]
    y_start = h - bar_h + 25
    for i, ln in enumerate(lines):
        cv2.putText(
            frame, ln, (20, y_start + i * 25),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return frame


def run(args: argparse.Namespace) -> None:
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open video source '{args.source}'")
        sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video source opened: {orig_w}x{orig_h}")

    seg_model = YOLO(args.yolo_model)
    print(f"YOLO model loaded: {args.yolo_model}")

    captioner = SceneCaptioner(
        base_url=args.lm_studio_url,
        model=args.gemma_model,
    )
    print(f"Gemma 4 captioner ready (model: {args.gemma_model})")
    print("Press 'q' to quit.\n")

    last_caption_time = 0.0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        results = seg_model(frame, verbose=False, conf=args.conf)
        annotated = results[0].plot()

        if now - last_caption_time >= args.caption_interval:
            captioner.request_caption(frame)
            last_caption_time = now

        annotated = draw_caption_bar(annotated, captioner.current_caption)
        annotated = draw_fps(annotated, fps)

        cv2.imshow("Gemma 4 + YOLO Vision", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time video vision pipeline")
    p.add_argument(
        "--source", default="0",
        help="Video source: 0 for webcam, or path to a video file (default: 0)",
    )
    p.add_argument(
        "--yolo-model", default="yolo11n-seg.pt",
        help="YOLO segmentation model (default: yolo11n-seg.pt)",
    )
    p.add_argument(
        "--gemma-model", default="google/gemma-4-26b-a4b",
        help="Gemma model name as loaded in LM Studio",
    )
    p.add_argument(
        "--lm-studio-url", default="http://localhost:1234/v1",
        help="LM Studio API endpoint (default: http://localhost:1234/v1)",
    )
    p.add_argument(
        "--caption-interval", type=float, default=3.0,
        help="Seconds between caption updates (default: 3.0)",
    )
    p.add_argument(
        "--conf", type=float, default=0.35,
        help="YOLO confidence threshold (default: 0.35)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
