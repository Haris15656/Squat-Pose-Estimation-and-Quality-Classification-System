import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def parse_time_range(value: str) -> tuple[float, float]:
    try:
        start_text, end_text = value.split("-")
        start = float(start_text)
        end = float(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid range '{value}'. Expected format start-end in seconds, e.g. 0-5."
        ) from exc

    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError(
            f"Invalid range '{value}'. Start must be >= 0 and end must be greater than start."
        )
    return start, end


def format_seconds(value: float) -> str:
    total_ms = int(round(value * 1000))
    seconds, milliseconds = divmod(total_ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}-{minutes:02d}-{seconds:02d}-{milliseconds:03d}"
    return f"{minutes:02d}-{seconds:02d}-{milliseconds:03d}"


def write_clip(input_path: Path, output_path: Path, start_sec: float, end_sec: float) -> bool:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Could not open video: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = max(start_frame + 1, int(round(end_sec * fps)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    written = 0

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1
        current_frame += 1

    writer.release()
    cap.release()

    if written == 0:
        if output_path.exists():
            output_path.unlink()
        print(f"Skipped empty clip: {output_path}")
        return False

    print(f"Saved {output_path} ({written} frames)")
    return True


def build_fixed_segments(duration_sec: float, clip_length_sec: float, overlap_sec: float) -> list[tuple[float, float]]:
    segments = []
    step = max(clip_length_sec - overlap_sec, 0.1)
    start = 0.0
    while start < duration_sec:
        end = min(start + clip_length_sec, duration_sec)
        if end - start > 0.01:
            segments.append((start, end))
        if end >= duration_sec:
            break
        start += step
    return segments


def main():
    parser = argparse.ArgumentParser(description="Split a long squat video into shorter clips for labeling/training.")
    parser.add_argument("input_video", type=str, help="Path to the input video")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save clips (default: beside input video)")
    parser.add_argument("--clip-length", type=float, default=6.0, help="Length of each clip in seconds when using fixed splitting")
    parser.add_argument("--overlap", type=float, default=1.0, help="Overlap between consecutive fixed clips in seconds")
    parser.add_argument("--segments", type=parse_time_range, nargs="*", help="Explicit time ranges like 0-5 5-10 10-15")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for generated clip filenames")
    args = parser.parse_args()

    input_path = Path(args.input_video)
    if not input_path.exists():
        raise SystemExit(f"Input video does not exist: {input_path}")
    if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise SystemExit(f"Unsupported video extension: {input_path.suffix}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / f"{input_path.stem}_clips"
    prefix = args.prefix or input_path.stem

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    cap.release()

    if args.segments:
        segments = list(args.segments)
    else:
        segments = build_fixed_segments(duration_sec, args.clip_length, args.overlap)

    if not segments:
        raise SystemExit("No segments to write. Try a shorter clip length or provide explicit --segments.")

    print(f"Input: {input_path}")
    print(f"Duration: {duration_sec:.2f}s | FPS: {fps:.2f} | Frames: {frame_count}")
    print(f"Writing {len(segments)} clips to: {output_dir}")

    written = 0
    for index, (start_sec, end_sec) in enumerate(segments, start=1):
        clip_name = f"{prefix}_{index:03d}_{format_seconds(start_sec)}_{format_seconds(end_sec)}.mp4"
        output_path = output_dir / clip_name
        if write_clip(input_path, output_path, start_sec, end_sec):
            written += 1

    print(f"Finished. Wrote {written}/{len(segments)} clips.")


if __name__ == "__main__":
    main()