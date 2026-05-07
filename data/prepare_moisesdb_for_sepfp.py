from __future__ import annotations

import argparse
import json
import os
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # PCM s16le
DEFAULT_THRESHOLD_DB = -20.0
DEFAULT_ACTIVE_RATIO = 0.02
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 1)))
DEFAULT_OUTPUT_ROOT = Path("/home/maclab/user_woojinkang/SepFP/data/multistem")
DEFAULT_DATASET_NAME = "moisesdb"


SIX_STEM_MAPPING = {
    "vocals": "vocals",
    "drums": "drums",
    "percussion": "drums",
    "bass": "bass",
    "guitar": "guitar",
    "piano": "piano",
    "other": "others",
    "other_keys": "others",
    "bowed_strings": "others",
    "wind": "others",
    "other_plucked": "others",
}

FOUR_STEM_MAPPING = {
    "vocals": "vocals",
    "drums": "drums",
    "percussion": "drums",
    "bass": "bass",
    "guitar": "others",
    "piano": "others",
    "other": "others",
    "other_keys": "others",
    "bowed_strings": "others",
    "wind": "others",
    "other_plucked": "others",
}

STEM_MODE_TO_MAPPING = {
    "six": SIX_STEM_MAPPING,
    "four": FOUR_STEM_MAPPING,
}

STEM_MODE_TO_ALLOWED_STEMS = {
    "six": frozenset(("vocals", "drums", "bass", "guitar", "piano", "others")),
    "four": frozenset(("vocals", "drums", "bass", "others")),
}

STEM_MODE_TO_OUTPUT_DIR = {
    "six": "6stem",
    "four": "4stem",
}


@dataclass(frozen=True)
class TrackSpec:
    raw_path: Path
    processed_relpath: Path
    canonical_stem: str


@dataclass(frozen=True)
class SongSpec:
    song_id: str
    tracks: tuple[TrackSpec, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw MoisesDB for SepFP training.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/maclab/user_woojinkang/Dataset/moisesdb/moisesdb_v0.1"),
        help="Root of raw MoisesDB v0.1.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Output root for converted 16kHz mono audio cropped to the shortest track per song. Overrides --output-root layout.",
    )
    parser.add_argument(
        "--meta-root",
        type=Path,
        default=None,
        help="Output root for .txt and .npy metadata built from cropped audio. Overrides --output-root layout.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for the multistem/<stem-mode>stem/<dataset-name>/ output layout.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Dataset name used under the multistem output layout.",
    )
    parser.add_argument(
        "--stem-mode",
        choices=sorted(STEM_MODE_TO_MAPPING),
        default="six",
        help="Canonical stem layout to prepare.",
    )
    parser.add_argument(
        "--limit-songs",
        type=int,
        default=None,
        help="Only process the first N songs after sorting. Useful for sample-first validation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel worker count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing converted audio and metadata.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run structural validation without creating new outputs.",
    )
    return parser.parse_args()


def resolve_output_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    stem_dir = STEM_MODE_TO_OUTPUT_DIR[args.stem_mode]
    dataset_root = args.output_root / stem_dir / args.dataset_name
    audio_root = args.audio_root or (dataset_root / "audio_16k_mono_cropped")
    meta_root = args.meta_root or (dataset_root / "meta_cropped")
    return audio_root, meta_root


def _validate_output_roots_for_mode(audio_root: Path, meta_root: Path, stem_mode: str) -> None:
    roots = (audio_root.resolve(), meta_root.resolve())
    if stem_mode == "four" and any("6stem" in root.parts for root in roots):
        raise ValueError(f"four-stem preprocessing must not write under a 6stem path: {roots}")
    if stem_mode == "six" and any("4stem" in root.parts for root in roots):
        raise ValueError(f"six-stem preprocessing must not write under a 4stem path: {roots}")


def _load_song_spec(song_dir: Path, stem_mapping: dict[str, str]) -> SongSpec | None:
    data_path = song_dir / "data.json"
    if not data_path.exists():
        return None

    payload = json.loads(data_path.read_text())
    song_id = song_dir.name
    tracks: list[TrackSpec] = []

    for stem in payload.get("stems", []):
        raw_stem = stem.get("stemName")
        canonical = stem_mapping.get(raw_stem)
        if canonical is None:
            continue
        for track in stem.get("tracks", []):
            track_id = track.get("id")
            extension = track.get("extension", "wav")
            if not track_id:
                continue
            raw_path = song_dir / raw_stem / f"{track_id}.{extension}"
            processed_relpath = Path(song_id) / canonical / f"{track_id}.wav"
            tracks.append(
                TrackSpec(
                    raw_path=raw_path,
                    processed_relpath=processed_relpath,
                    canonical_stem=canonical,
                )
            )

    if not tracks:
        return None

    tracks.sort(key=lambda item: (item.canonical_stem, item.processed_relpath.name))
    relpaths = [track.processed_relpath for track in tracks]
    if len(relpaths) != len(set(relpaths)):
        duplicates = sorted(str(path) for path in set(relpaths) if relpaths.count(path) > 1)
        raise ValueError(f"{song_id}: duplicate processed output paths after stem mapping: {duplicates[:3]}")
    return SongSpec(song_id=song_id, tracks=tuple(tracks))


def discover_songs(raw_root: Path, limit_songs: int | None = None) -> list[SongSpec]:
    return discover_songs_with_mapping(raw_root, STEM_MODE_TO_MAPPING["six"], limit_songs=limit_songs)


def discover_songs_with_mapping(
    raw_root: Path,
    stem_mapping: dict[str, str],
    limit_songs: int | None = None,
) -> list[SongSpec]:
    song_specs = []
    for song_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        spec = _load_song_spec(song_dir, stem_mapping=stem_mapping)
        if spec is not None:
            song_specs.append(spec)
    if limit_songs is not None:
        song_specs = song_specs[:limit_songs]
    return song_specs


def _pcm_bytes_to_float32(raw_bytes: bytes, sample_width: int, channels: int) -> np.ndarray:
    if sample_width == 1:
        samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        signed = (signed ^ sign_mask) - sign_mask
        samples = signed.astype(np.float32) / float(1 << 23)
    elif sample_width == 4:
        samples = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if channels == 1:
        return samples
    if channels == 2:
        return samples.reshape(-1, 2).mean(axis=1)
    raise ValueError(f"Unsupported channel count: {channels}")


def _resample_linear(signal: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return signal.astype(np.float32, copy=False)
    if signal.size == 0:
        return np.zeros((0,), dtype=np.float32)

    target_len = max(1, int(round(signal.shape[0] * dst_rate / src_rate)))
    src_positions = np.arange(signal.shape[0], dtype=np.float32)
    dst_positions = np.linspace(0.0, max(float(signal.shape[0] - 1), 0.0), num=target_len, dtype=np.float32)
    return np.interp(dst_positions, src_positions, signal).astype(np.float32)


def _float32_to_pcm16_bytes(signal: np.ndarray) -> bytes:
    clipped = np.clip(signal, -1.0, 1.0)
    encoded = np.round(clipped * 32767.0).astype("<i2")
    return encoded.tobytes()


def _converted_frame_count(raw_frames: int, src_rate: int, dst_rate: int = TARGET_SAMPLE_RATE) -> int:
    return max(1, int(round(raw_frames * dst_rate / src_rate)))


def get_converted_frame_count(raw_path: Path) -> int:
    with wave.open(str(raw_path), "rb") as src_handle:
        src_frames = src_handle.getnframes()
        src_rate = src_handle.getframerate()
    return _converted_frame_count(src_frames, src_rate)


def convert_track(
    raw_path: Path,
    output_path: Path,
    target_num_frames: int,
    overwrite: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(raw_path), "rb") as src_handle:
        nchannels = src_handle.getnchannels()
        sample_width = src_handle.getsampwidth()
        sample_rate = src_handle.getframerate()
        frames = src_handle.readframes(src_handle.getnframes())

    mono_float = _pcm_bytes_to_float32(frames, sample_width=sample_width, channels=nchannels)
    resampled = _resample_linear(mono_float, src_rate=sample_rate, dst_rate=TARGET_SAMPLE_RATE)
    resampled = resampled[:target_num_frames]
    if resampled.shape[0] != target_num_frames:
        raise ValueError(
            f"{raw_path}: converted length {resampled.shape[0]} does not match target {target_num_frames}"
        )
    converted = _float32_to_pcm16_bytes(resampled)

    with wave.open(str(output_path), "wb") as dst_handle:
        dst_handle.setnchannels(TARGET_CHANNELS)
        dst_handle.setsampwidth(TARGET_SAMPLE_WIDTH)
        dst_handle.setframerate(TARGET_SAMPLE_RATE)
        dst_handle.writeframes(converted)


def _read_mono_16k_wave(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        nchannels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        if nchannels != TARGET_CHANNELS:
            raise ValueError(f"{path} is not mono: {nchannels}")
        if sample_width != TARGET_SAMPLE_WIDTH:
            raise ValueError(f"{path} sample width is {sample_width}, expected {TARGET_SAMPLE_WIDTH}")
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"{path} sample rate is {sample_rate}, expected {TARGET_SAMPLE_RATE}")
        frames = handle.readframes(handle.getnframes())

    signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return signal


def compute_activation_matrix(wav_paths: Iterable[Path], threshold_db: float = DEFAULT_THRESHOLD_DB) -> np.ndarray:
    source_signals = [_read_mono_16k_wave(path) for path in wav_paths]
    if not source_signals:
        raise ValueError("No wave paths given for activation computation")

    sample_rate = TARGET_SAMPLE_RATE
    max_len = max(len(sig) for sig in source_signals)
    min_len = min(len(sig) for sig in source_signals)
    if max_len - min_len > sample_rate:
        print(f"Warning: large track length mismatch ({min_len} vs {max_len} samples)")

    num_seconds = min_len // sample_rate
    num_sources = len(source_signals)
    matrix = np.zeros((num_sources, num_seconds), dtype=bool)

    if num_seconds == 0:
        return matrix

    for idx, signal in enumerate(source_signals):
        if len(signal) < num_seconds * sample_rate:
            signal = np.pad(signal, (0, num_seconds * sample_rate - len(signal)))
        elif len(signal) > num_seconds * sample_rate:
            signal = signal[: num_seconds * sample_rate]

        magnitude = np.abs(signal)
        magnitude /= max(float(np.max(magnitude)), 1e-10)
        db = 20.0 * np.log10(magnitude + 1e-10)
        segments_db = db.reshape(num_seconds, sample_rate)
        matrix[idx] = np.sum(segments_db > threshold_db, axis=1) > DEFAULT_ACTIVE_RATIO * sample_rate

    return matrix


def process_song(
    spec: SongSpec,
    audio_root: Path,
    meta_root: Path,
    overwrite: bool = False,
    validate_only: bool = False,
    allowed_stems: frozenset[str] | None = None,
) -> dict[str, object]:
    processed_paths = [audio_root / track.processed_relpath for track in spec.tracks]
    converted_lengths = [get_converted_frame_count(track.raw_path) for track in spec.tracks]
    target_num_frames = min(converted_lengths)
    dropped_frames = max(converted_lengths) - target_num_frames

    if not validate_only:
        for track, output_path in zip(spec.tracks, processed_paths):
            if not track.raw_path.exists():
                raise FileNotFoundError(f"Missing raw track: {track.raw_path}")
            convert_track(
                track.raw_path,
                output_path,
                target_num_frames=target_num_frames,
                overwrite=overwrite,
            )

        meta_root.mkdir(parents=True, exist_ok=True)
        rel_paths = [str(track.processed_relpath) for track in spec.tracks]
        (meta_root / f"{spec.song_id}.txt").write_text("\n".join(rel_paths) + "\n")
        activations = compute_activation_matrix(processed_paths)
        np.save(meta_root / f"{spec.song_id}.npy", activations)

    txt_path = meta_root / f"{spec.song_id}.txt"
    npy_path = meta_root / f"{spec.song_id}.npy"
    validate_song_outputs(
        spec,
        audio_root=audio_root,
        txt_path=txt_path,
        npy_path=npy_path,
        allowed_stems=allowed_stems,
    )

    return {
        "song_id": spec.song_id,
        "num_tracks": len(spec.tracks),
        "num_seconds": int(np.load(npy_path).shape[1]),
        "target_num_frames": target_num_frames,
        "dropped_frames": dropped_frames,
    }


def validate_song_outputs(
    spec: SongSpec,
    audio_root: Path,
    txt_path: Path,
    npy_path: Path,
    allowed_stems: frozenset[str] | None = None,
) -> None:
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing txt metadata: {txt_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing activation metadata: {npy_path}")

    rel_paths = [line for line in txt_path.read_text().splitlines() if line]
    activations = np.load(npy_path)

    if len(rel_paths) != len(spec.tracks):
        raise ValueError(f"{spec.song_id}: txt rows {len(rel_paths)} != expected tracks {len(spec.tracks)}")
    if activations.shape[0] != len(rel_paths):
        raise ValueError(f"{spec.song_id}: npy rows {activations.shape[0]} != txt rows {len(rel_paths)}")

    expected_rel_paths = [str(track.processed_relpath) for track in spec.tracks]
    if rel_paths != expected_rel_paths:
        raise ValueError(f"{spec.song_id}: txt ordering does not match song spec ordering")

    for rel_path in rel_paths:
        rel_parts = Path(rel_path).parts
        if len(rel_parts) < 3:
            raise ValueError(f"{spec.song_id}: expected <song>/<stem>/<track>.wav relpath, got {rel_path}")
        canonical_stem = rel_parts[1]
        if allowed_stems is not None and canonical_stem not in allowed_stems:
            raise ValueError(f"{spec.song_id}: stem {canonical_stem!r} is not allowed for this stem mode")
        wav_path = audio_root / rel_path
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing converted wav: {wav_path}")
        with wave.open(str(wav_path), "rb") as handle:
            if handle.getnchannels() != TARGET_CHANNELS:
                raise ValueError(f"{wav_path}: expected mono")
            if handle.getframerate() != TARGET_SAMPLE_RATE:
                raise ValueError(f"{wav_path}: expected 16kHz")
            if handle.getsampwidth() != TARGET_SAMPLE_WIDTH:
                raise ValueError(f"{wav_path}: expected 16-bit PCM")
            if handle.getnframes() != activations.shape[1] * TARGET_SAMPLE_RATE:
                expected_frames = activations.shape[1] * TARGET_SAMPLE_RATE
                actual_frames = handle.getnframes()
                if actual_frames < expected_frames or actual_frames - expected_frames >= TARGET_SAMPLE_RATE:
                    raise ValueError(
                        f"{wav_path}: unexpected frame count {actual_frames}, expected close to {expected_frames}"
                    )

    saved_lengths = []
    for rel_path in rel_paths:
        wav_path = audio_root / rel_path
        with wave.open(str(wav_path), "rb") as handle:
            saved_lengths.append(handle.getnframes())
    if len(set(saved_lengths)) != 1:
        raise ValueError(f"{spec.song_id}: cropped track lengths are not identical: {sorted(set(saved_lengths))}")


def run_parallel(
    song_specs: list[SongSpec],
    audio_root: Path,
    meta_root: Path,
    workers: int,
    overwrite: bool,
    validate_only: bool,
    allowed_stems: frozenset[str] | None = None,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_song, spec, audio_root, meta_root, overwrite, validate_only, allowed_stems): spec.song_id
            for spec in song_specs
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["song_id"])
    return results


def summarize_outputs(results: list[dict[str, object]]) -> str:
    num_songs = len(results)
    num_tracks = sum(int(item["num_tracks"]) for item in results)
    seconds = [int(item["num_seconds"]) for item in results]
    dropped_seconds = [float(item["dropped_frames"]) / TARGET_SAMPLE_RATE for item in results]
    min_seconds = min(seconds) if seconds else 0
    max_seconds = max(seconds) if seconds else 0
    mean_seconds = float(sum(seconds) / len(seconds)) if seconds else 0.0
    max_dropped = max(dropped_seconds) if dropped_seconds else 0.0
    mean_dropped = float(sum(dropped_seconds) / len(dropped_seconds)) if dropped_seconds else 0.0
    return (
        f"processed_songs={num_songs} total_tracks={num_tracks} "
        f"min_seconds={min_seconds} max_seconds={max_seconds} mean_seconds={mean_seconds:.1f} "
        f"max_dropped_seconds={max_dropped:.3f} mean_dropped_seconds={mean_dropped:.3f}"
    )


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    audio_root, meta_root = resolve_output_roots(args)
    audio_root = audio_root.resolve()
    meta_root = meta_root.resolve()
    _validate_output_roots_for_mode(audio_root=audio_root, meta_root=meta_root, stem_mode=args.stem_mode)

    stem_mapping = STEM_MODE_TO_MAPPING[args.stem_mode]
    allowed_stems = STEM_MODE_TO_ALLOWED_STEMS[args.stem_mode]
    song_specs = discover_songs_with_mapping(raw_root, stem_mapping=stem_mapping, limit_songs=args.limit_songs)
    if not song_specs:
        raise SystemExit("No songs discovered. Check raw_root and data.json structure.")

    results = run_parallel(
        song_specs=song_specs,
        audio_root=audio_root,
        meta_root=meta_root,
        workers=max(1, args.workers),
        overwrite=args.overwrite,
        validate_only=args.validate_only,
        allowed_stems=allowed_stems,
    )
    print(summarize_outputs(results))


if __name__ == "__main__":
    main()
