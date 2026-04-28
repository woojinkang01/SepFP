from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible train/validation split for SepFP MoisesDB metadata.",
    )
    parser.add_argument(
        "--meta-root",
        type=Path,
        default=Path("/home/maclab/user_woojinkang/SepFP/data/moisesdb_meta_cropped"),
        help="Directory containing <song_id>.npy and <song_id>.txt metadata pairs.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("/home/maclab/user_woojinkang/SepFP/data/splits"),
        help="Directory where manifest files will be written.",
    )
    parser.add_argument(
        "--train-meta-root",
        type=Path,
        default=Path("/home/maclab/user_woojinkang/SepFP/data/moisesdb_meta_train208_seed0"),
        help="Output directory for training metadata files.",
    )
    parser.add_argument(
        "--val-meta-root",
        type=Path,
        default=Path("/home/maclab/user_woojinkang/SepFP/data/moisesdb_meta_val32_seed0"),
        help="Output directory for validation metadata files.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=32,
        help="Number of songs to assign to the validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible song sampling.",
    )
    return parser.parse_args()


def collect_song_ids(meta_root: Path) -> list[str]:
    song_ids = sorted(path.stem for path in meta_root.glob("*.npy"))
    if not song_ids:
        raise ValueError(f"No metadata .npy files found under {meta_root}")

    missing_txt = [song_id for song_id in song_ids if not (meta_root / f"{song_id}.txt").is_file()]
    if missing_txt:
        raise ValueError(f"Missing .txt pair for {len(missing_txt)} song(s), first={missing_txt[0]}")

    return song_ids


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_song_metadata(song_ids: list[str], src_root: Path, dst_root: Path) -> None:
    for song_id in song_ids:
        shutil.copy2(src_root / f"{song_id}.npy", dst_root / f"{song_id}.npy")
        shutil.copy2(src_root / f"{song_id}.txt", dst_root / f"{song_id}.txt")


def write_manifest(song_ids: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{song_id}\n" for song_id in song_ids), encoding="utf-8")


def main() -> None:
    args = parse_args()
    meta_root = args.meta_root.resolve()
    split_name = f"moisesdb_val{args.val_count}_seed{args.seed}"
    manifest_root = args.split_root.resolve() / split_name

    song_ids = collect_song_ids(meta_root)
    if args.val_count <= 0 or args.val_count >= len(song_ids):
        raise ValueError(f"val-count must be in [1, {len(song_ids) - 1}], got {args.val_count}")

    rng = random.Random(args.seed)
    val_song_ids = sorted(rng.sample(song_ids, args.val_count))
    val_song_set = set(val_song_ids)
    train_song_ids = [song_id for song_id in song_ids if song_id not in val_song_set]

    train_meta_root = args.train_meta_root.resolve()
    val_meta_root = args.val_meta_root.resolve()
    reset_dir(train_meta_root)
    reset_dir(val_meta_root)

    copy_song_metadata(train_song_ids, meta_root, train_meta_root)
    copy_song_metadata(val_song_ids, meta_root, val_meta_root)

    write_manifest(train_song_ids, manifest_root / "train_song_ids.txt")
    write_manifest(val_song_ids, manifest_root / "val_song_ids.txt")

    print(f"meta_root={meta_root}")
    print(f"train_meta_root={train_meta_root}")
    print(f"val_meta_root={val_meta_root}")
    print(f"manifest_root={manifest_root}")
    print(f"seed={args.seed}")
    print(f"total_songs={len(song_ids)}")
    print(f"train_songs={len(train_song_ids)}")
    print(f"val_songs={len(val_song_ids)}")


if __name__ == "__main__":
    main()
