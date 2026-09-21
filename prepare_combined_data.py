import argparse
import csv
from pathlib import Path


OUTPUT_COLUMNS = [
    "audio_filename",
    "speaker",
    "transcript",
    "language",
    "convo_id",
    "source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create combined manifests for the competition and Tetelancingo data."
    )
    parser.add_argument(
        "--competition-train-manifest",
        default="data/train_metadata.csv",
        help="Existing competition training CSV.",
    )
    parser.add_argument(
        "--competition-dev-manifest",
        default="data/dev_metadata.csv",
        help="Existing competition validation CSV.",
    )
    parser.add_argument(
        "--competition-audio-dir",
        default="nahuatl_dev/clips",
        help="Competition audio directory, relative to the repository root.",
    )
    parser.add_argument(
        "--tetelancingo-corpus",
        default="tetelancingo_nahuatl/corpus.tsv",
        help="Tetelancingo corpus.tsv path.",
    )
    parser.add_argument(
        "--tetelancingo-audio-dir",
        default="tetelancingo_nahuatl/audios",
        help="Tetelancingo audio directory, relative to the repository root.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/combined",
        help="Directory for the combined manifests.",
    )
    return parser.parse_args()


def read_manifest(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def competition_rows(
    manifest_path: Path, audio_dir: Path
) -> list[dict[str, str]]:
    rows = read_manifest(manifest_path, ",")
    result = []
    for row in rows:
        filename = row["audio_filename"]
        audio_path = audio_dir / filename
        if not audio_path.is_file():
            raise FileNotFoundError(f"Competition audio file not found: {audio_path}")
        result.append(
            {
                "audio_filename": audio_path.as_posix(),
                "speaker": row.get("speaker", ""),
                "transcript": row["transcript"],
                "language": row.get("language", ""),
                "convo_id": row.get("convo_id", ""),
                "source": "competition",
            }
        )
    return result


def tetelancingo_rows(
    corpus_path: Path, audio_dir: Path, split: str
) -> list[dict[str, str]]:
    rows = read_manifest(corpus_path, "\t")
    result = []
    for row in rows:
        if row["split"] != split:
            continue
        audio_path = audio_dir / row["audio_file"]
        if not audio_path.is_file():
            raise FileNotFoundError(f"Tetelancingo audio file not found: {audio_path}")
        result.append(
            {
                "audio_filename": audio_path.as_posix(),
                "speaker": row.get("speaker", ""),
                "transcript": row["sentence"],
                "language": "tetelancingo",
                "convo_id": "",
                "source": "tetelancingo",
            }
        )
    return result


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    competition_train = competition_rows(
        Path(args.competition_train_manifest), Path(args.competition_audio_dir)
    )
    competition_dev = competition_rows(
        Path(args.competition_dev_manifest), Path(args.competition_audio_dir)
    )
    tetelancingo_train = tetelancingo_rows(
        Path(args.tetelancingo_corpus),
        Path(args.tetelancingo_audio_dir),
        "train",
    )
    tetelancingo_test = tetelancingo_rows(
        Path(args.tetelancingo_corpus),
        Path(args.tetelancingo_audio_dir),
        "test",
    )

    output_dir = Path(args.output_dir)
    write_manifest(
        output_dir / "train_metadata.csv",
        competition_train + tetelancingo_train,
    )
    write_manifest(output_dir / "competition_train_metadata.csv", competition_train)
    write_manifest(output_dir / "tetelancingo_train_metadata.csv", tetelancingo_train)
    write_manifest(
        output_dir / "dev_metadata.csv",
        competition_dev + tetelancingo_test,
    )

    print(
        f"Train: {len(competition_train)} competition + "
        f"{len(tetelancingo_train)} Tetelancingo = "
        f"{len(competition_train) + len(tetelancingo_train)}"
    )
    print(
        f"Validation: {len(competition_dev)} competition + "
        f"{len(tetelancingo_test)} Tetelancingo = "
        f"{len(competition_dev) + len(tetelancingo_test)}"
    )


if __name__ == "__main__":
    main()
