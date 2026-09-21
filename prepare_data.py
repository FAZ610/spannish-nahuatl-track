import csv
import random
from pathlib import Path


def main() -> None:
    source = Path("nahuatl_dev/metadata.tsv")
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    with source.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    conversations = sorted({row["convo_id"] for row in rows})
    random.Random(42).shuffle(conversations)
    split_at = max(1, int(len(conversations) * 0.9))
    train_conversations = set(conversations[:split_at])

    columns = ["audio_filename", "speaker", "transcript", "language", "convo_id"]
    cleaned_rows = [
        {column: row[column] for column in columns}
        for row in rows
    ]
    train_rows = [
        row for row in cleaned_rows if row["convo_id"] in train_conversations
    ]
    dev_rows = [
        row for row in cleaned_rows if row["convo_id"] not in train_conversations
    ]

    for filename, split_rows in (
        ("train_metadata.csv", train_rows),
        ("dev_metadata.csv", dev_rows),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(split_rows)

    print(f"Training rows: {len(train_rows)}")
    print(f"Validation rows: {len(dev_rows)}")


if __name__ == "__main__":
    main()
