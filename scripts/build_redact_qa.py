import json

SRC = "/nas2/benchmarks/vpi/real/merged_qa/vpi_qa.json"
DST = "/nas2/benchmarks/vpi/real/merged_qa/vpi_qa_redact.json"

# (category, video_number) pairs to redact
REDACT_TARGETS = {
    ("tennis", "0001"),
    ("tennis", "0002"),
    ("basketball", "0002"),
    ("basketball", "0003"),
    ("soccer", "0002"),
}


def redact_video_path(video_path: str) -> str:
    if not video_path.endswith(".mp4"):
        raise ValueError(f"unexpected video_path: {video_path}")
    return video_path[: -len(".mp4")] + "_redacted.mp4"


def main():
    with open(SRC, "r") as f:
        data = json.load(f)

    n_redacted = 0
    for category, items in data["data"].items():
        for item in items:
            num = item["video_id"].split("_")[0]
            if (category, num) in REDACT_TARGETS:
                item["video_path"] = redact_video_path(item["video_path"])
                n_redacted += 1

    with open(DST, "w") as f:
        json.dump(data, f, indent=2)

    print(f"redacted {n_redacted} items -> {DST}")


if __name__ == "__main__":
    main()
