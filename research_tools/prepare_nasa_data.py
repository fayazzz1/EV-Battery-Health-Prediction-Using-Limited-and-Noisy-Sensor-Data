"""Install the project's four verified NASA MAT files from the official ZIP."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def files_in(archive, desired, depth=0):
    for entry in archive.infolist():
        name = Path(entry.filename).name
        if name in desired:
            yield name, archive.read(entry)
        elif depth < 3 and name.lower().endswith(".zip"):
            with ZipFile(io.BytesIO(archive.read(entry))) as nested:
                yield from files_in(nested, desired, depth + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Official NASA Battery Data Set ZIP")
    args = parser.parse_args()
    provenance = json.loads((ROOT / "data_source.json").read_text())
    wanted = {cell["battery_id"] + ".mat": cell["sha256"] for cell in provenance["cells"]}
    verified = {}
    with ZipFile(args.archive) as archive:
        for name, data in files_in(archive, wanted):
            if hashlib.sha256(data).hexdigest() == wanted[name]:
                verified[name] = data
    missing = wanted.keys() - verified.keys()
    if missing:
        parser.error("Missing or unexpected data checksum: " + ", ".join(sorted(missing)))
    out = ROOT / "python_app" / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in verified.items():
        (out / name).write_bytes(data)
    print("Verified and installed " + ", ".join(sorted(verified)))


if __name__ == "__main__":
    main()
