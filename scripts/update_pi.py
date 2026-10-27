from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    repo_dir = Path(__file__).resolve().parent.parent
    raw_dir = repo_dir / "data" / "raw"
    backup_root = Path.home() / ".local" / "share" / "horsewashing" / "backups"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_root / f"{timestamp}-{__import__('os').getpid()}"
    raw_backup = backup_dir / "raw"

    backup_dir.mkdir(parents=True, exist_ok=False)
    if raw_dir.is_dir():
        shutil.copytree(raw_dir, raw_backup)
        raw_dir.rename(backup_dir / "raw-staged")

    def restore_raw() -> None:
        if raw_backup.is_dir():
            shutil.copytree(raw_backup, raw_dir, dirs_exist_ok=True)

    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "pull", "--ff-only"], cwd=repo_dir, check=True)
        restore_raw()
    except (OSError, subprocess.CalledProcessError) as error:
        try:
            restore_raw()
        except OSError as restore_error:
            print(
                f"Update failed ({error}); raw-data restore also failed "
                f"({restore_error}). Backup remains at {backup_dir}",
                file=sys.stderr,
            )
            return 1
        print(
            f"Update failed ({error}). Raw data was restored; "
            f"backup remains at {backup_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"Update complete. Raw data backup is preserved at {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
