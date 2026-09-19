#!/usr/bin/env python3
"""Publish only this study's code, numeric evidence, and report to the authorized repo."""

import argparse
import json
import subprocess
import time

from analyze import DATA, REPORT, ROOT, STUDY, write


def publish(label):
    repository = STUDY.parents[1]
    paths = [
        STUDY / "scripts/native_disagg_flags",
        DATA,
        REPORT.with_suffix(".md"),
        REPORT.with_suffix(".html"),
    ]

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(repository),
                 "-c", "user.name=alisachen-google",
                 "-c", "user.email=alisachen-google@users.noreply.github.com", *args],
                text=True, stderr=subprocess.STDOUT, timeout=120,
            ).strip()
        except subprocess.CalledProcessError as error:
            error.add_note(error.output[-4000:])
            raise

    git("add", "--", *[str(path.relative_to(repository)) for path in paths])
    compact = sorted(DATA.glob("runs/*/request-metrics.csv.gz"))
    if compact:
        git("add", "-f", "--", *[str(path.relative_to(repository)) for path in compact])
    staged = git("diff", "--cached", "--name-only").splitlines()
    prefixes = [str(path.relative_to(repository)) for path in paths]
    if any(
        not any(name == p or name.startswith(p + "/") for p in prefixes)
        for name in staged
    ):
        raise RuntimeError("Unrelated staged files; refusing to publish them")
    if staged:
        git("commit", "-m", label)
    for attempt in range(3):
        git("fetch", "origin", "main")
        try:
            git("rebase", "origin/main")
        except subprocess.CalledProcessError:
            git("rebase", "--abort")
            raise
        try:
            git("push", "origin", "HEAD:main")
            break
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(3)
    commit = git("rev-parse", "HEAD")
    remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
    publication = {
        "state": "published",
        "updated_unix": time.time(),
        "commit": commit,
        "remote_main": remote,
        "report_url": "https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/"
        + str(REPORT.with_suffix(".md").relative_to(repository)),
        "html_url": "https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/"
        + str(REPORT.with_suffix(".html").relative_to(repository)),
    }
    write(ROOT / "publication.json", publication)
    return publication


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", default="Add native disagg C480 KV flag sweep progress"
    )
    print(json.dumps(publish(parser.parse_args().label)), flush=True)
