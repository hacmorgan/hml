"""
Git-interfacing utilities for hml models

Author:  Hamish Morgan
Date:    22/05/2022
License: BSD
"""


import os
import subprocess
import sys


def modified_files_in_git_repo() -> bool:
    """
    Ensure hml has no uncommitted files.

    It is required that the git hash is written to the model dir, to ensure the same
    code that was used for training can be retrieved later and paired with the trained
    parameters.

    Returns:
        True if there are modified files, False otherwise
    """
    result = subprocess.run(
        'cd "$HOME/src/hml" && git status --porcelain=v1 | grep -v -e "^??" -e "^M"',
        shell=True,
        stdout=subprocess.PIPE,
    )
    output = result.stdout.decode("utf-8").strip()
    if len(output) > 0:
        print(
            """
            ERROR: Uncommitted code in $HOME/src/hml

            Commit changes before re-running train
            """,
            file=sys.stderr,
        )
        return True
    return False


def write_commit_hash_to_model_dir(model_dir: str) -> None:
    """
    Write the commit hash used for training to the model dir
    """
    result = subprocess.run(
        'cd "$HOME/src/hml" && git rev-parse --verify HEAD',
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
    )
    commit_hash = result.stdout.decode("utf-8").strip()
    with open(os.path.join(model_dir, "commit-hash"), "w") as hashfile:
        hashfile.write(commit_hash)
