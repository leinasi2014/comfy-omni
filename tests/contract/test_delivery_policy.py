from pathlib import Path

from scripts.check_delivery_policy import check_delivery_policy


def test_repository_delivery_policy() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert check_delivery_policy(repo_root) == ()
