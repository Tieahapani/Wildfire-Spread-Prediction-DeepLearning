import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[3] / ".githooks" / "commit-msg"


def _run_hook(tmp_path: Path, message: str) -> subprocess.CompletedProcess:
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(message)
    return subprocess.run(["bash", str(HOOK), str(msg_file)], capture_output=True, text=True)


@pytest.mark.parametrize(
    "message",
    [
        "feat(eval): add masked PR-AUC metric\n\nWhy it matters.\n\n- Change\n",
        # Pair programming between people stays allowed.
        "fix(data): flip wind\n\n- Change\n\nCo-Authored-By: Jane Doe <jane@example.com>\n",
        "Merge branch 'main' into feat/x\n",
    ],
)
def test_hook_accepts_clean_messages(tmp_path, message):
    assert _run_hook(tmp_path, message).returncode == 0


@pytest.mark.parametrize(
    "message",
    [
        "feat(eval): add metric\n\n- Change\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n",
        "feat(eval): add metric\n\nco-authored-by: GitHub Copilot <copilot@github.com>\n",
        "docs(readme): rewrite\n\n- Change\n\nGenerated with Claude Code\n",
        # Merges and reverts skip the format check but not the attribution check.
        "Merge branch 'x'\n\nCo-Authored-By: ChatGPT <noreply@openai.com>\n",
    ],
)
def test_hook_rejects_ai_attribution(tmp_path, message):
    result = _run_hook(tmp_path, message)
    assert result.returncode == 1
    assert "AI attribution" in result.stderr


def test_hook_still_enforces_subject_format(tmp_path):
    assert _run_hook(tmp_path, "Added a metric.\n").returncode == 1
