"""run.sh's dotenv parser must never execute the file and must agree with pydantic-settings."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(env_text: str, tmp_path: Path, preset: dict[str, str] | None = None) -> dict[str, str]:
    env_file = tmp_path / "test.env"
    env_file.write_text(env_text)
    script = (
        'source <(sed -n "/^log()/p;/^warn()/p;/^load_env()/,/^}/p" "$ROOT/run.sh")\n'
        'load_env\n'
        'for k in A B C D E F G H; do printf "%s=[%s]\\n" "$k" "${!k-unset}"; done\n'
    )
    env = {"PATH": "/usr/bin:/bin", "ROOT": str(ROOT), "ENV_FILE": str(env_file), **(preset or {})}
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, check=True)
    return dict(line.split("=", 1) for line in out.stdout.splitlines())


def test_dotenv_parsing(tmp_path: Path):
    marker = tmp_path / "pwned"
    values = load(
        "# comment\n"
        "A=plain   # trailing comment\n"
        "B='{\"enable_thinking\": false}'\n"
        'C="spaced value"  # comment\n'
        "D=\n"
        "export E=exported\n"
        f"F=$(touch {marker})\n"
        "G=a#b\n"
        "not a valid line\n",
        tmp_path,
    )
    assert values["A"] == "[plain]"
    assert values["B"] == '[{"enable_thinking": false}]'
    assert values["C"] == "[spaced value]"
    assert values["D"] == "[]"
    assert values["E"] == "[exported]"
    assert values["F"] == f"[$(touch {marker})]" and not marker.exists()  # never executed
    assert values["G"] == "[a#b]"


def test_process_environment_wins(tmp_path: Path):
    assert load("A=from_file\n", tmp_path, {"A": "from_env"})["A"] == "[from_env]"


def test_bash_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "run.sh")], check=True)
