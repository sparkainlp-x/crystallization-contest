import csv
import json
import os

import pytest

from crystallization_contest.__main__ import main


@pytest.mark.parametrize("exp", ["sweep", "contest", "stress", "fcost"])
def test_cli_quick_smoke(exp, tmp_path, capsys):
    rc = main([exp, "--quick", "--widths", "16", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SYNTHETIC" in out and "smoke run" in out
    assert os.path.exists(tmp_path / f"{exp}_raw.csv")
    with open(tmp_path / f"{exp}_summary.csv") as f:
        assert len(list(csv.DictReader(f))) >= 2


def test_cli_rejects_bad_arguments():
    with pytest.raises(SystemExit):
        main(["sweep", "--jobs", "0"])
    with pytest.raises(SystemExit):
        main(["contest", "--configs", "XYZ", "--quick"])
    with pytest.raises(SystemExit):
        main(["sweep", "--seeds", "1"])
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_cli_v2_emits_strict_machine_readable_example(capsys):
    assert main(["v2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "illustrative_not_optimal"
    assert result["normalized_cooperative"]["score"] == pytest.approx(0.25)
    assert result["target_utility"]["per_episode_utility"] == pytest.approx(0.46)
