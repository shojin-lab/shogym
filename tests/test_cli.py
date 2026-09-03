"""The command line: one way to serve, one way to read, and a bare install that gets that far.

Three promises are checked here. Protocol version two is the only serving path, so there is no
flag that selects a version and the retired one cannot be asked for. Importing the package
or reading its help never pulls Temporal in: it is installed, and the import lives inside the
subcommand's body, which is the only place it is needed. And a run directory is read by naming
it, with a directory that has nothing to read answered rather than raised.

The import checks run in subprocesses. A test process that has already imported the durable
kernel for another test would answer for itself rather than for a fresh interpreter.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from shogym.cli import _build_parser, main
from shogym.serve.protocol_v2.rundir import create_run_directory


def _fresh(code: str) -> subprocess.CompletedProcess:
    """Run one snippet in an interpreter that has imported nothing of ours yet."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )


def test_importing_the_package_does_not_import_temporal() -> None:
    """Temporal is installed, and is still imported only where serving reaches it."""
    done = _fresh("import sys, shogym; print('temporalio' in sys.modules)")
    assert done.stdout.strip() == "False"


def test_reading_the_help_does_not_import_temporal() -> None:
    """Help is answered by the parser, which is built before anything durable is reached."""
    done = _fresh(
        "import sys\n"
        "from shogym.cli import main\n"
        "try:\n"
        "    main(['serve', '--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "print('temporalio' in sys.modules)\n"
    )
    assert done.stdout.splitlines()[-1] == "False"


def test_an_install_missing_temporal_is_told_what_to_install() -> None:
    """An install that did not finish, answered as an instruction.

    `pip install shogym` installs Temporal, so an install without it is a broken one, and a bare
    import traceback is a worse answer than the command that fixes it.
    """
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "class _Absent:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'temporalio' or name.startswith('temporalio.'):\n"
            "            raise ModuleNotFoundError(name=name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Absent())\n"
            "from shogym.cli import main\n"
            "main(['serve', 'wordle_v1'])\n",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2
    assert "pip install shogym" in done.stderr
    assert "Traceback" not in done.stderr


def test_the_service_this_command_starts_does_not_write_to_the_protocol_wire() -> None:
    """Whatever the embedded service prints goes to standard error, not to the stream.

    The service is another program and it announces itself when it starts. It inherits this
    process's descriptors, and this process's standard output is the transport, so a banner
    landing there is a line a strict client rejects before the handshake. The mechanism is
    checked with a child of its own rather than with the real service, because what is being
    checked is where a spawned child's output goes.
    """
    pytest.importorskip("temporalio")
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import subprocess, sys\n"
            "from shogym.serve.protocol_v2.kernel.runtime import _service_output_off_the_wire\n"
            "with _service_output_off_the_wire():\n"
            "    subprocess.run([sys.executable, '-c', \"print('a service banner')\"])\n"
            "print('the protocol')\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip() == "the protocol"
    assert "a service banner" in done.stderr


def test_there_is_no_flag_that_selects_a_serving_protocol() -> None:
    """The retired version cannot be asked for, and neither can the one that replaced it.

    A flag naming the version that is left would be a flag with one value, and a run that named
    the retired one would be a run asking for a path this package no longer has.
    """
    parser = _build_parser()
    for argv in (
        ["serve", "wordle_v1", "--protocol", "v1"],
        ["serve", "wordle_v1", "--protocol", "v2"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_a_run_directory_is_asked_for_without_naming_a_protocol() -> None:
    """The blobs and the resume manifest are what serving keeps, so the flag stands alone."""
    args = _build_parser().parse_args(
        ["serve", "wordle_v1", "--task", "3", "--run-dir", "runs/one"]
    )
    assert (args.command, args.env, args.task, args.run_dir) == (
        "serve",
        "wordle_v1",
        "3",
        "runs/one",
    )


def test_a_run_directory_is_read_by_naming_it() -> None:
    """Reading takes the directory and nothing else: what to read is what the run wrote."""
    args = _build_parser().parse_args(["results", "runs/one"])
    assert (args.command, args.run_dir) == ("results", "runs/one")


def test_reading_a_run_that_kept_no_history_says_so_and_writes_nothing(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run served without keeping its history has nothing to read, and that is the answer.

    The command starts no service to find that out and leaves the directory as it found it, so
    a directory that kept nothing costs a line of output rather than a download.
    """
    pytest.importorskip("temporalio")
    monkeypatch.delenv("SHOGYM_TEMPORAL_ADDRESS", raising=False)
    create_run_directory(
        tmp_path,
        workflow_id="stream/cli/1",
        task_queue="shogym-stream-v2",
        configuration_hash="c" * 64,
    )
    main(["results", str(tmp_path)])
    printed = capsys.readouterr().out
    assert printed.startswith("nothing to read:")
    assert not (tmp_path / "records.jsonl").exists()


def test_a_directory_that_is_not_a_run_is_refused_rather_than_reported_as_empty(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal is not an empty result, and something collecting runs has to tell them apart.

    A mistyped path, a half written directory, and one holding the retired protocol's logs all
    arrive here as refusals rather than as runs with nothing in them. Printing those as the
    answer and exiting nought would let a collection step gather no records and report that it
    had succeeded.
    """
    pytest.importorskip("temporalio")
    monkeypatch.delenv("SHOGYM_TEMPORAL_ADDRESS", raising=False)
    with pytest.raises(SystemExit) as refused:
        main(["results", str(tmp_path / "never-a-run")])
    assert refused.value.code == 1
    printed = capsys.readouterr()
    assert printed.out == ""
    assert printed.err.startswith("cannot read ")
    assert "never-a-run" in printed.err
