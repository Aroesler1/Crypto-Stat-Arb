"""Every command-line entry point must at least import and expose --help.

This exists because it did not. A patch to `run_residualization_ablation.py`
left a `for` statement with its body at the wrong indentation, the file was
committed, and nothing caught it: the unit tests never import the runners, so a
module that cannot be parsed passed the whole suite. Any script a reader is told
to run should fail here before it fails for them.

The check is deliberately shallow. It imports the module and asks its argument
parser for help, which exercises the module body and the parser wiring without
touching data, the network, or a backtest.
"""
import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ENTRYPOINTS = sorted(
    p.stem for p in (ROOT / "stat_arb").glob("*.py")
    if p.stem.startswith(("run_", "build_"))
)


def test_entrypoints_were_discovered():
    """A guard on the guard: an empty list would make every test below vacuous."""
    assert len(ENTRYPOINTS) >= 8, ENTRYPOINTS


@pytest.mark.parametrize("name", ENTRYPOINTS)
def test_entrypoint_imports(name):
    importlib.import_module(f"stat_arb.{name}")


@pytest.mark.parametrize("name", ENTRYPOINTS)
def test_entrypoint_help_parses(name):
    """--help must work on the scripts that take arguments.

    Not every runner does: `run_robustness.py` and `run_phase1/2/3.py` predate
    the argparse convention and their `main()` takes no argv. Those are skipped
    rather than forced to change, since the import test above already covers
    the failure this file exists to catch.
    """
    import inspect

    module = importlib.import_module(f"stat_arb.{name}")
    main = getattr(module, "main", None)
    if main is None:
        pytest.skip(f"{name} has no main()")
    if not inspect.signature(main).parameters:
        pytest.skip(f"{name}.main() takes no arguments")
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0


def test_every_package_module_imports():
    """The library modules too, not just the scripts."""
    import stat_arb
    for info in pkgutil.walk_packages(stat_arb.__path__, prefix="stat_arb."):
        if info.name.rsplit(".", 1)[-1].startswith(("run_", "build_")):
            continue          # covered above, and slower to import
        importlib.import_module(info.name)
