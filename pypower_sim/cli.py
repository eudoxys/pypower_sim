"""Command line interface

See `pypower_sim` for the command syntax.

Examples
--------

Print the bus, branch, generator, and cost tables of a case file:

    ppsim pypower_sim/_test/data/case9.py print

Use a case function whose name differs from the file name:

    ppsim mycases.py print -C case9

Caveats
-------

  - Only the `print` command is implemented. Plotting, solving, and
    time-series simulation are available from Python (see
    `pypower_sim.ppsolver.PPSolver` and `pypower_sim.ppplots.PPPlots`).
"""

import os
import sys
import argparse
import logging
import warnings
from importlib.util import spec_from_file_location, module_from_spec

from pypower_sim.ppmodel import PPModel

_logger = logging.getLogger(__file__)

E_OK = 0
"""Exit code on success"""

E_SYNTAX = 1
"""Exit code on syntax error"""

E_FAILED = 2
"""Exit code on failure"""

def _version() -> str:
    try:
        import importlib.metadata # pylint: disable=import-outside-toplevel
        return importlib.metadata.version("pypower_sim")
    except Exception: # pylint: disable=broad-exception-caught
        return "dev"

def load_case(filename:str,case:str|None=None) -> PPModel:
    """Load a model from a `pypower` case file

    Arguments
    ---------

      - `filename`: path to a Python file defining a case function

      - `case`: name of the case function (default is the file's basename)

    Returns
    -------

      - `pypower_sim.ppmodel.PPModel`: the model

    Exceptions
    ----------

      - `FileNotFoundError`: `filename` does not exist

      - `AttributeError`: the file does not define the `case` function
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found")
    name = os.path.splitext(os.path.basename(filename))[0]
    if case is None:
        case = name
    modspec = spec_from_file_location(name,filename)
    module = module_from_spec(modspec)
    modspec.loader.exec_module(module)
    if not hasattr(module,case):
        raise AttributeError(f"{filename} does not define case '{case}'")
    return PPModel(name=case,case=getattr(module,case))

def main(argv:list[str]|None=None) -> int:
    """Command line entry point

    Arguments
    ---------

      - `argv`: command line arguments (default `sys.argv[1:]`)

    Returns
    -------

      - `int`: exit code (`E_OK`, `E_SYNTAX`, or `E_FAILED`)
    """
    parser = argparse.ArgumentParser(prog="ppsim",
        description="PyPOWER time-series simulator",
        epilog="See https://www.eudoxys.com/pypower_sim for full documentation.")
    parser.add_argument("filename",
        help="pypower case file (Python)")
    parser.add_argument("command",
        choices=["print"],
        help="command to execute on the model")
    parser.add_argument("-C","--case",
        help="case function name in FILENAME (default is the file basename)")
    parser.add_argument("-d","--debug",action="store_true",
        help="enable debug logging and traceback on exceptions")
    parser.add_argument("-w","--warning",action="store_true",
        help="enable warning messages from python")
    parser.add_argument("--version",action="version",version=f"%(prog)s {_version()}")
    try:
        args = parser.parse_args(argv)
    except SystemExit as err:
        return E_OK if err.code in (0,None) else E_SYNTAX

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    if not args.warning:
        warnings.simplefilter("ignore")

    try:
        model = load_case(args.filename,args.case)
        match args.command:
            case "print":
                model.print(file=sys.stdout)
        return E_OK
    except Exception as err: # pylint: disable=broad-exception-caught
        if args.debug:
            raise
        print(f"ERROR [ppsim]: {err}",file=sys.stderr)
        return E_FAILED

if __name__ == "__main__":
    sys.exit(main())
