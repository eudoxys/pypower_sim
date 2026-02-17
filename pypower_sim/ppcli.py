"""PyPOWER Simulation CLI

# Usage
    
    ppsim [-h] [-C CASE] filename {print,plot,solve,simulate}

## Positional arguments

- `filename`: case filename

- `{print,plot,solve,simulate}`: command to execute on the model

## Options

- `-h`, `--help`: show this help message and exit

- `-C`, `--case CASE`: case name to use from `pypower_sim.ppcli.PPCLI.filename` specified

# Description

PyPOWER Simulation CLI

"""

import argparse
import importlib

from pypower_sim.ppmodel import PPModel

class PPCLI(argparse.ArgumentParser):
    """Main CLI implementation"""
    
    E_OK = 0
    """Exit code on success"""
    
    E_SYNTAX = 1
    """Exit code on syntax error"""

    E_FAILED = 2
    """Exit code on failure"""

    def __init__(self):
        """CLI constructor/processor"""
        super().__init__(
            description=__doc__.split("\n",maxsplit=1)[0],
            epilog="\n".join(__doc__.split("\n")[1:])
            )

        self.filename = None
        """PyPOWER case filename"""

        self.case = None
        """PyPOWER case data"""

        self.model = None
        """`pypower_sim.ppmodel.PPModel` object"""

        # required arguments
        self.add_argument("filename",
            help="case filename"
            )
        self.add_argument("command",
            choices=["print","plot","solve","simulate"],
            default="print",
            help="command to execute on the model"
            )

        # optional arguments
        self.add_argument("-C","--case",
            help="case name to use from `filename` specified"
            )
        # self.parser.add_argument("-v","--verbose",action="store_true")
        # self.parser.add_argument("-d","--debug",action="store_true")

        self.parse_args()

        if self.filename:
            if self.case is None:
                self.case = os.path.splitext(os.path.basename(self.filename))[0]
            module = importlib.import_module(self.filename)
            if self.case:
                self.model = PPModel(case=getattr(module,self.case))

        match self.command:

            case 'print':

                self.model.print()

            case '_':

                raise ValueError(f"{command=} is invalid")

        self.exitcode = E_OK
        """Exit code (see `pypower_sim.ppcli.PPCLI` exit codes `E_*`)"""
