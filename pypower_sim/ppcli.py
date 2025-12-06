"""PyPOWER Simulation CLI

"""

import argparse
import importlib

from .ppmodel import PPModel

class PPCLI(argparse.ArgumentParser):
    """Main CLI implementation"""
    E_OK = 0
    E_SYNTAX = 1

    def __init__(self):
        """CLI constructor/processor

        Arguments:

        (none)
        """
        super().__init__(
            description=__doc__.split("\n",maxsplit=1)[0],
            epilog="\n".join(__doc__.split("\n")[1:])
            )

        self.filename = None
        self.case = None
        self.model = None

        # required arguments
        self.add_argument("filename")
        self.add_argument("command",
            choices=["print","plot","solve","simulate"],
            default="print",
            )

        # optional arguments
        self.add_argument("-C","--case")
        # self.parser.add_argument("-v","--verbose",action="store_true")
        # self.parser.add_argument("-d","--debug",action="store_true")

        self.parse_args()

        if self.filename:
            module = importlib.import_module(self.filename)
            if self.case:
                self.model = PPModel(case=getattr(module,self.case))

        self.exitcode = 0
