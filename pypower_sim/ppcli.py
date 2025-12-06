"""PyPOWER Simulation CLI

"""

import sys
import argparse
import importlib

class PPCLI(argparse.ArgumentParser):

    E_OK = 0
    E_SYNTAX = 1

    def __init__(self):

        super().__init__(
            description=__doc__.split("\n")[0],
            epilog="\n".join(__doc__.split("\n")[1:])
            )

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

        module = importlib.import_module(self.filename)
        self.model = PPModel(case=getattr(module,self.case))

        self.exitcode = 0

if __name__ == "__main__":

    PPCLI()
