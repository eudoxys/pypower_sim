"""PyPOWER timeseries simulation"""

import sys

from .kml import KML
from .ppdata import PPData
from .ppgen import PPGen
from .ppmodel import PPModel
from .ppplots import PPPlots
from .ppsolver import PPSolver
from .ppcli import PPCLI
from .ppjson import PypowerModelDecoder, PypowerModelEncoder

def main(*args,**kwargs):
    """Main command line processor"""

    cli = PPCLI(*args,**kwargs)

    return cli.exitcode

if __name__ == "__main__":

    sys.exit(main())
