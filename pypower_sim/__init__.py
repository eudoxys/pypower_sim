"""PyPOWER timeseries simulation

Syntax: pypower_sim [OPTIONS...] CASENAME [ARGUMENTS ...]

Options:

Arguments:
"""

from .kml import KML
from .ppdata import PPData
from .ppgen import PPGen
from .ppmodel import PPModel
from .ppplots import PPPlots
from .ppsolver import PPSolver

def main(*args,**kwargs):
    """Main command line processor"""
    print([x for x in __doc__.split("\n") if x.startswith("Syntax: ")][0])
    
    return 1

if __name__ == "__main__":

    exit(main())