"""PyPower model plotting module

The `pypower_sim.ppplots.PPPlots` module is used to the generate the following
plots:

- `pypower_sim.ppplots.PPPlots.voltage` generates a voltage profile with bus
  index on the horizontal axis and voltage magnitude (per-unit bus kV) and
  angle (degrees) on the vertical axis.

- `pypower_sim.ppplots.PPPlots.generation` generates a generation profile with
  bus index on the horizontal axis and generation real power (MW) and
  reactive power (MVAr) on the vertical axis.

- `pypower_sim.ppplots.PPPlots.load` generates a load profile with bus index
  on the horizontal axis and load real power (MW) and reactive power(MVAr) on
  the vertical axis.
"""

from typing import TypeVar
import matplotlib.pyplot as plt
from .ppmodel import PPModel

class PPPlots:
    """`pypower_sim` plotting class implementation"""
    figsize = (15,8)
    """Default figure size"""

    def __init__(self,
        model:PPModel,
        ):
        """Plot creator constructor

        # Arguments

        - `model`: pypower_sim model
        """
        self.model = model
        """`pypower_sim.ppmodel.PPModel` object"""

    def voltage(self,
        figsize:tuple[int,int]=None,
        ) -> TypeVar('matplotlib.pyplot.figure'):
        """Generate voltage profile

        # Arguments

        - `figsize`: figure dimensions
        
        # Returns

        - `matplotlib.pyplot.figure`: voltage profile figure
        """

        bus = self.model.get_data("bus")

        #
        # Plot voltage errors by bus
        #
        fig = plt.figure(figsize=self.figsize if figsize is None else figsize)

        plt.subplot(2,1,1)
        plt.plot(bus.VM,label=self.model.name)
        plt.ylabel("Voltage Magnitude (pu.kV)")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(bus.VA,label=self.model.name)
        plt.ylabel("Voltage Angle (deg)")
        plt.xlabel("Bus ID")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.suptitle(f"{self.model.name} Voltage")

        return fig

    def generation(self,
        figsize:tuple[int,int]=None,
        ) -> TypeVar('matplotlib.pyplot.figure'):
        """Generate generation profile

        # Arguments

        - `figsize`: figure dimensions
        
        # Returns

        - `matplotlib.pyplot.figure`: voltage profile figure
        """

        bus = self.model.get_data("gen")

        #
        # Plot voltage errors by bus
        #
        fig = plt.figure(figsize=self.figsize if figsize is None else figsize)

        plt.subplot(2,1,1)
        plt.plot(bus.PG,label=self.model.name)
        plt.ylabel("Real power generation (pu.MVA)")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(bus.QG,label=self.model.name)
        plt.ylabel("Reactive power generation (pu.MVA)")
        plt.xlabel("Bus ID")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.suptitle(f"{self.model.name} Generation")

        return fig

    def load(self,
        figsize:tuple[int,int],
        ) -> TypeVar('matplotlib.pyplot.figure'):
        """Generate load profile

        # Arguments

        - `figsize`: figure dimensions
        
        # Returns

        - `matplotlib.pyplot.figure`: voltage profile figure
        """

        bus = self.model.get_data("bus")

        #
        # Plot voltage errors by bus
        #
        fig = plt.figure(figsize=self.figsize if figsize is None else figsize)

        plt.subplot(2,1,1)
        plt.plot(bus.PD,label=self.model.name)
        plt.ylabel("Real power demand (pu.MVA)")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(bus.QD,label=self.model.name)
        plt.ylabel("Reactive power demand (pu.MVA)")
        plt.xlabel("Bus ID")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.suptitle(f"{self.model.name} Generation")

        return fig
