"""PyPOWER Simulator GIS data manager"""

from typing import TypeVar
import warnings
import pandas as pd
from geohash import geohash
from pypower import idx_bus

class idx_gis:
    """Provide column index values for GIS data"""

    # pylint: disable=invalid-name,too-few-public-methods

    BUS_I = 0 # bus index
    """Bus index"""

    LAT = 1 # bus latitude
    """Bus latitude"""

    LON = 2 # bus longitude
    """Bus longitude"""

    GEOHASH = 3 # bus node id
    """Bus geohash"""

    NAME = 4 # bus name
    """Bus name"""

    GEN = 5 # generator count (nan: no gen allowed)
    """Total MW generation at bus"""

    LOAD = 6 # load count (nan: no load allowed)
    """Total MW load at bus"""

class PPGIS:
    """GIS manager class implementation

    GIS data can be loaded into a model by providing a dataframe
    or specifying a CSV file that can be read using `pandas.read_csv()`.

    If `columns` is specified, these columns will be used to select
    the columns to read in.

    The value for `update` must be specified to generate summary data and update the model:

    - `GEOHASH`: updates the `GEOHASH` column from `LAT` and `LON`

    - `GEN`: updates the `GEN` column from the aggreagated `gen.PMAX` data
      in the model.

    - `LOAD`: updates the `LOAD` column from the aggregated `bus.PD` data
      in the model.

    - `MODEL`: updates the model `gis` table from the data provided.
    """
    
    def __init__(self,
        model:TypeVar('PPModel'),
        data:str|pd.DataFrame,
        columns:list[str|int]=None,
        update:set[str]=None,
        **kwargs
        ):
        """GIS manager constructor

        # Arguments

        - `model`: pypower_sim model object

        - `data`: GIS data

        - `columns`: name of columns to read into the GIS data
          (see `ppgis.idx_gis`)

        - `update`: columns to update after loading data

        - `**kwargs`: Pandas read_csv() arguments to use
        """

        # check columns
        assert columns is None or len(columns) == len(model.get_header("gis")), \
            f"columns must be length={len(model.get_header("gis"))}"

        valid_updates = {"GEOHASH","GEN","LOAD","MODEL"}

        # check data
        if data is None:

            assert hasattr(model,"gis") and not model.gis is None, \
                "data must be provided if model does not contain GIS data already"

        # load data
        if isinstance(data,str):

            # ignore header row if columns are specified
            if not "skiprows" in kwargs and not columns is None:
                kwargs["skiprows"] = 1 

            # read CSV file
            data = pd.read_csv(data,
                usecols=columns,
                names=None if columns is None else model.get_header("gis"),
                **kwargs,
                )

            # force update if not already specified
            if update is None:
                update = valid_updates

        # check updates requested
        if update is None:
            update = set()
        assert isinstance(update,set), f"update must be a set or None"
        assert valid_updates - update == set(), f"update={{{valid_updates - update}}} " +\
            f"is not valid (must be one of {valid_updates})"

        # update geohash
        if "GEOHASH" in update:
            data["GEOHASH"] = [geohash(x.LAT,x.LON) for _,x in data.iterrows()]

        # update generators
        if "GEN" in update:

            gen = model.get_data("gen")
            nogis = set(gen.GEN_BUS) - set(data.BUS_I)
            assert nogis == set(), f"generators {nogis=} have no GIS data"
            
            gengis = pd.merge(gen,data,left_on="GEN_BUS",right_on="BUS_I")
            gengis.set_index("GEN_BUS",inplace=True)
            data.set_index("BUS_I",inplace=True)
            data["GEN"] = (gengis[["PMAX"]].groupby("GEN_BUS").sum()*model.case["baseMVA"]/1000).round(5)
            data.reset_index(inplace=True)

        # update loads
        if "LOAD" in update:

            load = model.get_data("bus")
            nogis = set(load.BUS_I) - set(data.BUS_I)
            assert nogis == set(), f"loads {nogis=} have no GIS data"

            loadgis = pd.merge(load,data,left_on="BUS_I",right_on="BUS_I")
            loadgis.set_index("BUS_I",inplace=True)
            data.set_index("BUS_I",inplace=True)
            data["LOAD"] = (loadgis[["PD"]].groupby("BUS_I").sum()*model.case["baseMVA"]/1000).round(5)
            data.reset_index(inplace=True)

        # save data to this object
        self.data = data[model.get_header("gis")]

        # update model
        if "MODEL" in update:
            model.case['gis'] = self.data.values
