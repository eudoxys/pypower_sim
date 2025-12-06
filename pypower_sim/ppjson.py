"""JSON support for PyPOWER simulation"""

import os
import sys
import io
import json
import pytz
from types import CodeType, FunctionType
import numpy as np
import pandas as pd

class PypowerModelEncoder(json.JSONEncoder):
    """Implements pypower_sim data encoder for JSON"""
    def default(self, obj):
        """Default JSON encoder for pypower_sim

        Caveat: tuples are converted by list to json before this call so you
        must encode tuples explicitly using `{"type":"tuple","data":data}`.
        """

        if isinstance(obj, np.integer):
        
            return int(obj)
        
        elif isinstance(obj, np.floating):
        
            return float(obj)
        
        elif isinstance(obj,set):

            return {"type": "set", "data": list(obj)}

        elif isinstance(obj,bytes):

            return {"type": "bytes", "data": " ".join(f"{x:x}" for x in obj)}
        
        elif isinstance(obj, np.ndarray):
        
            return {
                "type": "array", 
                "dtype": str(obj.dtype), 
                "data": obj.tolist(),
                }
        
        elif isinstance(obj,pd.DataFrame):
            return {
                "type": "dataframe",
                "frame" : {
                    "columns": list(obj.columns),
                    "index": {
                        "name": list(obj.index.names),
                        "dtype": str(obj.index.dtype),
                        "keys": obj.index.values.tolist(),
                        },
                    "rows":obj.values.tolist(),
                    },
                }
        
        elif isinstance(obj,dict) and [isinstance(x,tuple) for x in obj.keys()].any():
        
            return {"|".join(x):y for x,y in obj.items()}
        
        elif isinstance(obj,pd.Timestamp):
        
            return obj.strftime("%Y-%m-%d %H:%M:%S %Z")
        
        elif isinstance(obj,pd.Index):
        
            return obj.values.tolist()
        
        elif isinstance(obj,io.TextIOWrapper):
        
            return {
                "type": "file",
                "name": obj.name,
                "readable": obj.readable(),
                "writable": obj.writable(),
                "position": obj.tell(),
                }
        
        return json.JSONEncoder.default(self, obj)

def PypowerModelDecoder(data):
    """Convert JSON data back to pypower_sim data"""
    if isinstance(data,dict):

        # special encoding for PPModel data types
        if "type" in data:
            match data["type"]:

                case "file":

                    assert "name" in data, f"{data=} does not include 'name'"
                    assert "readable" in data, f"{data=} does not include 'readable'"
                    assert "writable" in data, f"{data=} does not include 'writable'"
                    assert "position" in data, f"{data=} does not include 'position'"
                    name = data["name"]
                    mode = None
                    if data["readable"]:
                        mode = "r"
                    elif data["writable"]:
                        mode = "a+" if mode == data["readable"] else "a"
                    pos = data["position"]
                    return open(name,mode,encoding="utf-8").seek(pos)

                case "array":

                    assert "dtype" in data, f"unable to decode array without 'dtype'"
                    dtype = data["dtype"] 
                    assert dtype in dir(np), f"{dtype=} is not a valid ndarray type"
                    assert "data" in data, f"unable to decode array without 'data'"
                    return np.array(data["data"],dtype=getattr(np,dtype))

                case "dataframe":

                    assert data["type"] == "dataframe", f"input {data=} is not a dataframe"
                    frame = data["frame"]
                    assert "keys" in frame["index"], f"input {data=} missing keys"
                    assert "dtype" in frame["index"], f"input {data=} missing dtype"
                    assert "name" in frame["index"], f"input {data=} missing name"
                    ndx = frame["index"]
                    assert ndx["dtype"] == "datetime64[ns, UTC]", f"input {data=} index is not datetime64[ns, UTC]"
                    df = pd.DataFrame(
                        data = frame["rows"],
                        index = pd.DatetimeIndex(ndx["keys"],tz=pytz.UTC),
                        )
                    df.index.names = frame["index"]["name"]
                    return df

                case "bytes":

                    return bytes([int(x,16) for x in data["data"].split()])

                case "tuple":

                    return tuple(data["data"])

                case "set":

                    return set(data["data"])

                case "_":

                    raise ValueError(f"type={spec['type']} is not recognized")

        return {x:PypowerModelDecoder(y) for x,y in data.items()}

    elif isinstance(data,list):

        return [PypowerModelDecoder(x) for x in data]

    return data

