"""JSON support for PyPOWER simulation"""

import io
import json
import pytz
import numpy as np
import pandas as pd

# pylink: disable=invalid-name
class PypowerModelEncoder(json.JSONEncoder):
    """Implements pypower_sim data encoder for JSON"""
    def default(self, o):
        """Default JSON encoder for pypower_sim

        Caveat: tuples are converted by list to json before this call so you
        must encode tuples explicitly using `{"type":"tuple","data":data}`.
        """

        # pylint: disable=too-many-return-statements

        if isinstance(o, np.integer):

            return int(o)

        if isinstance(o, np.floating):

            return float(o)

        if isinstance(o,set):

            return {"type": "set", "data": list(o)}

        if isinstance(o,bytes):

            return {"type": "bytes", "data": " ".join(f"{x:x}" for x in o)}

        if isinstance(o, np.ndarray):

            return {
                "type": "array",
                "dtype": str(o.dtype),
                "data": o.tolist(),
                }

        if isinstance(o,pd.DataFrame):
            return {
                "type": "dataframe",
                "frame" : {
                    "columns": list(o.columns),
                    "index": {
                        "name": list(o.index.names),
                        "dtype": str(o.index.dtype),
                        "keys": o.index.values.tolist(),
                        },
                    "rows":o.values.tolist(),
                    },
                }

        if isinstance(o,dict) and [isinstance(x,tuple) for x in o.keys()].any():

            return {"|".join(x):y for x,y in o.items()}

        if isinstance(o,pd.Timestamp):

            return o.strftime("%Y-%m-%d %H:%M:%S %Z")

        if isinstance(o,pd.Index):

            return o.values.tolist()

        if isinstance(o,io.TextIOWrapper):

            return {
                "type": "file",
                "name": o.name,
                "readable": o.readable(),
                "writable": o.writable(),
                "position": o.tell(),
                }

        return json.JSONEncoder.default(self, o)

# pylint: disable=invalid-name
def PypowerModelDecoder(data):
    """Convert JSON data back to pypower_sim data"""

    # pylint: disable=too-many-return-statements

    if isinstance(data,dict):

        # special encoding for PPModel data types
        if "type" in data:
            match data["type"]:

                case "file":

                    assert "name" in data, f"{data=} missing 'name'"
                    assert "readable" in data, f"{data=} missing 'readable'"
                    assert "writable" in data, f"{data=} missing 'writable'"
                    assert "position" in data, f"{data=} missing 'position'"
                    name = data["name"]
                    mode = None
                    if data["readable"]:
                        mode = "r"
                    elif data["writable"]:
                        mode = "a+" if mode == data["readable"] else "a"
                    pos = data["position"]
                    return open(name,mode,encoding="utf-8").seek(pos)

                case "array":

                    assert "dtype" in data, f"array {data=} missing 'dtype'"
                    dtype = data["dtype"]
                    assert dtype in dir(np), f"array {data=} 'dtype' is not valid"
                    assert "data" in data, f"array {data=} missing 'data'"
                    return np.array(data["data"],dtype=getattr(np,dtype))

                case "dataframe":

                    assert "frame" in data, f"dataframe {data=} missing 'frame'"
                    frame = data["frame"]
                    assert "keys" in frame["index"], f"dataframe {data=} missing frame 'keys'"
                    assert "dtype" in frame["index"], f"dataframe {data=} missing frame 'dtype'"
                    assert "name" in frame["index"], f"dataframe {data=} missing frame 'name'"
                    ndx = frame["index"]
                    assert ndx["dtype"] == "datetime64[ns, UTC]", \
                        f"dataframe {data=} index is not datetime64[ns, UTC]"
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

                    raise ValueError(f"type={data['type']} is not recognized")

        return {x:PypowerModelDecoder(y) for x,y in data.items()}

    if isinstance(data,list):

        return [PypowerModelDecoder(x) for x in data]

    return data
