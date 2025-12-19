# import os
# import pandas as pd


# dgen = pd.read_csv("../examples/wecc240_dgen.csv").set_index(["STATE","YEAR","MONTH"]).drop("STATUS",axis=1).sort_index()


# load = None
# for file in os.listdir("../examples/"):
#     if file.startswith("ca-") and file.endswith(".csv"):
#         pv = pd.read_csv(f"../examples/{file}")
#         pv.index = pd.date_range(start="2018-01-01 00:00:00",end="2018-12-31 23:59:59",freq="15min")
#         pv = pv['out.electricity.pv.energy_consumption'].to_frame().resample("1h").sum()
#         if load is None:
#             load = -pv
#         else:
#             load += -pv            

# load.columns = ["RES DPV"]

# print("WECC240 DPV")
# print(f"EIA861 CA 2018: {dgen.loc[("CA",2018),"RES_MWH"].sum()/1e6:.2f} TWH")
# print(f"RESSTOCK CA 2018: {-load['RES DPV'].sum()/1e9:.2f} TWH")

# res_dpv = (dgen.loc[("CA",2018),"RES_MW"]/1e3)
# res_dpv.index = pd.DatetimeIndex(f"2018-{x:02d}-01 00:00:00" for x in res_dpv.index)

# (load/1e6).plot(label="RESSTOCK",legend=True)
# fig = res_dpv.plot(label="EIA 861",
#     legend=True,
#     grid=True,
#     xlabel="Date/Time",
#     ylabel="GW",
#     title="2018 CA DPV"
#     ).figure
# fig.savefig("res_dpv.png")