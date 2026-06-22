"""Calculate powerflow transfer factors

A PFTF is a matrix that provides the fractional change in powerflow on each
line as a result of an outage of a line in the network. Each row represents a
line outage and each column represents the fraction change in powerflow on
the remaining lines.

Note: a row with all zeros indicates that powerflow solution failed. This
does mean that there is a blackout. It only means that the solver cannot find
the solution.
"""

import os
import warnings

import numpy as np
import pandas as pd

pd.options.display.width = None
pd.options.display.max_columns = None


from pypower_sim import PPModel, PPSolver
from pypower_sim.ppmodel import idx_branch as branch

test = PPModel("case9",case="case9.py")

solver = PPSolver(test)

test.case["branch"][:,branch.BR_STATUS] = 1.0
assert solver.solve_pf(), "original case does not solve"
result = test.get_data("branch")
baseflow = result["PF"].values
maxflow = result["RATE_A"].values

Nlines = len(test.case["branch"])

pftf = np.zeros((Nlines,Nlines))
testflow = np.zeros((Nlines,Nlines))
maxflow = np.vstack([[maxflow] for x in range(Nlines)])

for n in range(Nlines):
	test.case["branch"][:,branch.BR_STATUS] = 1.0
	test.case["branch"][n,branch.BR_STATUS] = 0.0
	try:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			if solver.solve_pf():
				result = test.get_data("branch")
				testflow[n,:] = result["PF"].values
				pftf[n,:] = testflow[n,:] / baseflow
	except:
		pass # no solution

# print(testflow.round(2))
# print(maxflow.round(1))
print(pftf.round(4))
if (testflow>maxflow).any():
	print("Line RATE_A violations detected!")
	print(testflow>maxflow)
else:
	print("No line rating violations detected")



