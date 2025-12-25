# pylint: disable=line-too-long
"""PyPOWER graph analysis module

The `pypower_sim.ppgraph` analysis module provides graph theoretic analysis tools for 
use with `pypower_sim` algorithms that require them.

# Description

The `pypower_sim.ppgraph.PPGraph` class implements methods to generate graph
analysis matrices such as the degree, adjacency, incidence, and Laplacian.
The module also supports spectral analysis to identify important network
properties such as graph Laplacian eigenvalues and the number of islands.

# Examples

If we define the Wheatstone network as the [`pypower`](https://eudoxys.com/pypower) case

    wheatstone = {
        "version": 2,
        "baseMVA": 100.0,
        "bus": array([
            [0, 3, 50,  30.99,  0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            [1, 1, 170, 105.35, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            [2, 1, 200, 123.94, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            [3, 2, 80,  49.58,  0, 0, 1, 1, 0, 230, 1, 1.1, 0.9]
            ]),
        "gen": array([
            [3, 318, 0, 100, -100, 1.02, 100, 1, 318, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0,   0, 100, -100, 1,    100, 1, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]),
        "branch": array([
            [0, 1, 0.01008, 0.0504, 0.1025, 250, 250, 250, 0, 0, 1, -360, 360],
            [0, 2, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 1, -360, 360],
            [1, 2, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 1, -360, 360],
            [1, 3, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 1, -360, 360],
            [2, 3, 0.01272, 0.0636, 0.1275, 250, 250, 250, 0, 0, 1, -360, 360]
            ]),
        }

then we have the following graph analysis results.

- The degree matrix is obtained using

        D = graph.degree()

  which `print(D.todense())` outputs as

        [[2. 0. 0. 0.]
         [0. 3. 0. 0.]
         [0. 0. 3. 0.]
         [0. 0. 0. 2.]]

- The adjacency matrix is obtained using

        A = graph.adjacency()

  which `print(A.todense())` outputs as

        [[0. 1. 1. 0.]
         [1. 0. 1. 1.]
         [1. 1. 0. 1.]
         [0. 1. 1. 0.]]

- The weighted incidence matrix is obtained using

        W = graph.incidence(weighted=True)

  which `print(W.todense())` outputs as

        [[ 3.41112845-2.79645642j  3.97046957-3.25500645j  0.        +0.j          0.        +0.j          0.        +0.j        ]
         [-3.41112845+2.79645642j  0.        +0.j          3.97046957-3.25500645j  3.97046957-3.25500645j  0.        +0.j        ]
         [ 0.        +0.j         -3.97046957+3.25500645j -3.97046957+3.25500645j  0.        +0.j          3.0365804 -2.48940046j]
         [ 0.        +0.j          0.        +0.j          0.        +0.j         -3.97046957+3.25500645j -3.0365804 +2.48940046j]]

- The graph Laplacian matrix is obtained using

        G = graph.laplacian(weighted=True)

  which `print(G.todense())` outputs as

        [[ 8.98519044-44.92595218j -3.81562882+19.07814408j -5.16956162+25.84780811j  0.         +0.j        ]
         [-3.81562882+19.07814408j 14.15475206-70.77376029j -5.16956162+25.84780811j -5.16956162+25.84780811j]
         [-5.16956162+25.84780811j -5.16956162+25.84780811j 13.3628291 -66.81414548j -3.02370585+15.11852927j]
         [ 0.         +0.j         -5.16956162+25.84780811j -3.02370585+15.11852927j  8.19326748-40.96633738j]]

- The spectral analysis results are obtained using

        S = graph.spectral()

  which `print(f"{S.E=}",f"{S.U=}",f"{S.K=}",sep="\n")` outputs as

        S.E=array([ 0.         +0.j        ,  8.25458374-41.2729187j , 17.12360749-85.61803747j, 19.31784783-96.58923916j])
        S.U=matrix([[ 0.5       +0.j,  0.5       +0.j,  0.5       +0.j,  0.5       +0.j],
                [-0.65950436-0.j,  0.08066363+0.j, -0.15274424-0.j,  0.73158497+0.j],
                [-0.56128403-0.j,  0.34531682+0.j,  0.62875016+0.j, -0.41278295-0.j],
                [-0.00377318-0.j,  0.79009472+0.j, -0.57562351+0.j, -0.21069803-0.j]])
        S.K=1

# Caveat

The result of graph analyses are cached in case they are requested more than
once. The cached values are based on the network bus and branch data at the
time the cache was refreshed. If the bus or branch values have changed since
the last refresh, then the `refresh=True` option should be included in the
request, e.g.

    graph.incidence(refresh=True)

which forces recalculation of all the graph analysis results by clearing the
cache of the outdated results from previous requests.
"""
# pylint: enable=line-too-long

from collections import Counter, namedtuple
from typing import TypeVar
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp

class PPGraph:
    """Network graph analysis class implementation"""
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
        model:TypeVar('pypower_sim.ppmodel.PPModel')
        ):
        """Construct graph analysis tools for a an $N$ bus $\\times$ $M$
        branch network model
        """
        self.model = model
        """Model object (`pypower_sim.ppmodel.PPModel`)"""

        # pylint: disable=invalid-name
        self.A = None # admittance matrix
        """Admittance matrix (`numpy.csr_matrix`)"""

        self.B = None # unweighted incidence matrix
        """Unweighted incidence matrix (`numpy.csr_matrix`)"""

        self.D = None # degree matrix
        """Degree matrix (`numpy.csr_matrix`)"""

        self.G = None # (weighted) graph Laplacian matrix
        """Graph Laplacian matrix (`numpy.csr_matrix`)"""

        self.L = None # (unweighted) Laplacian matrix
        """Laplacian matrix (`numpy.csr_matrix`)"""

        self.M = None
        """Number branches (`int`)"""

        self.N = None
        """Number of busses (`int`)"""

        self.S = None # spectral analysis results
        """Spectral analysis results (`collections.namedtuple('spectral',['E','U','K'])`)"""

        self.W = None # weighted incidence matrix
        """Weighted incidence matrix (`numpy.csr_matrix`)"""

        self.Z = None # impedance matrix
        """Branch impedance matrix (`numpy.nd_array`)"""


        self.refresh()

    def refresh(self):
        """Recompute all data analytics from source model

        This should be called after changes to the model bus or branch data.
        """

        # bus data
        self.bus = self.model.get_data("bus")
        self.bus_i = {y:x for x,y in self.bus["BUS_I"].to_dict().items()}
        self.N = len(self.bus)

        # branch data
        self.branch = self.model.get_data("branch")
        self.branch_ij = [(self.bus_i[x],self.bus_i[y])
            for x,y in self.branch[["F_BUS","T_BUS"]].values
            ]
        self.M = len(self.branch)

        # clear cache
        self.A = None # admittance matrix
        self.B = None # unweighted incidence matrix
        self.D = None # degree matrix
        self.G = None # (weighted) graph Laplacian matrix
        self.L = None # (unweighted) Laplacian matrix
        self.S = None # spectral analysis results
        self.W = None # weighted incidence matrix
        self.Z = None # impedance matrix

    def degree(self,
        refresh:bool=False,
        ) -> sp.csr_matrix:
        r"""Compute the degree matrix

        # Arguments

        - `refresh`: force regeneration of the matrix

        # Returns

        - `scipy.csr_matrix`: degree matrix

        # Description

        The degree matrix $D \in \mathbb R^{N \times N}$ is defined as the
        diagonal matrix having the elements $D_{ij} = \deg(v_i)$ when $i=j$
        where $\deg(v_i)$ is the count of branches that connect to or from
        the bus $i$, and all the elements $D_{ij} = 0$ when $i \ne j$.
        """
        if refresh:
            self.refresh()
        if not self.L is None:
            return self.D

        # count references to each bus
        counts = Counter([self.bus_i[x] for x in self.branch[["F_BUS","T_BUS"]].values.flatten()])

        # create diagonal matrix from counts
        self.D = sp.diags([counts[x] for x in range(self.N)])

        return self.D.tocsr()

    def adjacency(self,
        refresh:bool=False,
        ) -> sp.csr_matrix:
        r"""Compute the adjacency matrix

        # Arguments

        - `refresh`: force regeneration of the matrix

        # Returns

        - `scipy.csr_matrix`: adjacency matrix

        # Description

        The adjacency matrix $A \in \mathbb R^{N \times N}$ is non-negative
        symetric matrix where each element $A_{ij}$ is 1 when there is branch
        connecting bus $i$ and bus $j$ and a 0 otherwise.
        """
        if refresh:
            self.refresh()
        elif not self.L is None:
            return self.A


        self.A = sp.lil_matrix((self.N,self.N))
        for fbus,tbus in self.branch_ij:
            self.A[fbus,tbus] += 1
            self.A[tbus,fbus] += 1

        return self.A.tocsr()

    def impedance(self,
        refresh:bool=False,
        part:str|None=None,
        ):
        """Get network branch impedances

        # Arguments
        
        - `refresh`: force recalculation of previous results
        
        - `part`: get complex part `{'r','i','m','a','d'}
    
        # Returns
        
        - `np.ndarray`: branch impedance vector

        # Description
        
        Return the branch impedances for closed branches or zero for open branches.
        """        
        if refresh:
            self.refresh()
        elif self.Z is None:
            self.Z = np.array([(complex(x,y) if status == 1 else 0j)
                for x,y,status in self.branch[["BR_R","BR_X","BR_STATUS"]].values])

        match part:
            case 'r':
                return self.Z.real
            case 'i':
                return self.Z.imag
            case 'm':
                return np.abs(self.Z)
            case 'a':
                return np.angle(self.Z)
            case 'd':
                return np.angle(self.Z) * 180 / np.pi
            case None:
                return self.Z
            case '_':
                raise ValueError(f"{part=} is invalid")

    def incidence(self,
        refresh:bool=False,
        complex_flows:bool=True,
        weighted:bool=False,
        ) -> sp.csr_matrix:
        r"""Get network indicidence matrix

        # Arguments
        
        - `refresh`: force recalculation of previous results
        
        - `complex_flows`: if true produces incidence matrix using admittance
          instead of conductance (weighted only)
    
        - `weighted`: generated weighted incidence based on square root of
          line admittance (inverse of impedance $R+jX$)
    
        # Returns
        
        - `scipy.csr_matrix`: incidence matrix

        # Description

        The incidence matrix $B \in \mathbb R^{N \times M}$ has the element
        $B_{ij} = 1$ when the node $i$ in incident with the branch $j$, and 0
        otherwise. The weighted incidence matrix uses the value $\frac 1
        {R_j+jX_j}$ instead of 1 for closed branches and 0 for open
        branches.
        """
        if refresh:
            self.refresh()
        elif weighted and not self.W is None:
            return self.W
        elif not weighted and not self.B is None:
            return self.B

        fbus,tbus = zip(*self.branch_ij)

        if weighted:

            Z = np.array([(1/x if np.abs(x)!=0 else 0) for x in self.impedance()])
            if not complex_flows:
                Z = Z.real
            self.W = (sp.csr_matrix((Z,[range(self.M),fbus]),shape=(self.M,self.N)) -\
                sp.csr_matrix((Z,[range(self.M),tbus]),shape=(self.M,self.N))).T
            return self.W

        Z = self.branch["BR_STATUS"].values
        self.B = (sp.csr_matrix((Z,[range(self.M),fbus]),shape=(self.M,self.N)) -\
            sp.csr_matrix((Z,[range(self.M),tbus]),shape=(self.M,self.N))).T
        return self.B

    def laplacian(self,
        refresh:bool=False,
        weighted:bool=False,
        complex_flows:bool=False,
        ) -> sp.csr_matrix:
        r"""Get network graph Laplacian

        # Arguments

        - `refresh`: force recalculation of previous results

        - `weighted`: return graph Laplacian instead of Laplacian matrix

        - `complex_flows`: use incidence matrix with complex flow values
          (weighted only)
    
        # Returns

        - `scipy.csr_matrix`: (graph) Laplacian matrix

        # Description

        The (unweighted) Laplacian matrix $L \in \mathbb R^{N \times N}$ is
        defined as $L = D - A$, where $D$ is the degree matrix
        (see `pypower_sim.ppgraph.PPGraph.degree`) $A$ is the adjacency
        matrix(see `pypower_sim.ppgraph.PPGraph.adjacency`). The
        (weighted) graph Laplacian matrix is defined as $G = B~Y~B^T$, where
        $B$ is the unweighted incidence matrix
        (see `pypower_sim.ppgraph.PPGraph.incidence`) and $Y=1/Z$ is the
        branch admittance vector (see `pypower_sim.ppgraph.impedance`).
        """
        if refresh:
            self.refresh()
        elif weighted and not self.G is None:
            return self.G
        elif not weighted and not self.L is None:
            return self.L

        if weighted:

            # pylint: disable=invalid-name
            Y = sp.diags([(1/x if np.abs(x)!=0 else 0) for x in self.impedance()])
            if not complex_flows:
                Y = Y.real
            B = self.incidence(weighted=False)
            self.G = B @ Y @ B.T
            return self.G.tocsr()

        self.L = self.degree() - self.adjacency()
        return self.L.tocsr()

    def spectral(self,
        refresh:bool=False,
        precision:int=8,
        weighted:bool=False,
        complex_flows:bool=False,
        ) -> TypeVar("namedtuple('spectral',['E','U','K']"):
        """Get spectral analysis results

        # Arguments
        
        - `refresh`: force recalculation of previous results

        - `precision`: rounding of eigenvalues and eigenvector (default 8)

        - `weighted`: use graph Laplacian instead of Laplacian matrix

        - `complex_flows`: use incidence matrix with complex flow values
          (weighted only)
    
        # Returns
        
        - `namedtuple("spectral",["E","U","K"]`: Named tuple `spectral
          (E,U,K)` where `E` is the eigenvalues, `U` is the eigenvectors, and
          `K` is the number of independent networks (islands) found.

        # Description

        The spectral analysis of the network computes the eigenvalues $E$,
        eigenvector $U$, and count of islands $K$ using eigenvalue analysis.
        The number of islands in the network is the count of zero eigenvalues.

        """
        if refresh:
            self.refresh()
        elif not self.S is None:
            return self.S

        # pylint: disable=invalid-name
        G = self.laplacian(weighted=weighted,complex_flows=complex_flows)

        e,u = la.eig(G.todense())

        # sort eigenvalues and eigenvectors
        i = (e.real).argsort() # index of sorted real parts
        E,U = e[i].round(precision),u.T[i].round(precision)

        # count islands
        K = sum(1 if x==0 else 0 for x in np.abs(e[i]).round(precision))

        self.S = namedtuple('spectral',['E','U','K'])(E,U,K)
        return self.S

if __name__ == "__main__":

    from numpy import array
    wheatstone = {
        "version": '2',
        "baseMVA": 100.0,
        "bus": array([
            [0, 3, 50,  30.99,  0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            [1, 1, 170, 105.35, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            [2, 1, 200, 123.94, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            [3, 2, 80,  49.58,  0, 0, 1, 1, 0, 230, 1, 1.1, 0.9]
            ]),
        "gen": array([
            [3, 318, 0, 100, -100, 1.02, 100, 1, 318, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0,   0, 100, -100, 1,    100, 1, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]),
        "branch": array([
            [0, 1, 0.01008, 0.0504, 0.1025, 250, 250, 250, 0, 0, 1, -360, 360],
            [0, 2, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 1, -360, 360],
            [1, 2, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 0, -360, 360],
            [1, 3, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 1, -360, 360],
            [2, 3, 0.01272, 0.0636, 0.1275, 250, 250, 250, 0, 0, 1, -360, 360]
            ]),
        }

    # pip install -e ..
    from pypower_sim.ppmodel import PPModel
    test = PPModel(case=wheatstone)

    graph = PPGraph(test)

    np.set_printoptions(edgeitems=3,linewidth=10000)

    # print(graph.incidence(weighted=True))

    S = graph.spectral(weighted=False,complex_flows=False)
    print("*** Wheatstone ***",f"{S.E=}",f"{S.U=}",f"{S.K=}",sep="\n")

    # pip install git+https://github.com/eudoxys/wecc240
    from wecc240.wecc240_2011 import wecc240_2011 as wecc240
    test = PPModel(case=wecc240)

    graph = PPGraph(test)

    np.set_printoptions(edgeitems=3,linewidth=10000)

    S = graph.spectral()
    print("","*** WECC 240 ***",f"{S.E=}",f"{S.U=}",f"{S.K=}",sep="\n")
