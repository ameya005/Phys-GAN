"""
Finite element solvers for the displacement from stiffness matrix and force
vector. This version of the code is meant for global compliance minimization.
"""

import numpy as np
from scipy.sparse import coo_matrix

# Importing linear algabra solver for the SciPyFEA class
from scipy.sparse.linalg import spsolve



def ForwardTopo(Image:np.ndarray, load:list, fixity:list, E=1.0, nu=0.3, penal=3):
    
    def _node(nelx, nely, elx, ely):
        return (nely+1)*elx + ely
    
    def _nodes(nelx, nely, elx, ely):
        n1 = _node(nelx, nely, elx, ely)
        n2 = _node(nelx, nely, elx + 1, ely)
        n3 = _node(nelx, nely, elx + 1, ely + 1)
        n4 = _node(nelx, nely, elx, ely + 1)
        return n1, n2, n3, n4
    
    # element (local) stiffness matrix
    def ke(E=1, nu=0.3):
        """
        Calculates the local siffness matrix depending on E and nu.
        Parameters
        ---------
        E : float
            Youngs modulus of the material.
        nu : float
            Poisson ratio of the material.
        Returns
        -------
        ke : 2-D array size(8, 8)
            Local stiffness matrix.
        """
        k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        ke = E/(1-nu**2) * \
            np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                      [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                      [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                      [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                      [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                      [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                      [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                      [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
        return ke

    def _edof(nelx, nely):
        elx = np.repeat(range(nelx), nely).reshape((nelx*nely, 1))  # x position of element
        ely = np.tile(range(nely), nelx).reshape((nelx*nely, 1))  # y position of element
        n1, n2, n3, n4 = _nodes(nelx, nely, elx, ely)
        edof = np.array([dim*n1, dim*n1+1, dim*n2, dim*n2+1,
                         dim*n3, dim*n3+1, dim*n4, dim*n4+1])
        return edof

    def _compile_gk(Image, fixity, nelx, nely, dim, penal=3, kmin=1e-9):
        edof = _edof(nelx, nely)
        edof = edof.T[0]

        x_list = np.repeat(edof, 8)  # flat list pointer of each node in an element
        y_list = np.tile(edof, 8).flatten()  # flat list pointer of each node in element

        kd = Image.T.reshape(nelx*nely, 1, 1) ** penal  # knockdown factor

        value_list = ((np.tile(kmin, (nelx*nely, 1, 1)) + np.tile(ke()-kmin, (nelx*nely, 1, 1))*kd)).flatten()

        alldofs = [x for x in range(dim*(nely+1)*(nelx+1))]
        n= _node(nelx, nely, fixity[0], fixity[1])
        fixdofs = [dim*n, dim*n + 1]
        freedofs = list(set(alldofs) - set(fixdofs))

        dof = dim*(nelx+1)*(nely+1)
        k = coo_matrix((value_list, (y_list, x_list)), shape=(dof, dof)).tocsc()
        return k, np.array(freedofs)

    if len(Image.shape) != 2:
        raise ValueError('The image needs to be single channel but flattened to a bitmap')

    dim = len(Image.shape)
    nelx = Image.shape[0]
    nely = Image.shape[1]

    if load[0] > nelx or load[1] > nely or load[0] < 0 or load[1] < 0:
        raise ValueError('load location must be within the image space')

    if fixity[0] > nelx or fixity[1] > nely or fixity[0] < 0 or fixity[1] < 0:
        raise ValueError('fixity location must be within the image space')

    K, freedofs = _compile_gk(Image, fixity, nelx, nely, dim)
    force = np.zeros(dim*(nely+1)*(nelx+1))
    n = _node(nelx, nely, load[0], load[1])
    force[n] = load[2]
    force[n+1] = load[3]
    f_free = force[freedofs]
    K_free = K[freedofs, :][:, freedofs]

    # solving the system f = Ku with scipy
    u = np.zeros(dim*(nely+1)*(nelx+1))
    u[freedofs] = spsolve(K_free, f_free)

    edof = _edof(nelx, nely)
    ue = u[edof].T  # list with the displacements of the nodes of that element

    # calculating the compliance in 3 steps
    dot = np.dot(ke(), ue.reshape((nelx*nely, 8, 1)))
    ce = np.sum(ue.T*dot, axis=0).reshape(nelx,nely)  # element compliance
    c = np.sum(ce * Image**penal)  # total compliance
    return ce, c

def BackwardTopo(Image, Ce, penal=3):
    if len(Image.shape) != 2 or len(Ce.shape) != 2 :
        raise ValueError('The image needs to be single channel but flattened to a bitmap')
   
    Derivative = -penal * Image **(penal - 1) * Ce

    return Derivative



if __name__ == '__main__':
    density_matrix = np.ones((1600,400),dtype='float')
    load = [1599, 399, -1, -1]
    fixity =  [799, 199]

    Ce, C = ForwardTopo(density_matrix, load, fixity)
    Dc = BackwardTopo(density_matrix, Ce)

    import matplotlib.pyplot as plt
    plt.imshow(Ce, cmap='jet')