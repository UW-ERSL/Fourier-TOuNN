import numpy as np
import torch

def computeExtrusionMap(mesh, extrusion):
    if(mesh['type'] != 'grid'):
        return extrusion
    nelx, nely = mesh['nelx'], mesh['nely']
    if(extrusion['X']['isOn']):
         extrMap = np.zeros((nelx*nely,nelx*nely))
         extrElems = (1./nelx)*np.ones((nelx,nelx))
         for ey in range(nely):
             idxs = np.arange(ey, nelx*nely, nely)
             for rw in idxs:
                 for col in idxs:
                     extrMap[rw,col] =  (1./nelx)
         extrusion['X']['map'] = torch.tensor(extrMap).float()
                    
    if(extrusion['Y']['isOn']):
        extrMap = np.zeros((nelx*nely,nelx*nely))
        extrElems = (1./nely)*np.ones((nely,nely))
        for ex in range(nelx):
            extrMap[ex*nely:nely*(ex+1), ex*nely:nely*(ex+1)] = extrElems

        extrusion['Y']['map'] = torch.tensor(extrMap).float()
    return extrusion

def applyExtrusion(density, extrusion):
    if(extrusion['Y']['isOn']):
        extrDensity = torch.einsum('ij,j->i',extrusion['Y']['map'], density)
    elif(extrusion['X']['isOn']): # note only one map can be on at a time
        extrDensity = torch.einsum('ij,j->i',extrusion['X']['map'], density)
    else:
        extrDensity = density
    return extrDensity
# nelx, nely = 4, 4
# mesh = {'nelx':5, 'nely':5}
# extrusion = {'X':{'isOn':True},\
#              'Y':{'isOn':True} }
# extrusion = computeExtrusionMap(mesh, extrusion)

