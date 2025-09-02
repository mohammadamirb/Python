#@title _- Necessary packages to import -_
####################################################################
# Created By M.B @ 06-06-2025 for the joint work with Prof. T.D.K
# v2 01-07-2025
# v1 06-06-2025

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.linalg import eig,eigvals
import matplotlib.pyplot as plt
from typing import Sequence
import re

#@title _- Functions for the atomic structure -_

def UC(rux, ruy,acc):                                                               # This function creates the xyz coordinate for a perfect unit cell.
  # :: B_CONSTANT ::
  dx, dy = 0.5*acc*np.sqrt(3), 3*acc/2
  # :: E_CONSTANT ::

  n = rux*2                                                                     # number of atoms along a dimer line.
  tot_a = n*ruy                                                                 # Total number of atoms in the unit cell.
  var1 = np.arange(1, tot_a+1)                                                  # index of all atoms in UC00.
  UC00 = var1.reshape(ruy, n)                                                   # index of atoms in UC00, reshaped so that each dimer line (along zigzag direction) is in a row.
  xcoor = np.arange(0, dx*n, dx)
  xcoor = np.tile(xcoor, ruy)                                                   # Repeating the xcoor to match the size of n.
  y1, y2 = 0, acc/2
  ycoor = np.zeros((tot_a, 1))
  for ii in range(ruy):
    var = UC00[ii,:]
    if (ii % 2 == 0):
        swidx = 1 # switch index, if it is 1, then the first row is considered!
        idx1 = var[var % 2 != 0] -1                                             # These indices are corrected (I mean "-1").
        idx2 = var[var % 2 == 0] -1
    else:
        idx1 = var[var % 2 == 0] -1
        idx2 = var[var % 2 != 0] -1
    ycoor[idx1]=y1
    ycoor[idx2]=y2
    y1, y2 = y1 + dy, y2 + dy

  zcoor = np.zeros((tot_a, 1))                                                  # Just for CIF file!
  coords = np.column_stack((xcoor, ycoor, zcoor))
  xmm = np.array( [np.min(coords[:, 0]), np.max(coords[:, 0])] )
  ymm = np.array( [np.min(coords[:, 1]), np.max(coords[:, 1])] )
  al, bl = np.diff(xmm)+dx, np.diff(ymm)+acc                                    # a and b vector lengthes.
  acoord = np.array([ [0-dx/2,xmm[1]+dx/2],[0-acc/2, 0-acc/2] ])                # Coordination of the two points of vector a, ([x0 x1 y0 y1]).
  bcoord = np.array([ [0-dx/2,0-dx/2],[0-acc/2, ymm[1]+acc/2] ])                # Coordination of the two points of vector b, ([x0 x1 y0 y1]).
  return coords, [al,bl], np.hstack( (acoord,bcoord) )                          # XYZ Coordinates, a and b vectors, the coordinates of the unit cell corners.

def sublatt(xyz):                                                               # This function indicates the sublattice of the atoms. Note that the output contains only indices of the sublattices, not the coordinates. Also this function only works for the unit cell 0.
  all_idx = np.arange(xyz.shape[0])
  all_idx_mat = all_idx.reshape(ruy, rux*2)
  AA, A_Sub = [], []

  for ii in np.arange(ruy):
      if np.mod(ii,2) == 0:
          A_Sub = all_idx_mat[ii, np.mod(all_idx_mat[ii,:],2)!=0]
      elif np.mod(ii,2) == 1:
          A_Sub = all_idx_mat[ii, np.mod(all_idx_mat[ii,:],2)==0]
      AA.append( A_Sub )

  AA = (np.array(AA)).reshape(-1)                                               # Index of atoms on A sublattice.
  BB = np.delete(all_idx, AA)                                                   # Index of atoms on B sublattice.
  return AA, BB

def Vac_gen(xyz,nm):                                                            # This function creates vacancies.
  # IMPORTANT : input must be a list, e.g., [[1,1],[2,5],...]
  nm_del = np.array(nm)-1                                                       # Index correction for Python.
  cell_n = rux*2                                                                # n of the unit cell. See the text.
  A,B = sublatt(xyz)                                                            # Identifying the sublattices
  if nm_del.shape[0] > 1:
    at_del_idx = np.sum(np.column_stack( (nm_del[:,0] , nm_del[:,1]*cell_n ) ),1)
    if np.isin(A,at_del_idx).sum()==len(at_del_idx) or np.isin(B,at_del_idx).sum()==len(at_del_idx):
      ModelName = 'SS'
    else:
      ModelName= 'DS'
    spacings = np.diff(np.append( nm_del[:,0], cell_n+nm_del[0,0] ))-1          # Spacing between SVs base on n counting
    spacings[spacings == 1] = 0                                                 # Removing values of 1 (1AGNR is not considered)
    for ii in range(nm_del.shape[0]):
      ModelName += '_' + str(spacings[ii])
  else:
    at_del_idx = nm_del[0][0] + nm_del[0][1]*cell_n                             # Is nothing but n + m*n.
    spacings = cell_n-1
    if spacings == 1:
      spacings = 0
    ModelName = 'SS_' + str(spacings)

  print('You have a :', ModelName)
  out = np.delete(xyz, at_del_idx, axis=0)                                      # xyz coordinates of the atoms with vacancies.
  out_A = np.delete(A, np.isin(A,at_del_idx) )                                  # xyz coordinates of AA atoms with vacancies.
  out_B = np.delete(B, np.isin(B,at_del_idx) )                                  # xyz coordinates of BB atoms with vacancies.
  return out, [xyz[out_A,:], xyz[out_B]], ModelName

def distance(xyz1,xyz2):
  tot_a = np.shape(xyz1)[0]
  dist = np.zeros( (tot_a, tot_a) )
  for ii in range(tot_a):
    for jj in range(tot_a):
      dist[ii,jj] = np.sqrt( (xyz1[ii,0]-xyz2[jj,0])**2 + (xyz1[ii,1]-xyz2[jj,1])**2)
  return dist

def N_cell(xyz,vecs):                                                           # This function generates all the coordinates of all the 8 neighboring unit cells.
  a, b = vecs[0], vecs[1]
  tot = np.shape(xyz)[0]
  NC = np.zeros((tot,3,9))
  xcoor, ycoor, zcoor = xyz[:,0], xyz[:,1], xyz[:,2]
  NC[:,:,0]=np.column_stack((xcoor+a, ycoor, zcoor))
  NC[:,:,1]=np.column_stack((xcoor+a, ycoor+b, zcoor))
  NC[:,:,2]=np.column_stack((xcoor, ycoor+b, zcoor))
  NC[:,:,3]=np.column_stack((xcoor-a, ycoor+b, zcoor))
  NC[:,:,4]=np.column_stack((xcoor-a, ycoor, zcoor))
  NC[:,:,5]=np.column_stack((xcoor-a, ycoor-b, zcoor))
  NC[:,:,6]=np.column_stack((xcoor, ycoor-b, zcoor))
  NC[:,:,7]=np.column_stack((xcoor+a, ycoor-b, zcoor))
  NC[:,:,8]=np.column_stack((xcoor, ycoor, zcoor))                              # For convinience, the central unit cell is added to the end!
  return NC

def cifout(xyz,latvec,fn=None):                                                 # This function create a CIF file.
  leng = xyz.shape[0]
  a, b, c = latvec[0][0], latvec[1][0], 15                                      # Defining lattice vectors, a 15 Angstrom along c(z) is considered here.
  cif_hd = f"""# Crystal Data | The file is created by SV_TB python code, written by M.B. .
data_CsCode_Phase_1
_chemical_name_common         global
_cell_length_a                {a}
_cell_length_b                {b}
_cell_length_c                {15}
_cell_angle_alpha             90.0
_cell_angle_beta              90.0
_cell_angle_gamma             90.0
_cell_volume                  {a * b * c}
_space_group_name_H-M_alt     'P 1'
_space_group_IT_number        1

loop_
_space_group_symop_operation_xyz
  'x, y, z'

loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_occupancy
"""

  cl01 = ['C']
  cl01 = np.array( cl01*leng ).reshape(leng, 1)

  cl02 = np.round (xyz[:,0].reshape(leng, 1)/a, 6)                              # make x coordinate fractional, and round it to 6 decimals.
  cl03 = np.round (xyz[:,1].reshape(leng, 1)/b, 6)                              # make y coordinate fractional, and round it to 6 decimals.
  cl04 = xyz[:,2].reshape(leng, 1)

  exyz = np.hstack( (cl01, cl02, cl03, cl04) )                                  # Element, x, y, and z coordinates in fractional coordinates
    #x = " ".join(map(str, x))
    #x = ["{}\n".format(i) for i in x]
    #print(   x   )
  if fn is None:
    fn = f"{mname}_{rux*2}-{ruy}_{ int(100*round(np.random.rand(),3)) }"        # Generating a file name : ModelName_CellSize_A 2-digit random number if the file name is not specified.

  with open(f"{fn}.cif", "w") as f:
    f.write(cif_hd)
    for ii in range(leng):
      row = exyz[ii]
      label = f"{row[0]}{ii+1}"
      f.write("{:<5} {:<3} {:<15} {:<15} {:<5} {}\n".format(label, row[0], row[1], row[2], row[3], '1.0'))
  print( 'Data is saved in',fn,'.cif')

#@title _- Auxiliary functions for plotting -_
def labl(xyz):                                                                  # This function labels atoms, can be useful in checking the integrity of Hamiltonian
  for ii, (x, y) in enumerate(xyz[:,:2]):
    plt.text(x, y, f'{ii+1}', fontsize=12, ha='right', va='bottom', color='black')

def plothlp(xyz, bnd, sps=None, cmdd=None):                                     # This is a plot helper function!
    # xyz coordinates of the atoms, bondlength, scatter plot specification, command = 'noatom'
  if cmdd != 'noatom':
    if sps is None :
        sps = {'color': 'r','s': 100,'edgecolors': 'k', 'linewidths': 0.5,'alpha': 0.8,'marker': 'o'}
    plt.scatter(xyz[:, 0], xyz[:, 1], **sps, zorder=2)
  if bnd != 0:

    if sps is None :
      sps = {'color':[0.5,0.5,0.5,0.5], 'linewidth':0.5}

    dist = distance(xyz,xyz)
    r, c = np.where(np.round(dist/acc,1) == 1)
    bonds = np.column_stack((r, c))

    for ii in range(np.shape(bonds)[0]):                                         # Plotting bonds (NN).
      x = xyz[[bonds[ii,0],bonds[ii,1]],0]
      y = xyz[[bonds[ii,0],bonds[ii,1]],1]
      plt.plot(x, y, **sps, zorder=1)
  plt.gca().set_aspect('equal')

#@title _- Functions for Hamiltonian, v_F, DOS, and band structure -_
def H_Gen(xyz0, xyz1):
# :: B_CONSTANT ::
  t = -2.7                                                                      # Nearest-Neighbor hopping value (eV)
# :: E_CONSTANT ::
  tot_s = xyz0.shape[0]
  dist = distance(xyz0,xyz1)                                                    # Atomic distances are determined here
  H = np.zeros((tot_s, tot_s))
  r, c = np.where(np.round(dist/acc,1) == 1)
  H[r, c] = t
  return H

def Band_Str(kv, H_00, H_01, H_02=None):                                        # This is a function to calculate Bandstructure. If H_02 is not defined, the Hamiltonian is for a quasi-1D structure (GNR)
  tot_s = H_00.shape[0]

  if H_02 is None:
    H_02=np.zeros((tot_s, tot_s))

  E_eig = np.zeros((kv.shape[0], tot_s))
  V_eig = np.zeros((kv.shape[0], tot_s, tot_s), dtype=complex)

  for ii in range(kv.shape[0]):
    H_eff = (
      H_00 +
      H_01 * np.exp(1j * kv[ii, 0]) +
      np.conj(H_01 * np.exp(1j * kv[ii, 0])).T +
      H_02 * np.exp(1j * kv[ii, 1]) +
      np.conj(H_02 * np.exp(1j * kv[ii, 1])).T
    )
    eigenvalues, eigenvectors = eig(H_eff)  # Extract eigenvalues
    srt_idx = np.argsort(eigenvalues.real)                                      # Sorting eigenvalues in ascending order
    E_eig[ii] = eigenvalues.real[srt_idx]                                           # Sorted eigenvalues (Band sorting)
    V_eig[ii] = eigenvectors[:, srt_idx]                                            # Corresponding sorted eigenvectors - [k index-> eigenvalues <-band index]
  return E_eig, V_eig

def v_F(kv, E_eigs, latvec, kvidx, bndidx, nk):
# :: B_CONSTANT ::
  hbar = 1.0545718e-34;
  eV_to_J = 1.60218e-19;
# :: E_CONSTANT ::

  E_eigs=np.round(E_eigs,8)                                                     # Rounding eigenvalues to 8 decimals to avoid numerical problems in calculating v_F.
  nxtpnt = 3                                                                    # Next point in the k-path to assign to the the derivative, 3 is a safe value for large values of nk.
  if ( nk <= 3 ):                                                               # Checking the possibility of the calculation of the Fermi velocity.
    print("Increase nk, it is not possible to do the calculation.")
    raise sys.exit("Execution stopped ...")

  delta_ka = 2 * np.pi / (latvec[0] * nk)
  delta_kb = 2 * np.pi / (latvec[1] * nk)

  if 2*delta_ka > 0.05 or 2*delta_kb > 0.05:                                    # Checking the appropriateness of the sampling in k-space.
    print("WARNING: Higher nk will provide a more accurate estimate of v_F.")
  else:
    print("It seems that the nk is chosen appropriately.")

  if isinstance(kvidx, str) and kvidx.lower() == 'auto':
    print("Calculating v_F automatically at the corners provided")

    tmpvar = np.arange(0, kv.shape[0] + 1, nk)
    forw = np.sort( np.concatenate( [ tmpvar[:-1], tmpvar[:-1] + nxtpnt ] ) )
    backw = np.sort(np.concatenate( [ tmpvar[1:] - nxtpnt, np.append(tmpvar[1:-1], tmpvar[-1] - 1) ] ) )
    kvidx = []
    num_pairs = len(tmpvar)

    for i in range(1,num_pairs):
      kvidx.append(forw[:2])                                                    # Add first 2 elements from forw
      forw = forw[2:]                                                           # Remove them
      kvidx.append(backw[:2])                                                   # Add first 2 from backw
      backw = backw[2:]                                                         # Remove them

  elif isinstance(kvidx, str) and kvidx.lower() != 'auto':
      raise sys.exit("Command is unknown.")

  k = np.zeros(kv.shape)
  k[:, 0] = kv[:, 0] / (latvec[0] * 1e-10)
  k[:, 1] = kv[:, 1] / (latvec[1] * 1e-10)
  k = np.sum(k, axis=1)
  v_F = []
  for idx_pair in kvidx:
    dk = np.abs(np.diff(k[idx_pair]))
    dE = np.abs(np.diff(E_eigs[idx_pair, bndidx]))
    v_F.append(np.abs(dE * eV_to_J) / (hbar * dk))

  v_F = np.array(v_F)
  return np.round(v_F.T)

def DOS_band(E_eigs, sigma=0.01, nPoints=1000):

    E_vals = E_eigs.flatten()

    E_min = E_vals.min()
    E_max = E_vals.max()
    Ev = np.linspace(E_min, E_max, nPoints)

    DOS = np.zeros_like(Ev)

    p_f = 1 / (sigma * np.sqrt(2 * np.pi))                                      # Prefactor
    DOS = np.sum(p_f * np.exp(- (Ev[:, None] - E_vals[None, :])**2 / (2 * sigma**2)), axis=1) / len(E_vals)

    return Ev, DOS

#@title _- BZ and etc. -_
# :: B_CONSTANT ::
acc = 1.42                                                                      # C-C bond length in graphene in Angstrom, defined as a global variable.
# :: E_CONSTANT ::

# Brillouin Zone Points
reclat = np.array([
    [0, 0, 0],
    [np.pi, 0, 0],
    [np.pi, np.pi, 0],
    [0, np.pi, 0],
    [0, 0, 0]
])

bzticlabels = ['$\Gamma$', 'X', 'R', 'Y', '$\Gamma$']                           # Brillouin Zone tick labels

nk = 100                                                                        # Adjust resolution of the wavevectors.
k = np.vstack([
    np.linspace(reclat[ii, :2], reclat[ii+1, :2], nk)                           # It is a 2D structure!
    for ii in range(len(reclat) - 1)
])

ss = [{'color': 'r','edgecolors': 'None','linewidths': 0.5,'alpha': 1,'marker': 'o'},
      {'color': 'b','edgecolors': 'None','linewidths': 0.5,'alpha': 1,'marker': 'o'}]  # Sublattice colors.

#@title __-: Main :-__
# Created by M.B. The code can calculate electronic band structure and the
# Fermi velocity for a defected structutre, base on the manuscripts text.

figcod = int(input('Which figure do you want to plot?'))
#figcod = 3

if figcod==3:
  nm = [ [4,2],[6,2],[8,2] ]
  dlatm_c = [ [np.array([1,1])], [np.array([1,1])], [np.array([1,1])] ]
elif figcod==31:
  nm = [ [6,6] ]
  dlatm_c = [ [np.array([1,1])] ]
elif figcod==4:
  nm = [ [12,4],[10,4],[16,4] ]
  dlatm_c = [ np.array([ [1,1],[4,2] ]), np.array([ [1,1],[7,1] ]), np.array([ [1,1],[6,2],[10,2] ]) ]
elif figcod==41:
  nm = [ [6,2] ]
  dlatm_c = [ np.array([ [1,1],[4,2] ]) ]
elif figcod==42:
  nm = [ [8,2] ]
  dlatm_c = [ np.array([ [1,1],[4,2] ]) ]
elif figcod==5:
  nm = [ [12,4],[12,4] ]
  dlatm_c = [ np.array([ [1,1],[7,2] ]), np.array([ [1,1],[8,1] ]) ]
elif figcod==51:
  nm = [ [6,2] ]
  dlatm_c = [ np.array([ [1,1],[4,1] ]) ]
elif figcod==6:
  nm = [ [12,8],[12,8] ]
  dlatm_c = [ np.array([ [1,1],[7,2] ] ), np.array([ [1,1],[7,4] ]) ]

for ii in range(len(nm)):

  rux, ruy = int(nm[ii][0]/2), int(nm[ii][1])

  coords, vec, ucpoint = UC(rux,ruy,acc)                                        # The unit cell of a perfect structure is created here.
  dl = dlatm_c[ii]
  coor_v, pos_sublat, mname = Vac_gen(coords,dl)                                # Structural point defects are created here.
  NC_v = N_cell(coor_v,vec)                                                     # The coordinates of neighbouring cells are generated here.

  cifname = f"{mname}_{rux*2}-{ruy}"                                            # CIF file name.

  H00 = H_Gen(coor_v, coor_v)                                                   # Forming Hamiltonians.
  H01 = H_Gen(coor_v, NC_v[:,:,0])
  H02 = H_Gen(coor_v, NC_v[:,:,2])

  E, V = Band_Str(k, H00, H01, H02)                                             # Calculating band structutre.
  Ev, DOS = DOS_band(E)                                                         # Calculating DOS.
  plt.figure()
  plt.subplot(1,3,1)
  plothlp(coor_v, acc, sps=None, cmdd='noatom')
  plothlp(pos_sublat[0], 0, ss[0])
  plothlp(pos_sublat[1], 0, ss[1])
  plt.title(f'Unit cell size: ({nm[ii][0]},{nm[ii][1]}) \n {mname}'); plt.ylabel('Y (Å)'); plt.xlabel('X (Å)')

  plt.subplot(1,3,2)

  if "DS" in mname:                                                             # ::> Rough approximaton of the valence band index. Be careful!
    vband = int( (E.shape[1]-np.shape(dl)[0])/2  )                              # Valcence band index, considering Python indexing correction.
  else:
    vband = int( (E.shape[1]-np.shape(dl)[0])/2  ) -1                           # Valcence band index, considering Python indexing correction.

  ktmp = np.arange(0,k.shape[0])
  plt.plot(ktmp,E)
  plt.plot( ktmp,E[:,vband],'o',color=[0.5,0.5,0.5,0.1])                        # Plotting the first valence band, it is important to check if it is selected correct automatically.
  plt.xticks(np.arange(0,1+k.shape[0],nk),bzticlabels)
  plt.ylabel('Energy (eV)'); plt.ylim(-2.5, 2.5); plt.xlim(ktmp[0], ktmp[-1] + 1)

  plt.subplot(1,3,3)
  plt.plot( DOS, Ev,'-b', linewidth=2 )
  plt.ylabel('Energy (eV)'); plt.ylim(-2.5, 2.5); plt.xlabel('DOS (a.u.)')

  plt.tight_layout()
  plt.show()

  #kvidx = [ [0, 1],[k.shape[0] - 2, k.shape[0] - 1] ]                          # Manual k-point selectrion to calculate v_F
  kvidx = 'auto'
  Fermi_Velocity = v_F(k, E, vec, kvidx, vband, nk)                             # Band index is corrected in the function.
  print( Fermi_Velocity )
  cifout(coor_v,vec,fn=f"{cifname}")                                            # CIF export of the defected unit cell.
  cifout(coords,vec,fn=f"{cifname}_P")                                          # CIF export of the perfect unit cell.


# __.: Uncomment and move these lines to the "for" loop for perfect unit cell calculation. :.__
# H00_P = H_Gen(coords, coords)
# NC_P = N_cell(coords,vec)
# H01_P = H_Gen(coords, NC[:,:,0])
# H02_P = H_Gen(coords, NC[:,:,2])
# E_P, _ = Band_Str(k, H00, H01, H02)                                           # Energy eigenvalues for the perfect crystall

# Need a better plot from atomic structure? Then, uncomment the following lines.  
# alll=[]
# for jj in range(9):
#     alll.append(NC_v[:, :, jj])
# alll = np.vstack(alll)

# plt.figure()
# plothlp(alll, acc, sps=None, cmdd='noatom')
# plothlp(pos_sublat[0], 0, ss[0])
# plothlp(pos_sublat[1], 0, ss[1])

# pts = np.column_stack([    ucpoint[:, 0],    ucpoint[:, 1],
#       np.array([ucpoint[0, 1], ucpoint[1, 3]]),ucpoint[:, 3],ucpoint[:, 2], ucpoint[:, 0] ])

# plt.plot(pts[0, :], pts[1, :], '-k')
# plt.show()
