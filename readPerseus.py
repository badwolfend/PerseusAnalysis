import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

run = "../Charon/Power_Sims/run31/"
dir = run+"data_0_0"
time = 0
def getTile(t, pq, dir):
    """
    Returns the tiles for a time and p/q coordinate.
    :param t: Non-dimensional timestep
    :param pq: (mpi_P-1, mpi_Q-1)
    :return: data structure of all data included in this tile.
    """
    filename = open(dir+'/t_'+str(t)+'_mpip_'+str(pq[0])+'_mpiq_'+str(pq[1])+'.pickle', 'rb')
    return pickle.load(filename)

## Setup the dict for name to datatype (must do this since the F90 binary files are written
# in no particular order.  Just use whatever the order is at the end of your code. ##
header =	{
    0: ('test', np.int32),
    1: ('ktype', np.int32),
    2: ("ngu", np.int32),
    3: ("mpi_nx", np.int32),
    4: ("mpi_ny", np.int32),
    5: (None, np.int32),
    6: ("mpi_P-1", np.int32),
    7: ("mpi_Q-1", np.int32),
    8: (None, np.int32),
    9: ("nx", np.int32),
    10: ("ny", np.int32),
    11: (None, np.int32),
    12: ("T0*t", np.float32),
    13: ("L0*loc_lxd", np.float32),
    14: ("L0*loc_lyd", np.float32),
    15: (None, np.int32),
    16: ("L0/dxi", np.float32),
    17: ("L0/dyi", np.float32),
    18: (None, np.int32)
}

data_v = { "rh" : 0,
           "mx": 1,
           "my": 2,
           "mz": 3,
           "P": 4,
           "Tev": 5,
           "bx": 6,
           "by": 7,
           "bz": 8,
           "ex": 9,
           "ey": 10,
           "ez": 11,
           "jx": 12,
           "jy": 13,
           "jz": 14,
           "eta": 15,
          }

# Go through all the binary files in the ~/data directory and convert to pickle binary files #
if (True):
    for file in glob.glob(run+'**/*.bin'):
        print(file)
        with open(file, 'rb') as f:
            output = dict()
            nvar = 16
            ll = min(list(header.keys()))
            ul = max(list(header.keys()))

            for el in range(ll, ul):
                if header[el][0] is None:
                    np.fromfile(f, dtype=header[el][1], count=1)[0]
                else:
                    # print(str(header[el][0])+", "+str(header[el][1]))
                    output[header[el][0]]=np.fromfile(f, dtype=header[el][1], count=1)[0]

            dat = np.fromfile(f, dtype=np.float32, count=output['nx']*output['ny']*nvar)
            data = dat.reshape((output['nx'], output['ny'], nvar))
            output['data']=data
            t = int(file.split("t")[2].split(".")[0])
            d = file.split("\\")
            direct = d[0] + "/" + d[1]
            f.close()

        # with open(dir+'/'+'t_'+str(t)+'_mpip_'+str(output["mpi_P-1"])+'_mpiq_'+str(output["mpi_Q-1"])+'.pickle', 'wb') as handle:
        #     pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(direct+'/'+'t_'+str(t)+'_mpip_'+str(output["mpi_P-1"])+'_mpiq_'+str(output["mpi_Q-1"])+'.pickle', 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Assemble Tiles #
data = getTile(time, (0, 1), dir)
Z = data['data'][:, :, data_v['rh']+1]
fig, ax = plt.subplots()
im = ax.imshow(np.log(Z)/np.log(10), interpolation='gaussian', cmap='hot')
plt.axis('off')
plt.show()

mpi_nx = data['mpi_nx']
mpi_ny = data['mpi_ny']
nnr = 2
nnc = 2
ion_density = np.zeros((data['mpi_ny']*(Z.shape[1]-nnr), data['mpi_nx']*(Z.shape[0])-nnc))
ion_temp = np.zeros((data['mpi_ny']*(Z.shape[1]-nnr), data['mpi_nx']*(Z.shape[0])-nnc))
Tev = np.zeros((data['mpi_ny']*(Z.shape[1]-nnr), data['mpi_nx']*(Z.shape[0])-nnc))

# Physical Quantities #
for i in range(0, mpi_nx):
    for j in range(0, mpi_ny):
        data = getTile(time, (i, j), dir)
        Z_ion_dens = np.rot90(data['data'][:, :, data_v['rh']+1], 1)
        Z_Tev = np.rot90(data['data'][:, :, data_v['Tev']+1], 1)

        Z_ion_dens = Z_ion_dens[0:Z_ion_dens.shape[0]-nnr, int(nnc/2):Z_ion_dens.shape[1]-int(nnc/2)]
        Z_Tev = Z_Tev[0:Z_Tev.shape[0]-nnr, int(nnc/2):Z_Tev.shape[1]-int(nnc/2)]

        ion_density[(mpi_ny-j-1)*Z_ion_dens.shape[0]:(mpi_ny-j-1)*Z_ion_dens.shape[0]+Z_ion_dens.shape[0], i*Z_ion_dens.shape[1]:i*Z_ion_dens.shape[1]+Z_ion_dens.shape[1]]=Z_ion_dens
        Tev[(mpi_ny-j-1)*Z_Tev.shape[0]:(mpi_ny-j-1)*Z_Tev.shape[0]+Z_Tev.shape[0], i*Z_Tev.shape[1]:i*Z_Tev.shape[1]+Z_Tev.shape[1]]=Z_Tev

# Plot Results #
fig, ax = plt.subplots(nrows=1,ncols=1)
ax.imshow(np.log(ion_density)/np.log(10), interpolation='gaussian', cmap='hot')
# ax[1].imshow(np.log(Tev)/np.log(10), interpolation='gaussian', cmap='Greys')
# ax.set_title("log ion density")
# ax[1].set_title("log ion temperature")
plt.axis('off')
plt.show()
print('done')
