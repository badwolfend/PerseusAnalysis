import numpy as np
import matplotlib.pyplot as plt
import functions as ut
import scipy.signal as sps
import os
from scipy.stats import pearsonr

## Initialize Constants ##
las_loc_start=-220e-6
dl = 20e-6
dt0 = 0.45


def main(t=0, time=100, path="./"):
    t_las_begin = 86
    t_begin=80

    direct = os.listdir(path)
    peak_loc_array = np.array([])
    peak_loc_tuples = []
    x1ym = np.array([])
    x1y = np.array([])

    # For all files in directory #
    for d in direct:
        if d.split('_')[0] == "data":
            dir = d
            dir_parse = dir.split('_')
            YLoc = int(dir_parse[2])
            XLoc = int(dir_parse[1])
            dens_variance = np.zeros(time)

            # Assemble Tiles #
            for t in range(0, time):
                data0 = ut.getTile(t, (0, 1), path+"/"+dir)
                data1 = ut.getTile(t, (0, 2), path+"/"+dir)
                Z0 = np.log(data0['data'][:, :, ut.data_v['rh'] + 1])/np.log(10)
                Z1 = np.log(data1['data'][:, :, ut.data_v['rh'] + 1])/np.log(10)
                Z0 = Z0[:, Z0.shape[1]-int(Z0.shape[1]*(1/3)):]
                Z1 = Z1[:, :int(Z1.shape[1]*(1/3))]
                Z_ry0 = ut.reduce_y(Z0)
                Z_ry1 = ut.reduce_y(Z1)
                Z_ry_rod=np.zeros(( Z_ry0.size+Z_ry1.size))
                Z_ry_rod[0:Z_ry0.size]=Z_ry0
                Z_ry_rod[Z_ry0.size:]=Z_ry1
                Y = np.arange(-Z_ry_rod.size / 2, Z_ry_rod.size / 2)*data0['L0/dxi']
                dens_row_norm = Z_ry_rod/Z_ry_rod.max()
                dens_variance[t] = np.var(Z_ry_rod)
                if t == 0:
                    peak_dens = dens_row_norm
                else:
                    peak_dens = np.vstack((dens_row_norm, peak_dens))
                    # rho_crit_fact = ut.find_critical_dens_fact(532, Z, 6 * 10 ** 21)
                    # min = rho_crit_fact.min()
                    # print(rho_crit_fact.min())
                if False:
                    rho_crit_fact = ut.find_critical_dens_fact(532, Z, 6*10**21)
                    print(rho_crit_fact.min)
                    fig, ax = plt.subplots()
                    ax.imshow(peak_dens)
                    im = ax.imshow(rho_crit_fact, interpolation='gaussian', cmap='Greys')
                    # fig2, ax2 = plt.subplots()
                    # ax2.plot(Z_ry, color='k')
                    plt.show()


            # Smooth Data #
            smooth_variance = ut.smooth(dens_variance, window_len=7, window='hanning')
            smooth_var_t = np.arange(0, time, (time-0)/smooth_variance.size)

            # Find Inflection Points (first one) #
            i = ut.find_inflect(smooth_variance, thresh=20.0)
            i = len(dens_variance)-i
            # Use this inflection point as the time lock, now find the location of the peak density #
            x_max, dens_max = ut.find_peak(ut.smooth(peak_dens[i, :], window_len=2, window='hanning'))
            peak_loc_array = np.append(peak_loc_array, x_max*data0['L0/dxi'])
            peak_loc_tuples.append((XLoc, YLoc, x_max))

            x1y = np.append(x1y, dl*YLoc+las_loc_start)
            x1ym = np.append(x1ym, (x_max-Z_ry_rod.size/2)*data0['L0/dxi'])
            # ax.plot(XLoc, x_max-Z_ry_rod.size/2, 'xb')

            # Plot Results #
            if True:
                fig, ax11 = plt.subplots()
                im11 = ax11.plot(1000*Y, 255*peak_dens[i, :], 'k')
                ax11.plot(1000*Y[x_max], 255*dens_max, 'rx')

                print("x_laser= %f, x_peak= %f " %(1000*(las_loc_start + dl*YLoc), 1000*Y[x_max]))
                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
                ax1.imshow(255*peak_dens[:, :], extent=[1000*Y[0], 1000*Y[-1], 0, time*dt0], cmap='gray', interpolation='gaussian', aspect='auto')

                ax1.set_title('Time v. Ion Density')
                ax1.axvline(x=las_loc_start*1000 + dl*YLoc*1000, color='red')
                ax1.axvline(x=1000*Y[x_max], color='blue')
                tlas = (t_las_begin-t_begin)

                ax1.axhline(y=tlas, color='red')
                ax2.plot(smooth_variance, smooth_var_t*dt0, 'k')
                ax2.plot(smooth_variance[len(dens_variance)-i], smooth_var_t[len(dens_variance)-i]*dt0, 'xk')
                ax2.set_title('Ion Density Variance')
                plt.tight_layout()


    corry1, _ = pearsonr(x1y, x1ym)
    print(corry1)
    plt.show()

if __name__ == '__main__':
    main(path="../Charon/Power_Sims/run31")