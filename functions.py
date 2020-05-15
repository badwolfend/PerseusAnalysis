import numpy as np
import scipy.signal as sps
import pickle

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

def getTile(t, pq, dir):
    """
    Returns the tiles for a time and p/q coordinate.
    :param t: Non-dimensional timestep
    :param pq: (mpi_P-1, mpi_Q-1)
    :return: data structure of all data included in this tile.
    """
    filename = open(dir+'/t_'+str(t)+'_mpip_'+str(pq[0])+'_mpiq_'+str(pq[1])+'.pickle', 'rb')
    return pickle.load(filename)

def reduce_y(A):
    return np.mean(A,axis=0)

def reduce_x(A):
    return np.mean(A,axis=1)


def smooth(x, window_len=11, window='hanning'):

    if window_len < 3:
        return x

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

def derivative(x, order=1):
    x_ret = np.zeros_like(x)
    if order==1:
        x_ret[1:]=np.diff(x)
    elif order == 2:
        x_ret[2:] = np.diff(np.diff(x))
    return x_ret


def find_inflect(x, thresh):
    first_deriv = derivative(x, order=1)
    second_deriv = derivative(x, order=2)
    i_points = sps.find_peaks(second_deriv)
    for i in i_points[0]:
        if x[i+2] > (x.max()-x.min())/thresh and first_deriv[i+1]>0:
            return i-2
    return 0

def find_peak(x):
    i_points = sps.find_peaks(x)
    x_max = 0
    i_max = 0
    for i in i_points[0]:
        if x[i] >= x_max:
            x_max = x[i]
            i_max = i
    return i_max-2, x[i_max-2]


def find_critical_dens_fact(wl, dens, units):
    rho_crit = (10**21)/(wl**2) # cm^-3
    rho_crit_meters = rho_crit/(10**6)
    rho_crit_norm = rho_crit_meters*(10**9) # Normalized to 1 mm X 1 mm X 1 mm
    rho_crit_norm_log = np.log(rho_crit_norm)/np.log(10)

    rho_crit_factor = dens/rho_crit_norm_log
    return rho_crit_factor


def Icur_LTD(t, peak, tr, td):
    return peak*np.sin(np.pi*t/(2*tr))*np.exp(-t/td)


def flow_deriv(x_y_z, tspan, rho, dx, dy):
    x, y, z, w = x_y_z
    if x < 0:
        print(x)
    n, nx, ny = refindex(rho, dx, dy, x, y)

    yp = np.zeros(shape=(4,))
    yp[0] = z / n
    yp[1] = w / n
    yp[2] = nx
    yp[3] = ny

    return yp


def plasma_freq(ne):
    L0 = 1.0e-3
    t0 = 100.0e-9
    n0 = 6.0e28
    pi = 3.14159265
    mu0 = 4.e-7 * np.pi
    eps0 = 8.85e-12
    v0 = L0 / t0
    mu = 27
    b0 = np.sqrt(mu * 1.67e-27 * mu0 * n0) * v0
    e0 = v0 * b0
    j0 = b0 / (mu0 * L0)
    e = 1.6022e-19
    me = 9.1e-31
    ne_units = ne

    unit_fact = (L0**(3/2))/(j0*t0)
    omega = np.sqrt(ne*(e**2)/(eps0*me))
    return omega


def refindex(rho, dx, dy, i, j):
    omega_p = plasma_freq(rho)
    n = (1-((omega_p/5.63e14)**2))
    nxny = np.gradient(n, dx, dy)
    nx = nxny[1]
    ny = nxny[0]
    i=i/(dx*1000)
    j=j/(dy*1000)

    i = int(i)
    j=int(j)
    en = n[j][i]
    enx = nx[j][i]
    eny = ny[j][i]
    # print("n=%f, i=%i, j=%i" %(en, i, j))
    return [en, enx, eny]





