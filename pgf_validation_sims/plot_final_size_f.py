# -​*- coding: utf-8 -*​-
# @author: Laurent Hébert-Dufresne <lhebertd@uvm.edu>

# Packages
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import colorConverter as cc
import numpy as np


# =============================================================================
def G0(x, pk):
    return np.power(x[np.newaxis].T, pk[:, 0]).dot(pk[:, 1])

# =============================================================================
def G1(x, pk):
    return np.power(x[np.newaxis].T, pk[1:, 0] - 1).dot(
        np.multiply(pk[1:, 1], pk[1:, 0])) / pk[:, 0].dot(pk[:, 1])

# =============================================================================
def R0(x, T, pk):

    # Sample of the unit circle.
    N = 2000  # (needs to be large enough to minimize the effect of aliasing)
    z = np.exp(2 * np.pi * np.complex(0, 1) * np.arange(N) / N)
    R_at_x = G0(z*T+1-T, pk)

    return np.absolute(np.fft.ifft(R_at_x))[::-1]

# =============================================================================
def R0bar(x,y,P,R,f):
    summation = R[-1]
    for i in np.arange(1,max(R.shape)):
        summation += R[i-1]*((1-P)**i*x**i + (1-(1-P)**i)*(f*x + (1-f))**i)
    return summation

# =============================================================================
def R1(x, T, pk):

    # Sample of the unit circle.
    N = 2000  # (needs to be large enough to minimize the effect of aliasing)
    z = np.exp(2 * np.pi * np.complex(0, 1) * np.arange(N) / N)
    R_at_x = G1(z*T+1-T, pk)

    return np.absolute(np.fft.ifft(R_at_x))[::-1]

# =============================================================================
def Rbar1(x,y,P,R,f):
    summation = R[-1]
    for i in np.arange(1,max(R.shape)):
        summation += R[i-1]*((1-P)**i*x**i + (1-(1-P)**i)*(f*x + (1-f))**i)
    return summation

# =============================================================================
def yavg(P,R):
    summation = 0
    for i in np.arange(1,max(R.shape)):
        summation += i*R[i-1]*(1-(1-P)**i)
    return summation

# =============================================================================
def xavg(P,R):
    summation = 0
    for i in np.arange(1,max(R.shape)):
        summation += i*R[i-1]*(1-P)**i
    return summation

# =============================================================================
def solve_for_S(P,T,f):
    pk = np.loadtxt("./BA2_degreedis.dat")
    u_old = 0              # Dummy value
    R_new = R1(u_old,T,pk) # Dummy value
    u_new = 0.31416        # Dummy value
    while not np.allclose(u_old, u_new, rtol=1e-03, atol=1e-05):
        # print(P, f, u_new)
        u_old = u_new
        R_new = R1(u_old,T,pk)
        u_new = Rbar1(u_old, 1, P, R_new, f)
    R1_new = R1(u_new,T,pk)
    R0_new = R0(u_new,T,pk) # Dummy value
    return (1 - R0bar(u_new, 1, P, R0_new, f))

# =============================================================================
# Global parameters for the figures.
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Fira Sans", "PT Sans", "Open Sans", "Roboto", "DejaVu Sans", "Liberation Sans", "sans-serif"]
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["ytick.major.size"] = 8

#Make Figures Great Again color scheme... why am I still using this?
colors = [(11/255,26/255,69/255),
    (85/255,114/255,194/255),
    (195/255,177/255,137/255),
    (216/255,157/255,125/255),
    (175/255,90/255,59/255)]
n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=7)

#Load simulation results.
data = np.loadtxt('./validation_sims/ResultsBA2_f2.dat')

# Creates the figure and axis objects.
fig, ax = plt.subplots(1,1,figsize=(6,5.5), sharey=False)
fig.subplots_adjust(bottom=0.15)
#ax.set_xscale('log')
ax.set_xlim(0.000,0.5)
#ax.set_yscale('log')
ax.set_ylim(0.001,1)

# Parameter sets
F = np.linspace(0.00, 0.5, 50)
T = 0.3
P = np.arange(0,45,10)/100
Z = np.zeros((max(P.shape),max(F.shape)))

# Compute and plot final size
for p in np.arange(max(P.shape)-1,-1,-1):
    for f in np.arange(0,max(F.shape)):
        Z[p,f] = solve_for_S(P[p],T,F[f])
    ax.plot(F,Z[p,:], color = colors[-(p+1)], label=r'$P='+str(P[p])+'$')
    y = data[data[:,1] == P[p]]
    ax.plot(y[:,0], y[:,3], mec=colors[-(p+1)], ls='None', lw=3, marker='o', mfc='None', ms=3, mew=3)

# Labels
ax.set_xlabel(r'Failure probability')
ax.set_ylabel(r'Epidemic probability')

# Legend.
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper right', shadow=False, fancybox=False, prop={'size':20}, frameon=False, handlelength=1, numpoints=1)

# Save to file.
plt.tight_layout(0.1)
fig.savefig("Figure_BA2_f.pdf")
