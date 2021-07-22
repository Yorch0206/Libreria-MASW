import importlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import masw
importlib.reload(masw)

filename    = 'SampleData.dat'
HeaderLines = 7
fs          = 1000 # Hz
N           = 24
x1          = 10   # m
dx          = 1    # m
direction   = 'forward';
Amasw       = masw.masw(filename, 1/fs, fs, N, dx, x1, direction, header=6 )
A           = pd.read_csv(filename, header=6, delimiter="\t", skip_blank_lines=False)
A.head(10)
offset  = 0

dir(Amasw)
fig1, ax1 = Amasw.plot(scale=.8)

cT_min   = 50   # m/s
cT_max   = 400  # m/s
delta_cT = 1    # m/s

Amasw.dispersion_imaging(cT_min, cT_max, delta_cT)

resolution = 100   # No. de niveles
fmin = 0           # Hz
fmax = 50          # Hz
fig, ax = Amasw.plot_dispersion_image_2D(fmin, fmax, resolution)

f_receivers     = 4.5 # Frecuencia de los geófonos
select          = 'numbers'
up_low_boundary = 'yes'
p               = 95  # Porcentaje
Amasw.extract_dispersion_curve(fig, ax, f_receivers, select,up_low_boundary, p)

#fig, ax = plt.subplots(1,1)
#cn1     = ax.contourf(Amasw.fplot, Amasw.cplot, Amasw.Aplot.T, levels=resolution, cmap="RdBu_r")
#ax.plot(fvec, cvec,'y*')
#fig.colorbar(cn1,ax=ax)

Aabsnorm2 = (Amasw.Aplot.T/Amasw.Aplot.max(axis=1)).T
freq_ind, c_ind = np.where( Aabsnorm2== 1)

input("Press Enter to continue...")

plt.close(fig1)
plt.close(fig2)
fvec            = [Amasw.fplot[fi] for    fi in           freq_ind  if Amasw.fplot[fi] > f_receivers]
print('freq_ind: ', fvec)
print('c_Iind:   ', c_ind)