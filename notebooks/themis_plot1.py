import pyspedas
from pyspedas import tplot, options, tplot_options

probe  = 'a'
trange = ['2019-05-01', '2019-05-02']

pyspedas.projects.themis.esa(probe=probe, trange=trange, level='l2')

v = 'tha_peif_en_eflux'
options(v, 'spec', 1)
options(v, 'ylog', 1)
options(v, 'zlog', 1)
options(v, 'yrange', [5, 30000])
options(v, 'zrange', [1e3, 1e8])
options(v, 'Colormap', 'spedas')
options(v, 'ytitle', 'ESA i+ tha\nE [eV]')
options(v, 'ztitle', 'eflux\n[eV/cm^2-s-sr-eV]')

tplot_options('title', 'THEMIS-A ESA ion eflux 2019-05-01')
tplot(v, save_png='tha_peif_eflux_20190501', xsize=12, ysize=3)