import matplotlib.pyplot as plt
import numpy as np

from tick.hawkes import (
    SimuHawkesSumExpKernels,
    HawkesSumExpKern
)
from tick.hawkes import SimuHawkesExpKernels  # NOQA
from tick.hawkes import HawkesExpKern  # NOQA
from tick.plot import plot_point_process, qq_plots
import matplotlib

matplotlib.use('qt5agg')

##########################################################################
# fit
##########################################################################
order_event = np.load('example_arrays.npz')["arr_0"]
timestamps_list = []
timestamp = order_event[:, 0]
for i in range(1, 9):
    timestamps_list.append(timestamp[np.where(order_event[:, i] > 0)])

Fitter = HawkesExpKern
decays = [0.1, 0.5, 1.]
kwargs = {}
if Fitter == HawkesSumExpKern:
    if 'penalty' not in kwargs:
        kwargs['penalty'] = 'elasticnet'
        kwargs['elastic_net_ratio'] = 0.8
learner = Fitter(decays=0.1, **kwargs)
learner.fit(timestamps_list)

##########################################################################
# plot intensities
##########################################################################
t_min = 100
t_max = 200
show = True
learner.plot_estimated_intensity(timestamps_list, t_min=t_min,
                                 t_max=t_max)
plt.show()
