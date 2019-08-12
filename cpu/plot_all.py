import os
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,7))
cases = sorted([f.lstrip('output_gemm_').rstrip('.m') for f in os.listdir('.') if f.startswith('output_gemm_')])

num_plots = len(cases)

# Have a look at the colormaps here and decide which one you'd like:
# http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

for i in xrange(num_plots):
    fname = 'output_gemm_%s.m' % cases[i]
    x_series = []
    y_series = []
    with open(fname) as f:
        data = f.readlines()[2:-1]
    for l in data:
        x, y = l.split(' ')[:2]
        x, y = int(x), float(y)
        x_series.append(x)
        y_series.append(y)

    plt.plot(x_series, y_series)

# I'm basically just demonstrating several different legend options here...
plt.xlabel('m=k=n')
plt.ylabel('GFlops')
plt.legend(cases, ncol=4, loc='upper center', 
           bbox_to_anchor=[0.5, 1.1], 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)

plt.show()

