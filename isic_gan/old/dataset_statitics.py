import matplotlib.pyplot as plt
import dataset
import numpy as np
from numpy.random import rand

def plot_bar_chart(groups, groups_vals):
    plt.xticks(range(len(groups)), groups, rotation=0)
    thisplot = plt.bar(range(len(groups_vals)), groups_vals, color="gray")
    plt.show()

# def plot_scatter(x_data, y_data, x_label="", y_label="", title="", color = "r", yscale_log=False):
#     N = 10
#     colors = np.random.rand(N)
#     area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
#
#     plt.scatter(x_data, y_data, s=area, c=colors, alpha=0.5)
#     plt.show()
# plot_scatter(range(10), range(10))

def plotClassesProportion(size_x=600, size_y=450):
    index = dataset.loadIndex()

    classes = {}
    for v in index.values():

        # From the specific size
        if v['size_x'] != size_x and v['size_y'] != size_y:
            continue

        if v['type'] != 'dermoscopic':
            continue

        # Confirmed
        if v['diagnosis_confirm_type'] == None:
            continue

        vClass = '%s_%s' % (v['diagnosis'], v['benign_malignant'])

        if vClass not in classes.keys():
            classes[vClass] = 1
        else:
            classes[vClass] += 1

    classes_array = [c for c in classes.keys()]
    classes_vals = [classes[c] for c in classes_array]
    plot_bar_chart(classes_array, classes_vals)

def plotSizesProportion():
    index = dataset.loadIndex()

    sizes = {}
    for v in index.values():
        vSize = '%sx%s' % (v['size_x'], v['size_y'])
        if vSize not in sizes.keys():
            sizes[vSize] = 1
        else:
            sizes[vSize] += 1

    sizes_array = sorted([s for s in sizes.keys()], key=lambda s: sizes[s], reverse=True)[:15]
    sizes_vals = [sizes[s] for s in sizes_array]
    plot_bar_chart(sizes_array, sizes_vals)

# plotClassesProportion(600, 450)
# plotClassesProportion(3024, 2016)
# plotSizesProportion()
