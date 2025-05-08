import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from .constraints import WLS

COLORS = [
        "#ebac23",
        "#D80039",
        "#008cf9",
        "#AB358A",
        "#F5029F",
        "#0E00EE",
        "#00EB39",
        "#f05a28",
        "#15E4B1",
        "#D16174",
        "#136B8C",
        "#DD6E12",
        "#834C71",
    ]

markers = ["o", "s", "^", "v", "*", "D", "H"]

def plot_roi_img(img, rois):
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.frameon = False
    ax.set_axis_off()

    ax.imshow(img)

    color_i = 0
    # for i, coord in enumerate(rois):
    for (x1, y1, x2, y2) in rois:
        # update color
        if color_i == len(COLORS): color_i = 0
        curr_color = COLORS[color_i]
        color_i += 1

        # add roi rectangle
        roi = patches.Rectangle(
            (x1, y1), x2, y2, edgecolor=curr_color, facecolor="none", linewidth=2
        )
        ax.add_patch(roi)

    plt.show()

def plot_spectra(spectra, stds):
    """
    plot the averaged spectra for each rectangle.
    """
    plt.figure(figsize=(7, 7))

    bayer_sorted_indices = np.argsort(WLS[:3])
    non_bayer_sorted_indices = np.argsort(WLS[3:]) + 3

    color_i = 0
    marker_i = 0
    for i, spectrum in enumerate(spectra):

        # cycles colors if need be
        if color_i == len(COLORS):
            color_i = 0

        # cycles markers if need be
        if marker_i == len(markers):
            marker_i = 0

        curr_color = COLORS[color_i]

        # plot non-bayer bands
        nb_wls = np.array(WLS)[non_bayer_sorted_indices]
        nb_data = spectrum[non_bayer_sorted_indices]
        plt.errorbar(
            nb_wls,
            nb_data,
            yerr=stds[i][non_bayer_sorted_indices],
            fmt="-",
            ecolor=curr_color,
            capsize=3,
            color=curr_color,
            marker=markers[marker_i],
        )

        # plot bayer bands
        b_wls = np.array(WLS)[bayer_sorted_indices]
        b_data = spectrum[bayer_sorted_indices]
        plt.plot(b_wls, b_data, "+", color=curr_color)  # ,

        color_i += 1
        marker_i += 1

    plt.xlabel("wavelength (nm)")
    plt.ylabel("R* = IOF/cos(θ)")
    plt.ylim(top=0.5)  # makes y-axis max (0.5)
    plt.show()
