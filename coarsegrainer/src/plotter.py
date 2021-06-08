"""
Compact plotter for accuracy and loss series during training of 
coarse-grainer convnet and rsmimax net, density plots for weight series and 
the estimation of mutual information series for selected filters.

Author: Doruk Efe Gökmen
Date: 13/03/2020
"""


import math
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.pardir, os.pardir, "mi_estimator", "src"))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    MultipleLocator, FormatStrFormatter, AutoMinorLocator, FixedFormatter)
from matplotlib.transforms import Bbox, TransformedBbox, \
    blended_transform_factory
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector,\
    BboxConnectorPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from matplotlib.transforms import blended_transform_factory, TransformedBbox
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def mark_inset_hack(parent_axes, inset_axes, hack_axes, loc1, loc2, **kwargs):
    rect = TransformedBbox(hack_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1,
                       **kwargs, color='gray', alpha=0.1)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2,
                       **kwargs, color='gray', alpha=0.1)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2


def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = prop_lines.copy()
        prop_patches["alpha"] = prop_patches.get("alpha", 1)*0.2

    c1 = BboxConnector(bbox1, bbox2, loc1=loc1a,
                       loc2=loc2a, **prop_lines, color='gray', alpha=0.0)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox2, loc1=loc1b,
                       loc2=loc2b, **prop_lines, color='gray', alpha=0.0)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches, color='gray', alpha=0.1)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches, color='gray', alpha=0.1)

    p = BboxConnectorPatch(bbox1, bbox2,
                           # loc1a=3, loc2a=2, loc1b=4, loc2b=1,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           **prop_patches, color='gray', alpha=0.1)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect01(ax1, ax2, xmin, xmax, **kwargs):
    """
    ax1 : the main axes
    ax2 : the zoomed axes
    (xmin,xmax) : the limits of the colored area in both plot axes.

    connect ax1 & ax2. The x-range of (xmin, xmax) in both axes will
    be marked.  The keywords parameters will be used to create
    patches.

    """

    trans1 = blended_transform_factory(ax1.transData, ax1.transAxes)
    trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)

    bbox = Bbox.from_extents(-0.45, 0, ax1.get_xlim()[1]+0.45, 1)
    bbox2 = Bbox.from_extents(xmin, 0, xmax, 1)

    mybbox1 = TransformedBbox(bbox, trans1)
    mybbox2 = TransformedBbox(bbox2, trans2)

    prop_patches = kwargs.copy()
    prop_patches["ec"] = "none"
    #prop_patches["alpha"] = 0.1

    c1, c2, bbox_patch1, bbox_patch2, p = \
        connect_bbox(mybbox1, mybbox2,
                     loc1a=3, loc2a=2, loc1b=4, loc2b=1,
                     prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def plot_filter_series(coarse_grainer, series_skip=1, filter_index=None, filter_lim=0.5):

    ll = coarse_grainer.ll
    epochs = coarse_grainer.epochs

    w_series = coarse_grainer.cbk_filter.weights

    _, axs = plt.subplots(
        1, int(np.ceil(epochs/series_skip)))

    ii = 0
    for t in range(0, epochs, series_skip):
        if isinstance(filter_index, int):
            w = np.reshape(w_series[t][0].transpose()[
                           filter_index], tuple(reversed(ll)))
        else:
            w = np.reshape(w_series[t][0], tuple(reversed(ll)))

        axs[ii].imshow(w, clim=(-filter_lim, filter_lim), aspect=1,
                       interpolation='hanning', cmap='coolwarm')
        axs[ii].set_title("%i" % (t+1))

        axs[ii].set_xticks([i for i in range(ll[0])])
        axs[ii].set_yticks([i for i in range(ll[1])])
        plt.setp(axs[ii].get_xticklabels(), visible=False)
        plt.setp(axs[ii].get_yticklabels(), visible=False)

        ii += 1


def plot_fancy_rsmimax(estimates, filters, opt_params, CG_params, generator, 
                        mi_bound=r'$\rm InfoNCE$', series_skip=1, N_samples=10000, 
                        EMA_span=100, mi_max=1, filter_lim=0.5, 
                        fontsize=9, figsize=[8,6], font_family='helvetica',
                        interpolation='none', cmap='coolwarm', save=False):

    matplotlib.style.use('classic')
    plt.rc('text', usetex=True)
    params = {
        'text.latex.preamble': r'\usepackage{tgheros}'    # helvetica font
                           + r'\usepackage{sansmath}'   # math-font matching  helvetica
                           + r'\sansmath'                # actually tell tex to use it!
                           + r'\usepackage{siunitx}'    # micro symbols
                           + r'\sisetup{detect-all}',    # force siunitx to use the fonts
        'image.interpolation': interpolation,
        'image.cmap': cmap,
        'axes.grid': False,
        'savefig.dpi': 400,  # to adjust notebook inline plot size
        'axes.labelsize': fontsize,  # fontsize for x and y labels (was 10)
        'axes.titlesize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': fontsize,  # was 10
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'text.usetex': True,
        'figure.figsize': figsize,
        'font.family': font_family,
        'figure.facecolor': 'white',
    }
    matplotlib.rcParams.update(params)

    epochs = opt_params['iterations']*N_samples//opt_params['batch_size']
    num_hiddens = CG_params['num_hiddens']
    ll = CG_params['ll']

    # initiate width ratios for colorbar versus the filter weight density plots
    width_ratios = []
    for _ in range(int(np.ceil(epochs/series_skip))):
        width_ratios += [10]
    width_ratios += [0.5]

    fig = plt.figure(1)

    """
    1. Plot series for rsmi
    """
    mis = estimates
    mis_smooth = pd.Series(mis).ewm(span=EMA_span).mean()

    ax1 = plt.subplot(int(str(num_hiddens+1)+'1'+str(num_hiddens+1)))
    ax1.set_xlabel(r'$\rm{iterations}$')
    ax1.set_ylabel(r'$I_\Lambda(\mathcal{H}:\mathcal{E})$')
    p1 = ax1.plot(mis, label=mi_bound, color='black', alpha=0.3)[0]
    ax1.plot(mis_smooth, c=p1.get_color(), label=mi_bound+r' $\rm{EMA}$')
    #p1 = ax1.plot(mis, label='$\\rm{InfoNCE}$', color='lime', alpha=0.3)[0]
    #ax1.plot(mis_smooth, c=p1.get_color(), label='$\\rm{InfoNCE}$'+' $\\rm{EMA}$')
    ax1.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlim(-epochs/120, epochs+epochs/120)
    if mi_bound!='JS':
        ax1.set_ylim(0, round_up(max(mis_smooth), 1))
    ax1.set_yticks([i for i in np.linspace(0, round_up(max(mis), 1), 5)])

    w_series = filters

    for filter_index in range(num_hiddens):
        gs = gridspec.GridSpec(
            num_hiddens+1, int(np.ceil(epochs/series_skip))+1, width_ratios=width_ratios)

        ii = 0
        for t in np.arange(0, epochs, series_skip):
            """
            2. Show density plots for filter weights 
                for which mutual information series is estimated
            """
            if num_hiddens > 1:
                w = np.reshape(w_series[t].transpose()[
                               filter_index], tuple(reversed(ll)))
            else:
                w = np.reshape(w_series[t].transpose(), tuple(reversed(ll)))

            axf = fig.add_subplot(gs[filter_index, ii])
            axf.set_xlim(0, ll[0]-1)
            if filter_index == num_hiddens-1:
                # TODO: set adaptive value for 4
                zoom_effect01(axf, ax1, t-epochs/120, t+epochs/120)

            im = axf.imshow(w, clim=(-filter_lim, filter_lim), aspect=1)
            axf.xaxis.set_major_locator(plt.MaxNLocator(4))
            axf.yaxis.set_major_locator(plt.MaxNLocator(5))

            if filter_index == 0:
                axf.set_title("%i" % (t+1))

            axf.set_xticks([i for i in range(ll[0])])
            axf.set_yticks([i for i in range(ll[1])])
            axf.autoscale(enable=True)

            plt.setp(axf.get_xticklabels(), visible=False)
            plt.setp(axf.get_yticklabels(), visible=False)

            if t != 0:
                plt.setp(axf.get_yticklabels(), visible=False)
            else:
                if isinstance(filter_index, int):
                    axf.set_ylabel("$\\Lambda_%i$" % (filter_index+1))
                else:
                    axf.set_ylabel('$\\Lambda$')

                axf.xaxis.set_label_position('top')

            ii += 1

    cax = fig.add_subplot(gs[filter_index, ii])

    cbar = fig.colorbar(im, cax=cax,  fraction=0.036)
    cbar.ax.locator_params(nbins=5)
    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=None, hspace=0.4)

    if save:
        if generator.model == 'intdimer2d':
            plt.savefig(os.path.join(os.pardir, 'data', 'results', 
            'RSMImax'+generator.model+generator.lattice_type+'{0:.3f}'.format(generator.T)+'.pdf'))
        elif generator.model == 'ising2d':
            plt.savefig(os.path.join(os.pardir, 'data', 'results', 'RSMImax' +
                                    generator.model+generator.lattice_type+'{0:.3f}'.format(generator.J)+'.pdf'))
        else:
            plt.savefig(os.path.join(os.pardir, 'data', 'results', 'RSMImax' +
                                    generator.model+generator.lattice_type+'.pdf'))
    #plt.show()
