import nrrd
import numpy as np
import os
import tifffile as tiff
import pandas as pd
import sys

sys.path.insert(1, os.getcwd())
import TuningAnalysis

import pickle
import glob

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
import matplotlib

import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from skimage.transform import rotate, resize
from collections import defaultdict as ddict
import cv2
from skimage import measure
from scipy.signal import medfilt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from scipy.signal import resample, convolve
from scipy.interpolate import InterpolatedUnivariateSpline
from itertools import chain
import re
from scipy.stats import zscore
from scipy.stats import mannwhitneyu


class PaperFigures:

    def __init__(self, **kwargs):

        self.inToCm = 2.54
        self.red = '#c70039'
        self.paper()
        self.figpath = kwargs.get('figpath', os.getcwd())
        self.datapath = kwargs.get('datapath', os.getcwd())

        self.sd = None
        self.sd_bk = None
        self.sd_t = None
        self.zstack = None
        self.allmasks = None
        self.roi_df = None

    def paper(self):

        sns.set()
        sns.set_style("ticks")
        sns.set_context("paper")

        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['figure.autolayout'] = False
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['figure.titlesize'] = 10
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['font.size'] = 10
        plt.rcParams['lines.linewidth'] = 1.0
        plt.rcParams['lines.markersize'] = 10
        plt.rcParams['legend.fontsize'] = 8
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['figure.figsize'] = (3.5, 3.5)

        matplotlib.rcParams['font.family'] = ['sans-serif']
        matplotlib.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['svg.fonttype'] = 'none'

    def init_anatomy(self):

        self.maskdf = pd.read_excel(self.datapath + '/MaskComments_21dpf.xlsx', sheet_name='2p')
        self.maskdf.columns = ['area_idx', 'name', 'nameshort', 'namemini', 'confidence', 'comments']
        self.areadict = {}
        for name, idx in zip(self.maskdf.namemini, self.maskdf.area_idx):
            self.areadict[idx] = name
        self.areadict[0] = 'None'

        self.zstack, header = trans_nrrd(nrrd.read(self.datapath + '/21dpf_AVG_H2BGCaMP6s_gamma.nrrd'), header=True)
        self.allmasks = trans_nrrd(nrrd.read(self.datapath + '/2pMasks_final_20210518.nrrd'))

        self.micron_scaling = np.array([header['space directions'][i].max() for i in range(3)]).astype(float)

    def init_data_fig2a_g(self):

        sd = pickle.load(open(self.datapath + '/sumdict_fig2ag.p', 'rb'))
        self.roi_df = pickle.load(open(self.datapath + '/roidf_fig2ag.p', 'rb'))

        mean_coords = self.roi_df.groupby('unid').mean()
        sd['x'] = mean_coords.x.values * (1. / self.micron_scaling[0])
        sd['y'] = mean_coords.y.values * (1. / self.micron_scaling[1])
        sd['z'] = mean_coords.z.values * (
                1. / self.micron_scaling[1])  # not division by micron scaling because 2 microns/pix in and out.

        maskids = np.array(
            self.allmasks[
                sd.z.values.astype(int),
                sd.y.values.astype(int),
                sd.x.values.astype(int)
            ]).ravel()

        sd['area_idx'] = maskids
        sd['area'] = [self.areadict[int(i)] for i in maskids]

        rs = sd[[i for i in sd.keys() if i.startswith('rs_True') or i.startswith('rs_False')]].values
        sd['rs_mean'] = np.nanmean(rs, axis=1)
        sd['rs_max'] = np.nanmax(rs, axis=1)

        dstims = sorted([i for i in sd.keys() if i.startswith('m_')])
        dotstims = sorted([i for i in sd.keys() if i.startswith('m_False') or i.startswith('m_True')])
        dot_resp = sd[dotstims].values

        bpi = calc_bpi(dot_resp)
        sd['bpi'] = bpi

        # Adding mean continuous and bout response to medium sized dot for 1.5 Hz and 60 Hz
        contr = (sd['m_True_60.0_1.6'] +
                 sd['m_False_60.0_1.6']) / 2

        boutr = (sd['m_True_1.5_1.6'] +
                 sd['m_False_1.5_1.6']) / 2

        sd['_m_cont'] = contr
        sd['_m_bout'] = boutr

        nostim = sd['m_bl'].values

        sd['bri'] = (boutr - nostim) / (boutr + nostim)
        sd['cri'] = (contr - nostim) / (contr + nostim)

        self.sd_bk = sd  # backup dataframe with all ROIs for plotting pixel-based ROIs
        self.sd = sd[sd.iscell]
        print(sd.shape)

        # Thresholding based on 95th percentile of responses to each dot stimulus
        self.rstims = [i for i in dstims if 'True' in i or 'False' in i]
        self.thresh = np.nanpercentile(self.sd[self.rstims], 95, axis=0)
        self.sd_t = self.sd[np.any(self.sd[self.rstims].values > self.thresh, axis=1)]

    def plot_fig2h(self):

        sd = pickle.load(open(self.datapath + '/sumdict_fig2h.p', 'rb'))
        roi_df = pickle.load(open(self.datapath + '/roidf_fig2h.p', 'rb'))

        mean_coords = roi_df.groupby('unid').mean()

        sd['x'] = mean_coords.x.values * (1. / self.micron_scaling[0])
        sd['y'] = mean_coords.y.values * (1. / self.micron_scaling[1])
        sd['z'] = mean_coords.z.values * (1. / self.micron_scaling[1])

        maskids = np.array(
            self.allmasks[
                sd.z.values.astype(int),
                sd.y.values.astype(int),
                sd.x.values.astype(int)
            ]).ravel()

        sd['area_idx'] = maskids
        sd['area'] = [self.areadict[int(i)] for i in maskids]

        rs = sd[[i for i in sd.keys() if i.startswith('rs_') or i.startswith('rs_')]].values
        sd['rs_mean'] = np.nanmean(rs, axis=1)
        sd['rs_max'] = np.nanmax(rs, axis=1)

        dstims = [i for i in sd.keys() if i.startswith('m_')]
        br = sd[sorted(dstims)].values
        sd['dff'] = np.sum(br, axis=1)

        sd_t = sd[sd['iscell']]
        print(sd_t.shape)

        # exclude cells based on 95th percentile of any stim. response
        thresh = np.nanpercentile(sd_t[dstims], 95, axis=0)
        sd_t = sd_t[np.any(sd_t[dstims].values > thresh, axis=1)]

        accstims = sorted([i for i in sd_t.keys() if re.match('m_\d*.\d_1.5', i) is not None])
        speedstims = sorted([i for i in sd_t.keys() if re.match('m_\d*.\d_60.0', i) is not None],
                            key=lambda x: float(x.split('_')[1]))
        sigmastims = ['m_2.0_60.0_nan_nan'] + sorted(
            [i for i in sd_t.keys() if re.match('m_nan_1.5_True_\d*.\d.', i) is not None], reverse=True) + [
                         'm_2.0_1.5_nan_nan']
        bout = sd_t[accstims].values.mean(axis=1)
        cont = sd_t[speedstims].values.mean(axis=1)

        boutspeeds = [float(i.split('_')[1]) * 10 / 4 for i in accstims]
        contspeeds = [float(i.split('_')[1]) * 10 / 4 for i in speedstims]
        sigmasacc = [0.00001, 0.02, 0.5, 3, 12]

        sd_t['bpi'] = (bout - cont) / (bout + cont)
        print(sd_t.shape)

        col = sns.color_palette("Reds", 5)
        fig, ax = plt.subplots(3, 2, figsize=(2.5, 4), dpi=200, sharex=False, constrained_layout=True)
        lw = 2
        for sno, speed in enumerate(boutspeeds):
            rad = 1.8
            frate = 60.
            delay = 12
            size = .4
            boutrate = 1.5

            xys, params = circular_step(speed=speed / 10,
                                        ccw=False,
                                        shift=False,
                                        shuffle=False,
                                        rad=rad,
                                        boutrate=boutrate,
                                        frate=frate)
            xys = np.array(xys)
            dists = [np.linalg.norm(xys[i, :2] - xys[i - 1, :2]) for i in np.arange(1, xys.shape[0], 1)]

            print(sum(dists))

            ax[0, 0].plot(np.cumsum(dists), color=col[sno], lw=lw)
        ax[0, 0].set_yticks([0, 2])
        ax[0, 0].set_yticklabels([0, 4])
        ax[0, 0].set_ylim(0, 3)
        ax[0, 0].set_xlim(-50, 300)
        ax[0, 0].set_xticks([0, 300])
        ax[0, 0].set_xticklabels([0, 5])

        for sno, speed in enumerate(contspeeds):
            rad = 1.8
            frate = 60.
            delay = 12
            size = .4
            boutrate = 60.

            xys, params = circular_step(speed=speed / 10,
                                        ccw=False,
                                        shift=False,
                                        shuffle=False,
                                        rad=rad,
                                        boutrate=boutrate,
                                        frate=frate)
            xys = np.array(xys)
            dists = [np.linalg.norm(xys[i, :2] - xys[i - 1, :2]) for i in np.arange(1, xys.shape[0], 1)]
            print(sum(dists))

            ax[1, 0].plot(np.cumsum(dists), color=col[sno], lw=lw)
        ax[1, 0].set_yticks([0, 2])
        ax[1, 0].set_yticklabels([0, 4])
        ax[1, 0].set_ylim(0, 3)
        ax[1, 0].set_xlim(-50, 300)
        ax[1, 0].set_xticks([0, 300])
        ax[1, 0].set_xticklabels([0, 5])
        # The following sigma values for modulating instantaneous acceleration do not reflect the actual
        # accelerations estimated post-hoc from actual stimulus positions over time
        for sno, sigma in enumerate([0.00001, 0.025, 0.05, 0.1, None]):
            speed = .5
            rad = 1.8
            frate = 60.
            delay = 12
            boutrate = 60.

            xys, params = circular_step(speed=speed,
                                        ccw=False,
                                        shift=False,
                                        shuffle=False,
                                        rad=rad,
                                        boutrate=boutrate,
                                        frate=frate,
                                        sigma=sigma,
                                        delay=delay
                                        )
            xys = np.array(xys)
            dists = [np.linalg.norm(xys[i, :2] - xys[i - 1, :2]) for i in np.arange(1, xys.shape[0], 1)]

            ax[2, 0].plot(np.cumsum(dists), color=col[4 - sno], lw=lw)
        ax[2, 0].set_ylim(0, 1.7)

        ax[2, 0].set_xlabel('Time (s)')
        ax[2, 0].set_yticks([0, 1])
        ax[2, 0].set_yticklabels([0, 1])
        ax[2, 0].set_xlim(-5, 120)
        ax[2, 0].set_xticks([0, 120])
        ax[2, 0].set_xticklabels([0, 2])
        plt.setp(ax[1, 0],
                 ylabel='Distance (cm)')

        dt_bpn = sd_t[(sd_t.bpi > .5) & (sd_t.area_idx == 4)]
        xlabels = [['mm/sec', 'Bout Speed'], ['mm/sec', 'Cont. Speed'], ['mm/sec\u00b2', 'Acceleration']]

        for stims, params, xlabel, axno in zip(
                [accstims, speedstims, sigmastims],
                [boutspeeds, contspeeds, sigmasacc],
                xlabels,
                [0, 1, 2]
        ):
            axc = ax[axno, 1]
            dt_bivals = dt_bpn[stims + ['fid']]
            meanv = dt_bivals.groupby('fid').mean().reset_index()
            mvm = meanv.mean(axis=0).values[1:]
            stdvm = meanv.std(axis=0).values[1:]
            print(params)

            if axno == 0:

                x_vals = np.log2(params)
                print(x_vals)
                x_vals[np.isinf(x_vals)] = 0
                xnew = np.linspace(1, x_vals[-1], 1000)
                max_vals = np.zeros(dt_bivals.shape[0])

                for i in range(dt_bivals.shape[0]):
                    f = InterpolatedUnivariateSpline(x_vals, dt_bivals[stims].values[i], k=2)
                    ynew = f(xnew)
                    max_vals[i] = xnew[np.argmax(ynew)]

                mpeak, stdpeak = 2 ** (np.mean(max_vals)), 2 ** (np.std(max_vals))

            for i in range(meanv.shape[0]):
                axc.plot(params, meanv.values[i, 1:], color='black', marker='o', markersize=2, alpha=.5)
            axc.plot(params, mvm, color=self.red, marker='o', markersize=0, lw=2, mec='white')
            axc.scatter(params, mvm, c=col, zorder=4, edgecolors='white', alpha=1, s=25)
            axc.fill_between(params,
                             mvm - stdvm,
                             mvm + stdvm,
                             facecolor=self.red,
                             alpha=0.3)
            axc.set_xscale('log', base=2)
            if axno == 0:
                axc.errorbar(mpeak, 1.1, xerr=stdpeak, fmt='o', capsize=2, color='black', markersize=3)
                axc.axvline(mpeak, alpha=.2, linestyle='--', color='black')
            plt.setp(axc,
                     ylim=[-.1, 1.2],
                     xlabel=xlabel[0],
                     )
            axc.set_ylabel('Mean dF/F')

        ax[0, 1].set(xticks=[2 ** 0, 2 ** 2, 2 ** 4])
        ax[1, 1].set(xticks=[2 ** 2, 2 ** 4, 2 ** 6])
        ax[2, 1].set(xticks=[2 ** -16, 2 ** -6, 2 ** 0, 2 ** 4], xticklabels=['0', '$2^{-6}$', '$2^{0}$', '$2^{4}$'])

        fig.text(0.4, 1.05, 'Variable speed @ 1.5 Hz', ha='center', fontsize=10)
        fig.text(0.4, 0.63, 'Variable speed @ 60 Hz', ha='center', fontsize=10)
        fig.text(0.4, .22, 'Variable acceleration', ha='center', fontsize=10)
        ax[1, 1].set_ylabel('Mean dF/F')
        sns.despine(trim=True)
        plt.subplots_adjust(wspace=.8, hspace=.9, left=0, right=1, bottom=0, top=1)
        plt.savefig(self.figpath + '/SPEEDACCSIGMATUNING.svg', bbox_inches='tight')
        plt.close()

    def plot_fig2i(self):

        sd = pickle.load(open(self.datapath + '/sumdict_fig2i.p', 'rb'))
        roi_df = pickle.load(open(self.datapath + '/roidf_fig2i.p', 'rb'))
        mean_coords = roi_df.groupby('unid').mean()

        sd['x'] = mean_coords.x.values * (1. / self.micron_scaling[0])
        sd['y'] = mean_coords.y.values * (1. / self.micron_scaling[1])
        sd['z'] = mean_coords.z.values * (1. / self.micron_scaling[1])
        maskids = np.array(
            self.allmasks[
                sd.z.values.astype(int),
                sd.y.values.astype(int),
                sd.x.values.astype(int)
            ]).ravel()
        sd['area_idx'] = maskids
        sd['area'] = [self.areadict[int(i)] for i in maskids]

        rs = sd[[i for i in sd.keys() if i.startswith('rs_') or i.startswith('rs_')]].values
        sd['rs_mean'] = np.nanmean(rs, axis=1)
        sd['rs_max'] = np.nanmax(rs, axis=1)

        dstims = sorted([i for i in sd.keys() if i.startswith('m_')])
        br = sd[dstims].values
        sd['dff'] = np.sum(br, axis=1)

        sd_t = sd[sd['iscell']]
        print(sd_t.shape)

        # exclude cells based on 95th percentile of any stim. response
        thresh = np.nanpercentile(sd_t[dstims], 95, axis=0)
        sd_t = sd_t[np.any(sd_t[dstims].values > thresh, axis=1)]

        bout = sd_t[['m_True_1.5_1.6', 'm_False_1.5_1.6']].values.mean(axis=1)
        cont = sd_t[['m_True_60.0_1.6', 'm_False_60.0_1.6']].values.mean(axis=1)
        sd_t['_m_bout'] = bout
        sd_t['_m_cont'] = cont
        sd_t['_m_boutwf'] = sd_t[['m_Image_1.5_False', 'm_Image_1.5_True']].values.mean(axis=1)
        sd_t['_m_contwf'] = sd_t[['m_Image_60.0_False', 'm_Image_60.0_True']].values.mean(axis=1)

        sd_t['bpi'] = (bout - cont) / (bout + cont)
        print(sd_t.shape)

        stims = sorted([i for i in sd_t.keys() if i.startswith('_m_') or i.startswith('m_nan')]) + [
            'fid']
        stims_plot = [stims[0], stims[5], stims[2], stims[1], stims[3]]

        dt_bpn = sd_t[(sd_t.bpi > .5) & (sd_t.area_idx == 4)]
        ptn = sd_t[sd_t.area_idx == 7]

        print(dt_bpn.shape)

        stims = sorted([i for i in sd_t.keys() if i.startswith('_m_') or i.startswith('m_nan')]) + [
            'fid']

        fig, axes = plt.subplots(2, 1, figsize=(1, 2), dpi=200)
        axes = axes.reshape(-1)
        stims_plot = [stims[0], stims[4], stims[2], stims[1], stims[3]]
        slabels = ['Dot 1.5 Hz', 'Dot shoaling', 'Dot 60 Hz', 'Whole-field 1.5 Hz', 'Whole-field 60 Hz']

        for ngroup, ax, title, clr in zip([dt_bpn, ptn], axes, ['DT-BPNs', 'Pretectum'], ['gold', '#19E6D3']):

            mdf = pd.melt(ngroup[stims], id_vars='fid', var_name='stim', value_name='dff')
            meanf = pd.melt(mdf.groupby(['fid', 'stim']).mean().unstack())
            meanf['fid'] = [i for i in mdf['fid'].unique()] * int(meanf.shape[0] / mdf.fid.unique().shape[0])

            sns.barplot(data=meanf, x='stim', y='value', ci='sd', order=stims_plot, palette=[clr] * len(slabels),
                        saturation=1, ax=ax)  # , color='green', saturation=1, linestyles=['none']*len(slabels))
            sns.stripplot(data=meanf, x='stim', y='value', order=stims_plot, palette=['black'], jitter=True,
                          edgecolor='none', s=7, alpha=.5, ax=ax)
            ax.set_xticks(range(0, len(stims_plot)))
            ax.set_xticklabels(slabels, rotation=45, ha='right')
            ax.set_ylabel('Mean dF/F')
            ax.set_xlabel('')
            ax.set_title(title, fontsize=10, ha='center')
            ax.set_ylim(-.1, 1.7)
            sns.despine()
            ax.legend_.remove()
            ax.set_yticks([0, 1])
            if title.startswith('DT'):
                ax.set_xticklabels([])

        plt.subplots_adjust(wspace=0, hspace=0.4, left=0, right=1, bottom=0, top=1)
        plt.savefig(self.figpath + '/NATFLICRESP.svg', bbox_inches='tight', dpi=400)
        plt.close()

        mg = colors.LinearSegmentedColormap.from_list('mg', ['white', '#19E6D3'])
        ly = colors.LinearSegmentedColormap.from_list('mg', ['white', 'gold'])

        alphas = np.zeros(sd_t.shape[0]) + .5
        alphas[(sd_t.bpi > .5) & (sd_t.area_idx == 4)] = .9

        fig = plt.figure(figsize=(3, 3), dpi=200)
        grid = plt.GridSpec(3, 2, figure=fig)

        axes = [fig.add_subplot(grid[j, i]) for i in range(2) for j in range(3)]
        stimpairs = [

            ['m_False_1.5_1.6', 'm_True_1.5_1.6'],
            ['m_nan_fish_1.6'],
            ['m_False_60.0_1.6', 'm_True_60.0_1.6'],
            ['m_Image_1.5_False', 'm_Image_1.5_True'],
            ['m_Image_60.0_False', 'm_Image_60.0_True'],

        ]
        stimlabels = ['Dot 1.5 Hz', 'Dot shoaling', 'Dot 60 Hz', 'Whole-Field 1.5 Hz', 'Whole-Field 60 Hz']

        for sno in range(len(stimpairs)):
            plot_orthogonal_inset(
                axes[sno],
                self.zstack,
                sd_t,
                clim=(0, .8),
                cval=sd_t[stimpairs[sno]].values.mean(axis=1),
                roialpha=.7,
                roisize=2.,
                cmap='Reds',
                cmap1=mg,
                cmap2=ly,
                points='equal',
                allmasks=self.allmasks,
                mask=8,
                kde=None,
                fs=(2, 2),
                tag='NATFLIC',
                clb=False
            )

            axes[sno].set_title(stimlabels[sno].strip('m_'), fontsize=10)

        axes[2].add_patch(matplotlib.patches.Rectangle((215, 460), 50, 5, color="black", clip_on=False))

        axn = fig.add_subplot(grid[2, -1])
        ax2 = inset_axes(axn,
                         width="5%",
                         height="50%",
                         loc='lower left')
        cmap = 'Reds'
        norm = colors.Normalize(vmin=0, vmax=.8)

        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2)
        cbar.set_ticks([0, .8])
        ax2.yaxis.set_ticks_position('right')
        ax2.set_ylabel('Mean dF/F')
        ax2.yaxis.set_label_position('right')
        axn.set_axis_off()
        axn.text(3.2, 3.1, 'DT')

        axn.add_patch(matplotlib.patches.Rectangle((2.3, 3), .8, .5, color='gold', alpha=.3))

        axn.text(3.2, 2.1, 'Pretectum')
        axn.add_patch(matplotlib.patches.Rectangle((2.3, 2), .8, .5, color='#19E6D3'))

        axn.set_xlim(0, 4)
        axn.set_ylim(2, 6)
        for i in range(len(axes)):
            axes[i].set_axis_off()
        plt.subplots_adjust(wspace=0.05, hspace=0.0, left=0, right=1, bottom=0, top=1)
        plt.savefig(self.figpath + '/NATFLICRESP_ANATOMY.svg', bbox_inches='tight', dpi=800)
        plt.close()

    def plot_fig2j(self):

        sd = pickle.load(open(self.datapath + '/sumdict_fig2j.p', 'rb'))
        roi_df = pickle.load(open(self.datapath + '/roidf_fig2j.p', 'rb'))

        areadict = pickle.load(open(self.datapath + '/areadict_7dpf.p', 'rb'))
        areadict[0] = 'None'
        allmasks = tiff.imread(self.datapath + '/brainmasks_7dpf.tiff')[::-1, :, :]
        zstack = tiff.imread(self.datapath + '/7dpf_AVG_H2BGCaMP6s_atlas.tif')
        micron_scaling = [.99, .99, 1]
        mean_coords = roi_df.groupby('unid').mean()
        sd['x'] = mean_coords.x.values * (1. / micron_scaling[0])
        sd['y'] = mean_coords.y.values * (1. / micron_scaling[1])
        sd['z'] = mean_coords.z.values  # not division by micron scaling because 2 microns/pix in and out.
        for dno, dim in enumerate([sd.z, sd.y, sd.x]):
            dlim = zstack.shape[dno]
            dim[dim > dlim - 1] = dlim - 1

        maskids = np.array(
            allmasks[
                sd.z.values.astype(int),
                sd.y.values.astype(int),
                sd.x.values.astype(int)
            ]).ravel()
        sd['area_idx'] = maskids
        sd['area'] = [areadict[int(i)] for i in maskids]

        rs = sd[[i for i in sd.keys() if i.startswith('rs_True') or i.startswith('rs_False')]].values
        sd['rs_mean'] = np.nanmean(rs, axis=1)
        sd['rs_max'] = np.nanmax(rs, axis=1)

        dstims = sorted([i for i in sd.keys() if i.startswith('m_')])
        dotstims = sorted([i for i in sd.keys() if i.startswith('m_False') or i.startswith('m_True')])
        br = sd[dotstims].values
        bpi = calc_bpi(br)
        sd['bpi'] = bpi
        sd['dff'] = np.sum(br, axis=1)

        # Adding mean continuous and bout response
        contr = (sd['m_True_60.0_1.6'] +
                 sd['m_False_60.0_1.6']) / 2

        boutr = (sd['m_True_1.5_1.6'] +
                 sd['m_False_1.5_1.6']) / 2

        sd['_m_cont'] = contr
        sd['_m_bout'] = boutr

        nostim = sd['m_bl'].values

        sd['bri'] = (boutr - nostim) / (boutr + nostim)
        sd['cri'] = (contr - nostim) / (contr + nostim)

        sd_bk = sd
        sd = sd[sd.iscell]

        rstims = [i for i in dstims if 'True' in i or 'False' in i]
        thresh = np.nanpercentile(sd[rstims], 95, axis=0)
        sd_t = sd[np.any(sd[rstims].values > thresh, axis=1)]

        X = sd_t[(sd_t.bpi > .5) & (sd_t.area_idx == 3)][
            [i for i in sd_t.keys() if 'm_True' in i or 'm_False' in i]].values
        X[np.isnan(X)] = 0
        bpnbd = X[:, :9] + X[:, 9:]
        bpnbd[:, 2] = np.mean(bpnbd[:, 1:4])
        bpnbd[:, 7] = np.mean(bpnbd[:, 6:9])
        bpnbd = bpnbd[:, [0, 2, 4, 5, 7]]
        x_vals = np.array([0.75, 1.5, 3., 6., 60.])
        x_vals = np.log2(x_vals)
        max_vals = np.zeros(bpnbd.shape[0])
        xnew = np.linspace(-1, x_vals[-1], 1000)

        for i in range(bpnbd.shape[0]):
            f = InterpolatedUnivariateSpline(x_vals, bpnbd[i], k=2)
            ynew = f(xnew)
            max_vals[i] = xnew[np.argmax(ynew)]
            if xnew[np.argmax(ynew)] < np.log2(0.75):
                max_vals[i] = np.log2(0.75) + (np.random.random() - .5) / 10

        meanv = plot_tuning_per_fish(
            sd_t,
            [3],
            areadict,
            mpeak=2 ** max_vals.mean(),
            mstd=np.std(2 ** max_vals),
            min_bpns=30,
            figpath=self.figpath
        )
        bpncoords = sd_t[sd_t.bpi > .5][['x', 'y', 'z', 'bpi']]
        recompute = False

        kdestacks = []
        kdes = []
        for bpnc, xbound in zip([bpncoords[bpncoords.x <= 283], bpncoords[bpncoords.x > 283]], [[0, 283], [283, 597]]):

            xgrid = np.arange(xbound[0], xbound[1], 2)
            ygrid = np.arange(0, zstack.shape[1], 2)
            zgrid = np.arange(0, zstack.shape[0], 2)

            XX, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
            xyz = np.vstack([XX.ravel(), Y.ravel(), Z.ravel()]).T
            kde = KernelDensity(metric='euclidean', bandwidth=14,  # was 16 before
                                kernel='gaussian', algorithm='auto')
            kde.fit(bpnc[['x', 'y', 'z']])
            if recompute:
                kdehalf = np.exp(kde.score_samples(xyz)).reshape(xgrid.size, ygrid.size, zgrid.size)
                kdestacks.append(kdehalf / kdehalf.sum() * bpnc.shape[0])

            kdes.append(kde)

        if recompute:

            mk = np.concatenate(kdestacks, axis=0)
            np.save(self.datapath + '/kdestacks_fig2j.npy', mk)

        else:

            mk = np.load(self.datapath + '/kdestacks_fig2j.npy')

        kdevis = np.moveaxis(mk, 2, 0)
        kdevis = np.moveaxis(kdevis, 1, 2)
        kdevis_res = resize(kdevis, zstack.shape)

        # Fractions of BPNs in KDE, probably not needed anymore
        bpn_mask = np.zeros(kdevis_res.shape)
        bpn_mask[kdevis_res >= .06] = 1
        bpn_fid = sd_t[sd_t.bpi > .5].fid
        frs = list()
        for fid in bpn_fid.unique():
            inmask_bpn = bpn_mask[
                bpncoords[bpn_fid == fid].z.astype('int'),
                bpncoords[bpn_fid == fid].y.astype('int'),
                bpncoords[bpn_fid == fid].x.astype('int')
            ]
            bools_bpn, counts_bpn = np.unique(inmask_bpn, return_counts=True)
            inmask_all = bpn_mask[
                sd_t[sd_t.fid == fid].z.astype('int'),
                sd_t[sd_t.fid == fid].y.astype('int'),
                sd_t[sd_t.fid == fid].x.astype('int')
            ]
            bools_all, counts_all = np.unique(inmask_all, return_counts=True)
            if bools_bpn[-1] == 0:
                counts_bpn = [0]

            if bools_all[-1] == 0:
                frs.append(np.nan)
                continue
            fr = counts_bpn[-1] / counts_all[-1]
            # print(counts_bpn[-1], counts_all[-1], fr)

            frs.append(fr)
        # print(np.nanmean(frs))

        bpn_mask = np.zeros(kdevis_res.shape)
        bpn_mask[kdevis_res >= 0.0001] = kdevis_res[kdevis_res >= 0.0001]
        bpn_mask = (bpn_mask / bpn_mask.sum()) * sd_t[sd_t.bpi > .5].shape[0]
        cwalpha = abs(sd_t.bpi.values)
        cwalpha = np.clip(cwalpha, .0, .5)
        zyx = sd_t[sd_t.bpi > .5][['z', 'y', 'x']].values.astype(int)
        kdvals = bpn_mask[zyx[:, 0], zyx[:, 1], zyx[:, 2]]

        fs = np.array([(zstack.shape[0] - 50 + zstack.shape[2]) * 4 / 1000,
                       (zstack.shape[0] - 50 + zstack.shape[1] - 100) * 4 / 1000])
        plot_orthogonal_scatter_7dpf(
            zstack,
            sd_t[sd_t.bpi > .5],
            clim=(0.0001, 0.0006),
            cval=kdvals,
            roialpha=0.7,
            roisize=1.5,
            cmap='Reds',
            ctclrs='Reds',
            points='equal',
            allmasks=allmasks,
            mask=None,
            kde=None,
            fs=fs * .75,
            tag='BPIALL_7DPF',
            clb=True,
            tvals=np.array([0.0002, 0.0004, 0.0006]),
            bpn_mask=bpn_mask,
            save=True,
            figpath=self.figpath
        )

    def plot_fig2g(self):

        X = self.sd_t[(self.sd_t.bpi > .5) & (self.sd_t.area_idx == 4)][
            [i for i in self.sd_t.keys() if 'm_True' in i or 'm_False' in i]].values
        X[np.isnan(X)] = 0
        bpnbd = X[:, :9] + X[:, 9:]
        bpnbd[:, 2] = np.mean(bpnbd[:, 1:4])
        bpnbd[:, 7] = np.mean(bpnbd[:, 6:9])
        bpnbd = bpnbd[:, [0, 2, 4, 5, 7]]
        x_vals = np.array([0.75, 1.5, 3., 6., 60.])
        x_vals = np.log2(x_vals)
        max_vals = np.zeros(bpnbd.shape[0])
        xnew = np.linspace(-1, x_vals[-1], 1000)

        for i in range(bpnbd.shape[0]):
            f = InterpolatedUnivariateSpline(x_vals, bpnbd[i], k=2)
            ynew = f(xnew)
            max_vals[i] = xnew[np.argmax(ynew)]
            if xnew[np.argmax(ynew)] < np.log2(0.75):
                max_vals[i] = np.log2(0.75) + (np.random.random() - .5) / 10

        fig, ax = plt.subplots(figsize=(1, 1.), dpi=400)
        sns.violinplot(y=max_vals, palette=[self.red], alpha=0.2, ax=ax, inner=None)

        sns.stripplot(y=max_vals, color='grey', edgecolor='none', ax=ax, s=1., alpha=.7, zorder=1)
        sns.pointplot(y=max_vals, color='black', edgecolor='black', ax=ax, markers='.',
                      alpha=.5, estimator=np.mean, ci='sd', zorder=2, capsize=.1, errwidth=1)

        plt.setp(ax.lines, zorder=100)
        ax.set_alpha(0.02)
        ax.set_xticks([])
        ax.set_yticks([i for i in x_vals])
        ax.set_yticklabels([i for i in np.around(2 ** x_vals, decimals=2)])
        ax.set_ylabel('Peak rate (Hz)')
        ax.set_xlabel('DT-BPNs')

        sns.despine(trim=True)
        ax.spines['bottom'].set_visible(False)
        plt.savefig(self.figpath + '/DTBPNS_TUNING_PEAK.svg', bbox_inches='tight')
        plt.close()
        meanv = plot_tuning_per_fish(
            self.sd_t,
            [4],
            self.areadict,
            mpeak=2 ** max_vals.mean(),
            mstd=np.std(2 ** max_vals),
            min_bpns=30,
            figpath=self.figpath
        )
        print('Tuning peak: ', 2 ** max_vals.mean(), 2 ** max_vals.std())

    def print_fractions(self, areas):

        for i in areas:

            bafids = self.sd_t[self.sd_t.area_idx == i].fid.unique()
            prvals = np.zeros((bafids.shape[0], 10)) * np.nan
            prvals[:, 0] = bafids

            for fidno, fid in enumerate(bafids):
                n_ba = self.sd_t[(self.sd_t.fid == fid) & (self.sd_t.area_idx == i)].shape[0]
                n = self.sd_t[(self.sd_t.fid == fid)].shape[0]
                brn_ba = self.sd_t[(self.sd_t.fid == fid) & (self.sd_t.area_idx == i) & (self.sd_t.bri > .5)].shape[0]
                brn = self.sd_t[(self.sd_t.fid == fid) & (self.sd_t.bri > .5)].shape[0]

                crn_ba = self.sd_t[(self.sd_t.fid == fid) & (self.sd_t.area_idx == i) & (self.sd_t.cri > .5)].shape[0]
                crn = self.sd_t[(self.sd_t.fid == fid) & (self.sd_t.cri > .5)].shape[0]

                bpn_ba = self.sd_t[(self.sd_t.fid == fid) & (self.sd_t.area_idx == i) & (self.sd_t.bpi > .5)].shape[0]
                bpn = self.sd_t[(self.sd_t.fid == fid) & (self.sd_t.bpi > .5)].shape[0]

                bpi = self.sd_t[(self.sd_t.fid == fid)].bpi.mean()

                prvals[fidno, 1:] = np.array([n_ba, n, brn_ba, brn, crn_ba, crn, bpn_ba, bpn, bpi])
            prdf = pd.DataFrame(prvals.astype('float'), columns=(
                ['fid', 'n_ba', 'n', 'brn_ba', 'brn', 'crn_ba', 'crn', 'bpn_ba', 'bpn', 'bpi']))

            print('Number of recorded neurons')
            print('')
            print('Mean+-STD # of neurons in {}: '.format(self.areadict[i]), np.mean(prdf.n_ba), np.std(prdf.n_ba))
            print('Mean+-STD # of neurons total: ', np.mean(prdf.n), np.std(prdf.n))
            print('Mean+-STD fraction of {} neurons: '.format(self.areadict[i]), np.mean(prdf.n_ba / prdf.n),
                  np.std(prdf.n_ba / prdf.n))
            print('')

            print('Number of bout-responsive neurons (BRNs)')
            print('')
            print('Mean+-STD # of BRNs in {}: '.format(self.areadict[i]), np.mean(prdf.brn_ba), np.std(prdf.brn_ba))
            print('Mean+-STD # of BRNs total: ', np.mean(prdf.brn), np.std(prdf.brn))
            print('Mean+-STD fraction of BRNs in {}: '.format(self.areadict[i]), np.mean(prdf.brn_ba / prdf.n_ba),
                  np.std(prdf.brn_ba / prdf.n_ba))
            print('Mean+-STD fraction of BRNs total: ', np.mean(prdf.brn / prdf.n), np.std(prdf.brn / prdf.n))
            print('')

            print('Number of continuous-responsive neurons (CRNs)')
            print('')
            print('Mean+-STD # of CRNs in {}: '.format(self.areadict[i]), np.mean(prdf.crn_ba), np.std(prdf.crn_ba))
            print('Mean+-STD # of CRNs total: ', np.mean(prdf.crn), np.std(prdf.crn))
            print('Mean+-STD fraction of CRNs in {}: '.format(self.areadict[i]), np.mean(prdf.crn_ba / prdf.n_ba),
                  np.std(prdf.crn_ba / prdf.n_ba))
            print('Mean+-STD fraction of CRNs total: ', np.mean(prdf.crn / prdf.n), np.std(prdf.crn / prdf.n))
            print('')

            print('Number of Bout-preference-neurons (BPNs)')
            print('')
            print('Mean+-STD # of BPNs in {}: '.format(self.areadict[i]), np.mean(prdf.bpn_ba), np.std(prdf.bpn_ba))
            print('Mean+-STD # of BPNs total: ', np.mean(prdf.bpn), np.std(prdf.bpn))
            print('Mean+-STD fraction of BPNs in {}: '.format(self.areadict[i]), np.mean(prdf.bpn_ba / prdf.n_ba),
                  np.std(prdf.bpn_ba / prdf.n_ba))
            print('Mean+-STD fraction of BPNs total: ', np.mean(prdf.bpn / prdf.n), np.std(prdf.bpn / prdf.n))
            print('Mean+-STD BPI total: ', np.mean(prdf.bpi), np.std(prdf.bpi))
            return prdf

    def plot_figS4a(self):

        all_fids = list(self.sd.fid.unique())
        all_idcounts = np.zeros((len(all_fids), len([i for i in self.areadict.keys()])))
        all_idcountsbpn = np.zeros((len(all_fids), len([i for i in self.areadict.keys()])))

        for fid in self.sd.fid.unique():
            ids, idcounts = np.unique(self.sd[self.sd.fid == fid].area_idx.values, return_counts=True)
            ids_bpn, idcounts_bpn = np.unique(self.sd_t[(self.sd_t.fid == fid) & (self.sd_t.bpi > .5)].area_idx.values,
                                              return_counts=True)

            all_idcounts[all_fids.index(fid), ids] = idcounts
            all_idcountsbpn[all_fids.index(fid), ids_bpn] = idcounts_bpn

        fig, axes = plt.subplots(5, 1, figsize=(4., 12), dpi=200)
        pal = sns.color_palette('tab20')

        ax = axes[3]

        allbpn = pd.DataFrame(all_idcountsbpn)
        allbpn['fid'] = np.array(all_fids)
        keys = [i for i in allbpn.keys() if not (len(allbpn[i].unique()) == 1)]
        keys = [i for i in keys if not round(allbpn[i].mean(), 0) == 0]
        bpnkeys = keys

        allbm = pd.melt(allbpn[keys], id_vars='fid', value_name='count', var_name='area_idx')
        ax = sns.barplot(data=allbm, ci='sd', y='count', x='area_idx', ax=ax,
                         palette=[pal[i] for i in sorted(allbm['area_idx'].unique())])
        sns.stripplot(data=allbm, x='area_idx', y='count', palette=['black'], jitter=True, edgecolor='none', s=5,
                      alpha=.5, ax=ax)

        ax.set_xticks(range(len(keys) - 1))
        areas = [self.areadict[i] for i in range(all_idcounts.shape[1] - 1)]
        ax.set_xticklabels([areas[k] for k in keys if not isinstance(k, str)], rotation=45, ha='center')
        ax.set_ylabel('# cells')
        ax.set_title('Number of BPNs')
        ax.set_xlabel('')
        ax.set_ylim(0, 225)
        ax.bar_label(ax.containers[0], fmt='%.0f', rotation=0, fontsize=8, padding=20)

        ax = axes[4]

        allfractions = all_idcountsbpn / all_idcounts
        allfractions[np.isnan(allfractions)] = 0
        allf = pd.DataFrame(allfractions)
        allf['fid'] = np.array(all_fids)

        allfm = pd.melt(allf[keys], id_vars='fid', value_name='fraction', var_name='area_idx')
        ax = sns.barplot(data=allfm, ci='sd', y='fraction', x='area_idx', ax=ax,
                         palette=[pal[i] for i in sorted(allfm['area_idx'].unique())])
        sns.stripplot(data=allfm, x='area_idx', y='fraction', palette=['black'], jitter=True, edgecolor='none', s=5,
                      alpha=.5, ax=ax)

        ax.set_xticks(range(len(keys) - 1))

        areas = [self.areadict[i] for i in range(all_idcounts.shape[1] - 1)]
        ax.set_xticklabels([areas[k] for k in keys if not isinstance(k, str)], rotation=45, ha='center')
        ax.set_ylabel('rel. frequency')
        ax.set_title('Fraction of BPNs')
        ax.set_xlabel('')
        ax.bar_label(ax.containers[0], fmt='%.2f', rotation=0, fontsize=8, padding=30)
        ax.set_ylim(0, 0.12)

        ax = axes[0]

        alln = pd.DataFrame(all_idcounts)
        alln['fid'] = np.array(all_fids)
        keys = [i for i in alln.keys() if not (len(alln[i].unique()) == 1)]
        self.allkeys = keys
        allrn = pd.melt(alln[keys], id_vars='fid', value_name='count', var_name='area_idx')
        ax = sns.barplot(data=allrn, ci='sd', y='count', x='area_idx', ax=ax,
                         palette=[pal[i] for i in sorted(allrn['area_idx'].unique())])
        sns.stripplot(data=allrn, x='area_idx', y='count', palette=['black'], jitter=True, edgecolor='none', s=5,
                      alpha=.5, ax=ax)

        ax.set_xticks(range(len(keys) - 1))
        self.areas = [self.areadict[i] for i in range(all_idcounts.shape[1] - 1)]
        ax.set_xticklabels([self.areas[k] for k in keys if not isinstance(k, str)], rotation=45, ha='center')
        ax.set_ylabel('# cells')
        ax.set_title('Number of recorded neurons')
        ax.set_xlabel('')
        ax.bar_label(ax.containers[0], fmt='%.0f', rotation=0, fontsize=8, padding=25)
        ax.set_ylim(0, 4500)

        all_idcounts_r = np.zeros((len(all_fids), len([i for i in self.areadict.keys()])))
        for fid in self.sd.fid.unique():
            ids_r, idcounts_r = np.unique(self.sd_t[self.sd_t.fid == fid].area_idx.values, return_counts=True)
            all_idcounts_r[all_fids.index(fid), ids_r] = idcounts_r

        ax = axes[1]

        allnt = pd.DataFrame(all_idcounts_r)
        allnt['fid'] = np.array(all_fids)
        keys = [i for i in allnt.keys() if not (len(allnt[i].unique()) == 1)]

        allrnt = pd.melt(allnt[keys], id_vars='fid', value_name='count', var_name='area_idx')
        ax = sns.barplot(data=allrnt, y='count', ci='sd', x='area_idx', ax=ax,
                         palette=[pal[i] for i in sorted(allrnt['area_idx'].unique())])
        sns.stripplot(data=allrnt, x='area_idx', y='count', palette=['black'], jitter=True, edgecolor='none', s=5,
                      alpha=.5, ax=ax)

        ax.set_xticks(range(len(keys) - 1))
        areas = [self.areadict[i] for i in range(all_idcounts.shape[1] - 1)]
        ax.set_xticklabels([areas[k] for k in keys if not isinstance(k, str)], rotation=45, ha='center')
        ax.set_ylabel('# cells')
        ax.set_title('Number of recorded neurons above threshold')
        ax.set_xlabel('')
        ax.set_yticks([0, 1000, 2000])
        ax.bar_label(ax.containers[0], fmt='%.0f', rotation=0, fontsize=8, padding=25)

        ax = axes[2]

        allfractions = all_idcounts_r / all_idcounts
        allfractions[np.isnan(allfractions)] = 0
        allfractions[np.isinf(allfractions)] = 0
        allft = pd.DataFrame(allfractions)
        allft['fid'] = np.array(all_fids)

        keys = [i for i in allft.keys() if not (len(allft[i].unique()) == 1)]
        print(keys)
        allfnt = pd.melt(allft[keys], id_vars='fid', value_name='fraction', var_name='area_idx')
        ax = sns.barplot(data=allfnt, y='fraction', ci='sd', x='area_idx', ax=ax,
                         palette=[pal[i] for i in sorted(allfnt['area_idx'].unique())])
        sns.stripplot(data=allfnt, x='area_idx', y='fraction', palette=['black'], jitter=True, edgecolor='none', s=5,
                      alpha=.5, ax=ax)

        ax.set_xticks(range(len(keys) - 1))

        areas = [self.areadict[i] for i in range(all_idcounts.shape[1] - 1)]
        ax.set_xticklabels([areas[k] for k in keys if not isinstance(k, str)], rotation=45, ha='center')
        ax.set_ylabel('rel. frequency')
        ax.set_title('Fraction of recorded neurons above threshold')
        ax.set_xlabel('')
        ax.bar_label(ax.containers[0], fmt='%.2f', rotation=0, fontsize=8, padding=20)
        ax.set_ylim(0, 0.7)

        plt.subplots_adjust(hspace=.6)
        sns.despine(trim=True)

        plt.savefig(self.figpath + '/S4a.svg', bbox_inches='tight')
        plt.close()

    def plot_figS4b(self):

        pal = sns.color_palette('tab20')
        self.allkeys = [i for i in self.allkeys if not isinstance(i, str)]
        fig, ax = plt.subplots(1, 1, figsize=(4, 4.), dpi=200)
        for cno, k in enumerate(self.allkeys):
            ax.scatter(0, len(self.allkeys) - cno, color=pal[k])
            print('{} ({})'.format(self.areas[k], self.maskdf.name[k]))
            ax.text(2, len(self.allkeys) - cno - .15, '{} ({})'.format(self.areas[k], self.maskdf.name[k]))
            ax.set_xlim(-2, 10)
        ax.scatter(0, .01, s=0)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.figpath + '/figS5a.svg')
        plt.close()

        fig, ax = plt.subplots(5, 3, figsize=(7 * 12 / 15, 12), dpi=400)
        ax = ax.reshape(-1)

        for idx, key in enumerate(self.allkeys[:]):

            #     bamask = (allmasks==key).astype(int)
            #     bmean = bamask.mean(0).astype(float)
            #     bmean[bmean==0] = np.nan

            x1, x2, x3 = self.sd[(self.sd.area_idx == key)].x, self.sd[(self.sd.area_idx == key)].y, self.sd[
                (self.sd.area_idx == key)].z

            try:
                av = self.zstack[int(np.nanpercentile(x3, 5)):int(np.nanpercentile(x3, 95))].max(0)
            except:
                av = self.zstack[:].max(0)

            ax[idx].imshow(av, cmap='Greys', origin='lower', alpha=.9, aspect='equal', clim=(50, 400))
            #     if not idx==0:
            #         ax[idx].imshow(bmean, cmap='Reds', alpha=.6)
            ax[idx].scatter(x1, x2, color=pal[key], edgecolors='none', s=.8, alpha=.5, rasterized=True)
            ax[idx].set_axis_off()
            ax[idx].set_ylim(700, 25)
            ax[idx].set_xlim(25, 700)
            ax[idx].set_title('{}, n={}'.format(self.areadict[key], str(int(x1.shape[0]))))
        for i in range(ax.shape[0]):
            ax[i].set_axis_off()
        ax[12].add_patch(matplotlib.patches.Rectangle((500, 600), 100, 5, color="black"))

        # fig.suptitle('All recorded neurons per brain area')
        plt.subplots_adjust(hspace=.15)
        plt.savefig(self.figpath + '/figS5b.svg', bbox_inches='tight', dpi=800)
        plt.close()

    def plot_figS4c(self):

        fig, ax = plt.subplots(2, 2, figsize=(3, 3), constrained_layout=True, dpi=400)
        x = self.sd_t[(self.sd_t.bpi > .5) & (self.sd_t.area_idx == 8)].y
        y = self.sd_t[(self.sd_t.bpi > .5) & (self.sd_t.area_idx == 8)].z
        ax[1, 0].imshow(self.zstack[:, :, :].max(2), cmap='Greys', clim=(100, 400), aspect='equal')
        ax[1, 0].scatter(x, y, s=1, alpha=.5, color=self.red, edgecolors='none')
        ax[0, 0].hist(x, color=self.red, bins=35)
        ax[1, 1].hist(y, orientation='horizontal', color=self.red)
        ax[0, 1].set_axis_off()
        ax[0, 0].set_title('AP distribution')
        ax[1, 1].set_title('DV distribution')
        ax[1, 1].set_xlabel('# cells')
        ax[1, 0].set_xlabel('AP axis (µm)')
        ax[1, 1].yaxis.tick_right()
        ax[0, 0].set_ylabel('# cells')
        ax[1, 0].set_ylabel('DV axis (µm)')
        ax[1, 1].yaxis.set_label_position("right")
        ax[1, 0].set_ylim(280, 50)
        ax[1, 0].set_xlim(300, 520)
        ax[0, 0].set_xlim(300, 520)
        ax[0, 0].set_xticks([])
        ax[1, 1].set_yticks([])
        ax[1, 1].set_ylim(280, 50)
        ax[1, 0].set_aspect('equal')
        fig.suptitle('Anatomical distribution of tectal BPNs (n={})'.format(x.shape[0]))
        sns.despine()
        plt.subplots_adjust(wspace=.1, hspace=.1)
        plt.savefig(self.figpath + '/S_ANATOMICAL_DIST_BPN_TECTUM_DV.svg', bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(2, 2, figsize=(3, 2), constrained_layout=True, dpi=400)
        y = self.sd_t[(self.sd_t.bpi > .5) & (self.sd_t.area_idx == 8)].y
        x = self.sd_t[(self.sd_t.bpi > .5) & (self.sd_t.area_idx == 8)].x
        ax[1, 0].imshow(self.zstack[:, :, :].max(0), cmap='Greys', clim=(100, 400), aspect='auto')
        ax[1, 0].scatter(x, y, s=1, alpha=.5, color=self.red, edgecolors='none')
        ax[0, 0].hist(x, color=self.red, bins=35)
        ax[1, 1].hist(y, orientation='horizontal', color=self.red, bins=35)
        ax[0, 1].set_axis_off()
        ax[0, 0].set_title('ML distribution')
        ax[1, 1].set_title('AP distribution')
        ax[1, 1].set_xlabel('# cells')
        ax[1, 0].set_xlabel('ML axis (µm)')
        ax[1, 1].yaxis.tick_right()
        ax[0, 0].set_ylabel('# cells')
        ax[1, 0].set_ylabel('AP axis (µm)')
        ax[1, 1].yaxis.set_label_position("right")
        ax[1, 0].set_ylim(550, 275)
        # ax[1, 0].set_xlim(300, 520)
        ax[1, 0].set_xlim(125, 575)
        ax[0, 0].set_xticks([])
        ax[1, 1].set_yticks([])
        # ax[1, 1].set_ylim(550, 275)
        # ax[1, 0].set_aspect('equal')
        # fig.suptitle('Anatomical distribution of tectal BPNs (n={})'.format(x.shape[0]))
        sns.despine()
        plt.subplots_adjust(wspace=.1, hspace=.1)
        plt.savefig(self.figpath + '/S_ANATOMICAL_DIST_BPN_TECTUM_ML.svg', bbox_inches='tight')
        plt.close()

    def plot_figS5a(self):

        fig, ax = plt.subplots(2, 1, figsize=(1.5, 3), dpi=400)
        ax = ax.reshape(-1)
        s = 3
        ax[0].scatter(self.sd_t[np.invert(np.isnan(self.sd_t['m_grating_5.0']))].bpi.values,
                      self.sd_t[np.invert(np.isnan(self.sd_t['m_grating_5.0']))]['m_grating_5.0'].values, s=s,
                      edgecolors='none',
                      alpha=.3, color='black', rasterized='True')

        ax[1].scatter(self.sd_t[np.invert(np.isnan(self.sd_t['m_loom_1250.4_12.0']))].bpi.values,
                      self.sd_t[np.invert(np.isnan(self.sd_t['m_loom_1250.4_12.0']))]['m_loom_1250.4_12.0'].values, s=s,
                      edgecolors='none', alpha=.3, color='black', rasterized='True')

        bpns = self.sd_t[(self.sd_t.bpi > .5) & (self.sd_t.area_idx == 4)]
        ax[0].scatter(bpns[np.invert(np.isnan(bpns['m_grating_5.0']))].bpi.values,
                      bpns[np.invert(np.isnan(bpns['m_grating_5.0']))]['m_grating_5.0'].values, s=s, edgecolors='none',
                      color=self.red, alpha=.8, rasterized='True')

        ax[1].scatter(bpns[np.invert(np.isnan(bpns['m_loom_1250.4_12.0']))].bpi.values,
                      bpns[np.invert(np.isnan(bpns['m_loom_1250.4_12.0']))]['m_loom_1250.4_12.0'].values, s=s,
                      edgecolors='none', color=self.red, alpha=.8, rasterized='True')

        ax[0].set_ylabel('mean dF/F Grating')
        ax[1].set_ylabel('mean dF/F Loom')
        ax[1].set_xlabel('BPI')

        ax[1].scatter(3, 2, color='black', s=20)
        ax[1].text(4.5, 2, 'all (n={})'.format(self.sd_t[np.invert(np.isnan(self.sd_t['m_grating_5.0']))].shape[0]),
                   fontsize=8,
                   rasterized='True')

        ax[1].scatter(3, 2.5, color=self.red, s=20)
        ax[1].text(4.5, 2.5, 'DT-BPNs (n={})'.format(bpns[np.invert(np.isnan(bpns['m_loom_1250.4_12.0']))].shape[0]),
                   fontsize=8, rasterized='True')

        plt.setp(ax, xlim=(-5, 5))  # , ylim=(-1, 6), xlabel='BPI')
        ax[1].set_ylim(-1, 3)
        ax[0].set_ylim(-1, 6)
        # plt.tight_layout()
        sns.despine(trim=True)
        plt.subplots_adjust(wspace=.6, hspace=0.2, left=0, right=1, bottom=0, top=1)

        plt.savefig(self.figpath + '/S5a.svg', bbox_inches='tight', dpi=800)
        plt.close()

    def plot_fig2c(self):

        unids = np.array([int(i) for i in self.roi_df.unid.values])
        iscell_pix = self.sd_bk.iscell.values[unids]
        thresh_pix = np.any(self.sd_bk[self.rstims].values > self.thresh, axis=1)[unids]
        unids_t = unids[(iscell_pix == 1) & (thresh_pix)]
        roi_df_t = self.roi_df[(iscell_pix == 1) & (thresh_pix)]

        roistack_cont = create_roistack(self.zstack, self.micron_scaling, roi_df_t, unids_t, self.sd_bk,
                                        scoring='mcont')
        roistack_bout = create_roistack(self.zstack, self.micron_scaling, roi_df_t, unids_t, self.sd_bk,
                                        scoring='mbout')
        roistack_bl = create_roistack(self.zstack, self.micron_scaling, roi_df_t, unids_t, self.sd_bk, scoring='mbl')

        tmp = (self.allmasks == 4)
        tmp[np.isnan(tmp)] = 0
        dtMax = tmp.max(axis=0).astype('uint8') * 255  # max projection
        _, binary = cv2.threshold(255 - dtMax, 225, 255, cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(dtMax, cv2.COLOR_GRAY2RGB)
        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  # get contours
        image = cv2.drawContours(image, contours, -1, (0, 255, 0), 4)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype('float')
        image[image != 150] = np.nan
        fig = plt.figure(figsize=(6, 1.75), dpi=400, constrained_layout=False)
        spec = gridspec.GridSpec(ncols=14, nrows=4, figure=fig,
                                 width_ratios=(0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, .8, .8))

        ax0 = fig.add_subplot(spec[:, 2:5])
        ax1 = fig.add_subplot(spec[:, 5:8], sharey=ax0)
        ax2 = fig.add_subplot(spec[:, 8:11], sharey=ax0)

        axes = [ax0, ax1, ax2]
        for rs, ax in zip([roistack_bl, roistack_cont, roistack_bout], axes):
            mp = plot_dv_slices(
                self.zstack,
                ax,
                rs,
                cmap='Reds',
                clim=(0, 1.2),
                roialpha=.9,
                dims=(1, 1),
                slicerange=(120, 180),
                mask=image,
                fs=(5, 5)
            )
        new_labels = ['No Stim', 'Continuous', 'Bout-like']

        for gax, label in zip(axes, new_labels):
            gax.set_title(label)
        axes[0].add_patch(matplotlib.patches.Rectangle((450, 550), 100, 5, color="black"))
        ax = fig.add_subplot(spec[0:2, 0:2])
        clb = plt.colorbar(mp, cax=ax)
        clb.set_ticks([0., 0.5, 1.])
        clb.set_ticklabels([0., 0.5, 1.])
        ax.set(xticks=[])
        ax.yaxis.set_ticks_position('left')
        ax.set_ylabel('Mean dF/F')
        ax.yaxis.set_label_position('left')
        plt.subplots_adjust(wspace=0.1, hspace=0, left=0, right=1, bottom=0, top=1)

        plt.savefig(self.figpath + '/fig2c_1.svg', bbox_inches='tight')
        plt.close()

        min_bpns = 30
        cols = ['fid', 'm_bl', '_m_cont', '_m_bout']
        dt_bpn = pd.DataFrame(self.sd_t[(self.sd_t.bpi >= .5) & (self.sd_t.area_idx == 4)], columns=cols)
        dt_bpn['fid'] = self.sd_t[(self.sd_t.bpi >= .5) & (self.sd_t.area_idx == 4)]['fid'].values

        fids, fidc = np.unique(dt_bpn.fid.values, return_counts=True)
        fids = [fids[i] for i in range(fids.shape[0]) if fidc[i] > min_bpns]
        fidbl = []
        for fid in self.sd_t.fid.values:

            if fid in fids:
                fidbl.append(True)
            else:
                fidbl.append(False)

        sdm = self.sd_t[(self.sd_t.area_idx == 4) & (fidbl)]

        mdf = pd.melt(sdm[cols], id_vars="fid", var_name="feature", value_name="mean dF/F")
        fig, ax = plt.subplots(figsize=(.75, 1.75), dpi=200)

        g = sns.pointplot(x='feature', y='mean dF/F', hue='fid', data=mdf, ax=ax, zorder=-1, ci=None, color='black')

        ax.legend_.remove()

        ax.set_ylim(0., .5)
        plt.setp(g.collections, alpha=.2, edgecolors='none', zorder=-1)  # for the markers
        plt.setp(g.lines, alpha=.2, lw=.7, zorder=-1)

        meanf = pd.melt(mdf.groupby(['fid', 'feature']).mean().unstack())
        meanf['fid'] = [i for i in mdf['fid'].unique()] * 3
        fax = sns.pointplot(x='feature', y='value', data=meanf, capsize=.0, ax=ax, color='black', marker='none',
                            linestyles='none', order=['m_bl', '_m_cont', '_m_bout'])
        fax.set(yticks=[0., 0.5], yticklabels=[0., 0.5])
        fax.yaxis.set_label_position("right")
        fax.yaxis.tick_right()
        sns.despine(ax=fax, trim=True)
        fax.spines['left'].set_visible(False)
        fax.spines['right'].set_visible(True)

        ax.scatter(range(3), meanf.groupby('feature').mean().values[:, 0][::-1], zorder=0, color='black', s=2)
        ax.legend_.remove()
        ax.set_xticklabels(new_labels, rotation=45, ha='right')
        ax.set_ylabel('mean dF/F in DT')
        ax.set_xlabel(None)
        ax.set_xlim(-.5, 3)
        plt.savefig(self.figpath + '/fig2c_2.svg', bbox_inches='tight', transparent=True)
        plt.close()

    def generate_bpn_kde(
            self,
            recompute=False
    ):

        bpncoords = self.sd_t[self.sd_t.bpi > .5][['x', 'y', 'z', 'bpi']]

        kdestacks = []
        kdes = []
        for bpnc, xbound in zip([bpncoords[bpncoords.x <= 366], bpncoords[bpncoords.x > 366]], [[0, 366], [366, 725]]):

            xgrid = np.arange(xbound[0], xbound[1], 5)
            ygrid = np.arange(0, self.zstack.shape[1], 5)
            zgrid = np.arange(0, self.zstack.shape[0], 5)

            XX, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
            xyz = np.vstack([XX.ravel(), Y.ravel(), Z.ravel()]).T
            kde = KernelDensity(metric='euclidean', bandwidth=10,  # was 16 before
                                kernel='gaussian', algorithm='auto')
            kde.fit(bpnc[['x', 'y', 'z']])
            if recompute:
                kdehalf = np.exp(kde.score_samples(xyz)).reshape(xgrid.size, ygrid.size, zgrid.size)
                kdestacks.append(kdehalf / kdehalf.sum() * bpnc.shape[0])

            kdes.append(kde)

        if recompute:

            mk = np.concatenate(kdestacks, axis=0)
            np.save(self.datapath + '/kdestacks_fig2g.npy', mk)

        else:

            mk = np.load(self.datapath + '/kdestacks_fig2g.npy')

        kdevis = np.moveaxis(mk, 2, 0)
        kdevis = np.moveaxis(kdevis, 1, 2)
        kdevis_res = resize(kdevis, self.zstack.shape)

        bpn_mask = np.zeros(kdevis_res.shape)
        bpn_mask[kdevis_res >= 0.0001] = kdevis_res[kdevis_res >= 0.0001]
        self.bpn_mask = (bpn_mask / bpn_mask.sum()) * self.sd_t[self.sd_t.bpi > .5].shape[0]

    def plot_fig2ef(self):

        self.generate_bpn_kde()
        cwalpha = abs(self.sd_t.bpi.values)
        cwalpha = np.clip(cwalpha, .0, .5)

        plot_orthogonal_scatter(
            self.zstack,
            self.sd_t,
            clim=(-.5, .5),
            cval=self.sd_t.bpi,
            roialpha=cwalpha,
            roisize=.7,
            cmap='coolwarm',
            allmasks=self.allmasks,
            mask=None,
            kde=None,
            fs=(3.5, 3.5),
            tag='BPIALL',
            clb=True,
            tvals=np.array([0.0008, 0.0012, 0.0024]),
            bpn_mask=None,
            save=True,
            figpath=self.figpath
        )
        figheight = (3.5 / ((650 - 75) + self.zstack.shape[0])) * (650 - 75)

        zyx = self.sd_t[self.sd_t.bpi > .5][['z', 'y', 'x']].values.astype(int)
        kdvals = self.bpn_mask[zyx[:, 0], zyx[:, 1], zyx[:, 2]]

        plot_horizontal_scatter(
            self.zstack,
            self.sd_t[(self.sd_t.bpi > .5)],
            clim=(0.0004, 0.0024),
            cval=kdvals,
            roialpha=.8,
            roisize=1.,
            cmap='Reds',
            points='equal',
            allmasks=self.allmasks,
            mask=None,
            kde=None,
            fs=(2.8, figheight),
            tag='DTBPNs_contrs',
            clb=True,
            tvals=np.array([0.0008, 0.0012, 0.0024]),
            bpn_mask=self.bpn_mask,
            ctalpha=.8,
            ctclr='Reds',
            save=True,
            figpath=self.figpath
        )

    def plot_fig2b(self):

        cmap = plt.get_cmap('coolwarm')
        norm = colors.Normalize(vmin=-10., vmax=-.75)
        mp = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

        path = self.datapath + '/Imaging data'
        coolwarm = plt.get_cmap('coolwarm')
        fid = 4
        unids = []

        date = self.sd[self.sd.fid == fid].date.unique()[0]
        fno = self.sd[self.sd.fid == fid].fno.unique()[0]

        fig, axes = plt.subplots(2, 2, figsize=(2.5, 2.5), dpi=400)
        planedict = {0: [99179], 1: [53061], 5: [117764], 4: [64590]}

        for plane, cfac, axno, zrange in zip([0, 5], [1.3, 2], [0, 1], [[300, 450, 170, 20], [50, 250, 250, 50]]):
            print(plane)

            stats, iscell = [np.load(
                os.path.join(
                    path,
                    r'{0}/fish{1}/plane{2}/{3}.npy'.format(date, fno, plane, i)
                ), allow_pickle=True) for i in ['stat', 'iscell']]
            av = tiff.imread(glob.glob(os.path.join(
                path,
                r'{0}/fish{1}/downsampled*registered_plane{2}_av_std.tif'.format(date, fno, plane)))[0])
            av = rotate(av, angle=180, resize=False, preserve_range=True, cval=av.max())
            ax = axes[axno, 0]

            # neurons in SD and stats should be in same order with ISCELL boolean
            xys = np.array([stats[i]['med'] for i in range(stats.shape[0])])
            nids = list(np.arange(0, len(stats), 1))
            print(len(nids), stats.shape)
            xys, ori = pixel_rotate(xys, angle=180)

            unids_plane = planedict[plane]
            unids.extend(unids_plane)
            for unid in unids_plane:

                nid = self.sd[self.sd.unid == unid].nid.values[0]
                if plane == 0:
                    rc = coolwarm(0)
                    tx = '# 1'
                    axes[axno, 1].text(xys[nids.index(nid), 0] - 16, xys[nids.index(nid), 1] - 16, tx,
                                       color='white', fontsize=16)

                else:
                    rc = coolwarm(255)
                    tx = '# 2'
                    axes[axno, 1].text(xys[nids.index(nid), 0] - 16, xys[nids.index(nid), 1] - 16, tx,
                                       color='white', fontsize=16)

                print('nid ', nid, self.sd[self.sd.unid == unid].bpi)
                axes[axno, 1].add_patch(
                    matplotlib.patches.Rectangle((xys[nids.index(nid), 0] - 8, xys[nids.index(nid), 1] - 8), 16, 16,
                                                 facecolor='none', edgecolor=rc, lw=2, zorder=1))

            plot_rois_multicolor_scatter(av, stats[iscell[:, 0].astype(bool)], ax, cfac=cfac)

            ax.set(xlim=(12, 500), ylim=(500, 12))
            ax.add_patch(
                matplotlib.patches.Rectangle((zrange[0], zrange[3]), 200, 200, facecolor='none', edgecolor='white',
                                             lw=1., linestyle=(0, (3, 1, 1, 1))))
            if plane == 4:
                ax.arrow(400, 400, -50, -50, fc='white', head_width=20, head_length=20, )
                ax.arrow(400, 400, -50, 50, fc='white', head_width=20, head_length=20, )
                ax.text(250, 320, 'A', rotation=45, color='white')
                ax.text(250, 480, 'L', rotation=45, color='white')

            if plane == 0:
                ax.add_patch(
                    matplotlib.patches.Rectangle((350, 450), 100, 10, facecolor='white', edgecolor='white', lw=1.))

            ax = axes[axno, 1]

            xys = np.array([stats[i]['med'] for i in range(stats.shape[0])])[iscell[:, 0].astype(bool)]
            xys, ori = pixel_rotate(xys, angle=180)
            zoombl = (zrange[0] < xys[:, 0]) & (xys[:, 0] < zrange[1]) & (zrange[3] < xys[:, 1]) & (
                    xys[:, 1] < zrange[2])
            stats_alt = stats[iscell[:, 0].astype(bool)][zoombl]
            plot_rois_multicolor(av, stats_alt, ax, cfac=cfac,
                                 colorvals=np.where(zoombl)[0] / len(stats[iscell[:, 0].astype(bool)]))

            ax.set(xlim=(zrange[0], zrange[1]), ylim=(zrange[2], zrange[3]))

            ax.add_patch(matplotlib.patches.Rectangle((zrange[0] + 1, zrange[3] + 1), (zrange[1] - zrange[0]) - 2,
                                                      (zrange[2] - zrange[3]) - 2, facecolor='none',
                                                      edgecolor='white', lw=2., linestyle=(0, (3, 1, 1, 1))))

        fig.text(-.1, .15, 'Plane 6     ...       Plane 1', rotation=90, ha='left', fontsize=12)
        plt.axis('off')
        plt.subplots_adjust(wspace=0.0, hspace=.2, left=.0, right=1, bottom=0, top=1)
        plt.savefig(self.figpath + '/IMAGING_PLANES_EXAMPLE.svg', bbox_inches='tight', dpi=1200)

        plt.close()

        _ = plot_example_traces(
            [unids[1], unids[0]],
            self.sd,
            mp,
            path=self.datapath + '/Imaging data',
            stimpath=self.datapath + '/Stimuli data',
            tranges=([600, 840], [1900, 2140], [3300, 3450]),
            figpath=self.figpath
        )

        alltraces, allunids, allstimparams, alldelays, resp, all_fids = get_traces_stim(

            self.sd.iloc[np.any([self.sd.unid == i for i in unids], axis=0)],
            path=self.datapath + '/Imaging data',
            stimpath=self.datapath + '/Stimuli data',
            get_resp=True,
            reganalysis=False
        )
        fig = plt.figure(figsize=(2.7, 3.6), dpi=200)
        grid = plt.GridSpec(4, 5, figure=fig, width_ratios=(1, 1, 1, 1, 1), height_ratios=(.5, 1, 1, 1), wspace=0.4,
                            hspace=.3)
        axes = [fig.add_subplot(grid[j, i]) for j in range(1, 3) for i in range(5)]
        for pid, c in zip([0, 1], [coolwarm(i) for i in [0, 255]]):
            for nidx in range(1):
                plot_mean_resp(axes[5 * pid:5 + 5 * pid], resp[pid], ridxs=[0, 2, 4, 5, 7], color=c, idx=nidx)
        plt.setp(axes, ylim=(-1, 3))
        [axi.set_axis_off() for axi in axes[:5]]
        [axi.set_axis_off() for axi in axes[6:]]
        freq = ['0.75', '1.5', '3', '6', '60']
        [axi.set_title(freq[axno]) for axno, axi in enumerate(axes[:5])]
        axes[0].text(-70, 5.1, r'$\mathit{f}$(Hz)')
        ax = axes[5]

        sns.despine(ax=ax, trim=True)
        ax.set_ylabel('dF/F')
        ax.set_xlabel('time (s)')

        ax0 = fig.add_subplot(grid[0, :])
        ax0.plot([0., 2.9], [.5, .5], color='black')
        ax0.plot([3.1, 4.9], [.5, .5], color='black')
        ax0.text(1.5, 0.8, 'Bout-like', ha='center')
        ax0.text(4., 0.8, 'Continuous', ha='center')
        ax0.set_xlim(0, 5)
        ax0.set_ylim(0, 1)
        ax0.set_axis_off()

        plt.savefig(self.figpath + '/EXAMPLE_NEURONS_MEAN_RESP.svg', bbox_inches='tight', dpi=200)

        plt.close()

    def plot_fig4_figS8c(self):

        abldict = {
            '20211209': {1: 'ctr', 2: 'abl', 3: 'ctr', 4: 'abl', 5: 'ctr', 6: 'abl', 7: 'ctr', 8: 'abl', 9: 'ctr',
                         10: 'abl'},
            '20220210': {2: 'abl', 3: 'abl', 4: 'abl', 5: 'ctr', 6: 'ctr', 7: 'ctr', 8: 'ctr', 9: 'abl'}
        }

        sd = pickle.load(open(self.datapath + '/sumdict_figS8c.p', 'rb'))
        roi_df = pickle.load(open(self.datapath + '/roidf_figS8c.p', 'rb'))

        mean_coords = roi_df.groupby('unid').mean()
        sd['x'] = mean_coords.x.values * (1. / self.micron_scaling[0])
        sd['y'] = mean_coords.y.values * (1. / self.micron_scaling[1])
        sd['z'] = mean_coords.z.values * (
                1. / self.micron_scaling[1])  # not division by micron scaling because 2 microns/pix in and out.
        for ci, c in enumerate(['z', 'y', 'x']):
            sd[c] = np.clip(sd[c], 0, self.zstack.shape[ci] - 1)
        maskids = np.array(
            self.allmasks[
                sd.z.values.astype(int),
                sd.y.values.astype(int),
                sd.x.values.astype(int)
            ]).ravel()

        sd['area_idx'] = maskids
        sd['area'] = [self.areadict[int(i)] for i in maskids]

        dstims = sorted([i for i in sd.keys() if i.startswith('m_')])
        br = sd[dstims].values
        print(dstims)
        sd['dff'] = np.sum(br, axis=1)

        bout = sd[['m_True_1.5_1.6', 'm_False_1.5_1.6']].values.mean(axis=1)
        cont = sd[['m_True_60.0_1.6', 'm_False_60.0_1.6']].values.mean(axis=1)

        sd['bpi'] = (bout - cont) / (bout + cont)
        sd['bpi_old'] = (bout - cont)  # /(bout+cont)
        sd['_m_bout'] = bout
        sd['_m_cont'] = cont

        boutr = sd[['regsc_True_1.5_1.6', 'regsc_False_1.5_1.6']].values.mean(axis=1)
        contr = sd[['regsc_True_60.0_1.6', 'regsc_False_60.0_1.6']].values.mean(axis=1)
        sd['_regsc_bout'] = boutr
        sd['_regsc_cont'] = contr
        sd['bpi_reg'] = (boutr - contr) / (boutr + contr)

        group = np.empty(sd.shape[0], dtype='object')
        for fid in sd.fid.unique():
            group[sd.fid == fid] = abldict[sd[sd.fid == fid].date.unique()[0]][
                sd[sd.fid == fid].fno.unique()[0].astype(int)]

        sd['group'] = group
        sd_t = sd[sd.iscell]

        # Thresholding for all cells for cell counts
        mstims = ['_m_bout', '_m_cont', 'm_grating_5.0', 'm_loom_1250.4_12.0']

        mstimlabels = ['Dot 1.5 Hz', 'Dot 60 Hz', 'Grating', 'Looming']
        tval = 90

        fig = plt.figure(figsize=(8, 4), dpi=400)
        spec = gridspec.GridSpec(ncols=5, nrows=3, figure=fig, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1, 1])

        cyans = colors.LinearSegmentedColormap.from_list('mg', ['white', '#19E6D3'])
        ly = colors.LinearSegmentedColormap.from_list('ly', ['white', 'gold'])

        sdb_all = pd.DataFrame()
        pvals = []

        rylim = 600
        for stim, axno in zip(mstims, range(len(mstims))):

            stimthresh = np.nanpercentile(sd_t[stim].values, tval)
            if stim in ['_regsc_bout', '_regsc_cont', '_m_bout', '_m_cont', 'bpi']:

                sdb = sd_t[(sd_t.area_idx == 4) & (sd_t[stim] > stimthresh)]
                print(sdb.shape, stim, sdb.fid.unique())

                maskm = (self.allmasks == 4).mean(0)
                mclr = ly

            else:

                sdb = sd_t[(sd_t.area_idx == 7) & (sd_t[stim] > stimthresh)]
                print(sdb.shape, stim, sdb.fid.unique())

                maskm = (self.allmasks == 7).mean(0)
                mclr = cyans

            for gno, group in enumerate(['ctr', 'abl']):

                ax = fig.add_subplot(spec[gno, axno])
                print(gno, axno)
                av = np.nanmax(self.zstack[130: 180], 0)
                ax.imshow(av, cmap='Greys', origin='lower', alpha=.9, aspect='auto', clim=(100, 350))
                ax.imshow(maskm, cmap=mclr, alpha=np.clip(maskm / maskm.max(), 0, .3), aspect='equal')

                cvals = sdb[sdb.group == group][stim]
                cvals = zscore(cvals)
                ax.scatter(sdb[sdb.group == group].x,
                           sdb[sdb.group == group].y,
                           c=cvals,
                           cmap='Reds',
                           edgecolors='none',
                           vmin=-.5,
                           vmax=1,
                           alpha=.7,
                           s=5,
                           rasterized=True,
                           zorder=4
                           )

                ax.set_axis_off()
                plt.setp(ax, xticks=[], yticks=[], xlim=(275, 460), ylim=(450, 300))

                if gno == 0:
                    ax.set_title(mstimlabels[axno])
                    if axno == 0:
                        ax.add_patch(matplotlib.patches.Rectangle((280, 440), 50, 5, color="black"))

            sdbs = sdb.groupby(['fid', 'group']).sum()
            group = [i[1] for i in sdbs.index]
            sdbs['stim'] = stim
            sdbs['group'] = group
            sdb_all = pd.concat([sdb_all, sdbs])

            count_ctr = sdbs[sdbs.group == 'ctr'].iscell.values
            count_abl = sdbs[sdbs.group == 'abl'].iscell.values

            print(stim, np.median(count_ctr), np.median(count_abl), count_ctr.std(), count_abl.std())
            stats, pval = mannwhitneyu(count_ctr, count_abl, alternative='two-sided')
            print(stats, pval)
            pvals.append(pval)

        ax0 = fig.add_subplot(spec[2, 0:4])
        sns.stripplot(data=sdb_all, y='iscell', x='stim', hue='group', s=6, alpha=.5, dodge=.9, jitter=True, ax=ax0,
                      palette=['gray', 'r'], clip_on=False)

        g = sns.pointplot(data=sdb_all, y='iscell', ci='sd', estimator=np.median, x='stim', hue='group', dodge=.2,
                          ax=ax0, palette=['black', 'black'],
                          linestyles=['none'] * len(mstims), capsize=0, errwidth=1.5, markers='o', clip_on=False)

        ax0.legend_.remove()
        ax0.set_ylabel('# cells')
        ax0.set_ylim(0, rylim)
        ax0.set_xlabel('')
        sns.despine(ax=ax0, trim=False)
        ax0.spines['left'].set_visible(False)
        ax0.spines['right'].set_visible(True)

        ax0.yaxis.set_label_position("right")
        ax0.yaxis.tick_right()

        for pno, pval_all in enumerate(pvals):
            if pval_all > .05:
                sig = 'n.s.'
                rno = round(pval_all, 1)
            elif .05 > pval_all > .01:
                rno = round(pval_all, 3)
                sig = '*'.format(pval)
            elif .01 > pval_all > .001:
                rno = round(pval_all, 3)
                sig = '**'.format(pval)
            elif .001 > pval_all:
                rno = '<0.001'
                sig = '***'.format(pval)
            ax0.text(pno, rylim - 50, 'p={}'.format(rno), fontsize=10, ha='center', rotation=0)
        ax0.set_xticklabels(mstimlabels, rotation=45, ha='right')

        ax = fig.add_subplot(spec[0, -1])
        norm = colors.Normalize(vmin=0, vmax=1)
        axclb = inset_axes(ax,
                           width="10%",
                           height="30%",
                           loc='upper center')

        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='Reds'), cax=axclb)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['-0.5', '1'])
        axclb.set_ylabel('zscore', fontsize=10)
        axclb.yaxis.set_label_position('left')
        ax.set_axis_off()

        ax.text(3.2, 3.1, 'DT')
        ax.add_patch(matplotlib.patches.Rectangle((2.3, 3), .8, .5, color='gold', alpha=.3))

        ax.text(3.2, 2.1, 'Pretectum')
        ax.add_patch(matplotlib.patches.Rectangle((2.3, 2), .8, .5, color='#19E6D3'))
        ax.set_xlim(1, 4)
        ax.set_ylim(2, 6)

        ax = fig.add_subplot(spec[1, -1])
        ax.scatter([.2, .2], [.15, .35], c=['r', 'gray'], s=30)
        ax.text(.4, .1, 'Abl')
        ax.text(.4, .3, 'Ctr')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_axis_off()

        plt.subplots_adjust(wspace=0.05, hspace=0.01, left=0, right=1, bottom=0, top=1)
        plt.savefig(self.figpath + '/S8c.svg', bbox_inches='tight', dpi=800)
        plt.close()

        # Thresholding for all cells for cell counts
        mstims = ['_m_bout', '_m_cont', 'm_grating_5.0', 'm_loom_1250.4_12.0']

        tval = 90
        fig = plt.figure(figsize=(2.3, 2), dpi=200)
        spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, width_ratios=[1, .01, .7], height_ratios=[1, 1])
        ly = colors.LinearSegmentedColormap.from_list('ly', ['white', 'gold'])

        bpnylim = 250

        thresh = np.nanpercentile(sd_t[['m_True_1.5_1.6', 'm_False_1.5_1.6', 'm_True_60.0_1.6', 'm_False_60.0_1.6']],
                                  tval, axis=0)
        sdbpn = sd_t[
            np.any(sd_t[['m_True_1.5_1.6', 'm_False_1.5_1.6', 'm_True_60.0_1.6', 'm_False_60.0_1.6']] > thresh, axis=1)]
        glabels = ['Ctr', 'Abl']
        for gno, group in enumerate(['ctr', 'abl']):

            ax = fig.add_subplot(spec[gno, 0])
            print(gno, axno)
            s = sdbpn[(sdbpn.bpi > .5) & (sdbpn.group == group) & (sdbpn.area_idx == 4)]
            cvals = np.clip(s[s.group == group].bpi, -5, 5)

            cvals = zscore(cvals)
            ax.scatter(s.x,
                       s.y,
                       # color='black',
                       c=s['_m_bout'].values,
                       cmap='Reds',
                       vmin=0,
                       vmax=3,
                       edgecolors='none',
                       alpha=.8,
                       s=4,
                       rasterized=True,
                       zorder=4
                       )

            maskm = (self.allmasks == 4).mean(0)
            ax.imshow(av, cmap='Greys', origin='lower', alpha=.9, aspect='auto', clim=(100, 350))

            ax.imshow(maskm, cmap=ly, alpha=np.clip(maskm / maskm.max(), 0, .3), aspect='equal')

            plt.setp(ax, xticks=[], yticks=[], xlim=(275, 460), ylim=(450, 300))
            ax.set_ylabel(glabels[gno])
            ax.yaxis.set_label_position('left')

            for d in ['top', 'bottom', 'left', 'right']:
                ax.spines[d].set_visible(False)
            if gno == 0:
                ax.set_title('DT-BPNs')

            else:
                ax.add_patch(matplotlib.patches.Rectangle((280, 440), 50, 5, color="black"))

        ax = fig.add_subplot(spec[0, -1])
        norm = colors.Normalize(vmin=0, vmax=3)
        axclb = inset_axes(ax,
                           width="5%",
                           height="60%",
                           loc='upper left')

        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='Reds'), cax=axclb)
        cbar.set_ticks([0, 3])
        cbar.set_ticklabels(['0', '3'])
        axclb.set_ylabel('Mean dF/F', fontsize=10)
        axclb.yaxis.set_label_position('right')
        ax.set_axis_off()

        ax.set_xlim(1, 4)
        ax.set_ylim(2, 6)

        ax = fig.add_subplot(spec[1, -1])

        ax.set_ylim(-.2, .8)
        ax.set_xlim(.2, 1)
        ax.set_axis_off()

        plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0, right=1, bottom=0, top=1)
        plt.savefig(self.figpath + '/fig4_G.svg', bbox_inches='tight', dpi=800)
        plt.close()

        fig, ax = plt.subplots(figsize=(0.7, 1.8), dpi=200)
        fid_ctr, bpn_ctr = np.unique(sdbpn[(sdbpn.bpi > .5) & (sdbpn.group == 'ctr') & (sdbpn.area_idx == 4)].fid,
                                     return_counts=True)
        fid_abl, bpn_abl = np.unique(sdbpn[(sdbpn.bpi > .5) & (sdbpn.group == 'abl') & (sdbpn.area_idx == 4)].fid,
                                     return_counts=True)

        len_abl = len(sd[sd.group == 'abl'].fid.unique())
        len_ctr = len(sd[sd.group == 'ctr'].fid.unique())

        if len(bpn_abl) < len_abl:
            bpn_abl = list(bpn_abl) + [0] * (len_abl - len(bpn_abl))
        if len(bpn_ctr) < len_ctr:
            bpn_ctr = list(bpn_ctr) + [0] * (len_ctr - len(bpn_ctr))
            print(bpn_ctr)

        dat = pd.DataFrame(np.concatenate([np.stack([bpn_ctr, np.repeat(['ctr'], len_ctr)]).T.astype('object'),
                                           np.stack([bpn_abl, np.repeat(['abl'], len_abl)]).T.astype('object')
                                           ]),
                           columns=['ncells', 'group'])
        dat = dat.astype({'ncells': int, 'group': str})
        cd = cohen_d(bpn_ctr, bpn_abl)
        stats, pval = mannwhitneyu(bpn_ctr, bpn_abl, alternative='two-sided')
        print(pval)
        if pval < 0.001:
            pval = '<0.001'
        else:
            pval = np.around(pval, 3)
            print(pval)
        ax.text(-0.2, bpnylim, 'd=-{}'.format(round(cd, 2)), fontsize=10, ha='left', rotation=0)
        ax.text(-0.2, bpnylim - 20, 'p={}'.format(pval), fontsize=10, ha='left', rotation=0)
        sns.stripplot(data=dat, x=[0] * dat.shape[0], hue='group', y='ncells', alpha=.5, dodge=.9, jitter=True, ax=ax,
                      palette=['gray', 'r'], clip_on=False)
        sns.pointplot(data=dat, x=[0] * dat.shape[0], hue='group', y='ncells', ci='sd', markers='o',
                      estimator=np.median, dodge=.3, ax=ax, palette=['black', 'black'],
                      linestyles=['none'] * len(mstims), capsize=0, errwidth=1.5, zorder=5)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.scatter([.4, .4], [150, 130], c=['gray', 'r'], s=20, clip_on=False)
        ax.text(.6, 145, 'Ctr', fontsize=8)
        ax.text(.6, 125, 'Abl', fontsize=8)

        plt.setp(ax, ylabel='# cells', ylim=(0, 200), xlim=(-.4, .4))

        ax.set_ylabel('DT-BPNs')
        ax.set_xticklabels([''], ha='center', rotation=0)
        ax.legend_.remove()
        plt.savefig(self.figpath + '/fig4_H.svg', bbox_inches='tight', dpi=800)
        plt.close()

        slabels = [

            'Grating',
            '1.5 Hz',
            '1.5 Hz',
            '60 Hz',
            '60 Hz',
            '1.5 Hz',
            '1.5 Hz',
            '60 Hz',
            '60 Hz',
            'Grating',
            'Dim D-B',
            'Dim B-D',
            'Loom',
            'Loom'

        ]
        fig = plt.figure(figsize=(2, 1.8), dpi=200)
        spec = gridspec.GridSpec(ncols=1, nrows=4, figure=fig, height_ratios=[.7, 1, .7, 1])
        axes = [fig.add_subplot(spec[i]) for i in range(4)]

        for group, clr, sax in zip(['ctr', 'abl'], ['black', self.red], [axes[:2], axes[2:4]]):

            gbpn = sdbpn[(sdbpn.bpi > .5) & (sdbpn.group == group) & (sdbpn.area_idx == 4)]
            alltraces, allunids, allstimparams, alldelays, allresp, allfids = get_traces_stim(gbpn,
                                                                                              path=self.datapath + '/Imaging data',
                                                                                              stimpath=self.datapath + '/Stimuli data')
            traces_ = np.concatenate(alltraces, axis=0)
            ax = sax[0]
            ax1 = sax[1]

            for sno, sp in enumerate(allstimparams[0]):
                if not 0 < sno < 9:
                    continue
                if group == 'ctr':
                    ax.text(sp[1], .8, slabels[sno], rotation=45, fontsize=10, ha='left')
                else:
                    ax1.axvline(sp[1], color='black', lw=.5, ymin=0, ymax=3.5, clip_on=False, alpha=.5)

            ax.set_xlim(50, 400)
            ax1.set_xlim(50, 400)

            meantr = np.mean(traces_, axis=0)
            ax.plot(meantr, color=clr, lw=1.5)
            ax.fill_between(range(600),
                            meantr - np.std(traces_, axis=0),
                            meantr + np.std(traces_, axis=0),
                            alpha=.3,
                            edgecolor='none',
                            facecolor=clr
                            )
            tn = 0
            for tr in traces_[gbpn['_m_bout'].argsort()[::-1]][1:4]:
                ax1.plot(tr - tn, lw=.7, color=clr)
                tn += 1
            for tax in ['top', 'bottom', 'right']:
                ax.spines[tax].set_visible(False)
                ax.spines[tax].set(bounds=(0, .1))
            if group == 'ctr':

                ax1.set_xticks([])
                ax1.text(-50, -2.5, 'dF/F', rotation=90)
                for tax in ['top', 'bottom', 'right']:
                    ax1.spines[tax].set_visible(False)
            else:
                ax1.set_xticks([50, 75])
                ax1.set_xticklabels(['25 s', ''], ha='left')
                sns.despine(ax=ax1, trim=True)

            plt.setp(ax, yticks=[0, .5], yticklabels=[0, 0.5], xticks=[])
            plt.setp(ax1, yticks=[-1, 0], yticklabels=[0, 1])
            ax.spines['left'].set(bounds=(0, .5))
            ax1.spines['left'].set(bounds=(-1, -0))
            plt.subplots_adjust(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)
        plt.savefig(self.figpath + '/fig4_F.svg', bbox_inches='tight', dpi=800)

        plt.close()

    def plot_figs8d(self):

        sd1 = pickle.load(open(self.datapath + '/sumdict_figS8d_1.p', 'rb'))
        roi_df1 = pickle.load(open(self.datapath + '/roidf_figS8d_1.p', 'rb'))

        sd2 = pickle.load(open(self.datapath + '/sumdict_figS8d_2.p', 'rb'))
        roi_df2 = pickle.load(open(self.datapath + '/roidf_figS8d_2.p', 'rb'))

        sd2['unid'] += sd1['unid'].values.max() + 1
        roi_df2['unid'] += roi_df1['unid'].values.max() + 1
        sd2['fid'] += sd1['fid'].values.max() + 1

        sd = pd.concat([sd1, sd2])
        roi_df = pd.concat([roi_df1, roi_df2])

        zstack = tiff.imread(self.datapath + '/7dpf_AVG_H2BGCaMP6s_atlas.tif')
        allmasks = tiff.imread(self.datapath + '/brainmasks_7dpf.tiff')[::-1, :, :]
        micron_scaling = [.99, .99, 1]

        mean_coords = roi_df.groupby('unid').mean()
        sd['x'] = mean_coords.x.values * (1. / micron_scaling[0])
        sd['y'] = mean_coords.y.values * (1. / micron_scaling[1])
        sd['z'] = mean_coords.z.values * (
                1. / micron_scaling[2])  # not division by micron scaling because 2 microns/pix in and out.

        maskids = np.array(
            allmasks[
                sd.z.values.astype(int),
                sd.y.values.astype(int),
                sd.x.values.astype(int)
            ]).ravel()
        sd['area_idx'] = maskids

        dstims = sorted([i for i in sd.keys() if i.startswith('m_')])
        br = sd[dstims].values
        print(dstims)
        sd['dff'] = np.sum(br, axis=1)

        bout = sd[['m_True_1.5_1.6', 'm_False_1.5_1.6']].values.mean(axis=1)
        cont = sd[['m_True_60.0_1.6', 'm_False_60.0_1.6']].values.mean(axis=1)

        sd['bpi'] = (bout - cont) / (bout + cont)
        sd['bpi_old'] = (bout - cont)  # /(bout+cont)
        sd['_m_bout'] = bout
        sd['_m_cont'] = cont

        boutr = sd[['regsc_True_1.5_1.6', 'regsc_False_1.5_1.6']].values.mean(axis=1)
        contr = sd[['regsc_True_60.0_1.6', 'regsc_False_60.0_1.6']].values.mean(axis=1)
        sd['_regsc_bout'] = boutr
        sd['_regsc_cont'] = contr
        sd['bpi_reg'] = (boutr - contr) / (boutr + contr)

        group = np.empty(sd.shape[0], dtype='object')
        for fid in sd.fid.unique():
            print(fid)
            if sd[sd.fid == fid].date.unique()[0] != '20220203':

                print('unique groups', np.unique(sd[sd.fid == fid].is81c))
                if len(np.unique(sd[sd.fid == fid].is81c)) == 1:

                    if np.unique(sd[sd.fid == fid].is81c)[0] == 0:

                        group[sd.fid == fid] = 'ctr'

                    else:

                        group[sd.fid == fid] = 'abl'
                else:
                    print(fid, sd[sd.fid == fid].date.unique()[0])
                    print('Found more than one group in one fish!')
                    break

            else:

                if sd[sd.fid == fid].fno.unique()[0] % 2 == 0:

                    group[sd.fid == fid] = 'abl'

                else:

                    group[sd.fid == fid] = 'ctr'

        sd['group'] = group
        sd_t = sd[sd.iscell]
        mstims = ['_m_bout', '_m_cont', 'm_grating_5.0', 'm_loom_1250.4_12.0']
        mstimlabels = ['Dot 1.5 Hz', 'Dot 60 Hz', 'Grating', 'Looming']

        tval = 90

        fig = plt.figure(figsize=(9, 3.5), dpi=300)
        spec = gridspec.GridSpec(ncols=7, nrows=3, figure=fig, width_ratios=[1, .1, 1, 1, 1, 1, 1],
                                 height_ratios=[1, 1, 1])

        cyans = colors.LinearSegmentedColormap.from_list('mg', ['white', '#19E6D3'])
        ly = colors.LinearSegmentedColormap.from_list('ly', ['white', 'gold'])

        sdb_all = pd.DataFrame()
        pvals = []
        zslice = zstack[160:170, :, :]

        for stim, axno in zip(mstims, range(len(mstims))):

            if stim in ['_regsc_bout', '_regsc_cont', '_m_bout', '_m_cont', 'bpi']:

                stimthresh = np.nanpercentile(sd_t[stim].values, tval)
                sdb = sd_t[(sd_t.area_idx == 3) & (sd_t[stim] > stimthresh)]
                print(sdb.shape, stim, sdb.fid.unique())

                maskm = (allmasks == 3)[160:170].mean(0)
                mclr = ly
                ylim = (385, 225)

            else:

                stimthresh = np.nanpercentile(sd_t[stim].values, tval)
                sdb = sd_t[(sd_t.area_idx == 10) & (sd_t[stim] > stimthresh)]
                print(sdb.shape, stim, sdb.fid.unique())

                maskm = (allmasks == 10)[160:170].mean(0)
                mclr = cyans
                ylim = (410, 250)

            for gno, group in enumerate(['ctr', 'abl']):

                ax = fig.add_subplot(spec[gno, axno + 2])
                print(gno, axno)

                ax.imshow(zslice.mean(0), cmap='Greys', clim=(0, 180), aspect='equal')
                ax.imshow(maskm, cmap=mclr, alpha=np.clip(maskm / maskm.max(), 0, .3), aspect='equal')

                cvals = sdb[sdb.group == group][stim]
                cvals = zscore(cvals)
                ax.scatter(sdb[sdb.group == group].x,
                           sdb[sdb.group == group].y,
                           c=cvals,
                           cmap='Reds',
                           edgecolors='none',
                           vmin=-.5,
                           vmax=1.,
                           alpha=.7,
                           s=5,
                           rasterized=True,
                           zorder=4
                           )

                plt.setp(ax, ylim=ylim, xlim=(175, 400), xticks=[], yticks=[])
                ax.set_axis_off()
                if gno == 0:
                    ax.set_title(mstimlabels[axno])
            sdbs = sdb.groupby(['fid', 'group']).sum()
            group = [i[1] for i in sdbs.index]
            sdbs['stim'] = stim
            sdbs['group'] = group
            sdb_all = pd.concat([sdb_all, sdbs])

            count_ctr = sdbs[sdbs.group == 'ctr'].iscell.values
            count_abl = sdbs[sdbs.group == 'abl'].iscell.values

            print(stim, np.median(count_ctr), np.median(count_abl), count_ctr.std(), count_abl.std())
            stats, pval = mannwhitneyu(count_ctr, count_abl, alternative='two-sided')
            print(stats, pval)
            pvals.append(pval)

        ax0 = fig.add_subplot(spec[2, 2:6])
        sns.stripplot(data=sdb_all, y='iscell', x='stim', hue='group', hue_order=['ctr', 'abl'], s=6, alpha=.5,
                      dodge=.9, jitter=True, ax=ax0, palette=['gray', 'r'], clip_on=False)
        g = sns.pointplot(data=sdb_all, y='iscell', ci='sd', estimator=np.median, x='stim', hue='group',
                          hue_order=['ctr', 'abl'], dodge=.2, ax=ax0, palette=['black', 'black'],
                          linestyles=['none'] * len(mstims), capsize=0, errwidth=1.5, markers='o', clip_on=False)

        ax0.legend_.remove()
        ax0.set_ylabel('# cells')
        ax0.set_ylim(0, 200)
        ax0.set_xlabel('')
        sns.despine(ax=ax0, trim=False)
        ax0.spines['left'].set_visible(False)
        ax0.spines['right'].set_visible(True)

        ax0.yaxis.set_label_position("right")
        ax0.yaxis.tick_right()

        for pno, pval_all in enumerate(pvals):
            if pval_all > .05:
                sig = 'n.s.'
                rno = 2
            elif .05 > pval_all > .01:
                rno = 2
                sig = '*'.format(pval)
            elif .01 > pval_all > .001:
                rno = 3
                sig = '**'.format(pval)
            elif .001 > pval_all:
                rno = 4
                sig = '***'.format(pval)
            ax0.text(pno, 180, 'p={}'.format(round(pval_all, rno)), fontsize=10, ha='center', rotation=0)
        ax0.set_xticklabels(mstimlabels, rotation=45, ha='right')

        thresh = np.nanpercentile(sd_t[['m_True_1.5_1.6', 'm_False_1.5_1.6', 'm_True_60.0_1.6', 'm_False_60.0_1.6']],
                                  tval, axis=0)
        sdbpn = sd_t[
            np.any(sd_t[['m_True_1.5_1.6', 'm_False_1.5_1.6', 'm_True_60.0_1.6', 'm_False_60.0_1.6']] > thresh, axis=1)]
        for gno, group in enumerate(['ctr', 'abl']):

            ax = fig.add_subplot(spec[gno, 0])
            print(gno, axno)
            s = sdbpn[(sdbpn.bpi > .5) & (sdbpn.group == group) & (sdbpn.area_idx == 3)]

            ax.scatter(s[s.group == group].x,
                       s[s.group == group].y,
                       color='black',
                       edgecolors='none',
                       alpha=.7,
                       s=5,
                       rasterized=True,
                       zorder=4
                       )

            maskm = (allmasks == 3)[160:170].mean(0)
            ax.imshow(zslice.mean(0), cmap='Greys', clim=(0, 180), aspect='equal')
            ax.imshow(maskm, cmap=ly, alpha=np.clip(maskm / maskm.max(), 0, .3), aspect='equal')

            plt.setp(ax, ylim=(385, 225), xlim=(175, 400), xticks=[], yticks=[])
            ax.set_ylabel(group)
            ax.yaxis.set_label_position('left')

            for d in ['top', 'bottom', 'left', 'right']:
                ax.spines[d].set_visible(False)
            if gno == 0:
                ax.set_title('DT-BPNs')
            else:
                ax.add_patch(matplotlib.patches.Rectangle((185, 375), 100, 5, color="black"))

        fid_ctr, bpn_ctr = np.unique(sdbpn[(sdbpn.bpi > .5) & (sdbpn.group == 'ctr') & (sdbpn.area_idx == 3)].fid,
                                     return_counts=True)
        fid_abl, bpn_abl = np.unique(sdbpn[(sdbpn.bpi > .5) & (sdbpn.group == 'abl') & (sdbpn.area_idx == 3)].fid,
                                     return_counts=True)

        len_abl = len(sd[sd.group == 'abl'].fid.unique())
        len_ctr = len(sd[sd.group == 'ctr'].fid.unique())

        if len(bpn_abl) < len_abl:
            bpn_abl = list(bpn_abl) + [0] * (len_abl - len(bpn_abl))
        if len(bpn_ctr) < len_ctr:
            bpn_ctr = list(bpn_ctr) + [0] * (len_ctr - len(bpn_ctr))
            print(bpn_ctr)

        dat = pd.DataFrame(np.concatenate([np.stack([bpn_abl, np.repeat(['abl'], len_abl)]).T.astype('object'),
                                           np.stack([bpn_ctr, np.repeat(['ctr'], len_ctr)]).T.astype('object')]),
                           columns=['ncells', 'group'])
        dat = dat.astype({'ncells': int, 'group': str})
        ax = fig.add_subplot(spec[-1, 0])

        stats, pval = mannwhitneyu(bpn_ctr, bpn_abl, alternative='two-sided')
        ax.text(0., 80, 'p={}'.format(round(pval, 2)), fontsize=10, ha='center', rotation=0)
        sns.stripplot(data=dat, x=[0] * dat.shape[0], hue='group', hue_order=['ctr', 'abl'], y='ncells', s=6, alpha=.5,
                      dodge=.9, jitter=True, ax=ax, palette=['gray', 'r'], clip_on=False)
        sns.pointplot(data=dat, x=[0] * dat.shape[0], hue='group', hue_order=['ctr', 'abl'], y='ncells', ci='sd',
                      estimator=np.median, dodge=.3, ax=ax, palette=['black', 'black'],
                      linestyles=['none'] * len(mstims), capsize=0, errwidth=1.5, markers='o', clip_on=False)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)

        ax.set_ylabel('# cells')
        ax.set_ylim(0, 100)
        ax.set_xticklabels(['DT-BPNs'], ha='right', rotation=45)
        ax.legend_.remove()

        ax = fig.add_subplot(spec[0, -1])
        norm = colors.Normalize(vmin=0, vmax=1)
        axclb = inset_axes(ax,
                           width="10%",
                           height="30%",
                           loc='upper center')

        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='Reds'), cax=axclb)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['-0.5', '1'])
        axclb.set_ylabel('zscore', fontsize=10)
        axclb.yaxis.set_label_position('left')
        ax.set_axis_off()

        ax.text(3., 3.1, 'DT')
        ax.add_patch(matplotlib.patches.Rectangle((2., 3), .8, .5, color='gold', alpha=.3))

        ax.text(3., 2.1, 'Pretectum')
        ax.add_patch(matplotlib.patches.Rectangle((2., 2), .8, .5, color='#19E6D3', alpha=.3))
        ax.set_xlim(1, 4)
        ax.set_ylim(2, 7)

        ax = fig.add_subplot(spec[1, -1])
        ax.scatter([.2, .2], [.15, .35], c=['r', 'gray'], s=30)
        ax.text(.4, .1, 'abl')
        ax.text(.4, .3, 'ctr')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_axis_off()
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.savefig(self.figpath + '/S8d.svg', bbox_inches='tight', dpi=800)
        plt.close()

    def plot_figS5bd(self):

        sd = pickle.load(open(self.datapath + '/sumdict_fig2i.p', 'rb'))
        roi_df = pickle.load(open(self.datapath + '/roidf_fig2i.p', 'rb'))
        mean_coords = roi_df.groupby('unid').mean()

        sd['x'] = mean_coords.x.values * (1. / self.micron_scaling[0])
        sd['y'] = mean_coords.y.values * (1. / self.micron_scaling[1])
        sd['z'] = mean_coords.z.values * (1. / self.micron_scaling[1])
        maskids = np.array(
            self.allmasks[
                sd.z.values.astype(int),
                sd.y.values.astype(int),
                sd.x.values.astype(int)
            ]).ravel()
        sd['area_idx'] = maskids
        sd['area'] = [self.areadict[int(i)] for i in maskids]

        rs = sd[[i for i in sd.keys() if i.startswith('rs_') or i.startswith('rs_')]].values
        sd['rs_mean'] = np.nanmean(rs, axis=1)
        sd['rs_max'] = np.nanmax(rs, axis=1)

        dstims = sorted([i for i in sd.keys() if i.startswith('m_')])
        br = sd[dstims].values
        sd['dff'] = np.sum(br, axis=1)

        sd_t = sd[sd['iscell']]
        print(sd_t.shape)

        # exclude cells based on 95th percentile of any stim. response
        thresh = np.nanpercentile(sd_t[dstims], 95, axis=0)
        sd_t = sd_t[np.any(sd_t[dstims].values > thresh, axis=1)]

        bout = sd_t[['m_True_1.5_1.6', 'm_False_1.5_1.6']].values.mean(axis=1)
        cont = sd_t[['m_True_60.0_1.6', 'm_False_60.0_1.6']].values.mean(axis=1)
        sd_t['_m_bout'] = bout
        sd_t['_m_cont'] = cont
        sd_t['_m_boutwf'] = sd_t[['m_Image_1.5_False', 'm_Image_1.5_True']].values.mean(axis=1)
        sd_t['_m_contwf'] = sd_t[['m_Image_60.0_False', 'm_Image_60.0_True']].values.mean(axis=1)

        sd_t['bpi'] = (bout - cont) / (bout + cont)
        print(sd_t.shape)

        # all_xys = make_stimspeed_reg(stimpath=, ds_fac=5)
        all_xys = np.loadtxt(self.datapath + '/nat_stim_experiment_1min.txt')[::int(60 / 30), :]

        order = np.loadtxt(self.datapath + '/nat_stim_order_1min.txt')

        idx = 1
        num_stim = len(order)
        stim_len = int(all_xys.shape[0] / num_stim)
        xys = all_xys[idx * stim_len:(idx + 1) * stim_len, :]

        fig, ax = plt.subplots(1, 2, figsize=(4, 3), dpi=200, constrained_layout=True)
        ax[0].plot(xys[1:, 0], xys[1:, 1], linestyle='--', color='black', alpha=.5, lw=.5, rasterized='True')
        ax[0].scatter(xys[1:, 0], xys[1:, 1], zorder=4, alpha=.6, s=5, edgecolors='none', c=range(xys.shape[0] - 1),
                      cmap='viridis', rasterized='True')
        ax[0].set_xlim(-300, 500)
        ax[0].set_ylim(0, 700)

        ax1 = inset_axes(ax[1],
                         width="5%",
                         height="20%",
                         loc='center left')

        cmap = matplotlib.cm.viridis
        norm = colors.Normalize(vmin=0, vmax=1800)

        clb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax1)
        clb.set_label('Time (s)')
        clb.set_ticks([0, 1800])
        clb.set_ticklabels([0, 60])

        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[0].add_patch(matplotlib.patches.Rectangle((300, 0), 200, 10, color='black'))
        ax[0].text(400, -100, '2 cm', ha='center')
        ax[0].scatter(250, 250, s=25, color='black', marker='X')
        ax[0].set_aspect('equal')

        plt.savefig(self.figpath + '/S_NATURALISTIC_STIMULUS.svg', bbox_inches='tight')
        plt.close()

        stimulus_order = [[22, 19, 0, 25, 24, 27, 20, 23, 21, 1, 26, 18],
                          [25, 3, 27, 21, 23, 26, 18, 24, 19, 22, 20, 2],
                          [20, 24, 21, 19, 23, 22, 26, 5, 18, 25, 27, 4],
                          [6, 22, 23, 27, 20, 21, 7, 24, 18, 26, 19, 25],
                          [27, 25, 24, 26, 8, 9, 21, 18, 19, 23, 20, 22],
                          [25, 18, 26, 11, 19, 21, 27, 22, 10, 24, 20, 23],
                          [21, 13, 26, 12, 24, 23, 27, 25, 18, 22, 20, 19],
                          [14, 21, 25, 26, 20, 23, 22, 19, 27, 15, 24, 18],
                          [21, 19, 27, 17, 23, 25, 18, 16, 22, 20, 24, 26]]

        alltraces, allunids, allstimparams, alldelays, allresp, allfids = get_traces_stim_alt(
            sd_t[sd_t.bpi > .5],
            path=self.datapath + '/Imaging data',
            stimpath=self.datapath + '/Stimuli data'
        )

        for fidx in range(1):
            sno = -1
            recidx = 0
            tr_idxs = []
            for stim in allstimparams[fidx]:

                if stim[-1] != recidx:
                    print('reset')
                    sno = -1
                recidx = stim[-1]
                sno += 1

                if not stim[0].startswith('nan'):
                    continue
                print(recidx, sno)
                tr_idx = stimulus_order[recidx][sno]
                tr_idxs.append([tr_idx, stim[1], stim[2]])
                print(stim, tr_idx)

            stimreg_raw = add_stimspeed_reg(tr_idxs, all_xys, order, imaging_len=alltraces[fidx].shape[1],
                                            stimpath=self.datapath + '/nat_stim_experiment_1min.txt')
            for stim in allstimparams[fidx]:

                if stim[0].endswith('1.5_1.6'):
                    stimreg_raw = add_dot_speed(stimreg_raw, stim, boutrate=1.5)

                if stim[0].endswith('60.0_1.6'):
                    stimreg_raw[stim[1]:stim[2]] = .5 / 6

            stimreg = convolve_ts(stimreg_raw, delay=30, sampling_interval=1 / 6, toff=7)

            scores = get_regscores(stimreg, alltraces[fidx])
            traces_sort = alltraces[fidx][scores[:, 0].argsort()]
        meanr = np.median(alltraces[0], axis=0)
        meanr = (meanr - meanr.min()) / (meanr.max() - meanr.min())

        fig = plt.figure(figsize=(7, 5), dpi=400)
        grid = plt.GridSpec(4, 1, figure=fig, height_ratios=(1, 1, 1, 2), hspace=0.4)

        ax0 = fig.add_subplot(grid[0])
        ax01 = fig.add_subplot(grid[1])

        ax1 = fig.add_subplot(grid[2])
        ax2 = fig.add_subplot(grid[3])

        for stim in allstimparams[0]:

            if not 12000 < stim[1] < 18000:
                continue
            if stim[0].startswith('nan_fish'):
                ax0.text(stim[1], 10, 'Dot shoaling', rotation=45, ha='left')
                ax01.axvspan(stim[1], stim[2], facecolor=self.red, alpha=.1, edgecolor='none', zorder=0)

                ax0.axvspan(stim[1], stim[2], facecolor=self.red, alpha=.1, edgecolor='none', zorder=0)
                ax1.axvspan(stim[1], stim[2], facecolor=self.red, alpha=.1, edgecolor='none', zorder=0)
                ax2.axvspan(stim[1], stim[2], facecolor=self.red, alpha=.1, edgecolor='none', zorder=0)

            elif stim[0].endswith('1.5_1.6'):

                ax0.text(stim[1], 10, 'Dot 1.5 Hz', rotation=45, ha='left')
                ax01.axvspan(stim[1], stim[2], facecolor=self.red, alpha=.1, edgecolor='none', zorder=0)

                ax0.axvspan(stim[1], stim[2], facecolor=self.red, alpha=.1, edgecolor='none', zorder=0)
                ax1.axvspan(stim[1], stim[2], facecolor=self.red, alpha=.1, edgecolor='none', zorder=0)
                ax2.axvspan(stim[1], stim[2], facecolor=self.red, alpha=.1, edgecolor='none', zorder=0)

            elif stim[0].endswith('60.0_1.6'):

                ax0.text(stim[1], 10, 'Dot 60 Hz', rotation=45, ha='left')
                ax01.axvspan(stim[1], stim[2], facecolor='blue', alpha=.1, edgecolor='none', zorder=0)
                ax0.axvspan(stim[1], stim[2], facecolor='blue', alpha=.1, edgecolor='none', zorder=0)
                ax1.axvspan(stim[1], stim[2], facecolor='blue', alpha=.1, edgecolor='none', zorder=0)
                ax2.axvspan(stim[1], stim[2], facecolor='blue', alpha=.1, edgecolor='none', zorder=0)

            else:

                pass

        acc_conv = convolve_ts(np.abs(np.diff(stimreg_raw)), delay=30, sampling_interval=1 / 6, toff=7)

        ax01.plot(acc_conv * 6, alpha=1, color=self.red, lw=.5)
        ax0.plot(np.diff(stimreg_raw * 6), alpha=1, color=self.red, lw=.5)

        ax1.plot(meanr, lw=.3, alpha=.8, color='black')
        trids = [1, 2, 4, 5]
        for ax in [ax01, ax0, ax1]:
            for dirs in ['right', 'top', 'bottom']:
                ax.spines[dirs].set_visible(False)
                ax.set_xticks([])
        for dirs in ['right', 'top', 'left']:
            ax2.spines[dirs].set_visible(False)

        for tno, trid in enumerate(trids):
            ax2.plot(traces_sort[-trid] - tno - 2.5, alpha=.5, lw=.3, color='black')
        start = 12000
        ax0.set_ylim(-10, 10)
        ax01.set_ylim(-.1, 5)
        ax01.set_xlim(start, 18500)
        ax0.set_xlim(start, 18500)
        ax1.set_xlim(start, 18500)
        ax2.set_xlim(start, 18500)
        ax2.set_xticks(np.arange(12000, 19500, 300 * 5))
        ax2.set_xticklabels([int(i) for i in np.arange(0, 6300 / 300, 5)])
        ax2.set_xlabel('Time (min)')

        ax1.set_yticks([0.0, 0.5, 1.0])

        ax1.set_ylabel('DT-BPN \nmean dF/F', rotation=90, fontsize=8)
        ax0.set_ylabel('Acceleration \ncm/$\mathregular{s^2}$', rotation=90, fontsize=8)
        ax01.set_ylabel('convolved \nabs(Acc.)', rotation=90, fontsize=8)

        ax2.text(11500, -5, 'Example BPNs', rotation=90, fontsize=8, ha='center')
        ax2.set_yticks([])
        plt.savefig(self.figpath + '/S_NATDOT_SPEED_TRACES.svg', bbox_inches='tight', dpi=400)
        plt.close()

    def plot_figS5g(self):

        sd = pickle.load(open(self.datapath + '/sumdict_figS5g.p', 'rb'))
        br = sd[sorted([i for i in sd.keys() if i.startswith('m_')])].values
        sd['dff'] = np.sum(br, axis=1)

        bout = sd[['m_True_1.5_1.6', 'm_False_1.5_1.6']].values.mean(axis=1)
        cont = sd[['m_True_60.0_1.6', 'm_False_60.0_1.6']].values.mean(axis=1)

        bout = np.clip(bout, 0, None)
        cont = np.clip(cont, 0, None)

        sd['_m_bout'] = bout
        sd['_m_cont'] = cont

        sd['bpi'] = (bout - cont) / (bout + cont)

        dim = sd[['m_dim_True', 'm_dim_False']].values.mean(axis=1)
        dimr = sd[['regsc_dim_True', 'regsc_dim_False']].values.mean(axis=1)
        sd['_m_dim'] = dim
        sd['_regsc_dim'] = dimr

        boutr = sd[['regsc_True_1.5_1.6', 'regsc_False_1.5_1.6']].values.mean(axis=1)
        contr = sd[['regsc_True_60.0_1.6', 'regsc_False_60.0_1.6']].values.mean(axis=1)
        sd['_regsc_bout'] = boutr
        sd['_regsc_cont'] = contr

        sd = sd[sd.iscell]

        group = np.empty(sd.shape[0], dtype='object')
        group[sd.fno % 2 == 0] = 'ctr'
        group[sd.fno % 2 != 0] = 'iso'
        sd['group'] = group

        maskdict = {}
        for f in sd['fno'].unique():
            maskdict[f] = tiff.imread(self.datapath + '/Imaging data/20211130/fish{}/plane0/Mask.tif'.format(f))
        isthal = np.empty(sd.shape[0], dtype=bool)
        for fno in sd.fno.unique():
            xy = sd[sd.fno == fno][['x_coor', 'y_coor']].values
            tbool = maskdict[fno][xy[:, 1].astype(int), xy[:, 0].astype(int)].astype(bool)
            isthal[sd.fno == fno] = np.invert(tbool)

        sd['isthal'] = isthal

        # Thresholding for all cells for cell counts
        mstims = ['_m_bout', '_m_cont', 'm_grating_5.0', 'm_loom_1245.4_12.0']

        mstimlabels = ['Dot 1.5 Hz', 'Dot 60 Hz', 'Grating', 'Looming']
        tval = 50

        fig = plt.figure(figsize=(2, 1.), dpi=200)
        spec = gridspec.GridSpec(ncols=7, nrows=1, figure=fig, width_ratios=[1, 1, 1, 1, 2, 1, 3])

        sdb_all = pd.DataFrame()
        pvals = []

        for stim, axno in zip(mstims, range(len(mstims))):

            if stim in ['_m_bout', '_m_cont', 'bpi']:

                stimthresh = np.nanpercentile(sd[stim].values, tval)
                sdb = sd[(sd.isthal) & (sd[stim] > stimthresh)]

            else:

                stimthresh = np.nanpercentile(sd[stim].values, tval)
                sdb = sd[(sd.isthal == False) & (sd[stim] > stimthresh)]

            sdbs = sdb.groupby(['fid', 'group']).sum()
            group = [i[1] for i in sdbs.index]
            sdbs['stim'] = stim
            sdbs['group'] = group
            sdb_all = pd.concat([sdb_all, sdbs])

            count_ctr = sdbs[sdbs.group == 'ctr'].iscell.values
            count_abl = sdbs[sdbs.group == 'iso'].iscell.values

            print(stim, np.median(count_ctr), np.median(count_abl), count_ctr.std(), count_abl.std())
            stats, pval = mannwhitneyu(count_ctr, count_abl, alternative='greater')
            print(stats, pval)
            pvals.append(pval)

        ax0 = fig.add_subplot(spec[:5])
        sns.stripplot(data=sdb_all, y='iscell', x='stim', hue='group', s=5, alpha=.5, dodge=.9, jitter=True, ax=ax0,
                      palette=[self.red, 'black'])
        g = sns.pointplot(data=sdb_all, y='iscell', ci='sd', estimator=np.median, x='stim', hue='group', dodge=.2,
                          ax=ax0, palette=['black', 'black'],
                          linestyles=['none'] * len(mstims), capsize=0, errwidth=1.5, markers='.')

        ax0.legend_.remove()
        ax0.set_ylabel('# cells')
        ax0.set_ylim(-15, 150)
        ax0.set_yticks(np.arange(0, 200, 50))
        ax0.set_xlabel('')
        sns.despine(ax=ax0, trim=False)
        # ax0.spines['bottom'].set_visible(False)

        for pno, pval_all in enumerate(pvals):
            if pval_all > .05:
                sig = 'n.s.'
            elif .05 > pval_all > .01:
                sig = '*'.format(pval)
            elif .01 > pval_all > .001:
                sig = '**'.format(pval)
            elif .001 > pval_all:
                sig = '***'.format(pval)
            ax0.text(pno, 150, 'p={}'.format(round(pval_all, 2)), fontsize=10, ha='center', rotation=45)
        ax0.set_xticklabels(mstimlabels, rotation=45, ha='right')

        thresh = np.nanpercentile(sd[['m_True_1.5_1.6', 'm_False_1.5_1.6', 'm_True_60.0_1.6', 'm_False_60.0_1.6']],
                                  tval, axis=0)
        sdbpn = sd[
            np.any(sd[['m_True_1.5_1.6', 'm_False_1.5_1.6', 'm_True_60.0_1.6', 'm_False_60.0_1.6']] > thresh, axis=1)]

        _, bpn_ctr = np.unique(sdbpn[(sdbpn.bpi > .5) & (sdbpn.group == 'ctr') & (sdbpn.isthal)].fid,
                               return_counts=True)
        _, bpn_abl = np.unique(sdbpn[(sdbpn.bpi > .5) & (sdbpn.group == 'iso') & (sdbpn.isthal)].fid,
                               return_counts=True)

        if len(bpn_abl) < 5:
            bpn_abl = list(bpn_abl) + [0] * (5 - len(bpn_abl))
        if len(bpn_ctr) < 5:
            bpn_ctr = list(bpn_ctr) + [0] * (5 - len(bpn_ctr))
            print(bpn_abl)

        dat = pd.DataFrame({'iso': bpn_abl,
                            'ctr': bpn_ctr
                            })

        ax = fig.add_subplot(spec[5])

        stats, pval = mannwhitneyu(bpn_ctr, bpn_abl, alternative='greater')
        ax.text(0.1, 60, 'p={}'.format(round(pval, 2)), fontsize=10, ha='center', rotation=45)
        sns.stripplot(data=pd.melt(dat), x=[0] * 10, hue='variable', y='value', s=5, alpha=.5, dodge=.9, jitter=True,
                      ax=ax, palette=[self.red, 'black'])
        sns.pointplot(data=pd.melt(dat), x=[0] * 10, hue='variable', y='value', ci='sd', estimator=np.median, dodge=.3,
                      ax=ax, palette=['black', 'black'],
                      linestyles=['none'] * len(mstims), capsize=0, errwidth=1.5, markers='.')

        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)

        ax.set_ylabel('# cells')
        # ax[0].set_xticks([.5])
        ax.set_ylim(-5., 60)
        ax.set_yticks(np.arange(0, 61, 20))
        ax.set_xticklabels(['DT-BPNs'], ha='right', rotation=45)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.legend_.remove()

        ax = fig.add_subplot(spec[-1])

        ax.scatter(3, 2, color='black', s=20)
        ax.text(20, 1.9, 'ctr (N={})'.format(len(sd[sd.group == 'iso'].fid.unique())), fontsize=8)
        ax.scatter(3, 2.5, color=self.red, s=20)
        ax.text(20, 2.4, 'isolation (N={})'.format(len(sd[sd.group == 'iso'].fid.unique())), fontsize=8)
        ax.set_xlim(-100, 10)
        ax.set_ylim(0, 4)
        ax.set_axis_off()

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)
        plt.savefig(self.figpath + '/S_ISOLATION_IMAGING.svg', bbox_inches='tight')
        plt.close()


def get_regscores(reg, traces, lim=None):
    lreg = LinearRegression().fit(reg.T.reshape(-1, 1), traces[:, :lim].T)
    rsme = np.nanmean(np.sqrt((traces[:, :lim] - reg) ** 2), axis=1).reshape(-1, 1)
    v = ((reg - reg.mean()) ** 2).sum()
    r2 = 1 - (rsme / v)
    r2[np.isinf(r2)] = 0
    coefs = lreg.coef_
    scr = coefs * r2
    scores = np.concatenate([scr, coefs], axis=1)
    return scores


def add_stimspeed_reg(

        tr_idxs,
        all_xys,
        order,
        stimpath='',
        imaging_len=21000,
        fps=30,
        stim_fps=60

):
    stim_raw = np.loadtxt(stimpath)[::int(stim_fps / fps), :]
    stimreg = np.zeros(imaging_len)
    num_stim = len(order)
    stim_len = int(all_xys.shape[0] / num_stim)

    for tr_idx, start, stop in tr_idxs:
        xys = stim_raw[tr_idx * stim_len:(tr_idx + 1) * stim_len, :] / 100
        speed = np.sqrt((xys[1:, 0] - xys[:-1, 0]) ** 2 + (xys[1:, 1] - xys[:-1, 1]) ** 2)
        speed = resample(speed[1:], 360)
        speed = np.clip(speed, 0, speed.max())

        stimreg[start:stop - 1] = speed

    return stimreg


def add_dot_speed(
        reg,
        stim,
        rad=1.8,
        frate=6.,
        delay=30,
        size=.4,
        boutrate=1.5
):
    xys, params = circular_step(speed=.5,
                                ccw=False,
                                shift=False,
                                shuffle=False,
                                rad=rad,
                                boutrate=boutrate,
                                frate=frate)
    xys = np.array(xys)
    speed = np.sqrt((xys[1:, 0] - xys[:-1, 0]) ** 2 + (xys[1:, 1] - xys[:-1, 1]) ** 2)
    # speed = np.diff(speed)
    # dists = [np.linalg.norm(xys[i, :2] -  xys[i-1, :2]) for i in np.arange(1, xys.shape[0], 1)]
    print(speed.shape)
    # speed = np.clip(speed, 0, speed.max())
    reg[stim[1]:stim[2]] = speed[:stim[2] - stim[1]]

    return reg


def plot_orthogonal_scatter_7dpf(

        zstack,
        sd,
        cval='bpi',
        clim=(-1, 1),
        cmap='coolwarm',
        ctclrs='coolwarm',
        roialpha=.2,
        roisize=1,
        points='',
        allmasks=None,
        mask=None,
        kde=None,
        fs=(1, 1),
        clb=False,
        tvals=None,
        bpn_mask=None,
        tag='',
        save=False,
        figpath=''

):
    fig = plt.figure(constrained_layout=True, figsize=fs, dpi=120)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=(1, zstack.shape[0] / zstack.shape[2]),
                             height_ratios=(zstack.shape[0] / zstack.shape[1], 1))

    ax1 = fig.add_subplot(spec[0, 0])
    ax3 = fig.add_subplot(spec[1, 0], sharex=ax1)
    ax4 = fig.add_subplot(spec[1, 1], sharey=ax3)
    start, stop = 0, -1

    if isinstance(cval, str):

        cval = sd[cval]

    else:

        cval = cval

    if mask is not None:
        mask = (allmasks == mask)

    for axno, ax in zip([0, 1, 2], [ax3, ax1, ax4]):

        if axno == 0:

            av = np.nanmax(zstack[120: 180], axno)
            if mask is not None:
                avmask = np.nanmean(mask, axno)
            if kde is not None:
                avkde = np.nanmean(kde, axno)

            x1, x2 = sd.x, sd.y

        elif axno == 1:

            av = np.nanmax(zstack[:, start:stop, :], axno)
            if mask is not None:
                avmask = np.nanmean(mask, axno)
            if kde is not None:
                avkde = np.nanmean(kde, axno)

            x1, x2 = sd.x, sd.z

        elif axno == 2:

            av = np.nanmax(zstack[:, :, start:stop], axno).T
            if mask is not None:
                avmask = np.nanmean(mask, axno).T
            if kde is not None:
                avkde = np.nanmean(kde, axno).T

            x1, x2 = sd.z, sd.y
        ax.imshow(av, cmap='Greys', origin='lower', alpha=.5, aspect='auto', clim=(-10, 400))

        if mask is not None:
            ax.imshow(avmask, cmap='Greens', origin='lower', alpha=avmask / avmask.max(), aspect='auto')

        if kde is not None:
            ax.imshow(
                avkde,
                origin='lower',
                cmap='Reds',
                aspect='auto',
                alpha=avkde / avkde.max()
            )
        if bpn_mask is not None:

            col = sns.color_palette(ctclrs, 100)
            cvals = [0, 50, -1]
            cvals = [33, 66, 99]
            for tno, tval in enumerate(tvals):

                bpnm = bpn_mask.copy()
                bpnm[bpn_mask >= tval] = 1
                bpnm[bpn_mask < tval] = 0
                if axno == 2:
                    contours = measure.find_contours(bpnm.max(axno).T, .99)
                else:
                    contours = measure.find_contours(bpnm.max(axno), .99)

                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=.9, color=col[cvals[tno]], alpha=1, zorder=5)

        if points == 'equal':

            ax.scatter(x1, x2, s=roisize, c=cval, cmap=cmap, vmin=clim[0], vmax=clim[1], alpha=roialpha,
                       edgecolors='black', lw=.1, rasterized=True)

        elif points == 'rf':

            ax.scatter(x1, x2, s=1, c=cval, cmap='hsv', vmin=clim[0], vmax=clim[1], alpha=roialpha, edgecolors='none')

        ax.set(xticks=[], yticks=[])
        ax.set_axis_off()

    if clb:

        axn = fig.add_subplot(spec[0, 1])
        ax2 = inset_axes(axn,
                         width="5%",  # width = 50% of parent_bbox width
                         height="75%",  # height : 5%
                         loc='upper left')
        cmap = matplotlib.cm.Reds
        norm = colors.Normalize(vmin=clim[0] * 1000, vmax=clim[1] * 1000)

        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2)
        ax2.set(xticks=[], yticks=[clim[0] * 1000, clim[1] * 1000])
        ax2.yaxis.set_ticks_position('right')
        ax2.set_ylabel('BPNs/10\u00b3 µm\u00b3')
        ax2.yaxis.set_label_position('left')
        # axn.text(1.3, 4.5, 'BPNs/1000 µm\u00b3', fontsize=8)

        for tno, tval in enumerate(tvals):
            cv = tval / tvals.max()
            axn.add_patch(matplotlib.patches.Rectangle((1.3, tno * 2 - 2.2), .3, .6, color=col[cvals[tno]]))
            axn.text(1.8, tno * 2 - 2.6, '{}'.format(tval * 1000), fontsize=10)
            axn.set(ylim=(-8, 4), xlim=(0, 2), xticks=[], yticks=[])
            axn.set_axis_off()
    ax3.add_patch(matplotlib.patches.Rectangle((395, 700), 200, 10, color="black"))
    ax3.text(400, 675, '200 µm', fontsize=10)
    ax3.set_ylim(zstack.shape[1] - 50, 50)
    ax3.set(xticks=[], yticks=[])
    ax3.set_axis_off()
    ax1.set_ylim(zstack.shape[0] - 50, 0)
    ax4.set_xlim(zstack.shape[0] - 50, 0)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    if save:
        plt.savefig(figpath + '/ORTHVIEWS_{}.svg'.format(tag), bbox_inches='tight', dpi=800)

    plt.close()


def gaussian_t(x, sigma, mu):
    for x0 in x:
        yield 1. / (sigma * np.sqrt(2. * np.pi)) * np.exp(- (x0 - mu) ** 2. / (2. * sigma ** 2.))


def build_gaussians(freq, s, sigma, plot=False, fps=60, speed=.5 * 4., rad=1.8 * 4):
    print(freq, s)
    circ = float(2. * np.pi * rad)  # circumference
    period_rot = circ / float(speed)  # s/circle in cm
    s = float(s)
    freq = float(freq)
    mus = np.arange(4. * sigma, s + s / (freq * s), s / (freq * s))

    ngaussians = len(mus)
    gaussians = np.zeros((fps * int(s), ngaussians))
    acc = np.zeros(fps * int(s))
    for mno, mu in enumerate(mus):
        g = np.array([i for i in gaussian_t(np.arange(0, s, 1 / fps), sigma, mu)])
        gaussians[:, mno] = g

        for i, ms in enumerate(np.arange(0, s, 1 / fps)):
            acc[i] = gaussians[i, :].max()
    return acc


def circular_step(
        speed=10.,
        ccw=False,
        rad=10.,
        frate=60.,
        boutrate=2.,
        shift=False,
        shuffle=False,
        sigma=False,
        delay=12
):
    ccw = ccw  # direction
    rad = float(rad)  # cm
    speed = float(speed)  # cm/s
    boutrate = float(boutrate)  # 1/s

    circ = float(2. * np.pi * rad)  # circumference
    period_rot = circ / float(speed)  # s (sec per rotation)

    ang_velocity = float(2. * np.pi) / float(period_rot)  # radians/s
    ang_velocity_f = ang_velocity * (1. / float(frate))  # radians/frame
    angle = 90 * (np.pi * 2 / 360)  # start point (top), previously  1.5*np.pi

    x = rad * np.cos(angle)  # Starting x coordinate
    y = rad * np.sin(angle)  # Starting y coordinate

    xys = []
    bout = []

    nframes = 2 * np.pi / ang_velocity_f  # 2* pi for whole rotation
    if sigma:
        accs = build_gaussians(1.5, int(round(nframes / frate)), sigma, plot=False, fps=int(frate))
        norm_fac = np.sum(accs * ang_velocity_f) / (2 * np.pi)
    interval = round(frate / boutrate)

    if shift:
        new_angle = angle + ang_velocity_f * (interval / 2.)
    else:
        new_angle = angle

    for frame in range(int(round(nframes) + 1)):

        if sigma:
            acc = accs[frame] / norm_fac
        else:
            acc = 1
        if ccw:

            new_angle = new_angle - ang_velocity_f * acc

        else:
            new_angle = new_angle + ang_velocity_f * acc

        if frame % interval == 0:

            bout.append((x, y, new_angle))
            xys.append(bout)
            bout = []

            x = rad * np.cos(new_angle)
            y = rad * np.sin(new_angle)

        else:
            bout.append((x, y, new_angle))

    if shuffle:

        shuffle_xys = []

        for bn in range(0, int(round(len(xys) * .5))):

            shuffle_xys.append(xys[bn])

            if not bn + round(len(xys) * .5) >= len(xys):
                shuffle_xys.append(xys[int(bn + round(len(xys) * .5))])

        xys = shuffle_xys

    params = {

        'radius': rad,
        'speed': speed,
        'ccw': ccw,
        'frate': frate,
        'boutrate': boutrate,
        'delay': delay,
        'shuffle': shuffle,
        'shift': shift

    }

    return list(chain(*xys)), params


def plot_orthogonal_inset(

        ax,
        zstack,
        sd_t,
        cval='bpi',
        clim=(-1, 1),
        cmap='coolwarm',
        roialpha=.2,
        roisize=1,
        points='',
        allmasks=None,
        mask=None,
        kde=None,
        fs=(1, 1),
        zrange=(130, 180),
        tag='',
        clb=False,
        cmap1='Reds',
        cmap2='Greens'

):
    start, stop = 0, -1

    if isinstance(cval, str):

        cval = sd_t[cval]

    else:

        cval = cval

    if mask is not None:
        mask = (allmasks == mask)

    av = np.nanmax(zstack[zrange[0]: zrange[1]], 0)
    if mask is not None:
        avmask = np.nanmean(mask, 0)
    if kde is not None:
        avkde = np.nanmean(kde, 0)

    x1, x2 = sd_t.x, sd_t.y

    ax.imshow(av, cmap='Greys', origin='lower', alpha=.9, aspect='auto', clim=(100, 450))
    ax.scatter(x1, x2, zorder=3, s=roisize, c=cval, cmap=cmap, vmin=clim[0], vmax=clim[1], alpha=roialpha,
               edgecolors='none', rasterized=True)

    if mask is not None:
        avmask = ((allmasks == 7)[int(sd_t.z.min()):int(sd_t.z.max())].mean(0))
        ma = np.clip(avmask / avmask.max(), 0, .3)
        ax.imshow(avmask, cmap=cmap1, origin='lower', alpha=ma, aspect='auto')
        avmask = ((allmasks == 4)[int(sd_t.z.min()):int(sd_t.z.max())].mean(0))
        ma = np.clip(avmask / avmask.max(), 0, .3)
        ax.imshow(avmask, cmap=cmap2, origin='lower', alpha=ma, aspect='auto')

    if kde is not None:
        kdealpha = avkde / avkde.max()
        kdealpha = np.clip(kdealpha, 0, 0.7)
        ax.imshow(
            avkde,
            origin='lower',
            cmap='Reds',
            aspect='auto',
            alpha=kdealpha,
            zorder=6
        )
    ax.set(xticks=[], yticks=[])
    ax.set_axis_off()

    ax.set(xlim=(215, 500), ylim=(450, 300))
    ax.set_aspect('equal')
    return


def plot_rois_multicolor(avr, st, ax, cfac=2, colorvals=None):
    rois = np.zeros(shape=(avr.shape[0], avr.shape[1], 4)) * np.nan
    cm = matplotlib.cm.get_cmap('tab20')  # cc.cm.glasbey)
    if colorvals is None:
        colorvals = np.array(range(len(st))) / len(st)

    for nno, neuron in enumerate(st):
        rois[neuron['xpix'], neuron['ypix'], 3] = neuron['lam']
        rois[neuron['xpix'], neuron['ypix'], :3] = cm(colorvals[nno])[:3]

    rois[:, :, 3] -= np.nanmin(rois[:, :, 3])
    rois[:, :, 3] /= np.nanmax(rois[:, :, 3])
    rois[:, :, 3] = np.clip(rois[:, :, 3], .8, 1)

    rois = rotate(rois, angle=180, resize=False, preserve_range=True)
    avr = avr.max() - avr
    ax.imshow(avr.T, cmap='Greys', aspect='auto', \
              clim=(avr.max() / cfac, avr.max()), alpha=1, origin='lower', interpolation='none')
    ax.imshow(np.transpose(rois, [0, 1, 2]), interpolation='none', aspect='equal')
    ax.set_axis_off()
    return


def plot_traces_stims(traces, sp, prange, mp, ax, lw=1, text=True, bars=True, alpha=1):
    for sno, stim in enumerate(sp):

        freq = float(stim[0].split('_')[1])
        if stim[1] > prange[1] or stim[2] < prange[0]:
            continue

        if text:
            st = '{} Hz,\n{} mm'.format(stim[0].split('_')[1], stim[0].split('_')[2].strip(' '))
            ax.text(stim[1] + 20, .9, st, rotation=-45, fontsize=8, ha='right')
        if bars:
            ax.axvspan(stim[1], stim[2], alpha=.4, facecolor=mp.to_rgba(-freq), edgecolor='none')

    for t in range(traces.shape[0]):
        ax.plot(np.arange(prange[0], prange[1], 1), medfilt(traces[t][prange[0]:prange[1]] - t, 3), color='#333a36',
                lw=lw, alpha=alpha)
    ax.set(xlim=(prange[0] - 5, prange[1] + 5), ylim=[-.1, 0.9], yticks=(0, .9))
    ax.set_xticks([])
    return


def plot_example_traces(unids, sd, mp, tranges=([600, 740], [1600, 1840], [3300, 3540]), figpath='', path='',
                        stimpath=''):
    alltraces, allunids, allstimparams, alldelays, resp, all_fids = get_traces_stim(
        sd.iloc[np.any([sd.unid == i for i in unids], axis=0)],
        path=path,
        stimpath=stimpath,
        get_resp=True,
        reganalysis=True
    )

    sp = allstimparams[0]
    extraces = [alltraces[0][0].reshape(1, -1), alltraces[1][0].reshape(1, -1)]

    fig = plt.figure(figsize=(3, 3), constrained_layout=False, dpi=400)
    grid = plt.GridSpec(6, 4, figure=fig, width_ratios=(1, 1, 1, .5), height_ratios=(2.5, .01, 1, 2.5, .01, 1),
                        wspace=0.2, hspace=.8)
    for gno, extrace in zip([0, 3], extraces):

        ax0 = fig.add_subplot(grid[1 + gno:3 + gno, 0:3])
        # ax0.set_ylabel('Neuron #' + str(int(gno/3+1)))
        axf = fig.add_subplot(grid[1 + gno, 0:3])

        ax1 = fig.add_subplot(grid[0 + gno, 0])
        ax2 = fig.add_subplot(grid[0 + gno, 1], sharey=ax1)
        ax3 = fig.add_subplot(grid[0 + gno, 2], sharey=ax1)

        plot_traces_stims(extrace, sp, [0, extrace.size], mp, ax0, lw=.5, text=False)

        plot_traces_stims(extrace, sp, tranges[0], mp, ax1, lw=1, text=False)
        plot_traces_stims(extrace, sp, tranges[1], mp, ax2, lw=1, text=False)
        plot_traces_stims(extrace, sp, tranges[2], mp, ax3, lw=1, text=False)

        ax0.set(xlim=(0, extrace.size + 50))
        if gno == 3:
            ax0.set(xticks=np.arange(0, extrace.size + 50, 600))
            ax0.set_xticklabels([int(i) for i in np.arange(0, extrace.size + 50, 600) / 60], rotation=-45, ha='left')
            ax0.set_xlabel('time (min)')

        axf.set_xlim(0, extrace.size + 50)

        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)

        axf.set_axis_off()

        for pi in range(3):
            ax0.add_patch(matplotlib.patches.Rectangle((tranges[pi][0], .01), tranges[pi][1] - tranges[pi][0], .85,
                                                       facecolor=[.5, .5, .5, .1], edgecolor='grey', lw=1.))

        mark_inset(axf, ax1, 3, 4, fc='none', ec=[.1, .1, .1, 1], linestyle='dotted', lw=1, zorder=-1)
        mark_inset(axf, ax2, 3, 4, fc='none', ec=[.5, .5, .5, 1], linestyle='dotted', lw=1, zorder=-1)
        mark_inset(axf, ax3, 3, 4, fc='none', ec=[.5, .5, .5, 1], linestyle='dotted', lw=1, zorder=-1)

        sns.despine(ax=ax0, trim=True)
        sns.despine()
    # fig.text(-.001, .45, 'example\nneuron #1', rotation=90, ha='center', fontsize=10)
    # fig.text(-.001, .2, 'Example neurons', rotation=90, ha='center', fontsize=10)

    ax4 = fig.add_subplot(grid[1:4, 3])
    ax4.text(.6, 6, 'Bout\nfrequency\n(Hz)', fontsize=8, ha='center')
    for pos, freq in zip([0, 2, 4, 6, 8], [0.75, 1.5, 3., 6., 60.]):
        c = mp.to_rgba(-freq)
        c = np.array(c)
        c[-1] = .4  # alpha
        ax4.add_patch(matplotlib.patches.Rectangle((.1, pos / 2), .3, .6, facecolor=c, edgecolor='none'))
        ax4.text(.6, pos / 2, str(freq), fontsize=8)
        ax4.set_ylim(-2, 5)
        ax4.set_axis_off()

    fig.text(-.1, .4, 'Normalized dF', rotation=90, ha='center', fontsize=10)
    fig.text(-.02, .2, 'Neuron #2', rotation=90, ha='center', fontsize=10)
    fig.text(-.02, .6, 'Neuron #1', rotation=90, ha='center', fontsize=10)

    plt.savefig(figpath + '/NEURON_TRACES_PLANE_EXAMPLE.svg', bbox_inches='tight', dpi=400)
    plt.close()
    return resp


def plot_mean_resp(axes, resp, ridxs=[0, 2, 4, 5, 7], color='#FB7A7A', idx=None):
    rdict = ddict(list)

    for ridx, rs in enumerate(resp):

        if ridx >= 9:
            continue
            ridx = ridx - 9
        rs = [i for i in rs if i.shape[1] >= 40]
        if len(rs) == 0:
            continue
        if idx is not None:
            rs_mean = [i[idx, :41] for i in rs]
            rdict[ridx].extend(rs_mean)
        else:
            rs_mean = np.mean([i[:, :41] for i in rs], axis=0)
            rdict[ridx].extend([i[:41] for i in rs_mean])

    for ax, ridx in zip(axes, ridxs):
        ax.fill_between(range(41),
                        np.mean(rdict[ridx], axis=0) - np.std(rdict[ridx], axis=0),
                        np.mean(rdict[ridx], axis=0) + np.std(rdict[ridx], axis=0),

                        alpha=.3,
                        edgecolor='none',
                        facecolor=color
                        )
        ax.plot(np.mean(rdict[ridx], axis=0), color=color, lw=1)
        ax.set_ylim(-1.5, 3)
        ax.set_xlim(-5, 42)
        # ax.set_axis_off()
    # plt.savefig(figpath+'/MEAN_RESP_{}.svg'.format(tag), dpi=400, bbox_inches='tight')
    return


def plot_rois_multicolor_scatter(avr, st, ax, cfac=2):
    cm = matplotlib.cm.get_cmap('tab20')  # cc.cm.glasbey)
    clrs = np.array([cm(nno / len(st)) for nno in range(st.shape[0])])
    clrs[:, -1] = .9
    xys = np.array([i['med'] for i in st])
    xys, ori = pixel_rotate(xys, angle=180)
    avr = avr.max() - avr
    ax.imshow(avr.T, cmap='Greys', aspect='equal', \
              clim=(avr.max() / cfac, avr.max()), alpha=1, origin='lower', interpolation='none')
    ax.scatter(xys[:, 0], xys[:, 1], s=1.5, c=clrs, edgecolors='none', rasterized='True')
    ax.set_axis_off()

    return


def get_traces_stim(

        sdt,
        path='Imaging data',
        stimpath='Stimuli data',
        get_resp=False,
        reganalysis=True  # True for coloring example traces
):
    alltraces = []
    allunids = []
    allstimparams = []
    alldelays = []
    allresp = []
    allfids = []

    for fid in sdt.fid.unique():

        date = sdt[sdt.fid == fid].date.unique()[0]
        fno = sdt[sdt.fid == fid].fno.unique()[0]
        planes = sdt[sdt.fid == fid].plane.unique()
        fpath = os.path.join(path, date, 'fish{}'.format(fno))

        se = Tuninganalysis.StimuliExtraction(
            stimpath,
            path,
            frate=30.,
            ds_fac=5.,
            clip_resp_scores=2.,
            reganalysis=reganalysis
        )
        se.set_date_and_fish(date, fno, 6, (0, 6), 600)
        se.extract_stim(verbose=False)
        allstimparams.append(se.stimparams)
        delays = [se.protocols[i]['delay'] for i in range(len(se.protocols))]
        alldelays.append(delays)
        for plane in planes:

            nids = sdt[(sdt.fid == fid) & (sdt.plane == plane)].nid
            unids = sdt[(sdt.fid == fid) & (sdt.plane == plane)].unid
            traces = np.load(os.path.join(fpath, 'plane{}'.format(plane), 'F.npy'))[nids.values.astype(int)]
            tmin, tmax = traces.min(axis=1).reshape(-1, 1), traces.max(axis=1).reshape(-1, 1)
            traces = (traces - tmin) / (tmax - tmin)
            traces[np.isnan(traces)] = 0.

            if get_resp:

                scoreparams = se.score_neurons(traces)
                resp = scoreparams[-1]

            else:

                resp = None

            alltraces.append(traces)
            allunids.append(unids)
            allresp.append(resp)
            allfids.append(fid)

    return alltraces, allunids, allstimparams, alldelays, allresp, allfids


def plot_horizontal_scatter(

        zstack,
        sd,
        cval='bpi',
        clim=(-1, 1),
        cmap='coolwarm',
        roialpha=.2,
        roisize=1,
        points='',
        allmasks=None,
        mask=None,
        kde=None,
        fs=(1, 1),
        zrange=(120, 180),
        tag='',
        clb=False,
        tvals=None,
        bpn_mask=None,
        ctalpha=.7,
        ctclr='coolwarm',
        save=True,
        figpath='',

):
    fig = plt.figure(constrained_layout=False, figsize=fs, dpi=800)
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=(1, .25))

    ax = fig.add_subplot(spec[0])
    axno = 0
    start, stop = 0, -1

    if isinstance(cval, str):

        cval = sd[cval]

    else:

        cval = cval

    if mask is not None:
        mask = (allmasks == mask)

    av = np.nanmax(zstack[zrange[0]: zrange[1]], axno)
    if mask is not None:
        avmask = np.nanmean(mask, axno)
    if kde is not None:
        avkde = np.nanmean(kde, axno)

    x1, x2 = sd.x, sd.y
    ax.imshow(av, cmap='Greys', origin='lower', alpha=1, aspect='equal', clim=(50, 400), zorder=1)

    if points == 'equal':

        ax.scatter(x1, x2, s=roisize, c=cval, cmap=cmap, vmin=clim[0], vmax=clim[1], alpha=roialpha, rasterized=True,
                   zorder=3, edgecolors='black', linewidths=.1)

    elif points == 'rf':

        ax.scatter(x1, x2, s=1, c=cval, cmap='hsv', vmin=clim[0], vmax=clim[1], alpha=roialpha, edgecolors='none')

    if mask is not None:
        ax.imshow(avmask, cmap='Greens', origin='lower', alpha=avmask / avmask.max(), aspect='auto')

    if kde is not None:
        ax.imshow(
            avkde,
            origin='lower',
            cmap='Reds',
            aspect='auto',
            alpha=avkde / avkde.max(),
            zorder=2
        )
    ax.set(xticks=[], yticks=[])
    ax.set_axis_off()
    if bpn_mask is not None:

        col = sns.color_palette(ctclr, 100)
        cvals = [0, 50, -1]
        cvals = [33, 66, 99]
        for tno, tval in enumerate(tvals):

            bpnm = bpn_mask.copy()
            bpnm[bpn_mask >= tval] = 1
            bpnm[bpn_mask < tval] = 0
            if axno == 2:
                contours = measure.find_contours(bpnm.max(axno).T, .99)
            else:
                contours = measure.find_contours(bpnm.max(axno), .99)

            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.2, color=col[cvals[tno]], alpha=ctalpha, zorder=5)

    ax.set_xlim(125, 600)
    ax.set_ylim(650, 75)

    ax.set(xticks=[], yticks=[])
    ax.set_axis_off()

    ax.add_patch(matplotlib.patches.Rectangle((500, 600), 100, 5, color="black"))
    ax.text(500, 580, '200 µm', fontsize=10)

    if clb:

        axn = fig.add_subplot(spec[1])
        ax2 = inset_axes(axn,
                         width="15%",  # width = 50% of parent_bbox width
                         height="25%",  # height : 5%
                         loc='upper left')
        cmap = ctclr
        norm = colors.Normalize(vmin=clim[0] * 1000 / 8, vmax=clim[1] * 1000 / 8)

        cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                          cax=ax2)  # , ticks=[.0008*1000/8, .0012*1000/8, .0024*1000/8])
        cb.ax.locator_params(nbins=2)
        ax2.set(xticks=[])
        # ax2.yaxis.set_ticks_position('right')
        ax2.set_ylabel('BPNs/10\u00b3 µm\u00b3')
        ax2.yaxis.set_label_position('left')

        if bpn_mask is not None:
            # axn.text(1., -8, 'BPNs/1000 µm\u00b3', fontsize=10)
            # axn.text(-4., 2, 'BPNs/1000 µm\u00b3', fontsize=10)
            for tno, tval in enumerate(tvals):
                axn.add_patch(matplotlib.patches.Rectangle((0, tno * 2. - 10.2), .2, .6, color=col[cvals[tno]]))
                axn.text(.4, tno * 2. - 10.2, '{}'.format(tval * 1000 / 8), fontsize=10)
                axn.set(ylim=(-18, 4), xlim=(0, 2), xticks=[], yticks=[])

        axn.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    if save:
        plt.savefig(figpath + '/ORTHVIEWS_{}.svg'.format(tag), bbox_inches='tight', dpi=800)
    plt.close()


def plot_orthogonal_scatter(
        zstack,
        sd,
        cval='bpi',
        clim=(-1, 1),
        cmap='coolwarm',
        roialpha=.2,
        roisize=1,
        allmasks=None,
        mask=None,
        kde=None,
        fs=(1, 1),
        zrange=(120, 180),
        tag='',
        clb=False,
        tvals=None,
        bpn_mask=None,
        ctalpha=.7,
        ctclr='coolwarm',
        save=True,
        figpath=''

):
    fig = plt.figure(figsize=fs, dpi=800)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=(1, zstack.shape[0] / (650 - 75)),
                             height_ratios=(zstack.shape[0] / (650 - 75), 1))

    ax1 = fig.add_subplot(spec[0, 0])
    ax3 = fig.add_subplot(spec[1, 0], sharex=ax1)
    ax4 = fig.add_subplot(spec[1, 1], sharey=ax3)
    start, stop = 0, -1

    if isinstance(cval, str):

        cval = sd[cval]

    else:

        cval = cval

    if mask is not None:
        mask = (allmasks == mask)

    for axno, ax in zip([0, 1, 2], [ax3, ax1, ax4]):

        if axno == 0:

            av = np.nanmax(zstack[zrange[0]: zrange[1]], axno)
            if mask is not None:
                avmask = np.nanmean(mask, axno)
            if kde is not None:
                avkde = np.nanmean(kde, axno)

            x1, x2 = sd.x, sd.y

        elif axno == 1:

            av = np.nanmax(zstack[:, start:stop, :], axno)
            if mask is not None:
                avmask = np.nanmean(mask, axno)
            if kde is not None:
                avkde = np.nanmean(kde, axno)

            x1, x2 = sd.x, sd.z

        elif axno == 2:

            av = np.nanmax(zstack[:, :, start:stop], axno).T
            if mask is not None:
                avmask = np.nanmean(mask, axno).T
            if kde is not None:
                avkde = np.nanmean(kde, axno).T

            x1, x2 = sd.z, sd.y
        ax.imshow(av, cmap='Greys', origin='lower', alpha=1, aspect='auto', clim=(50, 400), zorder=1)

        sc = ax.scatter(x1, x2, s=roisize, c=cval, cmap=cmap, vmin=clim[0], vmax=clim[1], alpha=roialpha,
                        rasterized=True, zorder=3, edgecolors='none')

        if mask is not None:
            ax.imshow(avmask, cmap='Greens', origin='lower', alpha=avmask / avmask.max(), aspect='auto')

        if kde is not None:
            ax.imshow(
                avkde,
                origin='lower',
                cmap='Reds',
                aspect='auto',
                alpha=avkde / avkde.max(),
                zorder=2
            )
        ax.set(xticks=[], yticks=[])
        ax.set_axis_off()
        if bpn_mask is not None:

            col = sns.color_palette(ctclr, 100)
            cvals = [0, 50, -1]
            for tno, tval in enumerate(tvals):

                bpnm = bpn_mask.copy()
                bpnm[bpn_mask >= tval] = 1
                bpnm[bpn_mask < tval] = 0
                if axno == 2:
                    contours = measure.find_contours(bpnm.max(axno).T, .99)
                else:
                    contours = measure.find_contours(bpnm.max(axno), .99)

                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=col[cvals[tno]], alpha=ctalpha, zorder=5)

    ax3.set_xlim(75, 650)
    ax3.set_ylim(650, 75)
    ax4.set_xlim(zstack.shape[0], 0)
    ax1.set_ylim(zstack.shape[0], 0)
    ax3.set(xticks=[], yticks=[])
    ax3.set_axis_off()
    ax3.add_patch(matplotlib.patches.Rectangle((500, 600), 100, 5, color="black"))
    ax3.text(500, 580, '200 µm', fontsize=10)
    if clb:

        axn = fig.add_subplot(spec[0, 1])
        ax2 = inset_axes(axn,
                         width="10%",  # width = 50% of parent_bbox width
                         height="50%",  # height : 5%
                         loc='upper left')

        plt.colorbar(sc, cax=ax2)
        ax2.set(xticks=[])
        ax2.yaxis.set_ticks_position('right')
        ax2.set_ylabel('BPI')
        ax2.yaxis.set_label_position('left')

        if bpn_mask is not None:
            axn.text(1., 4.5, 'BPNs/1000 µm\u00b3', fontsize=10)

            for tno, tval in enumerate(tvals):
                axn.add_patch(matplotlib.patches.Rectangle((1., tno * 2 - 2.2), .3, .6, color=col[cvals[tno]]))
                axn.text(1.5, tno * 2 - 2.6, '{}'.format(tval * 1000 / 8), fontsize=10)
                axn.set(ylim=(-8, 4), xlim=(0, 2), xticks=[], yticks=[])
        axn.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    if save:
        plt.savefig(figpath + '/ORTHVIEWS_{}.svg'.format(tag), bbox_inches='tight', dpi=800)
    plt.close()


def plot_tuning_per_fish(
        sd_t,
        areas,
        areadict,
        mpeak=None,
        mstd=None,
        min_bpns=30,
        red='#c70039',
        figpath=''
):
    stims = np.array(sorted([i for i in sd_t.keys() if i.startswith('m_False') or i.startswith('m_True')]))
    for area in areas:

        print(areadict[area])
        dt_bpn = pd.DataFrame(sd_t[(sd_t.bpi >= .5) & (sd_t.area_idx == area)], columns=stims)
        dt_bpn['fid'] = sd_t[(sd_t.bpi >= .5) & (sd_t.area_idx == area)]['fid'].values

        fids, fidc = np.unique(dt_bpn.fid.values, return_counts=True)
        fids = [fids[i] for i in range(fids.shape[0]) if fidc[i] > min_bpns]
        print('# of animals: ', len(fids))

        fidbool = np.any([dt_bpn.fid == fid for fid in fids], axis=0)
        if not np.any(fidbool):
            continue
        dt_bpn = dt_bpn[fidbool]
        print(dt_bpn.shape, 'dtbpn #')
        fids, fidc = np.unique(dt_bpn.fid.values, return_counts=True)
        print(fidc.mean(), fidc.std())
        print(fidc)
        print(area, dt_bpn.shape)
        dt_bivals = dt_bpn[stims].values[:, :9] + dt_bpn[stims].values[:, 9:]
        dt_bivals /= 2
        dt_bivals = pd.DataFrame(dt_bivals)
        dt_bivals['fid'] = dt_bpn['fid'].values

        meanv = dt_bivals.groupby('fid').mean().reset_index()
        mvm = meanv.mean(axis=0).values[1:]
        stdvm = meanv.std(axis=0).values[1:]

        medsize = np.array([0, 2, 4, 5, 7])
        dsize = np.array([1, 3, 6, 8])
        freqs = np.array([0.75, 1.5, 1.5, 1.5, 3., 6., 60., 60., 60.])

        fig, ax = plt.subplots(figsize=(1.3, 1.3), dpi=400)

        for i in range(meanv.shape[0]):
            ax.plot(freqs[medsize], meanv.values[i, 1:][medsize], color='black', marker='o', markersize=2,
                    alpha=.5)

        ax.plot(freqs[medsize], mvm[medsize], color=red, marker='o', markersize=4, lw=2, mec='white')
        ax.fill_between(freqs[medsize],
                        mvm[medsize] + stdvm[medsize],
                        mvm[medsize] - stdvm[medsize],
                        facecolor=red,
                        alpha=0.3)

        ax.errorbar(mpeak, .9, xerr=mstd, fmt='o', capsize=2, color='black', markersize=3)
        ax.axvline(mpeak, alpha=.2, linestyle='--', color='black')
        ax.set_xscale('log', base=2)
        plt.setp(ax,
                 xticks=[0.75, 1.5, 3., 6., 60.],
                 xlim=(0.6, 75),
                 ylim=[-.1, 1],
                 yticks=[0, .4, .8],
                 ylabel='Mean dF/F',
                 xlabel='Bout frequency (Hz)',
                 title='DT-BPN response'
                 )

        ax.set_xticklabels([0.75, 1.5, 3., 6., 60.], rotation=60)
        sns.despine(trim=True)
        matplotlib.rcParams['xtick.minor.size'] = 0
        matplotlib.rcParams['xtick.minor.width'] = 0
        plt.savefig(figpath + '/BOUT_FREQUENCY_TUNING_DT.svg', bbox_inches='tight')
        plt.close()

        return meanv


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)


def get_traces_stim_alt(

        sdt,
        path='Imaging data',
        stimpath='Stimuli data',
        get_resp=False,
        reganalysis=True  # True for coloring example traces
):
    alltraces = []
    allunids = []
    allstimparams = []
    alldelays = []
    allresp = []
    allfids = []

    for fid in sdt.fid.unique():

        date = sdt[sdt.fid == fid].date.unique()[0]
        fno = sdt[sdt.fid == fid].fno.unique()[0]
        planes = sdt[sdt.fid == fid].plane.unique()
        fpath = os.path.join(path, date, 'fish{}'.format(fno))

        se = Tuninganalysis.StimuliExtraction(
            stimpath,
            path,
            frate=30.,
            ds_fac=5.,
            clip_resp_scores=2.,
            reganalysis=reganalysis
        )
        se.set_date_and_fish(date, fno, 1, (0, 6), 600)
        _, _, _, sparams_alt = se.extract_stim(verbose=False, altparams=True)
        allstimparams.append(sparams_alt)
        delays = [se.protocols[i]['delay'] for i in range(len(se.protocols))]
        alldelays.append(delays)
        for plane in planes:

            nids = sdt[(sdt.fid == fid) & (sdt.plane == plane)].nid
            unids = sdt[(sdt.fid == fid) & (sdt.plane == plane)].unid
            traces = np.load(os.path.join(fpath, 'plane{}'.format(plane), 'F.npy'))[nids.values.astype(int)]
            tmin, tmax = traces.min(axis=1).reshape(-1, 1), traces.max(axis=1).reshape(-1, 1)
            traces = (traces - tmin) / (tmax - tmin)
            traces[np.isnan(traces)] = 0.

            if get_resp:

                scoreparams = se.score_neurons(traces)
                resp = scoreparams[-1]

            else:

                resp = None

            alltraces.append(traces)
            allunids.append(unids)
            allresp.append(resp)
            allfids.append(fid)

    return alltraces, allunids, allstimparams, alldelays, allresp, allfids


def trans_nrrd(tup, header=False):
    im = tup[0].astype(np.uint8)
    im = np.moveaxis(im, 2, 0)
    im = np.moveaxis(im, 1, 2)
    if header:
        return im, tup[1]
    else:
        return im


def convolve_ts(

        ts,
        sampling_interval=1,
        toff=7,
        delay=30,

):
    t = np.linspace(0, delay, int(delay / sampling_interval))
    t = np.hstack((-np.ones(int(delay / sampling_interval)), t))
    e = np.exp(-t / toff) * (1 + np.sign(t)) / 2
    e = e / np.max(e)

    return np.convolve(ts, e, mode='same') / np.max(convolve(ts, e, mode='same'))


def pixel_rotate(

        points,
        angle=135,
        dims=(512, 512),
        rotate_first=False

):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    """
    angle = np.deg2rad(angle)
    ox, oy = dims[0] / 2., dims[1] / 2.
    corners = [[0, 0], [0, dims[1]], [dims[0], 0], [dims[0], dims[1]]]
    print(ox, oy)

    px, py = points[:, 0], points[:, 1]

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    if rotate_first:
        qx_min = min([ox + np.cos(angle) * (i - ox) - np.sin(angle) * (j - oy) for i, j in corners])
        qy_min = min([oy + np.sin(angle) * (i - ox) + np.cos(angle) * (j - oy) for i, j in corners])

        qx = qx - qx_min
        qy = qy - qy_min

    return np.concatenate((qx.reshape(-1, 1), qy.reshape(-1, 1)), axis=1), (ox, oy)


def calc_bpi(X):
    bout_r = np.mean(X[:, :5] + X[:, 9:14], axis=1)
    cont_r = np.mean(X[:, 5:9] + X[:, 14:], axis=1)
    bpi = (bout_r - cont_r) / (bout_r + cont_r)
    return bpi


def plot_dv_slices(

        zstack,
        axes,
        roistack,
        clim=(-1, 1),
        cmap='coolwarm',
        roialpha=.8,
        dims=(3, 3),
        slicerange=None,
        mask=None,
        fs=(15, 15)

):
    if slicerange is None:
        slicerange = np.array([92, 202])

    if dims == (1, 1):

        axes = [axes]
        slices = np.array(slicerange)

    else:

        axes = axes.reshape(-1)
        slices = np.linspace(slicerange[0], slicerange[1], axes.shape[0] + 1)

    for ax, sl in zip(axes, range(slices.shape[0])):

        start, stop = int(round(slices[sl])), int(round(slices[sl + 1]))
        print(start, stop)
        maxz = np.nanmax(zstack[start: stop]).max()
        rslc = np.nanmax(roistack[start: stop], 0)

        ax.imshow(np.nanmax(zstack[start: stop], 0), cmap='Greys', origin='lower', alpha=1, clim=(50, 400),
                  interpolation='none')

        mp = ax.imshow(rslc, cmap=cmap, clim=clim, origin='lower', alpha=roialpha, interpolation='none')
        if mask is not None:
            ax.imshow(mask, alpha=1, cmap='inferno', clim=(0, 150), origin='lower', interpolation='none')

        ax.set_axis_off()

        ax.set_xlim(150, 575)
        ax.set_ylim(575, 150)
        if not dims == (1, 1):
            ax.text(180, 180, '{} μm'.format(start * 2 + 40), fontdict={
                'size': 12})  # 40 is start of skin at average brain 21 dpf, *2 because of depth micron scaling
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    return mp


def create_roistack(

        zstack,
        micron_scaling,
        roi_df_t,
        unids_t,
        sd,
        scoring='bpi'

):
    roistack = np.zeros(shape=(zstack.shape[0], zstack.shape[1], zstack.shape[2])) * np.nan

    zyx_t = roi_df_t.values[:, :3][:, ::-1].astype(float)
    lam_t = roi_df_t.lam.values

    for i in range(3):
        zyx_t[:, i] *= (1. / micron_scaling[i])

    for i in range(3):
        shape_bool = zyx_t[:, i] < roistack.shape[i]
        zyx_t = zyx_t[shape_bool]
        lam_t = lam_t[shape_bool]
        unids_t = unids_t[shape_bool]

    zyx_t = np.round(zyx_t, 0).astype(int)
    if scoring == 'bpi':
        # BPI coloring
        scores = sd.bpi

    elif scoring == 'mcont':

        scores = (sd['m_True_60.0_1.6'] + sd['m_False_60.0_1.6']) / 2

    elif scoring == 'mbout':

        scores = (sd['m_True_1.5_1.6'] + sd['m_False_1.5_1.6']) / 2

    elif scoring == 'mbl':

        scores = sd['m_bl']
    scores_t = scores.values[unids_t]

    scores_t[np.isnan(scores_t)] = 0
    scores_t[np.isinf(scores_t)] = 0
    stackcolor = scores_t.reshape(-1)

    roistack[zyx_t[:, 0], zyx_t[:, 1], zyx_t[:, 2]] = stackcolor
    return roistack


if __name__ == '__main__':
    pf = PaperFigures(
        figpath='J:/_Projects/bpn_man/data/rawUpload/2p_data',
        datapath='J:/_Projects/bpn_man/data/rawUpload/2p_data'
    )

    pf.init_anatomy()

    pf.plot_figS5bd()
    pf.plot_fig4_figS8c()
    pf.plot_figs8d()

    pf.plot_fig2j()
    pf.plot_figS5g()

    pf.plot_fig2h()
    pf.plot_fig2i()

    pf.init_data_fig2a_g()
    pf.plot_fig2g()
    pf.plot_figS4a()
    pf.plot_figS4b()
    pf.plot_fig2c()
    pf.plot_figS5a()
    pf.plot_fig2b()
    pf.plot_figS4c()
    pf.plot_fig2ef()
