import numpy as np
import sys
import os
import pickle
import pandas as pd
import re


class StimuliExtraction:

    def __init__(self, stimpath, impath, **kwargs):

        self.stimpath = stimpath
        self.impath = impath

        self.frate = kwargs.get('frate', 30)
        self.ds_fac = kwargs.get('ds_fac', 5.)
        self.clip_resp_scores = kwargs.get('clip_resp_scores', 1.5)
        self.tau = kwargs.get('tau', 7.)
        self.reganalysis = kwargs.get('reganalysis', False)
        self.recs = kwargs.get('recs', None)
        self.micdelay = kwargs.get('micdelay', 0)

        self.date = None
        self.fno = None
        self.volrate = None
        self.rec_crop = None
        self.rec_offsets = None
        self.nplanes = None
        self.plane = None
        self.fstimpath = None
        self.pstimpath = None
        self.resp_scores = None
        self.magnitudes = None
        self.peak_t = None
        self.bad_stim = None
        self.bp_score = None
        self.stimparams = list()
        self.unique_p = set()
        self.protocols = list()
        self.regs = list()

    def set_date_and_fish(self, date, fno, nplanes, rec_crop, rec_offset, plane=None):

        self.date = date
        self.fno = fno
        self.fstimpath = os.path.join(self.stimpath, '{0}/fish{1}/'.format(str(self.date), str(self.fno)))
        self.nplanes = nplanes
        self.volrate = self.frate/(self.nplanes * self.ds_fac)
        self.rec_crop = rec_crop
        self.rec_offsets = [rec_offset] * (rec_crop[1] - rec_crop[0])
        if plane is not None:
            self.plane = plane
        self.resp_scores = None
        self.magnitudes = None
        self.peak_t = None
        self.bad_stim = None
        self.bp_score = None
        self.stimparams = list()
        self.protocols = list()
        self.regs = list()
        return

    def extract_stim_multirec(self, read_frate=False, fpath=None):

        stimdicts = {}
        for rec in self.recs:

            if read_frate:
                self.frate = ut.read_frate_singleplane(
                    os.path.join(fpath, 'rec{}'.format(rec)))  # hacky solution for single plane imaging on femtonics
                self.volrate = self.frate/(self.nplanes * self.ds_fac)
            self.fstimpath = os.path.join(self.stimpath, '{0}/fish{1}/rec{2}/'.format(self.date, self.fno, rec))
            stimparams, unique_p, nstimruns = self.extract_stim(rec=rec)
            stimdicts[rec] = {'stimparams': stimparams, 'unique_p': unique_p}

        return stimdicts

    def extract_stim(self, save=False, rec=None, verbose=False, altparams=False):
        """

        :param date: str, format YYYYMMDD
        :param fno: int, start at 1
        :param frate: float, imaging frame rate in fps
        :param offset: list of ints, # of frames per recording
        :param offset_mic: float, offset generated by microscope (femtonics)
        :return:

        TODO make this okay for flickers.

        """
        self.unique_p = set()
        sparams_alt = []
        if self.date is None or self.fno is None:
            sys.exit('Date and fish have to be set using set_date_and_fish.')

        rcount = -1
        # If individual planes were imaged one by one, the additional directory '/plane*/' is added
        if self.plane is None:
            stimfiles = sorted([os.path.join(self.fstimpath, i) for i in os.listdir(self.fstimpath)
                                if 'stimuli' in i or 'H2B_GCaMP6s' in i])
        elif rec is not None:

            self.pstimpath = os.path.join(self.fstimpath, 'rec{}/plane{}'.format(rec, self.plane))
            stimfiles = sorted([os.path.join(self.pstimpath, i) for i in os.listdir(self.pstimpath)
                                if 'stimuli' in i or 'H2B_GCaMP6s' in i])

        else:
            self.pstimpath = os.path.join(self.fstimpath, 'plane{}'.format(self.plane))
            stimfiles = sorted([os.path.join(self.pstimpath, i) for i in os.listdir(self.pstimpath)
                                if 'stimuli' in i or 'H2B_GCaMP6s' in i])

        if verbose:
            print(stimfiles)
        if self.rec_crop[1] == 0:
            return

        for sno, stimfile in enumerate(stimfiles[self.rec_crop[0]: self.rec_crop[1]]):
            if verbose:
                print(stimfile)
            protocol = pd.read_pickle(stimfile)

            if 'protocol' in protocol.keys():
                protocol = protocol['protocol']
            self.protocols.append(protocol)
            offset = self.rec_offsets[sno]

            for tno in range(len(protocol['tstart'])):
                if tno == 0:
                    rcount += 1
                    stop = int(np.round(protocol['delay'][0] * self.volrate)) + int(np.round(offset * self.volrate * rcount))
                if np.isnan(protocol['tstart'][tno])\
                        and ('start_flicker' in protocol and np.isnan(protocol['start_flicker'][tno]))\
                        and ('grating' in protocol and np.isnan(protocol['start_moving'][tno]))\
                        and ('loom' in protocol and np.isnan(protocol['start_moving'][tno]))\
                        and ('dim' in protocol and np.isnan(protocol['dim'][tno])):
                    print('Continuing')
                    continue

                if 'start_flicker' in protocol and np.isnan(protocol['tstart'][tno]) and not 'grating' in protocol and not 'loom' in protocol:
                    # Analysing the flicker stimuli
                    start_ = (protocol['start_flicker'][tno])
                    start = int(round((start_ + protocol['delay'][tno] + self.micdelay) * self.volrate,
                                      0)) + int(round(offset * self.volrate * rcount, 0))

                    stop_ = (protocol['stop_flicker'][tno])   # Time that passed since the previous stimulus finished
                    stop = int(round((stop_ + protocol['delay'][tno] / 2 + self.micdelay) * self.volrate,
                                      0)) + int(round(offset * self.volrate * rcount, 0))

                    if stop > int(round(offset * self.volrate * (rcount + 1), 0)):
                        continue

                    stim_p = 'flicker_' + str(protocol['t_on'][tno]) + '_' + str(protocol['pos'][tno][0] < 0)

                elif ('grating' in protocol and not np.isnan(protocol['grating'][tno])) or ('loom' in protocol and not np.isnan(protocol['loom'][tno]) or ('dim' in protocol and not np.isnan(protocol['dim'][tno]))):
                    # Analysing the grating stimuli
                    start_ = (protocol['start_moving'][tno])
                    start = int(round((start_ + self.micdelay) * self.volrate, 0)) + int(round(offset * self.volrate * rcount, 0))

                    stop_ = (protocol['stop'][tno])
                    stop = int(round((stop_ + protocol['delay'][tno] + self.micdelay) * self.volrate, 0))
                    if self.reganalysis:
                        stop = int(round((stop_ + self.micdelay) * self.volrate, 0))

                    if stop > offset * self.volrate:
                        continue

                    stop += int(round(offset * self.volrate * rcount, 0))
                    if not np.isnan(protocol['grating'][tno]):
                        stim_p = str('grating_' + str(protocol['speed'][tno]))

                    elif not np.isnan(protocol['loom'][tno]):

                        stim_p = str('loom_' + str(round(protocol['radius'][tno], 1)) + '_' + str(
                            round(protocol['speed'][tno], 1)))
                        # Sometimes the loom stimuli are so fast that the onset and stop time are rounded to the same frame #
                        if stop == start:
                            stop += 1

                    elif 'dim' in protocol.keys():
                        if not np.isnan(protocol['dim'][tno]):
                            stim_p = str('dim_' + str(protocol['inverse'][tno]))
                        else:
                            raise KeyboardInterrupt
                    else:
                        print('Something went wrong...')

                elif 'image' in protocol and not np.isnan(protocol['image'][tno]):
                    start_ = (protocol['tstart'][tno])
                    start = int(round((start_ + protocol['delay'][tno] + self.micdelay) * self.volrate,
                                      0)) + int(round(offset * self.volrate * rcount, 0))

                    stop_ = (protocol['tstop'][tno])
                    stop = int(round((stop_ + protocol['delay'][tno] + self.micdelay) * self.volrate, 0))

                    if self.reganalysis:
                        stop = int(round((stop_ + self.micdelay) * self.volrate, 0))

                    if stop > offset * self.volrate:
                        continue

                    stop += int(round(offset * self.volrate * rcount, 0))
                    if verbose:
                        print(start, stop)
                    stim_p = str('Image_' + str(protocol['boutrate'][tno]) + '_' + str(
                        protocol['ccw'][tno]))

                else:

                    if 'ccw' not in protocol.keys():
                        print('No fitting stimulus found!')
                        return False
                    # Analysing the moving dot stimuli
                    start_ = (protocol['tstart'][tno])
                    start = int(round((start_ + protocol['delay'][tno] + self.micdelay) * self.volrate,
                                      0)) + int(round(offset * self.volrate * rcount, 0))

                    stop_ = start_ + (float(protocol['tpoints'][tno][-1][0]) / 1000.)
                    stop = int(round((stop_ + protocol['delay'][tno] + self.micdelay) * self.volrate, 0))

                    if self.reganalysis:
                        stop = int(round((stop_ + self.micdelay) * self.volrate, 0))

                    if stop > offset * self.volrate:
                        continue

                    stop += int(round(offset * self.volrate * rcount, 0))
                    if verbose:
                        print(start, stop)
                    if 'sigma' in protocol.keys():

                        stim_p = '_'.join([str(protocol['speed'][tno]),
                                          str(protocol['boutrate'][tno]),
                                          str(protocol['acc'][tno]),
                                          str(protocol['sigma'][tno])])

                    else:

                        stim_p = str(protocol['ccw'][tno]) + '_' + str(protocol['boutrate'][tno]) + '_' + str(
                            protocol['size'][tno])

                self.stimparams.append([stim_p, start, stop])
                sparams_alt.append([stim_p, start, stop, sno])
                self.unique_p.add(stim_p)

        if save:
            pickle.dump([self.stimparams, sorted(self.unique_p), len(stimfiles)],
                        open(os.path.join(self.stimpath, r'{0}_F{1}_{2:.1f}Hz_params.p'.format(self.date, self.fno, self.volrate)),
                             'wb'))
        self.unique_p = sorted(list(self.unique_p))

        if altparams:

            return self.stimparams, self.unique_p, len(stimfiles), sparams_alt

        else:

            return self.stimparams, self.unique_p, len(stimfiles)

    def score_neurons(self, traces):

        if not self.stimparams or not self.unique_p:
            self.stimparams, self.unique_p, _ = pickle.load(
                open(os.path.join(self.stimpath, r'{0}_F{1}_{2:.1f}Hz_params.p'.format(self.date, self.fno, self.volrate)), 'rb'))

        resp = [list() for i in range(len(self.unique_p))]

        baselines = []
        for p in self.stimparams:

            idx = list(sorted(self.unique_p)).index(p[0])
            if p[2] > traces.shape[1]:
                continue

            bl = np.mean(traces[:, p[1] - int(round(5 * self.volrate)):p[1]], axis=1).reshape(-1, 1)
            blmin = np.mean(traces[:, p[1] - int(round(7 * self.volrate)):p[1] - int(round(5 * self.volrate))], axis=1).reshape(-1, 1)

            r = traces[:, p[1]:p[2] + 1]
            r = (r - bl) / bl
            resp[idx].append(r)
            baselines.append((bl-blmin)/blmin)

        self.baselines = np.concatenate(baselines, axis=1)
        self.resp_scores = np.zeros((traces.shape[0], len(self.unique_p)))
        self.magnitudes = np.empty((traces.shape[0], len(self.unique_p)))
        self.peak_t = np.empty((traces.shape[0], len(self.unique_p)))
        self.bad_stim = np.zeros(len(self.unique_p))

        meanmagn = []

        for rno, stim_r in enumerate(resp):
            if len(stim_r) == 0:
                self.resp_scores[:, rno] = 0.01 * np.ones(traces.shape[0])
                self.magnitudes[:, rno] = np.zeros(traces.shape[0])
                self.peak_t[:, rno] = np.zeros(traces.shape[0])
                self.bad_stim[rno] = True
            else:
                min_len = np.array([i[0].shape[0] for i in stim_r]).min()
                stim_r = np.array([i[:, :min_len] for i in stim_r])

                r_mean = np.mean(stim_r, axis=0) #TODO multipy instead of average
                errs = stim_r - r_mean
                rmse = np.mean(np.sqrt(np.mean(errs ** 2, axis=2)), axis=0)

                stds = np.std(r_mean, axis=1)
                self.resp_scores[:, rno] = stds / rmse
                self.magnitudes[:, rno] = r_mean.mean(axis=1)
                meanmagn.append(np.nanmean(r_mean))
                self.peak_t[:, rno] = r_mean.argmax(axis=1)
                # In case the stimulus only appeared once, scores will become infinite (rmse=0), change score to 0.01
                # not setting it to 0 because of a later division by the score.
                if not np.any(np.isfinite(self.resp_scores[:,rno])):
                    self.bad_stim[rno] = True
                    self.resp_scores[:, rno] = 0.01 * np.ones(traces.shape[0])

        return self.resp_scores, self.magnitudes, self.peak_t, self.bad_stim, np.mean(meanmagn), resp

def read_frate_singleplane(planepath):

    txtfile = [i for i in os.listdir(planepath) if i.endswith('metadata.txt')][0]
    txt = open(os.path.join(planepath, txtfile), 'rb')
    a = [str(i) for i in txt.readlines() if 'D3Step' in str(i)][0]
    print(a, planepath)
    x = re.findall("\d+\.\d+", a)
    frate = 1000. / float(x[0])
    print('Detected frame rate:', frate)
    return frate