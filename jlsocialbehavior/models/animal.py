# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:51:08 2016

@author: jlarsch
"""

import random
from models.AnimalTimeSeriesCollection import AnimalTimeSeriesCollection


class Animal(object):
    def __init__(self, ID=0, paired=True):
        self.ID = ID
        self.pair = []
        self.ts = []
        self.paired = paired
        self.neighbor = None
        self.experiment = None

    def joinPair(self, pair):  # use to add episode data into a pair.
        self.pair = pair
        pair.addAnimal(self)

    def joinExperiment(self, experiment):  # use to add all data to experiment.
        self.experiment = experiment
        experiment.addAnimal(self)
        self.paired = False
        self.wakeUp()

    def add_TimeSeriesCollection(self, ts):
        self.ts = ts
        return ts

    def add_BoutSeriesCollection(self, bs):
        self.bs = bs
        return bs

    def wakeUp(self):
        AnimalTimeSeriesCollection().linkAnimal(self)
        if self.paired:
            self.neighbor = self.pair.animals[1-self.ID]


