#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a part of the implementation of the stochastic registration algorithm based
on the following paper:
Andriy Myronenko and Xubo Song, "Point set registration: Coherent Point drift", IEEE
Transactions on Pattern Analysis and Machine Intelligence. 32 (2): 2262-2275, 2010.

The library is based on the python implementation of the paper in pycpd package.
"""

import numpy as np

from skeleton_refinement.expectation_maximization_registration import ExpectationMaximizationRegistration
from skeleton_refinement.utilities import gaussian_kernel

ALPHA = 2  # default value of alpha
BETA = 2  # default value of beta


class DeformableRegistration(ExpectationMaximizationRegistration):
    def __init__(self, alpha=ALPHA, beta=BETA, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = ALPHA if alpha is None else alpha
        self.beta = BETA if alpha is None else beta
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)

    def update_transform(self):
        A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
        B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)
        self.W = np.linalg.solve(A, B)

    def transform_point_cloud(self, Y=None):
        if Y is None:
            self.TY = self.Y + np.dot(self.G, self.W)
            return
        else:
            return Y + np.dot(self.G, self.W)

    def update_variance(self):
        qprev = self.sigma2

        xPx = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X, self.X), axis=1))
        yPy = np.dot(np.transpose(self.P1), np.sum(np.multiply(self.TY, self.TY), axis=1))
        trPXY = np.sum(np.multiply(self.TY, np.dot(self.P, self.X)))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10
        self.err = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        return self.G, self.W
