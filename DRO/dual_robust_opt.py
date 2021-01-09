"""
Some functions for solving the dual form of the robust optimization.

Recall that the dual objective is

\min_\theta c_k E[ [ l(\theta; X) -\eta ]_+^{k'} ] + \eta

What is provided here:

 Helper function for calculating c_k
 Binary search procedure for calculating optimal \eta

What is not provided here:
 
 Loss minimizers for [l(\theta;X) - \eta]_+^{k'}

(For recommender, we need to switch / use tensorrec).
"""

import scipy.optimize as sopt
import numpy as np

class robust_dual_opt:
    def __init__(self, ktype, eps):
        self.ktype = ktype
        self.eps = eps

    def fk(self, t):
        if self.ktype == 'chisq':
            return (t**2.0-2.0*t+2.0-1.0)/(2*2.0*(2.0-1.0))
        elif self.ktype =='cvar':
            return t

    def get_rho(self):
        return self.fk(1.0/self.eps)

    def get_ck(self):
        if self.ktype == 'chisq':
            return (2*self.get_rho()-1.0)**(1.0/2.0)
        elif self.ktype == 'cvar':
            return 1.0/self.eps

    def bisect_losses(self, fn, max_l):
        ck = self.get_ck()
        fstar = lambda eta: ck*fn(eta)+eta
        return sopt.brent(fstar, brack=(0, max_l))

