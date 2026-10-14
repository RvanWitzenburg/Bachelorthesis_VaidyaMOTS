r"""@package motsfinder.metric.analytical.schwarzschildksfunky

Schwarzschild slice in Kerr-Schild coordinates.

The class defined here can be used to produce the data for a slice of
Schwarzschild spacetime in Kerr-Schild form.
"""

import numpy as np
from scipy import linalg

from ..base import _ThreeMetric, trivial_dtlapse, trivial_dtshift


__all__ = [
    "SchwarzschildKSSliceFunky",
]


class SchwarzschildKSSliceFunky(_ThreeMetric):
    r"""Data for a slice of Schwarzschild spacetime in Kerr-Schild coordinates.

    The formulas implemented here are based on [1].

    @b References

    [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker. "Initial
        data and coordinates for multiple black hole systems." Physical Review
        D 59.2 (1998): 024015.
    """

    def __init__(self, fp,fpp, M=1):
        r"""Create a Schwarzschild slice in Kerr-Schild form.

        @param M
            ADM mass of the spacetime.
        """
        super().__init__()
        self._M = float(M)
        self._fp = fp
        self._fpp = fpp

    @property
    def M(self):
        r"""ADM mass of the spacetime (\ie of the black hole)."""
        return self._M
        
    @property
    def fp(self):
        r"""Slicing function"""
        return self._fp
        
    @property
    def fpp(self):
        r"""Derivative of the slicing function"""
        return self._fpp

    def H(self, point):
        r"""Scalar H function of [1]."""
        r = linalg.norm(point)
        return self._M/r

    def _mat_at(self, point):
        r = linalg.norm(point)
        M = self._M
        x = point
        fp = self._fp
        
        return np.asarray(np.identity(3) + [
            [2*M * x[i]*x[j] / r**3 for j in range(3)]
            for i in range(3)
        ] + [
        [0,0,2*M/r*x[0]*fp(x[2])/r],[0,0,2*M/r*x[1]*fp(x[2])/r],[2*M/r*x[0]*fp(x[2])/r,2*M/r*x[1]*fp(x[2])/r,2*M/r*2*x[2]*fp(x[2])/r]
        ]+[
        [0,0,0],[0,0,0],[0,0,2*M/r*(-r/(2*M)*fp(x[2])**2+fp(x[2])**2)]])
        

    def diff(self, point, inverse=False, diff=1):
        if inverse:
            return self._compute_inverse_diff(point, diff=diff)
        if diff == 0:
            return self._mat_at(point)
        if diff != 0:
            raise NotImplementedError

    def get_curv(self):
        return SchwarzschildKSSliceFunkyCurv(self)

    def get_lapse(self):
        return SchwarzschildKSSliceFunkyLapse(self)

    def get_dtlapse(self):
        return trivial_dtlapse

    def get_shift(self):
        return SchwarzschildKSSliceFunkyShift(self)

    def get_dtshift(self):
        return trivial_dtshift


class SchwarzschildKSSliceFunkyLapse():
    r"""Lapse function of the Schwarzschild slice in Kerr-Schild form."""
    
    def __init__(self, metric):
        self._g = metric
        self._fp = metric.fp
        self._fpp = metric.fpp

    def __call__(self, point, diff=0):
        ra = range(3)
        x = point
        M = self._g.M
        r = linalg.norm(point)
        fp = self._fp
        fpp = self._fpp
        
        if diff != 0:
            raise NotImplementedError

        return np.sqrt(-(r**3/(-(r**2*(2*M + r)) - 4*M*r*x[2]*fp(x[2]) + (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2)))
        
class SchwarzschildKSSliceFunkyShift():
    r"""Shift vector of the Schwarzschild slice in Kerr-Schild form."""
    
    def __init__(self, metric):
        self._g = metric
        self._fp = metric.fp
        self._fpp = metric.fpp


    def __call__(self, point, diff=0):
        if diff != 0:
            raise NotImplementedError
            
        r = linalg.norm(point)
        M = self._g.M
        x = point
        fp = self._fp
        fpp = self._fpp

        betal = [(2*M*x[0])/r**2,(2*M*x[1])/r**2,(2*M*x[2] + (2*M - r)*r*fp(x[2]))/r**2]
        g_inv = self._g.diff(point, inverse=True, diff=0)
        beta = np.einsum('ij,j->i', g_inv, betal)
        return np.asarray(beta)



class SchwarzschildKSSliceFunkyCurv():
    r"""Extrinsic curvature of the Schwarzschild slice in Kerr-Schild form."""
    def __init__(self, metric):
        self._g = metric
        self._fp = metric.fp
        self._fpp = metric.fpp

    def __call__(self, point, diff=0):
        x = point
        M = self._g.M
        r = linalg.norm(point)
        fp = self._fp
        fpp = self._fpp
        
        kxx = -((M*(2*(r**3*(x[0]**2 - x[1]**2 - x[2]**2) + M*x[0]**2*(x[0]**2 + x[1]**2 + x[2]**2)) + r*x[2]*(2*M*x[0]**2 + r*(x[0]**2 - 2*(x[1]**2 + x[2]**2)))*fp(x[2])))/(r**2*(x[0]**2 + x[1]**2 + x[2]**2)*np.sqrt(r**3/(r**2*(2*M + r) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2))*(r**3 + 2*M*(x[0]**2 + x[1]**2 + x[2]**2) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2)))
        kyy = -((M*(-2*r**3*(x[0]**2 - x[1]**2 + x[2]**2) + 2*M*x[1]**2*(x[0]**2 + x[1]**2 + x[2]**2) + r*x[2]*(2*M*x[1]**2 + r*(-2*x[0]**2 + x[1]**2 - 2*x[2]**2))*fp(x[2])))/(r**2*(x[0]**2 + x[1]**2 + x[2]**2)*np.sqrt(r**3/(r**2*(2*M + r) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2))*(r**3 + 2*M*(x[0]**2 + x[1]**2 + x[2]**2) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2)))
        kzz = -((2*M*(-(r**3*(x[0]**2 + x[1]**2 - x[2]**2)) + M*x[2]**2*(x[0]**2 + x[1]**2 + x[2]**2)) + M*r*x[2]*(2*r**3 + r*(-2*x[0]**2 - 2*x[1]**2 + x[2]**2) + 2*M*(2*x[0]**2 + 2*x[1]**2 + 3*x[2]**2))*fp(x[2]) + 2*M**2*r**2*(x[0]**2 + x[1]**2 + 3*x[2]**2)*fp(x[2])**2 + M*(2*M - r)*r**3*x[2]*fp(x[2])**3 + r**2*(x[0]**2 + x[1]**2 + x[2]**2)*(r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2 + x[2]**2))*fpp(x[2]))/(r**2*(x[0]**2 + x[1]**2 + x[2]**2)*np.sqrt(r**3/(r**2*(2*M + r) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2))*(r**3 + 2*M*(x[0]**2 + x[1]**2 + x[2]**2) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2)))
        kxy =  -((M*x[0]*x[1]*(4*r**3 + 2*M*(x[0]**2 + x[1]**2 + x[2]**2) + r*(2*M + 3*r)*x[2]*fp(x[2])))/(r**2*(x[0]**2 + x[1]**2 + x[2]**2)*np.sqrt(r**3/(r**2*(2*M + r) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2))*(r**3 + 2*M*(x[0]**2 + x[1]**2 + x[2]**2) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2)))
        kxz =  -((M*x[0]*(2*x[2]*(2*r**3 + M*(x[0]**2 + x[1]**2 + x[2]**2)) + r*(r**3 + 3*r*x[2]**2 + 2*M*(x[0]**2 + x[1]**2 + 2*x[2]**2))*fp(x[2]) + 2*M*r**2*x[2]*fp(x[2])**2))/(r**2*(x[0]**2 + x[1]**2 + x[2]**2)*np.sqrt(r**3/(r**2*(2*M + r) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2))*(r**3 + 2*M*(x[0]**2 + x[1]**2 + x[2]**2) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2))) 
        kyz =   -((M*x[1]*(2*x[2]*(2*r**3 + M*(x[0]**2 + x[1]**2 + x[2]**2)) + r*(r**3 + 3*r*x[2]**2 + 2*M*(x[0]**2 + x[1]**2 + 2*x[2]**2))*fp(x[2]) + 2*M*r**2*x[2]*fp(x[2])**2))/(r**2*(x[0]**2 + x[1]**2 + x[2]**2)*np.sqrt(r**3/(r**2*(2*M + r) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2))*(r**3 + 2*M*(x[0]**2 + x[1]**2 + x[2]**2) + 4*M*r*x[2]*fp(x[2]) - (r**3 + 2*M*(-r**2 + x[0]**2 + x[1]**2))*fp(x[2])**2)))
        
        if diff == 0:
            return np.asarray([[kxx,kxy,kxz],[kxy,kyy,kyz],[kxz,kyz,kzz]])
        if diff != 0:
            raise NotImplementedError
