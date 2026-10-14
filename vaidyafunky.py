r"""@package motsfinder.metric.analytical.vaidyafunky

Schwarzschild slice in Kerr-Schild coordinates.

The class defined here can be used to produce the data for a slice of
Schwarzschild spacetime in Kerr-Schild form.
"""

import numpy as np
from scipy import linalg

from ..base import _ThreeMetric, trivial_dtlapse, trivial_dtshift


__all__ = [
    "VaidyaSliceFunky",
]


class VaidyaSliceFunky(_ThreeMetric):
    r"""Data for a slice of Vaidya spacetime in Kerr-Schild coordinates.

    The formulas implemented here are based on [1].

    @b References

    [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker. "Initial
        data and coordinates for multiple black hole systems." Physical Review
        D 59.2 (1998): 024015.
    """

    def __init__(self, fp,fpp, M):
        r"""Create a Vaidya slice in Kerr-Schild form.

        @param M
            ADM mass of the spacetime.
        """
        super().__init__()
        self._M = M
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

    def _mat_at(self, point):
        r = linalg.norm(point)
        M = self._M
        x = point
        fp = self.fp

        gxx = 1 + (2*x[0]**2*M(x[0],x[1],x[2]))/r**3
        gyy = 1 + (2*x[1]**2*M(x[0],x[1],x[2]))/r**3
        gzz = 1 - fp(x[2])**2 + (2*M(x[0],x[1],x[2])*(x[2] + r*fp(x[2]))**2)/r**3
        gxy = (2*x[0]*x[1]*M(x[0],x[1],x[2]))/r**3
        gxz = (2*x[0]*M(x[0],x[1],x[2])*(x[2] + r*fp(x[2])))/r**3 
        gyz = (2*x[1]*M(x[0],x[1],x[2])*(x[2] + r*fp(x[2])))/r**3

        return np.asarray([[gxx,gxy,gxz],[gxy,gyy,gyz],[gxz,gyz,gzz]])   

    def diff(self, point, inverse=False, diff=1):
        if inverse:
            return self._compute_inverse_diff(point, diff=diff)
        if diff == 0:
            return self._mat_at(point)
        if diff != 0:
            raise NotImplementedError

    def get_curv(self):
        return VaidyaSliceFunkyCurv(self)

    def get_lapse(self):
        return VaidyaSliceFunkyLapse(self)

    def get_dtlapse(self):
        return VaidyaSliceFunkydtLapse(self)

    def get_shift(self):
        return VaidyaSliceFunkyShift(self)

    def get_dtshift(self):
        return VaidyaSliceFunkydtShift(self)


class VaidyaSliceFunkyLapse():
    r"""Lapse function of the Vaidya slice in Kerr-Schild form."""
    
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

        return np.sqrt(-(r**3/(r**3*(-1 + fp(x[2])**2) - 2*M(x[0],x[1],x[2])*(r**2 + 2*r*x[2]*fp(x[2]) + (r**2 - x[0]**2 - x[1]**2)*fp(x[2])**2))))

class VaidyaSliceFunkyShift():
    r"""Shift vector of the Vaidya slice in Kerr-Schild form."""
    
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
        h = 2*M(x[0],x[1],x[2]) / r
        
        betalx = (2*x[0]*M(x[0],x[1],x[2]))/r**2
        betaly = (2*x[1]*M(x[0],x[1],x[2]))/r**2
        betalz = -fp(x[2]) + (2*M(x[0],x[1],x[2])*(x[2] + r*fp(x[2])))/r**2
        
        betal = [betalx,betaly,betalz]
        g_inv = self._g.diff(point, inverse=True, diff=0)
        beta = np.einsum('ij,j->i', g_inv, betal)
        return np.asarray(beta)



class VaidyaSliceFunkyCurv():
    r"""Extrinsic curvature of the Vaidya slice in Kerr-Schild form."""
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
        h = 2*M(x[0],x[1],x[2]) / r
        
        kxx = (-2*r*x[0]**2*M(x[0],x[1],x[2])**2*(x[0]**2 + x[1]**2 + x[2]**2 + r*x[2]*fp(x[2])) - r**3*x[0]*(x[0]**2 + x[1]**2 + x[2]**2)*(-2*r*M(x[0],x[1],x[2],1,1) + fp(x[2])*(x[0]*M(x[0],x[1],x[2],1,3) - 2*x[2]*M(x[0],x[1],x[2],1,1)) + x[0]*M(x[0],x[1],x[2],1,0)) + M(x[0],x[1],x[2])*(2*(-(r**4*x[0]**2) + r**4*x[1]**2 + r**4*x[2]**2 + r*x[0]**2*x[2]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) + r*x[0]**2*x[1]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,2) + r*x[0]**5*M(x[0],x[1],x[2],1,1) + r*x[0]**3*x[1]**2*M(x[0],x[1],x[2],1,1) + r*x[0]**3*x[2]**2*M(x[0],x[1],x[2],1,1) - x[0]**6*M(x[0],x[1],x[2],1,0) - 2*x[0]**4*x[1]**2*M(x[0],x[1],x[2],1,0) - x[0]**2*x[1]**4*M(x[0],x[1],x[2],1,0) - 2*x[0]**4*x[2]**2*M(x[0],x[1],x[2],1,0) - 2*x[0]**2*x[1]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - x[0]**2*x[2]**4*M(x[0],x[1],x[2],1,0)) + fp(x[2])*(-2*x[0]**2*(-r**2 + x[0]**2 + x[1]**2)*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) + x[2]*(-(r**3*(x[0]**2 - 2*(x[1]**2 + x[2]**2))) + 2*x[0]**2*x[1]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,2) + 2*x[0]**3*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,1) - 2*r*x[0]**2*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,0)))))/(r**3*(x[0]**2 + x[1]**2 + x[2]**2)*np.sqrt(-(r**3/(r**3*(-1 + fp(x[2])**2) - 2*M(x[0],x[1],x[2])*(r**2 + 2*r*x[2]*fp(x[2]) + (r**2 - x[0]**2 - x[1]**2)*fp(x[2])**2))))*(r**3*(-1 + fp(x[2])**2) + 2*M(x[0],x[1],x[2])*(-x[0]**2 - x[1]**2 - x[2]**2 - 2*r*x[2]*fp(x[2]) + (-r**2 + x[0]**2 + x[1]**2)*fp(x[2])**2)))
        kyy = (-2*r*x[1]**2*M(x[0],x[1],x[2])**2*(x[0]**2 + x[1]**2 + x[2]**2 + r*x[2]*fp(x[2])) - r**3*x[1]*(x[0]**2 + x[1]**2 + x[2]**2)*(-2*r*M(x[0],x[1],x[2],1,2) + fp(x[2])*(x[1]*M(x[0],x[1],x[2],1,3) - 2*x[2]*M(x[0],x[1],x[2],1,2)) + x[1]*M(x[0],x[1],x[2],1,0)) + M(x[0],x[1],x[2])*(2*(r**4*x[0]**2 - r**4*x[1]**2 + r**4*x[2]**2 + r*x[1]**2*x[2]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) + r*x[1]**3*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,2) + r*x[0]**3*x[1]**2*M(x[0],x[1],x[2],1,1) + r*x[0]*x[1]**4*M(x[0],x[1],x[2],1,1) + r*x[0]*x[1]**2*x[2]**2*M(x[0],x[1],x[2],1,1) - x[0]**4*x[1]**2*M(x[0],x[1],x[2],1,0) - 2*x[0]**2*x[1]**4*M(x[0],x[1],x[2],1,0) - x[1]**6*M(x[0],x[1],x[2],1,0) - 2*x[0]**2*x[1]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - 2*x[1]**4*x[2]**2*M(x[0],x[1],x[2],1,0) - x[1]**2*x[2]**4*M(x[0],x[1],x[2],1,0)) + fp(x[2])*(-2*x[1]**2*(-r**2 + x[0]**2 + x[1]**2)*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) + x[2]*(r**3*(2*x[0]**2 - x[1]**2 + 2*x[2]**2) + 2*x[1]**3*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,2) + 2*x[0]*x[1]**2*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,1) - 2*r*x[1]**2*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,0)))))/(r**3*(x[0]**2 + x[1]**2 + x[2]**2)*np.sqrt(-(r**3/(r**3*(-1 + fp(x[2])**2) - 2*M(x[0],x[1],x[2])*(r**2 + 2*r*x[2]*fp(x[2]) + (r**2 - x[0]**2 - x[1]**2)*fp(x[2])**2))))*(r**3*(-1 + fp(x[2])**2) + 2*M(x[0],x[1],x[2])*(-x[0]**2 - x[1]**2 - x[2]**2 - 2*r*x[2]*fp(x[2]) + (-r**2 + x[0]**2 + x[1]**2)*fp(x[2])**2)))
        kzz = ((x[2] + r*fp(x[2]))**2*M(x[0],x[1],x[2],1,0) + (r*(-(r**4*(2*r*x[2]*(x[2]*M(x[0],x[1],x[2]) - (x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) - (x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])*M(x[0],x[1],x[2],1,0)) + x[2]**2*fp(x[2])*(x[2]*M(x[0],x[1],x[2]) - (x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) - (x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])*M(x[0],x[1],x[2],1,0)) + r**2*(r*(x[0]**2 + x[1]**2 + x[2]**2)*fpp(x[2]) + fp(x[2])*(2 - fp(x[2])**2)*(x[2]*M(x[0],x[1],x[2]) - (x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) - (x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])*M(x[0],x[1],x[2],1,0))))) + 2*r*M(x[0],x[1],x[2])*(-(x[2]**2*(x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])*((x[0]**2 + x[1]**2)*M(x[0],x[1],x[2],1,3) - x[1]*x[2]*M(x[0],x[1],x[2],1,2) - x[0]*x[2]*M(x[0],x[1],x[2],1,1) + x[0]**2*fp(x[2])*M(x[0],x[1],x[2],1,0) + x[1]**2*fp(x[2])*M(x[0],x[1],x[2],1,0))) + r**3*(r*(x[0]**2 + x[1]**2 + x[2]**2) - fp(x[2])**2*((x[0]**2 + x[1]**2 + 3*x[2]**2)*M(x[0],x[1],x[2]) - (x[0]**2 + x[1]**2 + x[2]**2)*(3*x[2]*M(x[0],x[1],x[2],1,3) + x[1]*M(x[0],x[1],x[2],1,2) + x[0]*M(x[0],x[1],x[2],1,1))) + 3*x[2]*(x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])**3*M(x[0],x[1],x[2],1,0)) - r**2*(r*(r*x[2]**2 + (x[0]**2 + x[1]**2 + x[2]**2)**2*fpp(x[2])) + (x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])**3*((x[0]**2 + x[1]**2)*M(x[0],x[1],x[2],1,3) - x[2]*(x[1]*M(x[0],x[1],x[2],1,2) + x[0]*M(x[0],x[1],x[2],1,1))) - x[2]*fp(x[2])*(-((2*x[0]**2 + 2*x[1]**2 + 3*x[2]**2)*M(x[0],x[1],x[2])) + (x[0]**2 + x[1]**2 + x[2]**2)*(r + 3*x[2]*M(x[0],x[1],x[2],1,3) + 2*x[1]*M(x[0],x[1],x[2],1,2) + 2*x[0]*M(x[0],x[1],x[2],1,1))) - 3*x[2]**2*(x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])**2*M(x[0],x[1],x[2],1,0) + (x[0]**2 + x[1]**2)*(x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])**4*M(x[0],x[1],x[2],1,0)) - r*x[2]*(-(x[2]*(x[0]**2 + x[1]**2 + x[2]**2)*(-M(x[0],x[1],x[2]) + x[2]*M(x[0],x[1],x[2],1,3) + x[1]*M(x[0],x[1],x[2],1,2) + x[0]*M(x[0],x[1],x[2],1,1))) + 2*(x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])**2*((x[0]**2 + x[1]**2)*M(x[0],x[1],x[2],1,3) - x[2]*(x[1]*M(x[0],x[1],x[2],1,2) + x[0]*M(x[0],x[1],x[2],1,1))) + 2*(x[0]**2 + x[1]**2)*(x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])**3*M(x[0],x[1],x[2],1,0) + x[2]**2*fp(x[2])*(r**2 - (x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,0))) + r**4*(-(x[2]*M(x[0],x[1],x[2])*fp(x[2])**3) + (x[0]**2 + x[1]**2 + x[2]**2)*(r*fpp(x[2]) + fp(x[2])**3*(M(x[0],x[1],x[2],1,3) + fp(x[2])*M(x[0],x[1],x[2],1,0)))))))/((x[0]**2 + x[1]**2 + x[2]**2)*(r**5*(-1 + fp(x[2])**2) - 2*r**2*M(x[0],x[1],x[2])*(x[0]**2 + x[1]**2 + x[2]**2 + 2*r*x[2]*fp(x[2]) + (r**2 - x[0]**2 - x[1]**2)*fp(x[2])**2))))/(r**3*np.sqrt(-(r**3/(r**3*(-1 + fp(x[2])**2) - 2*M(x[0],x[1],x[2])*(r**2 + 2*r*x[2]*fp(x[2]) + (r**2 - x[0]**2 - x[1]**2)*fp(x[2])**2)))))
        kxy = (-2*r*x[0]*x[1]*M(x[0],x[1],x[2])**2*(x[0]**2 + x[1]**2 + x[2]**2 + r*x[2]*fp(x[2])) - r**3*(x[0]**2 + x[1]**2 + x[2]**2)*(-(r*x[0]*M(x[0],x[1],x[2],1,2)) - r*x[1]*M(x[0],x[1],x[2],1,1) + fp(x[2])*(x[0]*x[1]*M(x[0],x[1],x[2],1,3) - x[2]*(x[0]*M(x[0],x[1],x[2],1,2) + x[1]*M(x[0],x[1],x[2],1,1))) + x[0]*x[1]*M(x[0],x[1],x[2],1,0)) + x[0]*x[1]*M(x[0],x[1],x[2])*(2*(-2*r**4 + r*x[2]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) + r*x[1]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,2) + r*x[0]**3*M(x[0],x[1],x[2],1,1) + r*x[0]*x[1]**2*M(x[0],x[1],x[2],1,1) + r*x[0]*x[2]**2*M(x[0],x[1],x[2],1,1) - x[0]**4*M(x[0],x[1],x[2],1,0) - 2*x[0]**2*x[1]**2*M(x[0],x[1],x[2],1,0) - x[1]**4*M(x[0],x[1],x[2],1,0) - 2*x[0]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - 2*x[1]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - x[2]**4*M(x[0],x[1],x[2],1,0)) + fp(x[2])*(2*(r**2 - x[0]**2 - x[1]**2)*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) + x[2]*(-3*r**3 + 2*x[1]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,2) + 2*x[0]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,1) - 2*r*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,0)))))/(r**3*(x[0]**2 + x[1]**2 + x[2]**2)*np.sqrt(-(r**3/(r**3*(-1 + fp(x[2])**2) - 2*M(x[0],x[1],x[2])*(r**2 + 2*r*x[2]*fp(x[2]) + (r**2 - x[0]**2 - x[1]**2)*fp(x[2])**2))))*(r**3*(-1 + fp(x[2])**2) + 2*M(x[0],x[1],x[2])*(-x[0]**2 - x[1]**2 - x[2]**2 - 2*r*x[2]*fp(x[2]) + (-r**2 + x[0]**2 + x[1]**2)*fp(x[2])**2)))
        kxz = (-2*r*x[0]*M(x[0],x[1],x[2])**2*(x[2] + r*fp(x[2]))*(x[0]**2 + x[1]**2 + x[2]**2 + r*x[2]*fp(x[2])) + r**3*(x[0]**2 + x[1]**2 + x[2]**2)*(-(r*x[0]*(-1 + fp(x[2])**2)*M(x[0],x[1],x[2],1,3)) + (r*x[2] + (r**2 + x[2]**2)*fp(x[2]) + r*x[2]*fp(x[2])**2)*M(x[0],x[1],x[2],1,1) + x[0]*x[2]*(-1 + fp(x[2])**2)*M(x[0],x[1],x[2],1,0)) + x[0]*M(x[0],x[1],x[2])*(2*x[2]*(-2*r**4 + r*x[2]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) + r*x[1]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,2) + r*x[0]**3*M(x[0],x[1],x[2],1,1) + r*x[0]*x[1]**2*M(x[0],x[1],x[2],1,1) + r*x[0]*x[2]**2*M(x[0],x[1],x[2],1,1) - x[0]**4*M(x[0],x[1],x[2],1,0) - 2*x[0]**2*x[1]**2*M(x[0],x[1],x[2],1,0) - x[1]**4*M(x[0],x[1],x[2],1,0) - 2*x[0]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - 2*x[1]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - x[2]**4*M(x[0],x[1],x[2],1,0)) + fp(x[2])*(-r**5 - 3*r**3*x[2]**2 + 2*(2*r**2 - x[0]**2 - x[1]**2)*x[2]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) + 2*x[1]*(r**2 + x[2]**2)*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,2) + 2*r**2*x[0]**3*M(x[0],x[1],x[2],1,1) + 2*r**2*x[0]*x[1]**2*M(x[0],x[1],x[2],1,1) + 2*r**2*x[0]*x[2]**2*M(x[0],x[1],x[2],1,1) + 2*x[0]**3*x[2]**2*M(x[0],x[1],x[2],1,1) + 2*x[0]*x[1]**2*x[2]**2*M(x[0],x[1],x[2],1,1) + 2*x[0]*x[2]**4*M(x[0],x[1],x[2],1,1) - 2*r*x[0]**4*M(x[0],x[1],x[2],1,0) - 4*r*x[0]**2*x[1]**2*M(x[0],x[1],x[2],1,0) - 2*r*x[1]**4*M(x[0],x[1],x[2],1,0) - 6*r*x[0]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - 6*r*x[1]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - 4*r*x[2]**4*M(x[0],x[1],x[2],1,0)) + 2*r*(x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])**2*((r**2 - x[0]**2 - x[1]**2)*M(x[0],x[1],x[2],1,3) + x[2]*(x[1]*M(x[0],x[1],x[2],1,2) + x[0]*M(x[0],x[1],x[2],1,1) - r*M(x[0],x[1],x[2],1,0)))))/(r**3*(x[0]**2 + x[1]**2 + x[2]**2)*np.sqrt(-(r**3/(r**3*(-1 + fp(x[2])**2) - 2*M(x[0],x[1],x[2])*(r**2 + 2*r*x[2]*fp(x[2]) + (r**2 - x[0]**2 - x[1]**2)*fp(x[2])**2))))*(r**3*(-1 + fp(x[2])**2) + 2*M(x[0],x[1],x[2])*(-x[0]**2 - x[1]**2 - x[2]**2 - 2*r*x[2]*fp(x[2]) + (-r**2 + x[0]**2 + x[1]**2)*fp(x[2])**2)))
        kyz = (-2*r*x[1]*M(x[0],x[1],x[2])**2*(x[2] + r*fp(x[2]))*(x[0]**2 + x[1]**2 + x[2]**2 + r*x[2]*fp(x[2])) + r**3*(x[0]**2 + x[1]**2 + x[2]**2)*(-(r*x[1]*(-1 + fp(x[2])**2)*M(x[0],x[1],x[2],1,3)) + (r*x[2] + (r**2 + x[2]**2)*fp(x[2]) + r*x[2]*fp(x[2])**2)*M(x[0],x[1],x[2],1,2) + x[1]*x[2]*(-1 + fp(x[2])**2)*M(x[0],x[1],x[2],1,0)) + x[1]*M(x[0],x[1],x[2])*(2*x[2]*(-2*r**4 + r*x[2]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) + r*x[1]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,2) + r*x[0]**3*M(x[0],x[1],x[2],1,1) + r*x[0]*x[1]**2*M(x[0],x[1],x[2],1,1) + r*x[0]*x[2]**2*M(x[0],x[1],x[2],1,1) - x[0]**4*M(x[0],x[1],x[2],1,0) - 2*x[0]**2*x[1]**2*M(x[0],x[1],x[2],1,0) - x[1]**4*M(x[0],x[1],x[2],1,0) - 2*x[0]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - 2*x[1]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - x[2]**4*M(x[0],x[1],x[2],1,0)) + fp(x[2])*(-r**5 - 3*r**3*x[2]**2 + 2*(2*r**2 - x[0]**2 - x[1]**2)*x[2]*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,3) + 2*x[1]*(r**2 + x[2]**2)*(x[0]**2 + x[1]**2 + x[2]**2)*M(x[0],x[1],x[2],1,2) + 2*r**2*x[0]**3*M(x[0],x[1],x[2],1,1) + 2*r**2*x[0]*x[1]**2*M(x[0],x[1],x[2],1,1) + 2*r**2*x[0]*x[2]**2*M(x[0],x[1],x[2],1,1) + 2*x[0]**3*x[2]**2*M(x[0],x[1],x[2],1,1) + 2*x[0]*x[1]**2*x[2]**2*M(x[0],x[1],x[2],1,1) + 2*x[0]*x[2]**4*M(x[0],x[1],x[2],1,1) - 2*r*x[0]**4*M(x[0],x[1],x[2],1,0) - 4*r*x[0]**2*x[1]**2*M(x[0],x[1],x[2],1,0) - 2*r*x[1]**4*M(x[0],x[1],x[2],1,0) - 6*r*x[0]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - 6*r*x[1]**2*x[2]**2*M(x[0],x[1],x[2],1,0) - 4*r*x[2]**4*M(x[0],x[1],x[2],1,0)) + 2*r*(x[0]**2 + x[1]**2 + x[2]**2)*fp(x[2])**2*((r**2 - x[0]**2 - x[1]**2)*M(x[0],x[1],x[2],1,3) + x[2]*(x[1]*M(x[0],x[1],x[2],1,2) + x[0]*M(x[0],x[1],x[2],1,1) - r*M(x[0],x[1],x[2],1,0)))))/(r**3*(x[0]**2 + x[1]**2 + x[2]**2)*np.sqrt(-(r**3/(r**3*(-1 + fp(x[2])**2) - 2*M(x[0],x[1],x[2])*(r**2 + 2*r*x[2]*fp(x[2]) + (r**2 - x[0]**2 - x[1]**2)*fp(x[2])**2))))*(r**3*(-1 + fp(x[2])**2) + 2*M(x[0],x[1],x[2])*(-x[0]**2 - x[1]**2 - x[2]**2 - 2*r*x[2]*fp(x[2]) + (-r**2 + x[0]**2 + x[1]**2)*fp(x[2])**2)))
       
        if diff == 0:
            return np.asarray([[-kxx,-kxy,-kxz],[-kxy,-kyy,-kyz],[-kxz,-kyz,-kzz]])
        if diff != 0:
            raise NotImplementedError

class VaidyaSliceFunkydtLapse():
    r"""Time derivative of the lapse function of the Vaidya slice in Kerr-Schild form."""
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
            
        return (r**6*(r**2 + 2*r*x[2]*fp(x[2]) + (r**2 - x[0]**2 - x[1]**2)*fp(x[2])**2)*M(x[0],x[1],x[2],1,0))/((-(r**3/(r**3*(-1 + fp(x[2])**2) - 2*M(x[0],x[1],x[2])*(r**2 + 2*r*x[2]*fp(x[2]) + (r**2 - x[0]**2 - x[1]**2)*fp(x[2])**2))))**1.5*(r**3*(-1 + fp(x[2])**2) - 2*M(x[0],x[1],x[2])*(r**2 + 2*r*x[2]*fp(x[2]) + (r**2 - x[0]**2 - x[1]**2)*fp(x[2])**2))**3)

class VaidyaSliceFunkydtShift():
    r"""Time derivative of the shift vector of the Vaidya slice in Kerr-Schild form."""
    def __init__(self, metric):
        self._g = metric
        self._fp = metric.fp
        self._fpp = metric.fpp

    def __call__(self, point, diff=0):
        x = point
        M = self._g.M
        r = linalg.norm(point)
        fp = self._fp

        if diff != 0:
            raise NotImplementedError

        Mdot = M(x[0], x[1], x[2], 1, 0)       # ∂_τ M

        # Covariant shift time derivatives: ∂_τβ_i
        betaldt = np.array([
            (2*x[0]*Mdot) / r**2,
            (2*x[1]*Mdot) / r**2,
            (2*(x[2] + r*fp(x[2]))*Mdot) / r**2,
        ])

        # Covariant shift: β_i
        betal = np.array([
            (2*x[0]*M(x[0],x[1],x[2])) / r**2,
            (2*x[1]*M(x[0],x[1],x[2])) / r**2,
            -fp(x[2]) + (2*M(x[0],x[1],x[2])*(x[2] + r*fp(x[2]))) / r**2,
        ])

        # Funky null vector (unnormalised spatial part): x^(f)_i = (x, y, z + r f')
        x_f = np.array([x[0], x[1], x[2] + r*fp(x[2])])

        # ∂_τγ_{ij} = 2 x^(f)_i x^(f)_j ∂_τM / r^3
        dtgam = (2*Mdot / r**3) * np.outer(x_f, x_f)

        g_inv = self._g.diff(point, inverse=True, diff=0)

        # ∂_τβ^i = γ^{ij} ∂_τβ_j  −  γ^{ik}(∂_τγ_{kl})γ^{lj}β_j
        term1 = np.einsum('ij,j->i',         g_inv, betaldt)
        term2 = np.einsum('ik,kl,lj,j->i',  g_inv, dtgam, g_inv, betal)

        return np.asarray(term1 - term2)
            