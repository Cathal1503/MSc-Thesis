"""
dv_complex.py
=============
Dyakonov / Dyakonov-Voigt surface-wave machinery for *complex* permittivities
with *real* propagation angle psi, following Mackay-Zhou-Lakhtakia
(Proc. R. Soc. A 475, 20190317) and the exceptional-points Letter
(arXiv:2004.02260), normalized to k0 = 1 and exp(-i w t).

Frame: medium A uniaxial, optic axis along x in the interface plane,
    eps_A = eps_s I + (eps_t - eps_s) xhat xhat ;
medium B isotropic eps_B.  Propagation at angle psi to the optic axis.

Key facts (all analytic, hence valid by continuation for complex eps):
 * EP (Voigt) locus:  alpha_A1 = alpha_A2  <=>  q = sqrt(eps_s)/cos psi.
 * DV angle (closed form, eq. 2.48-type):
       tan^2 psi_DV = (eps_B - eps_s)(eps_s + eps_t)^2
                      / [ 4 eps_s (eps_s + eps_B)(eps_t - eps_B) ].
 * For complex (eps_s, eps_t), requiring psi REAL turns that relation into a
   QUADRATIC for eps_B at each real psi -- the "partner locus" eps_B(psi):
       -t x^2 + [t(eps_t - eps_s) - C] x + t eps_s eps_t + C eps_s = 0,
       t = tan^2 psi,  C = (eps_s + eps_t)^2/(4 eps_s).
"""
from __future__ import annotations
import cmath
import numpy as np


# ------------------------------------------------------------- decay constants
def alphas(q, psi, es, et, eB):
    """Normalized decay constants; branches Im(a1), Im(a2) > 0, Im(aB) < 0."""
    a1 = 1j * cmath.sqrt(q * q - es)
    if a1.imag < 0:
        a1 = -a1
    arg = (q * q * (es + et - (es - et) * cmath.cos(2 * psi)) - 2 * es * et) / (2 * es)
    a2 = 1j * cmath.sqrt(arg)
    if a2.imag < 0:
        a2 = -a2
    aB = -1j * cmath.sqrt(q * q - eB)
    if aB.imag > 0:
        aB = -aB
    return a1, a2, aB


def Fdisp(q, psi, es, et, eB):
    """Corrected eq. (2.27) dispersion residual (complex-capable)."""
    a1, a2, aB = alphas(q, psi, es, et, eB)
    L = -es * (es * aB - eB * a1) * (aB - a2) * cmath.tan(psi) ** 2
    R = a1 * (aB - a1) * (es * aB * a2 - eB * a1 * a1)
    return L - R


def q_locus(psi, es):
    """EP locus q/k0 = sqrt(eps_s)/cos(psi); complex for complex eps_s."""
    return cmath.sqrt(es) / cmath.cos(psi)


# ------------------------------------------------------------------ DV closed forms
def tan2_psi_DV(es, et, eB):
    return (eB - es) * (es + et) ** 2 / (4 * es * (es + eB) * (et - eB))


def psi_DV(es, et, eB):
    """Complex DV angle from the closed form (radians)."""
    return cmath.atan(cmath.sqrt(tan2_psi_DV(es, et, eB)))


def eB_partner_locus(es, et, psi):
    """Both complex roots eps_B such that a DV wave exists at REAL angle psi."""
    t = np.tan(psi) ** 2
    C = (es + et) ** 2 / (4 * es)
    a = -t
    b = t * (et - es) - C
    c = t * es * et + C * es
    disc = cmath.sqrt(b * b - 4 * a * c)
    return (-b + disc) / (2 * a), (-b - disc) / (2 * a)


def dv_point(es, et, eB_root, psi):
    """Assemble the DV solution at real psi: q, decay constants, diagnostics."""
    q = q_locus(psi, es)
    a1, a2, aB = alphas(q, psi, es, et, eB_root)
    return dict(psi_deg=np.degrees(psi), q=q, a1=a1, a2=a2, aB=aB,
                ok=(a1.imag > 0 and a2.imag > 0 and aB.imag < 0 and q.imag >= -1e-12),
                res_check=abs(Fdisp(q, psi, es, et, eB_root)))


# ------------------------------------------------- complex-q Dyakonov branch
def _newton_q(q0, psi, es, et, eB, deflate=True, tol=1e-12, itmax=60):
    q = complex(q0)
    for _ in range(itmax):
        if deflate:
            f = lambda x: Fdisp(x, psi, es, et, eB) / (x - q_locus(psi, es))
        else:
            f = lambda x: Fdisp(x, psi, es, et, eB)
        F = f(q)
        h = 1e-7 * max(1.0, abs(q))
        dF = (f(q + h) - f(q - h)) / (2 * h)
        if dF == 0:
            return None
        step = F / dF
        q -= step
        if abs(step) < tol * max(1.0, abs(q)):
            return q
    return None


def dyakonov_branch(psi_grid, es, et, eB, q_seed, psi_seed):
    """Continuation of the complex-q Dyakonov root over a real psi grid,
    seeded at (psi_seed, q_seed) and marched outward in both directions.
    Returns complex q array (nan where lost/unphysical)."""
    psi_grid = np.asarray(psi_grid, float)
    out = np.full(psi_grid.shape, np.nan + 0j, complex)
    i0 = int(np.argmin(np.abs(psi_grid - psi_seed)))

    def admissible(q, psi):
        a1, a2, aB = alphas(q, psi, es, et, eB)
        return a1.imag > 1e-9 and a2.imag > 1e-9 and aB.imag < -1e-9 and q.imag > -1e-9

    for rng in (range(i0, len(psi_grid)), range(i0 - 1, -1, -1)):
        q = complex(q_seed)
        first = True
        for i in rng:
            # near the seed the genuine root sits on the EP locus: nudge off it
            guess = q if not first else q + 1e-4 * (1 + 1j)
            sol = _newton_q(guess, psi_grid[i], es, et, eB)
            if sol is None or not admissible(sol, psi_grid[i]):
                break
            out[i] = sol
            q = sol
            first = False
    return out
