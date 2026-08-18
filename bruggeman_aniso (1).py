"""
bruggeman_aniso.py
==================
Extension of `bruggeman.py` to *uniaxial* constituents whose optic axes are
aligned with the HCM symmetry (z) axis.

Because constituent permittivity dyadics, depolarization dyadics and the HCM
dyadic are then all diagonal in the same (x,y,z) frame with x/y degeneracy,
every dyadic equation of Iga-Buitron, Mackay & Lakhtakia (JPCM 37, 045703,
2025) splits into independent t- and z-components; the only change from the
isotropic-constituent code is that eps_incl / eps_host are replaced by pairs
(eps_t, eps_z).  Setting eps_t == eps_z recovers `bruggeman.eps_hcm` exactly
(verified in self-tests).

Constituent convention: pass (eps_ordinary, eps_extraordinary) i.e. the
material's (eps_s, eps_t) in the book's notation, optic axis along z.

Public API
----------
    eps_hcm_aniso(epsA, epsB, fA, shape=..., kappa=..., alpha=...) -> (eps_t, eps_z)
    img_hcm_aniso(epsA, epsB, fA, shape=..., kappa=..., alpha=..., N=10)
    gamma_wiener_bound(epsA, epsB)  -> (gamma_max, f_at_max)
"""
from __future__ import annotations
import numpy as np
from bruggeman import (Lt_sphere, Lt_doubly_truncated_sphere,
                       Lt_singly_truncated_sphere, Lt_doubly_truncated_spheroid,
                       Lt_hemispheroid)

_SHAPES = {
    "sphere":       lambda **p: (lambda g: Lt_sphere(g)),
    "dt_sphere":    lambda **p: (lambda g: Lt_doubly_truncated_sphere(g, p["kappa"])),
    "st_sphere":    lambda **p: (lambda g: Lt_singly_truncated_sphere(g, p["kappa"])),
    "dt_spheroid":  lambda **p: (lambda g: Lt_doubly_truncated_spheroid(g, p["kappa"], p["alpha"])),
    "hemispheroid": lambda **p: (lambda g: Lt_hemispheroid(g, p["alpha"])),
}


def _pair(e):
    """Accept scalar (isotropic) or (eps_t, eps_z) pair."""
    if np.isscalar(e) or isinstance(e, complex):
        return complex(e), complex(e)
    et, ez = e
    return complex(et), complex(ez)


def bruggeman_aniso(epsA, epsB, fA, LtA, LtB=Lt_sphere, tol=1e-9, maxit=4000):
    """Bruggeman eq. (15) with aligned-uniaxial constituents A and B.

    epsA, epsB : scalar or (eps_t, eps_z); optic axes || z.
    Returns (eps_t, eps_z, iters); iters=-1 flags non-convergence.
    """
    eAt, eAz = _pair(epsA)
    eBt, eBz = _pair(epsB)
    fB = 1.0 - fA
    et = fA * eAt + fB * eBt          # componentwise volume-average start
    ez = fA * eAz + fB * eBz
    for it in range(maxit):
        g = ez / et
        LAt = LtA(g); LAz = 1.0 / g - 2.0 * LAt
        LBt = LtB(g); LBz = 1.0 / g - 2.0 * LBt
        NAt, NAz = g * LAt / et, g * LAz / ez
        NBt, NBz = g * LBt / et, g * LBz / ez
        KAt = 1.0 / (1.0 + NAt * (eAt - et)); KAz = 1.0 / (1.0 + NAz * (eAz - ez))
        KBt = 1.0 / (1.0 + NBt * (eBt - et)); KBz = 1.0 / (1.0 + NBz * (eBz - ez))
        etn = (fA * eAt * KAt + fB * eBt * KBt) / (fA * KAt + fB * KBt)
        ezn = (fA * eAz * KAz + fB * eBz * KBz) / (fA * KAz + fB * KBz)
        if max(abs(etn - et), abs(ezn - ez)) < tol:
            return etn, ezn, it + 1
        et, ez = etn, ezn
    return et, ez, -1


def eps_hcm_aniso(epsA, epsB, fA, shape="dt_spheroid", *,
                  kappa=None, alpha=None,
                  shapeB=None, kappaB=None, alphaB=None,
                  tol=1e-9, return_iters=False):
    """Uniaxial HCM (eps_t, eps_z) from aligned-uniaxial constituents.

    shape / kappa / alpha     : geometry of the A particles.
    shapeB / kappaB / alphaB  : geometry of the B particles.  Default
        shapeB=None keeps the paper's convention (B = spheres).  Set e.g.
        shapeB="dt_spheroid", alphaB=<..> to give the second phase prolate
        (alphaB<1) or oblate (alphaB>1) spheroids co-aligned with z, i.e.
        a fully particulate two-shaped morphology.
    """
    if shape not in _SHAPES:
        raise ValueError(f"unknown shape {shape!r}")
    LtA = _SHAPES[shape](kappa=kappa, alpha=alpha)
    if shapeB is None:
        LtB = Lt_sphere
    else:
        if shapeB not in _SHAPES:
            raise ValueError(f"unknown shapeB {shapeB!r}")
        LtB = _SHAPES[shapeB](kappa=kappaB, alpha=alphaB)
    et, ez, n = bruggeman_aniso(epsA, epsB, fA, LtA, LtB=LtB, tol=tol)
    return (et, ez, n) if return_iters else (et, ez)


def img_hcm_aniso(epsA, epsB, fA, shape="dt_spheroid", *,
                  kappa=None, alpha=None, N=10):
    """Incremental Maxwell Garnett, eqs (20)-(21), aligned-uniaxial constituents.
    A = shaped inclusions added into host B in N stages."""
    eAt, eAz = _pair(epsA)
    eBt, eBz = _pair(epsB)
    LtA = _SHAPES[shape](kappa=kappa, alpha=alpha)
    beta = 1.0 - (1.0 - fA) ** (1.0 / N)
    et, ez = eBt, eBz                      # stage 0: the (possibly uniaxial) host
    for _ in range(N):
        g = ez / et
        LtAg = LtA(g); LzAg = 1.0 / g - 2.0 * LtAg
        Lts = Lt_sphere(g); Lzs = 1.0 / g - 2.0 * Lts
        DAt, DAz = g * LtAg / et, g * LzAg / ez
        Dst, Dsz = g * Lts / et, g * Lzs / ez
        at = (eAt - et) / (1.0 + DAt * (eAt - et))
        az = (eAz - ez) / (1.0 + DAz * (eAz - ez))
        et = et + beta * at / (1.0 - beta * Dst * at)
        ez = ez + beta * az / (1.0 - beta * Dsz * az)
    return et, ez


def gamma_wiener_bound(epsA, epsB, nf=20001):
    """Rigorous upper bound on gamma = eps_z/eps_t over *all* microstructures
    of an aligned two-phase composite (real, positive constituents):
       eps_z <= f eAz + (1-f) eBz        (Wiener arithmetic bound)
       eps_t >= [f/eAt + (1-f)/eBt]^-1   (Wiener harmonic bound)
    Returns (max_f gamma_bound(f), argmax f)."""
    eAt, eAz = [x.real for x in _pair(epsA)]
    eBt, eBz = [x.real for x in _pair(epsB)]
    f = np.linspace(0.0, 1.0, nf)
    gb = (f * eAz + (1 - f) * eBz) * (f / eAt + (1 - f) / eBt)
    i = int(np.argmax(gb))
    return float(gb[i]), float(f[i])


if __name__ == "__main__":
    from bruggeman import eps_hcm
    print("bruggeman_aniso.py self-test")
    # (1) isotropic reduction: pair (e,e) must reproduce the scalar solver
    et0, ez0 = eps_hcm(1.5 + 0.1j, 5 + 0.6j, 0.3, "dt_spheroid", kappa=0.9, alpha=0.3)
    et1, ez1 = eps_hcm_aniso((1.5 + 0.1j,) * 2, (5 + 0.6j,) * 2, 0.3,
                             "dt_spheroid", kappa=0.9, alpha=0.3)
    ok = abs(et0 - et1) < 1e-10 and abs(ez0 - ez1) < 1e-10
    print(f"  [{'ok' if ok else 'FAIL'}] isotropic reduction (Bruggeman)")
    # (2) fA -> 1 recovers constituent A; fA -> 0 recovers B
    et, ez = eps_hcm_aniso((3.88, 7.02), 1.0, 0.9999, "dt_spheroid", kappa=0.99, alpha=0.4)
    ok = abs(et - 3.88) < 1e-2 and abs(ez - 7.02) < 1e-2
    print(f"  [{'ok' if ok else 'FAIL'}] fA->1 limit -> calomel")
    # (3) Bruggeman lies inside the Wiener bound
    gmax, fst = gamma_wiener_bound((3.88, 7.02), 1.0)
    et, ez = eps_hcm_aniso((3.88, 7.02), 1.0, fst, "dt_spheroid", kappa=0.999, alpha=0.05)
    print(f"  [{'ok' if (ez/et).real <= gmax + 1e-9 else 'FAIL'}] "
          f"Br gamma={(ez/et).real:.3f} <= bound {gmax:.3f} (calomel/void)")
    # (4) IMG isotropic reduction against notebook-style scalar IMG
    et2, ez2 = img_hcm_aniso((1.5 + 0.1j,) * 2, (5 + 0.6j,) * 2, 0.3,
                             "dt_spheroid", kappa=0.9, alpha=0.3, N=10)
    print(f"  IMG aniso (iso-reduced): eps_t={et2:.4f}, eps_z={ez2:.4f}")
