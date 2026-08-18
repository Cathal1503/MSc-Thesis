r"""
homog_axisym.py
===============
Bruggeman homogenization with a *general* axisymmetric particle shape and
fully complex constitutive parameters.

Conventions follow Iga-Buitron, Mackay & Lakhtakia, JPCM 37 (2025) 045703,
as validated in earlier sessions:

    eps_HCM = diag(eps_t, eps_t, eps_z),  gamma = eps_z/eps_t  (optic axis z)
    trace(L) = 1/gamma,   L = diag(L_t, L_t, L_z),  L_z = 1/gamma - 2 L_t
    i*omega*D = gamma * L * eps^{-1}   (normalized units, eps0 = 1)

Key identity used for the shape factor (verified against the closed-form
sphere result below): scaling z -> z/sqrt(gamma) maps the uniaxial
comparison medium to an isotropic one, and

    L(gamma) = N_iso(scaled shape) / gamma ,

where N_iso is the ordinary isotropic depolarization dyadic (trace 1) of the
z-scaled particle evaluated at its centre.  For an axisymmetric shape the
whole dyadic reduces to one scalar quadrature over the profile:

    N_z = (1/2) \oint  rho * z~ * (rho^2 + z~^2)^{-3/2}  d rho ,
    z~ = z / sqrt(gamma)  (complex-capable analytic continuation),

with the profile oriented top-centre -> top edge -> side -> bottom edge ->
bottom-centre (then n_z dl = d rho).  Complex gamma is handled by principal
branches; valid for passive media (|arg gamma| < pi).

Shapes provided: sphere, spheroid, doubly truncated spheroid, superspheroid
    (rho/alpha)^p + |z|^p = 1   (p = 2 recovers the spheroid).
Any user shape can be added by supplying profile segments
u in [0,1] -> (rho, z, rho*drho/du).
"""
from __future__ import annotations
import cmath
import numpy as np

_ISO_GUARD = 1e-7


# ---------------------------------------------------------------- quadrature
def _gauss01(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


# ------------------------------------------------- closed form for validation
def Lt_sphere_closed(gamma):
    """Closed-form L_t of a full sphere in a uniaxial comparison medium."""
    g = complex(gamma)
    if abs(g - 1.0) < _ISO_GUARD:
        g += _ISO_GUARD
    r = cmath.sqrt(g - 1.0)
    return (cmath.atan(r) - r / g) / (2.0 * (g - 1.0) ** 1.5)


def Nz_spheroid_closed(alpha):
    """Isotropic N_z of a spheroid with semi-axes (alpha, alpha, 1), real alpha."""
    a = float(alpha)
    if abs(a - 1.0) < 1e-9:
        return 1.0 / 3.0
    if a < 1.0:                       # prolate
        e = np.sqrt(1.0 - a * a)
        return (1 - e**2) / e**2 * (np.log((1 + e) / (1 - e)) / (2 * e) - 1.0)
    e = np.sqrt(1.0 - 1.0 / a**2)     # oblate  (semi-axes a,a,1 with a>1)
    return (1.0 / e**2) * (1.0 - np.sqrt(1 - e**2) / e * np.arcsin(e))


# ------------------------------------------------------------ shape profiles
def segments_spheroid(alpha=1.0):
    """Full spheroid, semi-axes (alpha, alpha, 1). Param: z = 1 - 2u."""
    def seg(u):
        z = 1.0 - 2.0 * u
        rho = alpha * np.sqrt(np.clip(1.0 - z * z, 0.0, None))
        rho_drho = 2.0 * alpha**2 * z          # rho * drho/du  (smooth)
        return rho, z, rho_drho
    return [seg]


def segments_dt_spheroid(alpha, kappa):
    """Doubly truncated spheroid: semi-axes (alpha, alpha, 1), cut at z = +/-kappa."""
    rk = alpha * np.sqrt(1.0 - kappa**2)      # cap radius

    def top(u):     # centre -> edge, z = +kappa
        rho = u * rk
        return rho, np.full_like(u, kappa), rk**2 * u
    def side(u):    # z: +kappa -> -kappa along the spheroid
        z = kappa * (1.0 - 2.0 * u)
        rho = alpha * np.sqrt(1.0 - z * z)
        return rho, z, 2.0 * kappa * alpha**2 * z
    def bot(u):     # edge -> centre, z = -kappa
        rho = (1.0 - u) * rk
        return rho, np.full_like(u, -kappa), -rk**2 * (1.0 - u)
    return [top, side, bot]


def segments_superspheroid(alpha, p):
    """Superspheroid (rho/alpha)^p + |z|^p = 1, exponent p > 1.
    p = 2 -> spheroid; p > 2 -> boxy/cylindrical; 1 < p < 2 -> pointy."""
    def upper(u):   # pole (rho=0, z=1) -> equator (rho=alpha, z=0)
        rho = u * alpha
        z = (np.clip(1.0 - u**p, 0.0, None)) ** (1.0 / p)
        return rho, z, alpha**2 * u
    def lower(u):   # equator -> pole (z=-1)
        rho = (1.0 - u) * alpha
        z = -((np.clip(1.0 - (1.0 - u)**p, 0.0, None)) ** (1.0 / p))
        return rho, z, -alpha**2 * (1.0 - u)
    return [upper, lower]


SHAPES = {
    "sphere":        lambda **kw: segments_spheroid(1.0),
    "spheroid":      lambda **kw: segments_spheroid(kw["alpha"]),
    "dt_spheroid":   lambda **kw: segments_dt_spheroid(kw["alpha"], kw["kappa"]),
    "superspheroid": lambda **kw: segments_superspheroid(kw["alpha"], kw["p"]),
}


# --------------------------------------------------------- depolarization L_t
def Nz_iso(segments, gamma=1.0, n=400):
    """Isotropic-space N_z of the z-scaled shape (complex gamma allowed)."""
    sg = cmath.sqrt(complex(gamma))
    u, w = _gauss01(n)
    total = 0.0 + 0.0j
    for seg in segments:
        rho, z, rho_drho = seg(u)
        zt = z / sg
        total += 0.5 * np.sum(w * rho_drho * zt * (rho**2 + zt**2) ** (-1.5))
    return total


def Lt_shape(gamma, shape="sphere", n=400, **kw):
    """Transverse depolarization factor L_t(gamma) for a named shape."""
    g = complex(gamma)
    if abs(g - 1.0) < _ISO_GUARD:
        g += _ISO_GUARD
    Nz = Nz_iso(SHAPES[shape](**kw), g, n=n)
    return (1.0 - Nz) / (2.0 * g)


# -------------------------------------------------------- Bruggeman solver
def _residual(et, ez, eA, eB, fA, segsA, segsB, n):
    """Bruggeman residual (fA a^A_c + fB a^B_c) for c = t, z."""
    fB = 1.0 - fA
    g = ez / et
    if abs(g - 1.0) < _ISO_GUARD:
        g *= (1.0 + _ISO_GUARD)
    NzA = Nz_iso(segsA, g, n=n); LtA = (1 - NzA) / (2 * g); LzA = 1 / g - 2 * LtA
    NzB = Nz_iso(segsB, g, n=n); LtB = (1 - NzB) / (2 * g); LzB = 1 / g - 2 * LtB

    def a(ej, Lc, ec):
        return (ej - ec) / (1 + (g * Lc / ec) * (ej - ec))

    Rt = fA * a(eA, LtA, et) + fB * a(eB, LtB, et)
    Rz = fA * a(eA, LzA, ez) + fB * a(eB, LzB, ez)
    return Rt, Rz


def _newton2(et, ez, eA, eB, fA, segsA, segsB, n, tol=1e-11, itmax=80):
    """Damped Newton on the 2-component complex Bruggeman residual."""
    for _ in range(itmax):
        Rt, Rz = _residual(et, ez, eA, eB, fA, segsA, segsB, n)
        if max(abs(Rt), abs(Rz)) < tol * max(1.0, abs(et), abs(ez)):
            return et, ez, True
        h = 1e-6 * max(1.0, abs(et), abs(ez))
        Rt_t, Rz_t = _residual(et + h, ez, eA, eB, fA, segsA, segsB, n)
        Rt_z, Rz_z = _residual(et, ez + h, eA, eB, fA, segsA, segsB, n)
        J = np.array([[(Rt_t - Rt) / h, (Rt_z - Rt) / h],
                      [(Rz_t - Rz) / h, (Rz_z - Rz) / h]], complex)
        try:
            d = np.linalg.solve(J, np.array([Rt, Rz], complex))
        except np.linalg.LinAlgError:
            return et, ez, False
        # damp large steps
        s = 1.0
        nrm = max(abs(d[0]) / max(abs(et), 1e-9), abs(d[1]) / max(abs(ez), 1e-9))
        if nrm > 0.5:
            s = 0.5 / nrm
        et, ez = et - s * d[0], ez - s * d[1]
    return et, ez, False


def bruggeman(eps_incl, eps_host, fA, shape="superspheroid", *,
              shapeB="sphere", kwA=None, kwB=None,
              n=320, nsteps=None, return_ok=False):
    """Bruggeman HCM (eps_t, eps_z) for shaped A-particles (eps_incl, fraction fA)
    dispersed with B-particles (eps_host).  Fully complex-capable.

    Solves fA a^A_c + fB a^B_c = 0 (c = t, z) by Newton with continuation in
    fA from the dilute limit, which tracks the physical branch (the one
    continuously connected to eps_host, with Im >= 0 for passive
    constituents) through plasmonic resonance bands.
    """
    kwA = dict(kwA or {})
    kwB = dict(kwB or {})
    eA, eB = complex(eps_incl), complex(eps_host)
    segsA = SHAPES[shape](**kwA)
    segsB = SHAPES[shapeB](**kwB)

    if nsteps is None:
        nsteps = max(8, int(np.ceil(fA / 0.02)))
    et = ez = eB
    ok = True
    for f in np.linspace(fA / nsteps, fA, nsteps):
        et, ez, ok = _newton2(et, ez, eA, eB, f, segsA, segsB, n)
        if not ok:  # refine continuation locally
            et2, ez2 = eB, eB
            for f2 in np.linspace(f / (4 * nsteps), f, 4 * nsteps):
                et2, ez2, ok = _newton2(et2, ez2, eA, eB, f2, segsA, segsB, n)
            et, ez = et2, ez2
    return (et, ez, ok) if return_ok else (et, ez)


def bruggeman_curve(eps_incl, eps_host, fA_grid, shape="superspheroid", *,
                    shapeB="sphere", kwA=None, kwB=None, n=320):
    """Vector version: continuation along an increasing fA grid (fast, robust)."""
    kwA = dict(kwA or {})
    kwB = dict(kwB or {})
    eA, eB = complex(eps_incl), complex(eps_host)
    segsA = SHAPES[shape](**kwA)
    segsB = SHAPES[shapeB](**kwB)
    fA_grid = np.asarray(fA_grid, float)
    ets = np.empty(fA_grid.shape, complex)
    ezs = np.empty(fA_grid.shape, complex)
    oks = np.empty(fA_grid.shape, bool)
    et = ez = eB
    # pre-walk to the first grid point
    for f in np.linspace(fA_grid[0] / 6, fA_grid[0], 6):
        et, ez, _ = _newton2(et, ez, eA, eB, f, segsA, segsB, n)
    for i, f in enumerate(fA_grid):
        et, ez, ok = _newton2(et, ez, eA, eB, f, segsA, segsB, n)
        ets[i], ezs[i], oks[i] = et, ez, ok
    return ets, ezs, oks
