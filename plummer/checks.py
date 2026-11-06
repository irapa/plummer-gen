"""Testes de qualidade de uma realização do modelo de Plummer.

Cada função compara uma propriedade das partículas com a previsão analítica
do modelo de Plummer:

* massa acumulada M(r) (teste de Kolmogorov-Smirnov dos raios);
* distribuição de q = V/Ve, g(q) (teste de Kolmogorov-Smirnov);
* perfil de densidade rho(r);
* dispersão de velocidades sigma_r(r) e anisotropia beta(r);
* razão virial Q = K/|W| e raio virial;
* centro de massa.
"""
import numpy as np
from scipy import integrate, special, stats


# ---------------------------------------------------------------- energias

def kinetic_energy(vel, mass):
    return 0.5 * np.sum(mass * np.sum(vel ** 2, axis=1))


def potential_energy(pos, mass, G=1.0, chunk=256):
    """Energia potencial por soma direta sobre os pares, O(N^2).

    Os pares são somados em blocos de `chunk` linhas para limitar a memória.
    Para N = 4x10^4 leva alguns segundos.
    """
    n = len(mass)
    w = 0.0
    for i0 in range(0, n - 1, chunk):
        i1 = min(i0 + chunk, n)
        # linhas i0..i1-1 contra as colunas j = i0+1..n-1; só os pares j > i contam
        d2 = sum((pos[i0:i1, k, None] - pos[None, i0 + 1:, k]) ** 2 for k in range(3))
        upper = np.arange(i0 + 1, n)[None, :] > np.arange(i0, i1)[:, None]
        inv_r = np.where(upper, 1.0 / np.sqrt(np.where(upper, d2, 1.0)), 0.0)
        w -= mass[i0:i1] @ inv_r @ mass[i0 + 1:]
    return G * w


def virial_ratio(model):
    """Q = K/|W| (0,5 no equilíbrio virial)."""
    return (kinetic_energy(model.vel, model.mass)
            / abs(potential_energy(model.pos, model.mass, model.G)))


def virial_radius(model):
    """R_V = G M^2 / (2|W|); vale 16a/(3 pi) = 1,70a para o modelo completo."""
    w = potential_energy(model.pos, model.mass, model.G)
    return model.G * np.sum(model.mass) ** 2 / (2.0 * abs(w))


def expected_virial_ratio(rmax_over_a):
    """Q esperado para o algoritmo truncado em rmax, sem reescala.

    As posições seguem rho de Plummer até rmax, com a massa total M
    redistribuída nesse volume; as velocidades vêm do modelo completo, com
    <V^2>(r) = <q^2> Ve^2 = (1/4) 2GM/sqrt(r^2 + a^2). Em unidades G = M = a = 1:

        K = (1/2) int <V^2>(x) dm(x),   W = -int m(x)/x dm(x).
    """
    if rmax_over_a is None or np.isinf(rmax_over_a):
        return 0.5
    xm = rmax_over_a
    c = xm ** 3 / (1.0 + xm ** 2) ** 1.5           # fração de massa até rmax

    def m(x):
        return x ** 3 / (1.0 + x ** 2) ** 1.5 / c

    def dm(x):
        return 3.0 * x ** 2 / (1.0 + x ** 2) ** 2.5 / c

    k = integrate.quad(lambda x: 0.25 / np.sqrt(1.0 + x ** 2) * dm(x), 0.0, xm)[0]
    w = integrate.quad(lambda x: m(x) / x * dm(x), 0.0, xm)[0]
    return k / w


# ---------------------------------------------------- distribuições teóricas

def mass_cdf(r, a, rmax=None):
    """Fração da massa dentro de r, para o modelo truncado em rmax."""
    s = (r / a) ** 2
    f = (s / (1.0 + s)) ** 1.5
    if rmax is not None:
        smax = (rmax / a) ** 2
        f = np.minimum(f / (smax / (1.0 + smax)) ** 1.5, 1.0)
    return f


def q_cdf(q):
    """Distribuição acumulada de q para g(q) = q^2 (1 - q^2)^(7/2).

    Com u = q^2, g(q) dq é proporcional a u^(1/2) (1 - u)^(7/2) du, ou seja,
    u = q^2 segue uma distribuição beta com parâmetros (3/2, 9/2).
    """
    return special.betainc(1.5, 4.5, np.clip(q, 0.0, 1.0) ** 2)


def density(r, M, a):
    """rho(r) = 3M / (4 pi a^3) (1 + r^2/a^2)^(-5/2)."""
    return 3.0 * M / (4.0 * np.pi * a ** 3) * (1.0 + (r / a) ** 2) ** -2.5


def sigma_1d(r, G, M, a):
    """Dispersão de velocidades unidimensional do modelo isotrópico:
    sigma^2 = G M / (6 sqrt(r^2 + a^2))."""
    return np.sqrt(G * M / (6.0 * np.sqrt(r ** 2 + a ** 2)))


# ---------------------------------------------------------------- medidas

def radii(model):
    return np.sqrt(np.sum(model.pos ** 2, axis=1))


def q_values(model):
    """q = V/Ve(r) de cada partícula, descontada a reescala virial."""
    from .generate import escape_velocity
    v = np.sqrt(np.sum(model.vel ** 2, axis=1)) / model.vscale
    return v / escape_velocity(radii(model), model.G, model.total_mass, model.a)


def ks_radii(model):
    """Teste KS dos raios contra M(r). Devolve (D, p)."""
    return stats.kstest(radii(model), lambda r: mass_cdf(r, model.a, model.rmax))


def ks_q(model):
    """Teste KS de q = V/Ve contra g(q). Devolve (D, p)."""
    return stats.kstest(q_values(model), q_cdf)


def density_profile(model, edges):
    """Densidade em cascas esféricas. Devolve (r_médio, rho, erro poissoniano)."""
    r = radii(model)
    counts, _ = np.histogram(r, bins=edges)
    mass, _ = np.histogram(r, bins=edges, weights=model.mass)
    vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rmid = np.sqrt(edges[1:] * edges[:-1])
    rho = mass / vol
    return rmid, rho, rho / np.sqrt(np.maximum(counts, 1))


def velocity_profile(model, edges):
    """sigma_r, sigma_t (por componente) e beta = 1 - sigma_t^2/sigma_r^2
    em cascas esféricas. Devolve (r_médio, sigma_r, sigma_t, beta)."""
    r = radii(model)
    rhat = model.pos / r[:, None]
    vr = np.sum(model.vel * rhat, axis=1)
    vt2 = np.sum(model.vel ** 2, axis=1) - vr ** 2
    idx = np.digitize(r, edges) - 1
    nb = len(edges) - 1
    sr = np.full(nb, np.nan)
    st = np.full(nb, np.nan)
    for k in range(nb):
        sel = idx == k
        if np.count_nonzero(sel) > 1:
            sr[k] = np.sqrt(np.mean(vr[sel] ** 2))
            st[k] = np.sqrt(0.5 * np.mean(vt2[sel]))
    return np.sqrt(edges[1:] * edges[:-1]), sr, st, 1.0 - st ** 2 / sr ** 2


def center_of_mass(model):
    """Posição e velocidade do centro de massa (devem ser ~0)."""
    w = model.mass / np.sum(model.mass)
    return w @ model.pos, w @ model.vel


def report(model):
    """Resumo dos testes de qualidade, como texto."""
    d_r, p_r = ks_radii(model)
    d_q, p_q = ks_q(model)
    xcm, vcm = center_of_mass(model)
    w = potential_energy(model.pos, model.mass, model.G)
    q = kinetic_energy(model.vel, model.mass) / abs(w)
    rv = model.G * np.sum(model.mass) ** 2 / (2.0 * abs(w))
    lines = [
        repr(model),
        "KS raios vs M(r):   D = {:.4f}  p = {:.3f}".format(d_r, p_r),
        "KS q vs g(q):       D = {:.4f}  p = {:.3f}".format(d_q, p_q),
        "Q = K/|W|:          {:.4f}".format(q),
        "Q esperado s/ reescala: {:.4f}".format(
            expected_virial_ratio(None if model.rmax is None else model.rmax / model.a)),
        "R_V = {:.4f}  (16a/3pi = {:.4f})".format(rv, 16.0 * model.a / (3.0 * np.pi)),
        "|x_cm| = {:.2e}  |v_cm| = {:.2e}".format(np.linalg.norm(xcm), np.linalg.norm(vcm)),
    ]
    return "\n".join(lines)
