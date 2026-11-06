"""Unidades astrofísicas e unidades de N-corpos.

Unidades astrofísicas usadas pelo pacote: massa em massas solares,
comprimento em parsecs e velocidade em km/s. Nesse sistema
G = 4,3009 x 10^-3 pc (km/s)^2 / Msun e a unidade de tempo pc/(km/s)
vale 0,9778 Myr.

Unidades de N-corpos (Hénon; Heggie & Mathieu 1986): G = M = 1 e energia
total E = -1/4, o que equivale a raio virial R_V = 1 quando Q = 0,5.
"""
import numpy as np

from .checks import potential_energy

G_ASTRO = 4.300917e-3      # pc (km/s)^2 / Msun
MYR_PER_TIME_UNIT = 0.9778   # 1 pc/(km/s) em Myr


def nbody_scales(model):
    """Escalas de comprimento, velocidade e tempo das unidades de N-corpos.

    Devolve (R_V, V*, T*) nas unidades do modelo, com R_V = G M^2 / (2|W|),
    V* = sqrt(G M / R_V) e T* = R_V / V*.
    """
    m = model.total_mass
    rv = model.G * m ** 2 / (2.0 * abs(potential_energy(model.pos, model.mass, model.G)))
    vstar = np.sqrt(model.G * m / rv)
    return rv, vstar, rv / vstar
