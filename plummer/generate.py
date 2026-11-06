"""Geração de esferas de Plummer isotrópicas.

Algoritmo de Aarseth, Hénon & Wielen (1974, A&A 37, 183):

* raios pela inversão da massa acumulada, M(r)/M = X1, que dá
  r = a (X1^(-2/3) - 1)^(-1/2);
* direções isotrópicas a partir de dois números aleatórios;
* módulo da velocidade V = q Ve(r), com Ve = sqrt(-2 Phi(r)) e q sorteado
  por rejeição de von Neumann na distribuição g(q) = q^2 (1 - q^2)^(7/2),
  cujo máximo (0,092) é menor que 0,1.

Todas as grandezas estão no sistema de unidades em que G, M e a são dados
pelo usuário (o padrão é G = M = a = 1).
"""
import warnings

import numpy as np

from .checks import kinetic_energy, potential_energy

G_BOUND = 0.1   # limitante de g(q) usado na rejeição


class PlummerModel(object):
    """Realização de uma esfera de Plummer: posições, velocidades e massas.

    Atributos
    ---------
    pos, vel : arrays (N, 3)
    mass : array (N,)
    a, total_mass, G : parâmetros do modelo
    rmax : raio de truncamento (None se não truncado)
    vscale : fator aplicado às velocidades para levar Q a 0,5 (1 se não houve)
    seed : semente do gerador de números aleatórios
    """

    def __init__(self, pos, vel, mass, a, total_mass, G, rmax, vscale, seed):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.a = a
        self.total_mass = total_mass
        self.G = G
        self.rmax = rmax
        self.vscale = vscale
        self.seed = seed

    @property
    def n(self):
        return len(self.mass)

    def __repr__(self):
        return ("PlummerModel(N={}, M={:g}, a={:g}, G={:g}, rmax={}, vscale={:.5f}, seed={})"
                .format(self.n, self.total_mass, self.a, self.G, self.rmax, self.vscale, self.seed))


def x1_max(rmax, a):
    """Maior valor de X1 = M(r)/M permitido para que r <= rmax."""
    if rmax is None:
        return 1.0
    s2 = (rmax / a) ** 2
    return (s2 / (1.0 + s2)) ** 1.5


def isotropic(length, rng):
    """Vetores de módulo `length` com direções uniformes na esfera."""
    z = (1.0 - 2.0 * rng.random(len(length))) * length
    phi = 2.0 * np.pi * rng.random(len(length))
    rho = np.sqrt(length ** 2 - z ** 2)
    return np.column_stack((rho * np.cos(phi), rho * np.sin(phi), z))


def sample_radii(n, a, rmax, rng):
    """Raios distribuídos segundo M(r) de Plummer, truncados em rmax."""
    x1 = rng.random(n) * x1_max(rmax, a)
    with np.errstate(divide="ignore"):
        return a / np.sqrt(x1 ** (-2.0 / 3.0) - 1.0)


def sample_q(n, rng):
    """q = V/Ve sorteado por rejeição em g(q) = q^2 (1 - q^2)^(7/2).

    A taxa de aceitação é de 43 %, por isso sorteamos em blocos de ~2,5 vezes
    o número de valores que ainda faltam.
    """
    q = np.empty(n)
    filled = 0
    while filled < n:
        m = int(2.5 * (n - filled)) + 16
        x4 = rng.random(m)
        x5 = rng.random(m)
        accepted = x4[G_BOUND * x5 < x4 ** 2 * (1.0 - x4 ** 2) ** 3.5][:n - filled]
        q[filled:filled + len(accepted)] = accepted
        filled += len(accepted)
    return q


def escape_velocity(r, G, M, a):
    """Ve(r) = sqrt(-2 Phi(r)), com Phi(r) = -G M / sqrt(r^2 + a^2)."""
    return np.sqrt(2.0 * G * M / np.sqrt(r ** 2 + a ** 2))


def generate(n, mass=1.0, a=1.0, G=1.0, rmax=None, virialize=None, seed=None):
    """Gera uma esfera de Plummer isotrópica com N partículas de mesma massa.

    Parâmetros
    ----------
    n : número de partículas
    mass : massa total M
    a : raio de escala de Plummer
    G : constante gravitacional no sistema de unidades escolhido
    rmax : raio de truncamento (None = sem truncamento)
    virialize : se True, multiplica as velocidades por um fator constante
        para que Q = K/|W| = 0,5 exatamente. O padrão (None) é True quando
        há truncamento e False quando não há. Veja a nota abaixo.
    seed : semente do gerador de números aleatórios (inteiro), para que o
        modelo possa ser reproduzido

    Nota sobre o truncamento
    ------------------------
    As velocidades são sorteadas da função distribuição do modelo de Plummer
    completo, cujo potencial inclui a massa que está além de rmax. Ao cortar
    as posições em rmax e redistribuir essa massa entre as N partículas, a
    energia potencial do sistema muda e o modelo deixa de estar em
    equilíbrio virial: Q = 0,486 para rmax = 6,6a e Q = 0,41 para rmax = 2a
    (veja checks.expected_virial_ratio). Por isso, quando rmax é dado, as
    velocidades são reescaladas por padrão, como fazem os códigos de N-corpos
    (no NBODY6++GPU, parâmetro Q do bloco INSCALE).

    O sistema é posto no referencial do centro de massa, em posição e em
    velocidade.
    """
    if rmax is not None and rmax <= 0:
        raise ValueError("rmax deve ser positivo")
    if virialize is None:
        virialize = rmax is not None
    elif rmax is not None and not virialize:
        warnings.warn("modelo truncado sem reescala das velocidades: Q < 0,5 "
                      "(veja a documentação de generate)")

    rng = np.random.default_rng(seed)
    r = sample_radii(n, a, rmax, rng)
    pos = isotropic(r, rng)
    vel = isotropic(sample_q(n, rng) * escape_velocity(r, G, mass, a), rng)
    m = np.full(n, mass / n)

    pos -= np.average(pos, axis=0, weights=m)
    vel -= np.average(vel, axis=0, weights=m)

    vscale = 1.0
    if virialize:
        q = kinetic_energy(vel, m) / abs(potential_energy(pos, m, G))
        vscale = np.sqrt(0.5 / q)
        vel *= vscale

    return PlummerModel(pos, vel, m, a, mass, G, rmax, vscale, seed)
