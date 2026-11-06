"""Figuras dos testes de qualidade do gerador (Seção 3.1 do artigo).

Uso: python scripts/validate.py [N] [semente]

Gera validacao_perfis.pdf (M(r), rho(r), g(q), sigma_r e beta) e
validacao_truncamento.pdf (Q em função de rmax/a, teoria e realizações).
Unidades: G = M = a = 1.
"""
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plummer  # noqa: E402
from plummer import checks  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
model = plummer.generate(n, seed=seed)
print(checks.report(model))

fig, ax = plt.subplots(2, 2, figsize=(9, 7.5))

# (a) massa acumulada
r = np.sort(checks.radii(model))
rr = np.logspace(-2, 2, 400)
ax[0, 0].plot(r, np.arange(1, n + 1) / n, "C0", lw=2, label="partículas")
ax[0, 0].plot(rr, checks.mass_cdf(rr, 1.0), "k--", label="teoria")
ax[0, 0].set(xscale="log", xlim=(0.03, 50), xlabel="$r/a$", ylabel="$M(r)/M$")
ax[0, 0].set_title("KS: D = {:.4f}, p = {:.2f}".format(*checks.ks_radii(model)), fontsize=10)
ax[0, 0].legend()

# (b) densidade
edges = np.logspace(-1.5, 1.5, 25)
rmid, rho, err = checks.density_profile(model, edges)
ax[0, 1].errorbar(rmid, rho, err, fmt="o", ms=4, label="partículas")
ax[0, 1].plot(rr, checks.density(rr, 1.0, 1.0), "k--", label="teoria")
ax[0, 1].set(xscale="log", yscale="log", xlim=(0.03, 50), xlabel="$r/a$",
             ylabel=r"$\rho\,a^3/M$")
ax[0, 1].legend()

# (c) distribuição de q = V/Ve
q = checks.q_values(model)
qq = np.linspace(0, 1, 300)
g = qq ** 2 * (1 - qq ** 2) ** 3.5
ax[1, 0].hist(q, bins=40, range=(0, 1), density=True, alpha=0.6, label="partículas")
ax[1, 0].plot(qq, g / np.trapz(g, qq), "k--", label="$g(q)$ normalizada")
ax[1, 0].set(xlabel="$q = V/V_e$", ylabel="densidade de probabilidade")
ax[1, 0].set_title("KS: D = {:.4f}, p = {:.2f}".format(*checks.ks_q(model)), fontsize=10)
ax[1, 0].legend()

# (d) dispersão de velocidades e anisotropia
edges = np.logspace(-0.7, 1.0, 13)   # r < 0,2a tem poucas partículas
rmid, sr, st, beta = checks.velocity_profile(model, edges)
ax[1, 1].plot(rmid, sr, "o", ms=4, label=r"$\sigma_r$")
ax[1, 1].plot(rmid, st, "s", ms=4, label=r"$\sigma_t$")
ax[1, 1].plot(rr, checks.sigma_1d(rr, 1.0, 1.0, 1.0), "k--", label="teoria")
ax[1, 1].plot(rmid, beta, "^", ms=4, color="C3", label=r"$\beta$")
ax[1, 1].axhline(0, color="gray", lw=0.5)
ax[1, 1].set(xscale="log", xlim=(0.15, 15), xlabel="$r/a$",
             ylabel=r"$\sigma\,(GM/a)^{-1/2}$,  $\beta$")
ax[1, 1].legend(ncol=2)

fig.tight_layout()
fig.savefig("validacao_perfis.pdf")

# Q em função do raio de truncamento, sem reescala das velocidades
warnings.simplefilter("ignore", UserWarning)
xm = np.logspace(np.log10(1.5), 2, 60)
fig, ax = plt.subplots(figsize=(5, 3.8))
ax.plot(xm, [checks.expected_virial_ratio(x) for x in xm], "k-", label="teoria")
for x in (2.0, 3.0, 6.6, 15.0, 50.0):
    qs = [checks.virial_ratio(plummer.generate(20000, rmax=x, virialize=False, seed=s))
          for s in range(3)]
    ax.errorbar(x, np.mean(qs), np.std(qs), fmt="o", color="C0")
ax.axhline(0.5, color="gray", ls=":")
ax.axvline(6.6, color="gray", ls="--", lw=0.8)
ax.text(6.9, 0.39, r"$\omega$ Cen", color="gray")
ax.set(xscale="log", xlabel="$r_{max}/a$", ylabel="$Q = K/|W|$")
fig.tight_layout()
fig.savefig("validacao_truncamento.pdf")
print("gravado: validacao_perfis.pdf, validacao_truncamento.pdf")
