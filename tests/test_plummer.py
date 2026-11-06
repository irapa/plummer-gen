import numpy as np
import pytest

import plummer
from plummer import checks, io, units
from plummer.__main__ import main, read_params
from plummer.generate import G_BOUND

N = 20000


@pytest.fixture(scope="module")
def full():
    return plummer.generate(N, seed=1)


@pytest.fixture(scope="module")
def truncated():
    return plummer.generate(N, rmax=6.6, seed=2)


def test_seed_reproducible():
    m1 = plummer.generate(500, seed=42)
    m2 = plummer.generate(500, seed=42)
    assert np.array_equal(m1.pos, m2.pos) and np.array_equal(m1.vel, m2.vel)


def test_center_of_mass(full):
    xcm, vcm = checks.center_of_mass(full)
    assert np.allclose(xcm, 0.0, atol=1e-12) and np.allclose(vcm, 0.0, atol=1e-12)


@pytest.mark.parametrize("name", ["full", "truncated"])
def test_ks(name, request):
    model = request.getfixturevalue(name)
    assert checks.ks_radii(model)[1] > 0.01
    assert checks.ks_q(model)[1] > 0.01


def test_rmax_respected(truncated):
    # o recentramento desloca as partículas de ~a/sqrt(N)
    assert checks.radii(truncated).max() < 6.6 * 1.01


def test_q_bound():
    q = np.linspace(0.0, 1.0, 100001)
    assert np.max(q ** 2 * (1 - q ** 2) ** 3.5) < G_BOUND


def test_q_cdf_normalized():
    q = np.linspace(0.0, 1.0, 200001)
    g = q ** 2 * (1 - q ** 2) ** 3.5
    cdf = np.cumsum(g) / np.sum(g)
    assert np.max(np.abs(cdf - checks.q_cdf(q))) < 1e-4


def test_expected_virial_ratio():
    assert checks.expected_virial_ratio(None) == 0.5
    assert checks.expected_virial_ratio(1e4) == pytest.approx(0.5, abs=1e-3)
    assert checks.expected_virial_ratio(6.6) == pytest.approx(0.486, abs=1e-3)


def test_truncation_without_rescale_is_subvirial():
    with pytest.warns(UserWarning):
        m = plummer.generate(N, rmax=2.0, virialize=False, seed=3)
    assert checks.virial_ratio(m) == pytest.approx(checks.expected_virial_ratio(2.0), abs=0.01)


def test_virialize(truncated):
    assert checks.virial_ratio(truncated) == pytest.approx(0.5, rel=1e-10)
    assert truncated.vscale == pytest.approx(1.0 / np.sqrt(2 * 0.486), abs=0.01)


def test_untruncated_near_equilibrium(full):
    assert checks.virial_ratio(full) == pytest.approx(0.5, abs=0.01)
    assert checks.virial_radius(full) == pytest.approx(16 / (3 * np.pi), rel=0.02)


def test_dispersion_profile(full):
    edges = np.array([0.1, 0.3, 0.6, 1.0, 1.5, 2.5])
    rmid, sr, st, beta = checks.velocity_profile(full, edges)
    expected = checks.sigma_1d(rmid, 1.0, 1.0, 1.0)
    assert np.allclose(sr, expected, rtol=0.06)
    assert np.all(np.abs(beta) < 0.1)


def test_density_profile(full):
    edges = np.logspace(-1, 1, 9)
    rmid, rho, err = checks.density_profile(full, edges)
    # compara com a média analítica em cada casca: [M(r2) - M(r1)] / volume
    vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    expected = np.diff(checks.mass_cdf(edges, 1.0)) / vol
    assert np.all(np.abs(rho - expected) < 4 * err)


def test_potential_energy_brute_force():
    m = plummer.generate(300, seed=4)
    d = np.sqrt(((m.pos[:, None] - m.pos[None]) ** 2).sum(-1))
    i, j = np.triu_indices(300, 1)
    brute = -np.sum(m.mass[i] * m.mass[j] / d[i, j])
    assert checks.potential_energy(m.pos, m.mass, chunk=37) == pytest.approx(brute, rel=1e-12)


def test_gadget_roundtrip(tmp_path):
    m = plummer.generate(1000, mass=5e6, a=4.0, G=units.G_ASTRO, rmax=26.4, seed=5)
    fname = str(tmp_path / "ic.g2")
    io.write_gadget(fname, m)
    pos, vel, mass = io.read_gadget(fname)
    assert np.allclose(pos, m.pos, rtol=1e-6, atol=1e-5)
    assert np.allclose(vel, m.vel, rtol=1e-6, atol=1e-5)
    assert np.allclose(mass, m.mass)


def test_cli(tmp_path):
    out = tmp_path / "teste"
    par = tmp_path / "par.txt"
    par.write_text("n = 2000  # comentário\nmass = 5e6\na = 4.0\nrmax = 26.4\n"
                   "seed = 7\noutput = '{}'\nformats = ['ascii', 'nbody6', 'gadget']\n"
                   .format(out))
    assert main([str(par)]) == 0
    tab = np.loadtxt(str(out) + ".txt")
    assert tab.shape == (2000, 7)
    assert np.allclose(np.loadtxt(str(out) + ".dat.10"), tab)


def test_params_reject_code(tmp_path):
    par = tmp_path / "par.txt"
    par.write_text("n = __import__('os').getcwd()\nmass = 1\na = 1\n")
    with pytest.raises(ValueError):
        read_params(str(par))
