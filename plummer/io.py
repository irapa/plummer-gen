"""Gravação das condições iniciais em formatos lidos por códigos de N-corpos.

As funções nbody6 e gadget supõem que o modelo foi gerado em unidades
astrofísicas (Msun, pc, km/s, com G = units.G_ASTRO).
"""
import struct

import numpy as np

COLUMNS = "m x y z vx vy vz"


def _table(model):
    return np.column_stack((model.mass, model.pos, model.vel))


def write_ascii(fname, model):
    """Tabela de texto com cabeçalho: m x y z vx vy vz."""
    header = "{}\n{}".format(repr(model), COLUMNS)
    np.savetxt(fname, _table(model), fmt="%.10e", header=header)


def write_nbody6(fname, model):
    """Arquivo dat.10 do NBODY6++GPU: m x y z vx vy vz, sem cabeçalho.

    Com KZ(22) = 6 o código lê massas em Msun, posições em pc e velocidades
    em km/s, e converte para unidades de N-corpos (veja examples/).
    """
    np.savetxt(fname, _table(model), fmt="%.10e", delimiter="\t")


# ------------------------------------------------------------------ GADGET

KPC = 1000.0     # pc
MASS_UNIT = 1e10  # Msun


def _block(f, label, payload):
    """Bloco do formato 2 do GADGET: rótulo de 4 caracteres + bloco Fortran."""
    f.write(struct.pack("<i4sii", 8, label.encode(), len(payload) + 8, 8))
    f.write(struct.pack("<i", len(payload)))
    f.write(payload)
    f.write(struct.pack("<i", len(payload)))


def write_gadget(fname, model, ptype=1):
    """Snapshot no formato binário 2 do GADGET (ICFormat = 2).

    Unidades: kpc, 10^10 Msun e km/s (os valores padrão do GADGET). Todas as
    partículas têm a mesma massa, gravada na MassTable do cabeçalho.
    """
    n = model.n
    if not np.allclose(model.mass, model.mass[0]):
        raise ValueError("write_gadget só trata partículas de mesma massa")
    npart = [0] * 6
    npart[ptype] = n
    massarr = [0.0] * 6
    massarr[ptype] = model.mass[0] / MASS_UNIT
    head = struct.pack("<6I", *npart)
    head += struct.pack("<6d", *massarr)
    head += struct.pack("<dd", 0.0, 0.0)            # time, redshift
    head += struct.pack("<ii", 0, 0)                # flag_sfr, flag_feedback
    head += struct.pack("<6I", *npart)              # npartTotal
    head += struct.pack("<ii", 0, 1)                # flag_cooling, num_files
    head += struct.pack("<4d", 0.0, 0.0, 0.0, 1.0)  # BoxSize, Omega0, OmegaLambda, h
    head += struct.pack("<ii", 0, 0)                # flag_stellarage, flag_metals
    head += struct.pack("<6I", *([0] * 6))          # npartTotalHighWord
    head += struct.pack("<i", 0)                    # flag_entropy
    head += b"\0" * (256 - len(head))
    with open(fname, "wb") as f:
        _block(f, "HEAD", head)
        _block(f, "POS ", (model.pos / KPC).astype("<f4").tobytes())
        _block(f, "VEL ", model.vel.astype("<f4").tobytes())
        _block(f, "ID  ", np.arange(1, n + 1, dtype="<u4").tobytes())


def read_gadget(fname):
    """Lê um arquivo escrito por write_gadget. Devolve (pos_pc, vel_kms, m_msun)."""
    with open(fname, "rb") as f:
        data = f.read()
    blocks = {}
    i = 0
    while i < len(data):
        label = data[i + 4:i + 8].decode()
        i += 16
        size = struct.unpack("<i", data[i:i + 4])[0]
        blocks[label] = data[i + 4:i + 4 + size]
        i += size + 8
    head = blocks["HEAD"]
    npart = struct.unpack("<6I", head[:24])
    massarr = struct.unpack("<6d", head[24:72])
    ptype = int(np.argmax(npart))
    n = npart[ptype]
    pos = np.frombuffer(blocks["POS "], "<f4").reshape(n, 3).astype(float) * KPC
    vel = np.frombuffer(blocks["VEL "], "<f4").reshape(n, 3).astype(float)
    return pos, vel, np.full(n, massarr[ptype] * MASS_UNIT)
