"""Linha de comando: python -m plummer arquivo_de_parametros.txt

O arquivo de parâmetros tem uma atribuição por linha (chave = valor), com
valores literais do Python; '#' inicia um comentário. Exemplo em
examples/omega_cen.txt. Chaves:

    n          número de partículas (obrigatória)
    mass       massa total (obrigatória)
    a          raio de escala de Plummer (obrigatória)
    rmax       raio de truncamento (padrão: None, sem truncamento)
    virialize  True/False (padrão: True se rmax for dado)
    seed       semente do gerador de números aleatórios (padrão: None)
    units      "astro" (Msun, pc, km/s) ou "nbody" (G = 1); padrão "astro"
    output     prefixo dos arquivos de saída (padrão: "plummer")
    formats    lista entre "ascii", "nbody6" e "gadget" (padrão: ["ascii"])
"""
import argparse
import ast
import sys

from . import checks, io, units
from .generate import generate

DEFAULTS = {"rmax": None, "virialize": None, "seed": None, "units": "astro",
            "output": "plummer", "formats": ["ascii"]}
REQUIRED = ("n", "mass", "a")
WRITERS = {"ascii": (io.write_ascii, ".txt"),
           "nbody6": (io.write_nbody6, ".dat.10"),
           "gadget": (io.write_gadget, ".g2")}


def read_params(fname):
    """Lê o arquivo chave = valor sem executar código (ast.literal_eval)."""
    params = dict(DEFAULTS)
    with open(fname) as f:
        for num, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError("{}:{}: esperado 'chave = valor'".format(fname, num))
            key, value = (s.strip() for s in line.split("=", 1))
            if key not in DEFAULTS and key not in REQUIRED:
                raise ValueError("{}:{}: chave desconhecida '{}'".format(fname, num, key))
            params[key] = ast.literal_eval(value)
    missing = [k for k in REQUIRED if k not in params]
    if missing:
        raise ValueError("faltam as chaves: " + ", ".join(missing))
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m plummer",
                                     description="Gera uma esfera de Plummer.")
    parser.add_argument("params", help="arquivo de parâmetros (chave = valor)")
    args = parser.parse_args(argv)
    p = read_params(args.params)

    if p["units"] == "astro":
        G = units.G_ASTRO
    elif p["units"] == "nbody":
        G = 1.0
    else:
        raise ValueError("units deve ser 'astro' ou 'nbody'")
    for fmt in p["formats"]:
        if fmt not in WRITERS:
            raise ValueError("formato desconhecido: " + fmt)
        if fmt != "ascii" and p["units"] != "astro":
            raise ValueError("o formato '{}' exige units = 'astro'".format(fmt))

    model = generate(int(p["n"]), mass=p["mass"], a=p["a"], G=G, rmax=p["rmax"],
                     virialize=p["virialize"], seed=p["seed"])
    for fmt in p["formats"]:
        writer, ext = WRITERS[fmt]
        writer(p["output"] + ext, model)
        print("gravado:", p["output"] + ext)
    print(checks.report(model))
    if p["units"] == "astro":
        rv, vstar, tstar = units.nbody_scales(model)
        print("Unidades de N-corpos: R_V = {:.4f} pc, V* = {:.3f} km/s, T* = {:.4f} Myr"
              .format(rv, vstar, tstar * units.MYR_PER_TIME_UNIT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
