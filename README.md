# plummer-gen

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24015458.svg)](https://doi.org/10.5281/zenodo.24015458)

Gerador de condições iniciais de esferas de Plummer isotrópicas para
simulações de N-corpos. Implementa o algoritmo de Aarseth, Hénon & Wielen
(1974) e acompanha o artigo

> S. Moraes e I. Rodrigues, *Geração de Modelos Esféricos de Sistemas
> Estelares com a Utilização do Perfil de Densidade de Plummer*,
> Revista Brasileira de Ensino de Física (submetido).

## Instalação

Requer Python ≥ 3.6, NumPy ≥ 1.17 e SciPy (matplotlib só para as figuras).

```
pip install .
```

## Uso

Como biblioteca (unidades G = M = a = 1 por padrão):

```python
import plummer
from plummer import checks

model = plummer.generate(40000, seed=1)       # modelo completo
print(checks.report(model))                   # testes de qualidade
```

Em unidades astrofísicas (Msun, pc, km/s), truncado em `rmax`:

```python
from plummer import io, units

model = plummer.generate(40000, mass=5e6, a=4.0, G=units.G_ASTRO,
                         rmax=26.37, seed=20241105)
io.write_nbody6("dat.10", model)     # NBODY6++GPU (KZ(22) = 6)
io.write_gadget("ic.g2", model)      # GADGET-2/3/4 (ICFormat = 2)
```

Pela linha de comando, com um arquivo de parâmetros:

```
python -m plummer examples/omega_cen.txt
```

## Truncamento e equilíbrio virial

As velocidades são sorteadas da função distribuição do modelo de Plummer
completo. Quando as posições são cortadas em `rmax`, a massa que estaria
além desse raio é redistribuída entre as N partículas, a energia potencial
muda e o sistema deixa de estar em equilíbrio virial: Q = K/|W| = 0,486
para `rmax = 6,6a` e 0,41 para `rmax = 2a`
(`checks.expected_virial_ratio`). Por isso, quando `rmax` é dado, as
velocidades são multiplicadas por um fator constante que leva Q a 0,5
(`virialize=True`, o padrão nesse caso). É o mesmo procedimento aplicado
pelo NBODY6++GPU com `Q = 0.5` no bloco `INSCALE`.

## Testes de qualidade

`plummer.checks` compara cada realização com as previsões analíticas:
massa acumulada M(r) e distribuição de q = V/Ve (testes de
Kolmogorov-Smirnov), perfil de densidade, dispersão de velocidades e
anisotropia β(r), razão virial, raio virial e centro de massa.
`scripts/validate.py` produz as figuras do artigo. Os testes automáticos
rodam com

```
python -m pytest tests
```

## English summary

`plummer-gen` generates isotropic Plummer-sphere initial conditions for
N-body codes (Aarseth, Hénon & Wielen 1974 algorithm), with optional
truncation radius and virial rescaling, writers for NBODY6++GPU and GADGET,
and quality checks against the analytic model. See the paper above.

## Licença

MIT. Veja `LICENSE`. Para citar, veja `CITATION.cff`.

## Como citar

S. Moraes e I. Rodrigues, *plummer-gen: gerador de esferas de Plummer
para simulações de N-corpos*, v1.0.0, Zenodo (2026),
[doi:10.5281/zenodo.24015458](https://doi.org/10.5281/zenodo.24015458).
