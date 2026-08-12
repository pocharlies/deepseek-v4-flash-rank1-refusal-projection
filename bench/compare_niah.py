#!/usr/bin/env python3
"""Puerta de igualdad: compara dos corridas de bench_niah.py celda a celda.

Uso previsto:
  A = niah_baseline_lambda0.json   corrido contra la IMAGEN BASE (sin parche)
  B = niah_rank1_lambda0.json      corrido contra la imagen PARCHEADA con lambda=0

MEDIDO EL 12-08-2026, Y CAMBIA EL CRITERIO: `exact` NO SIRVE COMO PUERTA.

La idea original era que con lambda=0 el hook es `y - 0*r*(r.y)`, bit-exacto a `y`
(verificado con torch.equal), asi que B deberia reproducir A literalmente. Se probo
comparando una corrida de lambda=0 CONSIGO MISMA — misma imagen, mismo deployment,
misma config — y salio distinta:

    L=128000 d=25%   A: 'SK-7734-QX'   B: 'SK-773-QX'    (30/30 -> 29/30)

O sea: **vLLM no es determinista run-a-run a temperatura 0**. El batching continuo
cambia el orden de reduccion en los GEMM y eso mueve el argmax en empates cerrados.
El modo `exact` marca ese ruido como defecto del hook: es un falso positivo, y de
hecho disparo contra una corrida comparada con ella misma.

CRITERIO BUENO: `hits`. Y con una banda de ruido medida de ~1 celda de 30 por corrida,
asi que una diferencia de 1-2 celdas NO es una regresion — hace falta repetir.

`exact` se deja porque sigue siendo util como DESCRIPCION (ver cuanto se mueve la
redaccion), nunca como puerta.
"""
from __future__ import annotations

import argparse
import json
import sys


def key(r):
    return (r["target_tokens"], r["depth_pct"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="corrida de referencia (base)")
    ap.add_argument("--b", required=True, help="corrida a validar")
    ap.add_argument(
        "--mode",
        choices=["exact", "hits"],
        default="exact",
        help="exact = texto identico (lambda=0); hits = solo compara aciertos (lambda=1)",
    )
    args = ap.parse_args()

    A = {key(r): r for r in json.load(open(args.a))["results"]}
    B = {key(r): r for r in json.load(open(args.b))["results"]}

    common = sorted(set(A) & set(B))
    if not common:
        print("SIN CELDAS COMUNES — las dos corridas no son comparables")
        return 2
    if set(A) != set(B):
        print(f"aviso: A tiene {len(A)} celdas, B tiene {len(B)}; se comparan {len(common)}")

    diffs, hits_a, hits_b, n = [], 0, 0, 0
    for k in common:
        a, b = A[k], B[k]
        hits_a += a["hits"]
        hits_b += b["hits"]
        n += a["n"]
        for i, (ca, cb) in enumerate(zip(a["cells"], b["cells"])):
            if args.mode == "exact":
                if ca.get("answer") != cb.get("answer"):
                    diffs.append((k, i, ca.get("answer", "")[:60], cb.get("answer", "")[:60]))
            elif ca.get("hit") != cb.get("hit"):
                diffs.append((k, i, f"hit={ca.get('hit')}", f"hit={cb.get('hit')}"))

    print(f"celdas comparadas : {len(common)}  ({n} llamadas)")
    print(f"aciertos A / B    : {hits_a}/{n}  ->  {hits_b}/{n}   delta={hits_b-hits_a:+d}")
    print(f"modo              : {args.mode}")
    if not diffs:
        print(
            "\nIDENTICAS."
            if args.mode == "exact"
            else "\nMISMOS ACIERTOS en todas las celdas."
        )
        return 0

    print(f"\n{len(diffs)} DIFERENCIAS:")
    for (L, d), i, va, vb in diffs[:20]:
        print(f"  L={L} d={d}% aguja{i}:")
        print(f"    A: {va!r}")
        print(f"    B: {vb!r}")
    if len(diffs) > 20:
        print(f"  ... y {len(diffs)-20} mas")
    if args.mode == "exact":
        print(
            "\nNO es un veredicto: el servidor no es determinista a temp 0 y este modo\n"
            "marca ese ruido como si fuera del hook (medido: una corrida comparada\n"
            "CONSIGO MISMA tambien difiere). Juzga con --mode hits."
        )
    else:
        print(
            f"\nDelta de aciertos {hits_b-hits_a:+d}. Ruido medido run-a-run: ~1 celda\n"
            "de 30, asi que |delta| <= 2 NO es concluyente — repetir la corrida."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
