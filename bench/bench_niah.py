#!/usr/bin/env python3
"""Needle-in-a-haystack a profundidad, con lambda de la proyeccion rank-1 como brazo.

Por que existe: el model card de cebeuq/DeepSeek-V4-Flash-0731-abliterated NO validó
contexto largo. Las direcciones de refusal se capturaron con prompts de ~60 tokens y
ninguna abliteracion publicada de DeepSeek-V4 se ha medido mas alla de 32k. Aqui se
sirve a 262 144. Esto es exactamente el hueco.

Mide recuperacion exacta de un dato insertado a distintas PROFUNDIDADES dentro de
contextos de distintas LONGITUDES, comparando lambda=0 (base intacto) contra lambda=1
(proyeccion ortogonal exacta) en el MISMO deployment, con los MISMOS prompts.

Temperatura 0 a proposito: esto es recuperacion, no generacion. Un fallo tiene que ser
del modelo, no del sampler. (El bench de tool-calling usa 1.0/0.95 porque alli si
importa el comportamiento agentico.)

El cambio de lambda invalida el prefix cache por diseno -- lambda entra en la clave de
hash de bloque -- asi que los dos brazos no se contaminan. Sin eso, un prefijo cacheado
con lambda=0 y reusado con lambda=1 daria estados corruptos en silencio.

Uso:
  python3 bench_niah.py --base http://<head>:8888 --lambdas 0,1 --out niah.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

MODEL = "deepseek-v4-flash-0731"
TEMP, TOP_P = 0.0, 1.0
# 512 y no 96: este modelo razona ANTES de contestar y emite esos tokens en un campo
# `reasoning` aparte. Con 96 se agotaba el presupuesto razonando y `content` volvia
# vacio con finish_reason="length" -- que parece un fallo de recuperacion del modelo
# y NO lo es. Se distingue explicitamente mas abajo: `truncated` != `miss`.
MAX_TOKENS = 512

# Agujas: hechos concretos, verificables por substring, que no pueden salir del
# conocimiento previo del modelo. Tres por celda para que una suerte no marque la
# diferencia.
NEEDLES = [
    (
        "El codigo de autorizacion del almacen de Getafe es SK-7734-QX.",
        "cual es el codigo de autorizacion del almacen de Getafe",
        "SK-7734-QX",
    ),
    (
        "La revision de firmware aprobada para las basculas Dibal es la 9.42-beta3.",
        "que revision de firmware esta aprobada para las basculas Dibal",
        "9.42-beta3",
    ),
    (
        "El limite de peso facturable pactado con el transportista es de 31,5 kg por bulto.",
        "cual es el limite de peso facturable pactado por bulto",
        "31,5",
    ),
]

# Relleno tecnico y anodino: no debe contener nada que compita con la aguja ni que
# se parezca a una respuesta.
FILLER = (
    "El procedimiento de recepcion en muelle exige cotejar el albaran contra la orden "
    "de compra antes de mover la mercancia a la zona de ubicacion. Cada palet recibe "
    "una etiqueta interna con su identificador de entrada. Las incidencias de rotura se "
    "anotan en el parte diario y se fotografian antes de retirar el retractilado. El "
    "responsable de turno firma el cierre de muelle al terminar la jornada. Los envases "
    "reutilizables se separan en la zona de retorno y se cuentan al cierre de semana. "
)


def post(base: str, path: str, payload: dict | None, timeout: int = 900):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(base: str, path: str, timeout: int = 60):
    req = urllib.request.Request(base.rstrip("/") + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def count_tokens(base: str, text: str) -> int | None:
    """Usa /tokenize del propio servidor. None si la ruta no esta."""
    try:
        return int(post(base, "/tokenize", {"model": MODEL, "prompt": text})["count"])
    except Exception:
        return None


def build_haystack(base: str, target_tokens: int) -> str:
    """Relleno calibrado con el tokenizador del servidor; si no esta, por caracteres."""
    per_block = count_tokens(base, FILLER)
    if per_block is None or per_block <= 0:
        # ~3,6 caracteres por token en español tecnico. Solo es el punto de partida:
        # la longitud REAL que se reporta sale de usage.prompt_tokens.
        n = max(1, int(target_tokens * 3.6 / len(FILLER)))
        return FILLER * n
    n = max(1, target_tokens // per_block)
    text = FILLER * n
    # ajuste fino
    for _ in range(24):
        got = count_tokens(base, text) or 0
        if got >= target_tokens * 0.98:
            break
        text += FILLER * max(1, (target_tokens - got) // per_block)
    return text


def set_lambda(base: str, lam: float) -> dict:
    r = post(base, "/admin/refusal_lambda", {"lambda": lam})
    chk = get(base, "/admin/refusal_lambda")
    # OJO: `chk.get("lambda") or X` NO vale — 0.0 es falsy en Python y el caso
    # lambda=0 es precisamente el brazo de control. Hay que comparar contra None.
    got = chk.get("lambda")
    if not chk.get("consistent") or got is None or abs(got - lam) > 1e-9:
        raise RuntimeError(f"lambda no quedo fijado: pedido={lam} leido={chk}")
    return chk


def run_cell(base: str, haystack: str, depth_pct: int, needle: tuple) -> dict:
    fact, question, expect = needle
    cut = int(len(haystack) * depth_pct / 100)
    # cortar en frontera de frase para no partir una palabra
    if 0 < cut < len(haystack):
        j = haystack.find(". ", cut)
        cut = (j + 2) if j != -1 else cut
    doc = haystack[:cut] + fact + " " + haystack[cut:]

    payload = {
        "model": MODEL,
        "temperature": TEMP,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{doc}\n\n"
                    f"Basandote UNICAMENTE en el documento anterior, responde: "
                    f"{question}? Responde solo con el dato, sin explicaciones."
                ),
            }
        ],
    }
    t0 = time.time()
    try:
        r = post(base, "/v1/chat/completions", payload)
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read()[:200]!r}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)[:200]}
    dt = time.time() - t0

    choice = r["choices"][0]
    msg = choice["message"]
    text = (msg.get("content") or "").strip()
    # El modelo emite su razonamiento en un campo aparte. Se guarda para poder
    # distinguir "no lo encontro" de "lo encontro pero se quedo sin presupuesto
    # para escribirlo".
    reason = (msg.get("reasoning") or msg.get("reasoning_content") or "") or ""
    finish = choice.get("finish_reason")
    usage = r.get("usage", {})

    hit = expect.lower() in text.lower()
    hit_reasoning = (not hit) and (expect.lower() in reason.lower())
    return {
        "ok": True,
        "hit": hit,
        "hit_only_in_reasoning": hit_reasoning,
        "expected": expect,
        "answer": text[:200],
        "reasoning_tail": reason[-160:] if reason else "",
        "finish_reason": finish,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_s": round(dt, 2),
        # respuesta vacia = fallo, nunca exito. Pero se separa la causa:
        "empty": len(text) == 0,
        "truncated": len(text) == 0 and finish == "length",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="http://<head>:8888")
    ap.add_argument("--lengths", default="32000,128000,256000")
    ap.add_argument("--depths", default="0,25,50,75,100")
    ap.add_argument("--lambdas", default="0,1")
    ap.add_argument("--out", default="niah.json")
    ap.add_argument(
        "--no-lambda-control",
        action="store_true",
        help="No llamar a /admin/refusal_lambda. Para medir la linea base contra un "
        "deployment SIN parchear: el brazo se etiqueta igual pero no se fija nada.",
    )
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]

    print(f"construyendo pajares para {lengths} ...")
    hay = {}
    for L in lengths:
        hay[L] = build_haystack(args.base, L)
        got = count_tokens(args.base, hay[L])
        print(f"  {L:>7} -> {got if got else '?'} tokens ({len(hay[L])} chars)")

    results = []
    for lam in lambdas:
        if args.no_lambda_control:
            print(f"\n=== lambda={lam} (SIN control: no se fijo nada) ===")
        else:
            st = set_lambda(args.base, lam)
            print(f"\n=== lambda={lam} (ranks={len(st.get('per_rank', []))}) ===")
        for L in lengths:
            for d in depths:
                cell = [run_cell(args.base, hay[L], d, n) for n in NEEDLES]
                good = [c for c in cell if c.get("ok")]
                hits = sum(1 for c in good if c["hit"])
                empt = sum(1 for c in good if c.get("empty"))
                trunc = sum(1 for c in good if c.get("truncated"))
                only_r = sum(1 for c in good if c.get("hit_only_in_reasoning"))
                lat = round(sum(c["latency_s"] for c in good) / max(1, len(good)), 1)
                ptok = good[0].get("prompt_tokens") if good else None
                results.append(
                    {
                        "lambda": lam,
                        "target_tokens": L,
                        "prompt_tokens": ptok,
                        "depth_pct": d,
                        "hits": hits,
                        "n": len(cell),
                        "errors": len(cell) - len(good),
                        "empty": empt,
                        "truncated": trunc,
                        "hit_only_in_reasoning": only_r,
                        "avg_latency_s": lat,
                        "cells": cell,
                    }
                )
                flag = "" if hits == len(cell) else "   <-- FALLO"
                extra = f" trunc={trunc}" if trunc else ""
                extra += f" solo-en-reasoning={only_r}" if only_r else ""
                print(
                    f"  L={L:>7} d={d:>3}%  {hits}/{len(cell)}"
                    f"  err={len(cell)-len(good)} vacias={empt}{extra}  {lat}s{flag}"
                )

    # ── comparativa
    print("\n================ NIAH: lambda=0 vs resto ================")
    base_lam = lambdas[0]
    for lam in lambdas[1:]:
        print(f"\n  lambda={lam} contra lambda={base_lam}:")
        for L in lengths:
            b = sum(r["hits"] for r in results if r["lambda"] == base_lam and r["target_tokens"] == L)
            c = sum(r["hits"] for r in results if r["lambda"] == lam and r["target_tokens"] == L)
            tot = len(depths) * len(NEEDLES)
            print(f"    L={L:>7}  {b}/{tot} -> {c}/{tot}   delta={c-b:+d}")

    json.dump(
        {"model": MODEL, "temp": TEMP, "lengths": lengths, "depths": depths,
         "lambdas": lambdas, "results": results},
        open(args.out, "w"),
        indent=2,
        ensure_ascii=False,
    )
    print(f"\nescrito {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
