#!/usr/bin/env python3
"""Bateria de velocidad para DeepSeek-V4-Flash-0731 en TP=2.

Mide NON-STREAMING y a temperatura 0, que es lo que pide el criterio de
aceptacion. Contar chunks SSE con decodificacion especulativa mide steps/s, no
tok/s, y las cifras salen ~4x bajas: aqui se usa usage.completion_tokens del
propio servidor, que es el unico numero que no miente.

El acceptance rate NO se estima desde el tok/s: se lee de /metrics, que expone
los contadores de spec-decode de vLLM. Se toma un delta alrededor de cada fase
para no promediar con el warmup.

Uso:  python3 bench_speed.py --base http://<head>:8888 [--out resultados.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

MODEL = "deepseek-v4-flash-0731"

# ── prompts ───────────────────────────────────────────────────────────────────
# Dos familias, porque el criterio las separa: "codigo/estructurado" y "media
# sobre prompts variados". Los textos son deliberadamente largos de RESPUESTA
# (no de prompt) para que el decode domine y el tok/s sea el del steady state.
CODE_PROMPTS = [
    ("py-refactor",
     "Escribe una clase Python `RateLimiter` con token bucket thread-safe, "
     "type hints completos, docstrings y tres tests de pytest. Codigo completo."),
    ("sql-window",
     "Escribe una consulta PostgreSQL que, por cliente, saque el ticket medio "
     "movil de 3 pedidos usando window functions, y explica cada CTE."),
    ("ts-types",
     "Implementa en TypeScript un tipo `DeepReadonly<T>` recursivo que funcione "
     "con arrays, tuplas, Map y Set, con casos de prueba que compilen."),
    ("k8s-yaml",
     "Escribe un Deployment de Kubernetes con initContainer, probes, "
     "securityContext sin privilegios y limites, y comenta cada bloque."),
    ("algo",
     "Implementa Dijkstra con heap binario en Rust, con manejo de errores "
     "idiomatico y tests. Codigo completo, sin elidir."),
]

VARIED_PROMPTS = [
    ("explica-tecnico",
     "Explica como funciona la atencion dispersa comprimida frente a la "
     "atencion densa, y por que reduce el KV cache. Detallado."),
    ("resumen-largo",
     "Resume las implicaciones operativas de servir un MoE de 284B con 13B "
     "activos en dos nodos unidos por RoCE, y los modos de fallo."),
    ("prosa",
     "Escribe un informe de incidencia de 600 palabras sobre una caida de un "
     "servicio de inferencia por agotamiento de memoria unificada."),
    ("razonamiento",
     "Tengo 119 GiB por nodo, pesos de 83.5 GB por nodo y un KV cache "
     "comprimido. Calcula el contexto maximo servible y razona los pasos."),
    ("traduccion",
     "Traduce al ingles tecnico y luego al aleman: 'el arbitro de GPU escala a "
     "cero los residentes del nodo elegido antes de arrancar la carga'."),
]

# Profundidades para la curva de prefill. El relleno es texto real repetido, no
# un token repetido: un token repetido se comprime distinto en la atencion
# dispersa y da una curva optimista.
FILLER = ("El planificador de vLLM agrupa peticiones en lotes continuos y el "
          "cache de KV se pagina en bloques de tamano fijo para evitar la "
          "fragmentacion externa de la memoria del acelerador. ")
PREFILL_DEPTHS = [1_000, 4_000, 16_000, 64_000, 131_072, 262_144]


def _post(base: str, path: str, payload: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _metrics(base: str) -> dict[str, float]:
    """Contadores de spec-decode de /metrics. Devuelve {} si no estan."""
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as r:
            body = r.read().decode()
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] /metrics no accesible: {e}", file=sys.stderr)
        return {}
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        m = re.match(r"^(\S+?)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def _spec_delta(before: dict, after: dict) -> dict:
    """acceptance rate y mean acceptance length a partir del delta.

    vLLM v1 expone:
      vllm:spec_decode_num_draft_tokens_total     tokens propuestos
      vllm:spec_decode_num_accepted_tokens_total  tokens aceptados
      vllm:spec_decode_num_drafts_total           numero de drafts
    mean acceptance length = aceptados/drafts + 1  (el +1 es el token del target,
    que siempre sale aunque se rechace todo el draft).
    """
    def d(k: str) -> float:
        for name in (f"vllm:spec_decode_num_{k}_total", f"vllm:spec_decode_num_{k}"):
            if name in after:
                return after.get(name, 0.0) - before.get(name, 0.0)
        return 0.0

    draft, acc, drafts = d("draft_tokens"), d("accepted_tokens"), d("drafts")
    res: dict[str, float | None] = {
        "draft_tokens": draft, "accepted_tokens": acc, "drafts": drafts,
    }
    res["acceptance_rate"] = round(acc / draft, 4) if draft > 0 else None
    res["mean_acceptance_length"] = round(acc / drafts + 1, 3) if drafts > 0 else None
    return res


def run_group(base: str, name: str, prompts, max_tokens: int) -> dict:
    print(f"\n=== {name} ===")
    before = _metrics(base)
    rows = []
    for pid, text in prompts:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,          # NON-streaming, a proposito
        }
        t0 = time.perf_counter()
        try:
            r = _post(base, "/v1/chat/completions", payload)
        except Exception as e:  # noqa: BLE001
            print(f"  {pid:18} FALLO: {e}")
            rows.append({"id": pid, "error": str(e)})
            continue
        dt = time.perf_counter() - t0
        u = r.get("usage") or {}
        ct = u.get("completion_tokens") or 0
        # Respuesta vacia = fallo, no exito. Pero OJO: con reasoning_parser
        # deepseek_v4 el texto puede venir en reasoning_content, y si max_tokens
        # corta dentro del bloque de pensamiento los dos campos salen vacios con
        # tokens > 0. Eso es truncamiento, no un backend muerto: solo es fallo de
        # verdad cuando el servidor no genero NADA.
        msg = r["choices"][0].get("message") or {}
        body = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
        if ct == 0:
            print(f"  {pid:18} FALLO: 0 tokens generados")
            rows.append({"id": pid, "error": "empty_response", "wall_s": round(dt, 2)})
            continue
        truncated = not body.strip()
        tps = ct / dt
        rows.append({
            "id": pid, "completion_tokens": ct, "prompt_tokens": u.get("prompt_tokens"),
            "wall_s": round(dt, 3), "tok_s": round(tps, 2), "truncated_in_think": truncated,
        })
        print(f"  {pid:18} {ct:6d} tok  {dt:7.2f}s  {tps:6.2f} tok/s"
              f"{'  (cortado en el bloque de think)' if truncated else ''}")
    after = _metrics(base)
    ok = [r["tok_s"] for r in rows if "tok_s" in r]
    return {
        "rows": rows,
        "tok_s_mean": round(statistics.mean(ok), 2) if ok else None,
        "tok_s_median": round(statistics.median(ok), 2) if ok else None,
        "spec": _spec_delta(before, after),
    }


def run_ttft(base: str, n: int = 5) -> dict:
    """TTFT de prompt corto. Es el unico caso donde SI se hace streaming: es la
    unica forma de cronometrar el primer token. No se usa para medir tok/s."""
    print("\n=== TTFT (prompt corto) ===")
    lat = []
    for i in range(n):
        payload = {"model": MODEL, "messages": [{"role": "user", "content": "Di 'hola'."}],
                   "temperature": 0, "max_tokens": 16, "stream": True}
        req = urllib.request.Request(
            base.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                for raw in r:
                    s = raw.decode(errors="ignore").strip()
                    if not s.startswith("data:") or s.endswith("[DONE]"):
                        continue
                    try:
                        ch = json.loads(s[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    d = (ch.get("choices") or [{}])[0].get("delta") or {}
                    # OJO: esta build emite el pensamiento en `reasoning`, NO en
                    # `reasoning_content`, y el primer chunk trae content:"" (falsy)
                    # solo para abrir el rol. Mirar solo content/reasoning_content
                    # hace que TTFT no capture NADA y salga vacio sin error.
                    if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                        lat.append(time.perf_counter() - t0)
                        break
        except Exception as e:  # noqa: BLE001
            print(f"  intento {i+1}: FALLO {e}")
    for i, v in enumerate(lat):
        print(f"  intento {i+1}: {v:.3f}s")
    return {"samples_s": [round(v, 3) for v in lat],
            "median_s": round(statistics.median(lat), 3) if lat else None}


def run_prefill(base: str, depths) -> dict:
    """Curva de prefill por profundidad de contexto. max_tokens=1 para aislar
    el prefill del decode."""
    print("\n=== curva de prefill ===")
    rows = []
    unit = max(1, len(FILLER) // 4)  # ~4 chars/token aprox
    for d in depths:
        filler = FILLER * max(1, d // unit)
        prompt = (filler + "\n\nEn una sola palabra, di 'listo'.")
        payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": 1, "stream": False}
        t0 = time.perf_counter()
        try:
            r = _post(base, "/v1/chat/completions", payload)
        except Exception as e:  # noqa: BLE001
            print(f"  {d:>8} objetivo  FALLO: {e}")
            rows.append({"target_depth": d, "error": str(e)})
            continue
        dt = time.perf_counter() - t0
        pt = (r.get("usage") or {}).get("prompt_tokens") or 0
        rows.append({"target_depth": d, "prompt_tokens": pt, "wall_s": round(dt, 3),
                     "prefill_tok_s": round(pt / dt, 1) if dt > 0 else None})
        print(f"  {d:>8} objetivo -> {pt:>8} tok reales  {dt:7.2f}s  {pt/dt:8.1f} tok/s")
    return {"rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="p.ej. http://100.73.153.70:8888")
    ap.add_argument("--out", default="bench_speed_results.json")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--skip-prefill", action="store_true")
    a = ap.parse_args()

    try:
        with urllib.request.urlopen(a.base.rstrip("/") + "/v1/models", timeout=30) as r:
            served = [m["id"] for m in json.loads(r.read()).get("data", [])]
        print(f"modelos servidos: {served}")
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: el servidor no responde en {a.base}: {e}", file=sys.stderr)
        return 1

    res = {
        "base": a.base,
        "served_models": served,
        "code": run_group(a.base, "codigo / estructurado", CODE_PROMPTS, a.max_tokens),
        "varied": run_group(a.base, "prompts variados", VARIED_PROMPTS, a.max_tokens),
        "ttft": run_ttft(a.base),
    }
    if not a.skip_prefill:
        res["prefill"] = run_prefill(a.base, PREFILL_DEPTHS)

    # Veredicto contra los criterios de aceptacion.
    spec = res["code"]["spec"]
    ar = spec.get("acceptance_rate")
    mal = spec.get("mean_acceptance_length")
    verdict = {
        "tok_s_code":    (res["code"]["tok_s_mean"],   45, 55),
        "tok_s_varied":  (res["varied"]["tok_s_mean"], 40, 45),
        "acceptance_rate": (ar, 0.55, 0.70),
        "mean_accept_len": (mal, 3.5, 4.5),
    }
    print("\n=== VEREDICTO ===")
    gates = {}
    for k, (val, mn, obj) in verdict.items():
        if val is None:
            print(f"  {k:18} SIN DATO"); gates[k] = None; continue
        ok = val >= mn
        gates[k] = ok
        print(f"  {k:18} {val:>8}  min {mn}  obj {obj}   {'OK' if ok else 'POR DEBAJO'}")
    if res["ttft"]["median_s"] is not None:
        ok = res["ttft"]["median_s"] < 3.0
        gates["ttft"] = ok
        print(f"  {'ttft_median_s':18} {res['ttft']['median_s']:>8}  max 3.0        {'OK' if ok else 'POR DEBAJO'}")
    res["gates"] = gates

    if gates.get("acceptance_rate") is False:
        print("\n*** PARAR: acceptance < 55%. Es fallo de carga del drafter, no de "
              "sampling. NO tocar LiteLLM hasta resolverlo. ***")

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nescrito {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
