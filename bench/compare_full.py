#!/usr/bin/env python3
"""Comparacion extensa lambda=0 vs lambda=1.5: acceptance, velocidad y calidad.

Por que existe: todas las medidas anteriores tienen n=1 o n=3 por brazo, y este
banco tiene un ruido que se come cualquier veredicto de una sola corrida (medido:
el MISMO lambda=1 dio acceptance 0,5383 y 0,5944 en dos pasadas). Esto corre lo
mismo muchas veces y ALTERNANDO los brazos, para que ninguna deriva del sistema
—termica, cache, carga— se cuele como si fuera efecto de lambda.

Ejes:
  1. acceptance de DSpark  (bench_speed, contador de /metrics)  -> puerta 0,55
  2. velocidad             (tok/s codigo y variado, TTFT)
  3. calidad               NIAH 32k/128k y tool-calling 5 fases

Salida: JSON con todas las corridas + resumen con media, desviacion y t de Welch.
Un delta sin su desviacion al lado no significa nada; por eso van juntos.

Uso: python3 compare_full.py --base http://<head>:8888 --reps 6
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def post(base, path, payload, timeout=120):
    r = urllib.request.Request(
        base.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())


def get(base, path, timeout=60):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as f:
        return json.loads(f.read())


def set_lambda(base, lam):
    post(base, "/admin/refusal_lambda", {"lambda": lam})
    chk = get(base, "/admin/refusal_lambda")
    got = chk.get("lambda")
    # `or` NO vale aqui: 0.0 es falsy y 0 es justo el brazo de control.
    if not chk.get("consistent") or got is None or abs(got - lam) > 1e-9:
        raise RuntimeError(f"lambda no quedo fijado: {chk}")


def run_speed(base, tag):
    out = HERE / f"cf_speed_{tag}.json"
    subprocess.run([sys.executable, str(HERE / "bench_speed.py"), "--base", base,
                    "--out", str(out)], capture_output=True, timeout=1800)
    try:
        d = json.loads(out.read_text())
    except Exception:
        return None
    # Estructura real de bench_speed: code/varied llevan tok_s_mean y un sub-dict
    # `spec` con el acceptance leido de los contadores de /metrics. El acceptance
    # de la puerta es el de `code`, que es el perfil con el que se calibro el
    # suelo de 0,55.
    code = d.get("code") or {}
    varied = d.get("varied") or {}
    spec = code.get("spec") or {}
    return {
        "acceptance": spec.get("acceptance_rate"),
        "accept_len": spec.get("mean_acceptance_length"),
        "acceptance_varied": (varied.get("spec") or {}).get("acceptance_rate"),
        "tok_s_code": code.get("tok_s_mean"),
        "tok_s_varied": varied.get("tok_s_mean"),
        "ttft": (d.get("ttft") or {}).get("median_s"),
    }


def run_niah(base, tag):
    out = HERE / f"cf_niah_{tag}.json"
    subprocess.run([sys.executable, str(HERE / "bench_niah.py"), "--base", base,
                    "--lengths", "32000,128000", "--depths", "0,25,50,75,100",
                    "--lambdas", "0", "--no-lambda-control", "--out", str(out)],
                   capture_output=True, timeout=3000)
    try:
        rs = json.loads(out.read_text())["results"]
    except Exception:
        return None
    return {"hits": sum(r["hits"] for r in rs), "n": sum(r["n"] for r in rs),
            "invalid": sum(r.get("errors", 0) + r.get("empty", 0) for r in rs)}


def run_tooling(base, tag):
    out = HERE / f"cf_tool_{tag}.json"
    p = subprocess.run([sys.executable, str(HERE / "bench_tooling.py"), "--base", base,
                        "--out", str(out)], capture_output=True, text=True, timeout=1800)
    for line in (p.stdout or "").splitlines():
        if "SCORE TOOL-CALLING" in line:
            return line.strip()
    return None


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    se = math.sqrt(st.stdev(a) ** 2 / len(a) + st.stdev(b) ** 2 / len(b))
    return (st.mean(a) - st.mean(b)) / se if se else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--lambdas", default="0,1.5")
    ap.add_argument("--out", default="compare_full.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lambdas.split(",")]
    data = {str(l): {"speed": []} for l in lams}
    t0 = time.time()

    # ── 1+2) acceptance y velocidad, ALTERNANDO brazos
    for i in range(args.reps):
        for lam in lams:
            set_lambda(args.base, lam)
            r = run_speed(args.base, f"{lam}_{i}")
            if r:
                data[str(lam)]["speed"].append(r)
            acc = (r or {}).get("acceptance")
            print(f"  rep{i+1} lambda={lam:<4} acceptance={acc} tok_s={(r or {}).get('tok_s_code')}",
                  flush=True)

    # ── 3) calidad, una pasada por brazo (son caras)
    for lam in lams:
        set_lambda(args.base, lam)
        data[str(lam)]["niah"] = run_niah(args.base, str(lam))
        data[str(lam)]["tooling"] = run_tooling(args.base, str(lam))
        print(f"  lambda={lam}: niah={data[str(lam)]['niah']} tooling={data[str(lam)]['tooling']}",
              flush=True)

    set_lambda(args.base, 0.0)

    # ── resumen
    print("\n" + "=" * 74)
    print(f"{'metrica':22s} " + " ".join(f"{'l='+str(l):>16s}" for l in lams) + "   t")
    print("=" * 74)
    for key, label in (("acceptance", "acceptance (code)"),
                       ("acceptance_varied", "acceptance (variado)"),
                       ("accept_len", "accept_len"),
                       ("tok_s_code", "tok/s codigo"), ("tok_s_varied", "tok/s variado"),
                       ("ttft", "TTFT mediana")):
        cols, series = [], []
        for l in lams:
            vals = [s[key] for s in data[str(l)]["speed"] if s.get(key) is not None]
            series.append(vals)
            cols.append(f"{st.mean(vals):.4f}±{st.stdev(vals):.4f}" if len(vals) > 1
                        else (f"{vals[0]:.4f}" if vals else "—"))
        t = welch(series[0], series[-1]) if len(series) >= 2 else None
        print(f"{label:22s} " + " ".join(f"{c:>16s}" for c in cols) +
              (f"   {t:+.2f}" if t is not None else ""))
    print("-" * 74)
    for l in lams:
        n = data[str(l)]["niah"]
        print(f"{'NIAH l='+str(l):22s} " +
              (f"{n['hits']}/{n['n']} (invalidas {n['invalid']})" if n else "—"))
        print(f"{'tool-calling l='+str(l):22s} {data[str(l)]['tooling'] or '—'}")

    print(f"\n|t| < 2 => indistinguible. Suelo de acceptance: 0.55")
    print(f"tiempo total: {(time.time()-t0)/60:.1f} min")
    json.dump(data, open(args.out, "w"), indent=2)
    print(f"escrito {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
