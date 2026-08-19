#!/usr/bin/env python3
"""Barrido de calidad por lambda SIN tocar el dial global.

PARA QUE. Hoy `REFUSAL_MAX_LAMBDA` esta en 2.5 y no hay ni una medida completa
que lo respalde: en la ventana del 16-08 el arm de 2.5 corrio gsm8k (0.74) pero
MMLU-Pro murio al 42% con ConnectionRefusedError, asi que la unica suite que
discrimina razonamiento dificil no tiene numero a 2.5. Este runner existe para
producir ese numero y poner el techo por evidencia y no por criterio.

QUE CAMBIA RESPECTO A bench_quality_multilambda.py. Aquel movia el dial GLOBAL
por arm: ponia el servidor entero en lambda 2.5 mientras duraba la medicion, y
por eso tenia que serializar y restaurar produccion en un `finally`. Aqui el
lambda viaja POR PETICION en `cache_salt`, inyectado por refusal_salt_proxy.py.
El dial global se queda en 0 y se COMPRUEBA que sigue en 0 antes y despues de
cada arm — si algo lo moviera, el barrido para en vez de atribuir a lambda una
degradacion que no es suya.

DOS GUARDAS QUE PUEDEN FALLAR, que es el punto:

  sello    Al acabar el arm se lee /proxy/stats. Si `sealed` no cuadra con las
           peticiones vistas, o es 0, el arm se marca invalido. Un lambda que no
           llega es exactamente el fallo que este mecanismo acaba de arreglar en
           el servidor; medir sin comprobarlo seria repetirlo.
  runaway  Politica del arnes de Qwen del 17-08: a lambda alto el modelo entra en
           generacion desbocada (alli, 3 respuestas en 300 s donde a 1.5 hacia 100
           en 277). Un arm que supera su presupuesto de pared se ABORTA y se
           reporta como arm FALLIDO, sin nota de accuracy. Un arm desbocado no es
           un 0%: es una medicion que no existe, y ponerle un numero mentiria.

La salida usa el MISMO layout que el runner viejo (lambda_<L>/<task>/…), asi que
analyze_quality_ab.py da el analisis pareado con McNemar sin tocar nada.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROXY = HERE / "refusal_salt_proxy.py"


def get_json(url: str, timeout: float = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def global_dial(base: str) -> dict[str, Any]:
    return get_json(base.rstrip("/") + "/admin/refusal_lambda")


class HeadDown(Exception):
    """El head no contesta. NO es un resultado de calidad: es infraestructura."""


def head_alive(base: str) -> dict[str, Any] | None:
    try:
        return global_dial(base)
    except Exception:  # noqa: BLE001  (URLError, timeout, socket, JSON a medias)
        return None


def wait_for_head(base: str, timeout: float, when: str) -> dict[str, Any]:
    """Espera a que el head vuelva. Devuelve su dial YA REARRANCADO.

    POR QUE EXISTE: el 18-08 un commit ajeno (`17a7d1a`) rolo los pods a mitad del
    barrido y se lo llevo por delante con `Connection refused`. El head es
    produccion: se reinicia cuando le toca, no cuando conviene a la medida. Un
    barrido de horas que muere en el minuto 20 por eso no mide nada.

    Lo que NO hace: dar por bueno el arm que atravesaba el reinicio. Ese se
    descarta y se repite entero — mezclar mitad-antes/mitad-despues seria peor
    que no medir.
    """
    deadline = time.monotonic() + timeout
    first = True
    while time.monotonic() < deadline:
        st = head_alive(base)
        if st is not None:
            if not first:
                print(f"[sweep] head de vuelta ({when}): dial={st.get('lambda')}",
                      flush=True)
            return st
        if first:
            print(f"[sweep] head CAIDO ({when}); esperando hasta {timeout:.0f}s",
                  flush=True)
            first = False
        time.sleep(10)
    raise HeadDown(f"el head no volvio en {timeout:.0f}s ({when})")


def assert_dial_untouched(base: str, expect: float, when: str) -> dict[str, Any]:
    st = head_alive(base)
    if st is None:
        raise HeadDown(f"el head no contesta ({when})")
    got = st.get("lambda")
    if got is None or abs(float(got) - expect) > 1e-9:
        raise RuntimeError(
            f"el dial GLOBAL vale {got} y deberia valer {expect} ({when}). Este "
            f"barrido no lo toca, asi que lo ha movido otra cosa: cualquier "
            f"resultado de aqui en adelante mezclaria dos lambdas. Se para."
        )
    if st.get("consistent") is False:
        raise RuntimeError(f"los ranks TP no coinciden en el dial global: {st}")
    return st


def parse_suites(value: str) -> list[dict[str, Any]]:
    out = []
    for chunk in value.split(","):
        task, _, limit = chunk.partition(":")
        out.append({"task": task.strip(), "limit": int(limit or 100)})
    return out


def result_exists(results_dir: Path, lam: float, task: str) -> bool:
    return any((results_dir / f"lambda_{lam}" / task).glob("*/results_*.json"))


def start_proxy(upstream: str, lam: float, timeout: float
                ) -> tuple[subprocess.Popen, int]:
    proc = subprocess.Popen(
        ["python3", str(PROXY), "--upstream", upstream, "--lambda", str(lam),
         "--port", "0", "--timeout", str(timeout)],
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    if not line.isdigit():
        proc.kill()
        raise RuntimeError(f"el proxy no publico puerto; dijo {line!r}")
    return proc, int(line)


def run_arm(args, lam: float, suite: dict[str, Any]) -> dict[str, Any]:
    task, limit = str(suite["task"]), int(suite["limit"])
    if task == "mmlu_pro_llama" and args.mmlu_pro_llama_per_category is not None:
        # mmlu_pro_llama es un GRUPO de 14 tareas y lm-eval aplica --limit a cada
        # categoria: --limit 100 lanzaria 1.400 peticiones sin avisar.
        limit = args.mmlu_pro_llama_per_category

    if result_exists(args.results_dir, lam, task):
        print(f"[sweep] lambda={lam} {task}: ya completo, se reutiliza", flush=True)
        # El dict tiene que traer las MISMAS claves que un arm corrido. Que no las
        # trajera tumbo el barrido entero con KeyError('wall_seconds') en el primer
        # resume, 30 min despues de relanzarlo y sin que se notara.
        return {"lambda": lam, "task": task, "status": "reused", "reason": "",
                "wall_seconds": 0.0, "proxy": {}}

    partial = args.results_dir / f"lambda_{lam}" / task
    if partial.exists():
        shutil.rmtree(partial)

    assert_dial_untouched(args.base, args.expect_global, f"antes de lambda={lam} {task}")
    proxy, port = start_proxy(args.base, lam, args.request_timeout)
    log_path = args.results_dir / f"lambda_{lam}_{task}.log"
    started = time.monotonic()
    status, reason = "ok", ""
    try:
        cmd = [
            args.python, "-m", "lm_eval",
            "--model", "local-chat-completions",
            "--model_args",
            (f"model={args.model},"
             f"base_url=http://127.0.0.1:{port}/v1/chat/completions,"
             f"num_concurrent={args.concurrency},tokenized_requests=False,"
             f"max_retries={args.retries},timeout={args.request_timeout}"),
            "--apply_chat_template",
            "--tasks", task,
            "--limit", str(limit),
            "--num_fewshot", "0",
            "--gen_kwargs", f"max_gen_toks={args.max_gen_toks},temperature=0",
            "--seed", "0,1234,1234,1234",
            "--output_path", str(args.results_dir / f"lambda_{lam}" / task),
            "--log_samples",
        ]
        env = os.environ.copy()
        args.hf_home.mkdir(parents=True, exist_ok=True)
        env.update({
            "OPENAI_API_KEY": "bench",
            "HF_HOME": str(args.hf_home),
            "HF_DATASETS_CACHE": str(args.hf_home / "datasets"),
            "PYTHONUNBUFFERED": "1",
        })
        print(f"[sweep] lambda={lam} {task} limit={limit} "
              f"presupuesto={args.arm_budget_seconds}s", flush=True)
        with log_path.open("w") as log:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, env=env,
                                 bufsize=1, start_new_session=True)
            assert p.stdout is not None
            for line in p.stdout:
                log.write(line); log.flush()
                if "100%" in line or "|Tasks|" in line or "|    Groups" in line:
                    print(line.rstrip(), flush=True)
                if time.monotonic() - started > args.arm_budget_seconds:
                    # RUNAWAY. Se aborta el grupo entero de procesos.
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    status = "failed_runaway"
                    reason = (f"supero el presupuesto de {args.arm_budget_seconds}s; "
                              f"arm reportado como FALLIDO, sin nota de accuracy")
                    break
            rc = p.wait()
        if status == "ok" and rc:
            status, reason = "failed_lm_eval", f"lm-eval devolvio rc={rc}"

        stats = get_json(f"http://127.0.0.1:{port}/proxy/stats", timeout=10)
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proxy.kill()

    # El sello tiene que haber llegado a TODAS las peticiones del arm.
    if status == "ok":
        if stats["sealed"] == 0:
            status, reason = "failed_unsealed", "el proxy no sello ni una peticion"
        elif stats["sealed"] != stats["seen"]:
            status, reason = "failed_unsealed", (
                f"selladas {stats['sealed']} de {stats['seen']} vistas")

    # Un arm que ATRAVESO un reinicio del head no se puntua, aunque lm-eval haya
    # devuelto 0: la mitad de las respuestas vendrian de otro proceso. Se marca
    # invalido para que el main lo repita entero.
    if head_alive(args.base) is None:
        status, reason = "failed_head_restart", (
            "el head no contesta al cerrar el arm; se descarta y se repite")
    else:
        try:
            assert_dial_untouched(args.base, args.expect_global,
                                  f"despues de lambda={lam} {task}")
        except HeadDown as e:
            status, reason = "failed_head_restart", str(e)

    return {
        "lambda": lam, "task": task, "status": status, "reason": reason,
        "wall_seconds": round(time.monotonic() - started, 1), "proxy": stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="head de vLLM, p.ej. http://ip:8888")
    ap.add_argument("--model", required=True)
    ap.add_argument("--python", default="python3", help="python del venv con lm-eval")
    ap.add_argument("--hf-home", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    # 2.0 entra a proposito: entre 1.5 (medido, 0.66 en MMLU) y 2.5 (sin medir) no
    # hay ningun punto, y el techo hay que ponerlo donde este el codo.
    ap.add_argument("--lambdas", default="0,1.0,1.5,2.0,2.5")
    ap.add_argument("--expect-global", type=float, default=0.0,
                    help="valor en el que DEBE quedarse el dial global todo el rato")
    ap.add_argument("--suites", type=parse_suites,
                    default=parse_suites("mmlu_pro_llama:112,gsm8k:100"))
    ap.add_argument("--mmlu-pro-llama-per-category", type=int, default=8)
    ap.add_argument("--max-gen-toks", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--request-timeout", type=int, default=1200)
    # Referencia del 17-08 en Qwen: gsm8k 100 tardo 277-385 s segun lambda. 3600 s
    # deja margen de sobra para MMLU y sigue cazando una desbocada en minutos.
    ap.add_argument("--arm-budget-seconds", type=int, default=3600)
    # El head es produccion y se reinicia cuando le toca (18-08: un commit ajeno
    # rolo los pods a mitad del barrido). Un arm atravesado por un reinicio se
    # DESCARTA y se repite; no se puntua a medias.
    ap.add_argument("--arm-retries", type=int, default=3)
    ap.add_argument("--head-wait-seconds", type=float, default=1800)
    args = ap.parse_args()

    lambdas = [float(v) for v in args.lambdas.split(",")]
    if len(lambdas) != len(set(lambdas)):
        raise SystemExit("--lambdas no admite repetidos")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    st = assert_dial_untouched(args.base, args.expect_global, "al arrancar")
    print(f"[sweep] dial global {st.get('lambda')} per_rank={st.get('per_rank')} "
          f"— NO se toca en todo el barrido", flush=True)

    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base": args.base, "model": args.model,
        "lambdas": lambdas, "suites": args.suites,
        "max_gen_toks": args.max_gen_toks, "temperature": 0,
        "seeds": [0, 1234, 1234, 1234], "paired": True,
        "mmlu_pro_llama_per_category": args.mmlu_pro_llama_per_category,
        "method": "per-request cache_salt via refusal_salt_proxy (dial global intacto)",
        "arm_budget_seconds": args.arm_budget_seconds,
    }
    (args.results_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    arms = []
    for suite in args.suites:
        for lam in lambdas:
            arm = None
            for attempt in range(1, args.arm_retries + 1):
                try:
                    arm = run_arm(args, lam, suite)
                except HeadDown as e:
                    # El head estaba caido ANTES de empezar. Se espera y se repite;
                    # no se anota nada, porque no se ha medido nada.
                    print(f"[sweep] {e}", flush=True)
                    st = wait_for_head(args.base, args.head_wait_seconds,
                                       f"antes de lambda={lam} {suite['task']}")
                    args.expect_global = float(st["lambda"])
                    print(f"[sweep] expectativa del dial re-tomada: "
                          f"{args.expect_global} (el head rearranco con su INIT)",
                          flush=True)
                    continue

                if arm["status"] != "failed_head_restart":
                    break

                # Arm invalido por reinicio: esperar, re-sincronizar la expectativa
                # del dial (al arrancar el head toma VLLM_REFUSAL_LAMBDA_INIT) y
                # repetirlo desde cero.
                print(f"[sweep] arm lambda={lam} {suite['task']} invalidado por "
                      f"reinicio del head (intento {attempt}/{args.arm_retries})",
                      flush=True)
                st = wait_for_head(args.base, args.head_wait_seconds,
                                   f"reintento de lambda={lam} {suite['task']}")
                args.expect_global = float(st["lambda"])
                shutil.rmtree(args.results_dir / f"lambda_{lam}" / suite["task"],
                              ignore_errors=True)

            if arm is None:
                # Se agotaron los intentos sin llegar a correr NI UNO: el head
                # nunca volvio. Se anota como fallo; no se sigue como si nada.
                arm = {"lambda": lam, "task": suite["task"],
                       "status": "failed_head_down", "wall_seconds": 0.0,
                       "reason": f"el head no volvio en {args.arm_retries} intentos",
                       "proxy": {}}

            arm["attempts"] = attempt
            arms.append(arm)
            print(f"[sweep] -> lambda={lam} {suite['task']}: {arm['status']} "
                  f"{arm.get('reason','')} ({arm['wall_seconds']}s)", flush=True)
            (args.results_dir / "arms.json").write_text(json.dumps(arms, indent=2))

    bad = [a for a in arms if a["status"].startswith("failed")]
    print("\n=== resumen ===")
    for a in arms:
        print(f"  lambda={a['lambda']:<4} {a['task']:<16} {a['status']}"
              + (f"  ({a['reason']})" if a.get("reason") else ""))
    print(f"\narms fallidos: {len(bad)} de {len(arms)}")
    print(f"analisis pareado:  python3 analyze_quality_ab.py {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
