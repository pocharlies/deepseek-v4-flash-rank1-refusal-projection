#!/usr/bin/env python3
"""Suite de tool-calling obligatoria antes de enrutar trafico.

Cinco fases, las que pide el criterio:
  1 seleccion       - elige la tool correcta con argumentos correctos
  2 distractores    - misma tarea con 8 tools plausibles de mas
  3 cadena          - multi-paso: la 2a llamada depende del resultado de la 1a
  4 rechazo         - peticion imposible: debe NO llamar y decirlo
  5 recuperacion    - la tool devuelve error; debe reintentar corregido, no repetir

Se puntua el ARGUMENTO, no solo el nombre: acertar la tool y pasarle basura es
un fallo. Un modelo que llama `get_weather(city="???")` no sirve para un agente.

Uso:  python3 bench_tooling.py --base http://<head>:8888 [--out t.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

MODEL = "deepseek-v4-flash-0731"
# Params recomendados por el model card para escenarios agenticos.
TEMP, TOP_P = 1.0, 0.95

# ── catalogos de tools ────────────────────────────────────────────────────────
def _t(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}

CORE = [
    _t("get_order_status", "Consulta el estado de un pedido por su numero.",
       {"order_id": {"type": "string", "description": "Numero de pedido, p.ej. SK-10492"}},
       ["order_id"]),
    _t("create_shipping_label", "Crea una etiqueta de envio para un pedido.",
       {"order_id": {"type": "string"},
        "weight_kg": {"type": "number", "description": "Peso en kilogramos"},
        "carrier": {"type": "string", "enum": ["correos", "seur", "gls"]}},
       ["order_id", "weight_kg", "carrier"]),
    _t("get_stock", "Devuelve el stock disponible de un SKU.",
       {"sku": {"type": "string"}}, ["sku"]),
]

DISTRACTORS = [
    _t("get_invoice_pdf", "Descarga el PDF de una factura emitida.",
       {"invoice_id": {"type": "string"}}, ["invoice_id"]),
    _t("cancel_order", "Cancela un pedido que no ha salido de almacen.",
       {"order_id": {"type": "string"}}, ["order_id"]),
    _t("get_customer", "Ficha de un cliente por email.",
       {"email": {"type": "string"}}, ["email"]),
    _t("list_returns", "Lista devoluciones abiertas en un rango de fechas.",
       {"since": {"type": "string"}, "until": {"type": "string"}}, ["since"]),
    _t("update_price", "Cambia el precio de venta de un SKU.",
       {"sku": {"type": "string"}, "price_eur": {"type": "number"}}, ["sku", "price_eur"]),
    _t("send_email", "Envia un correo a un cliente.",
       {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
       ["to", "subject", "body"]),
    _t("get_shipment_tracking", "Estado de seguimiento de un envio ya creado.",
       {"tracking_number": {"type": "string"}}, ["tracking_number"]),
    _t("reserve_stock", "Reserva unidades de un SKU para un pedido.",
       {"sku": {"type": "string"}, "qty": {"type": "integer"}}, ["sku", "qty"]),
]


def call(base: str, messages: list, tools: list, timeout: float = 900.0) -> dict:
    payload = {"model": MODEL, "messages": messages, "tools": tools,
               "tool_choice": "auto", "temperature": TEMP, "top_p": TOP_P,
               "max_tokens": 1500, "stream": False}
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tool_calls(resp: dict) -> list:
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    return msg.get("tool_calls") or []


def text_of(resp: dict) -> str:
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("content") or "").strip()


def parse_args_of(tc: dict) -> dict:
    try:
        a = tc["function"]["arguments"]
        return json.loads(a) if isinstance(a, str) else (a or {})
    except Exception:  # noqa: BLE001
        return {}


def phase(n: str, ok: bool, detail: str, results: list) -> None:
    print(f"  [{'OK  ' if ok else 'FALLO'}] {n}: {detail}")
    results.append({"case": n, "ok": ok, "detail": detail})


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="bench_tooling_results.json")
    a = ap.parse_args()
    R: list = []

    # ── 1 seleccion ───────────────────────────────────────────────────────────
    print("\n=== 1. seleccion de tool ===")
    for name, prompt, want_tool, check in [
        ("sel-order", "Que sabes del pedido SK-10492?", "get_order_status",
         lambda ar: ar.get("order_id") == "SK-10492"),
        ("sel-stock", "Cuantas unidades quedan del SKU BB-3300?", "get_stock",
         lambda ar: ar.get("sku") == "BB-3300"),
    ]:
        try:
            r = call(a.base, [{"role": "user", "content": prompt}], CORE)
        except Exception as e:  # noqa: BLE001
            phase(name, False, f"excepcion: {e}", R); continue
        tc = tool_calls(r)
        if len(tc) != 1:
            phase(name, False, f"esperaba 1 tool_call, hubo {len(tc)}", R); continue
        got, ar = tc[0]["function"]["name"], parse_args_of(tc[0])
        ok = got == want_tool and check(ar)
        phase(name, ok, f"tool={got} args={ar}", R)

    # ── 2 distractores ────────────────────────────────────────────────────────
    print("\n=== 2. resistencia a distractores (3 utiles + 8 senuelos) ===")
    for name, prompt, want_tool, check in [
        ("dis-order", "Necesito saber en que anda el pedido SK-77120.", "get_order_status",
         lambda ar: ar.get("order_id") == "SK-77120"),
        ("dis-label", "Crea la etiqueta del pedido SK-88001, pesa 2.4 kg, mandalo por SEUR.",
         "create_shipping_label",
         lambda ar: ar.get("order_id") == "SK-88001"
                    and abs(float(ar.get("weight_kg", 0)) - 2.4) < 1e-6
                    and str(ar.get("carrier", "")).lower() == "seur"),
    ]:
        try:
            r = call(a.base, [{"role": "user", "content": prompt}], CORE + DISTRACTORS)
        except Exception as e:  # noqa: BLE001
            phase(name, False, f"excepcion: {e}", R); continue
        tc = tool_calls(r)
        if len(tc) != 1:
            phase(name, False, f"esperaba 1 tool_call, hubo {len(tc)}: "
                               f"{[c['function']['name'] for c in tc]}", R); continue
        got, ar = tc[0]["function"]["name"], parse_args_of(tc[0])
        ok = got == want_tool and check(ar)
        phase(name, ok, f"tool={got} args={ar}", R)

    # ── 3 cadena multi-paso ───────────────────────────────────────────────────
    # get_stock -> create_shipping_label con el peso que devuelve el paso 1.
    print("\n=== 3. cadena multi-paso ===")
    try:
        msgs = [{"role": "user", "content":
                 "Para el pedido SK-90210: mira el stock del SKU AR-15X y, si hay "
                 "existencias, crea la etiqueta por GLS usando el peso unitario que "
                 "te devuelva la consulta de stock."}]
        r1 = call(a.base, msgs, CORE + DISTRACTORS)
        tc1 = tool_calls(r1)
        if len(tc1) != 1 or tc1[0]["function"]["name"] != "get_stock":
            phase("chain-paso1", False,
                  f"esperaba get_stock, hubo {[c['function']['name'] for c in tc1]}", R)
        else:
            phase("chain-paso1", parse_args_of(tc1[0]).get("sku") == "AR-15X",
                  f"args={parse_args_of(tc1[0])}", R)
            msgs.append((r1["choices"][0].get("message")))
            msgs.append({"role": "tool", "tool_call_id": tc1[0].get("id"),
                         "content": json.dumps({"sku": "AR-15X", "available": 14,
                                                "unit_weight_kg": 3.15})})
            r2 = call(a.base, msgs, CORE + DISTRACTORS)
            tc2 = tool_calls(r2)
            if len(tc2) != 1 or tc2[0]["function"]["name"] != "create_shipping_label":
                phase("chain-paso2", False,
                      f"esperaba create_shipping_label, hubo "
                      f"{[c['function']['name'] for c in tc2]}", R)
            else:
                ar = parse_args_of(tc2[0])
                # El peso TIENE que venir del paso 1. Si lo inventa, es fallo.
                ok = (ar.get("order_id") == "SK-90210"
                      and abs(float(ar.get("weight_kg", 0)) - 3.15) < 1e-6
                      and str(ar.get("carrier", "")).lower() == "gls")
                phase("chain-paso2", ok, f"args={ar} (weight debe ser 3.15 del paso 1)", R)
    except Exception as e:  # noqa: BLE001
        phase("chain", False, f"excepcion: {e}", R)

    # ── 4 rechazo de imposible ────────────────────────────────────────────────
    # No hay tool de reembolsos. Debe NO llamar nada y decir que no puede.
    print("\n=== 4. rechazo de peticion imposible ===")
    try:
        # Lo que importa NO es que se abstenga de llamar en el primer turno:
        # consultar el pedido antes de decir "no puedo" es razonable. Importa el
        # turno FINAL: que admita la carencia y que NO afirme falsamente haber
        # hecho el reembolso. La version anterior puntuaba el turno 1 y suspendia
        # al modelo por buscar informacion primero — fallo del test, no del modelo.
        msgs = [{"role": "user", "content":
                 "Emite un reembolso de 40 EUR al pedido SK-10492 y "
                 "confirmame que el dinero ha salido."}]
        r = call(a.base, msgs, CORE + DISTRACTORS)
        tc = tool_calls(r)
        if tc:
            msgs.append((r.get("choices") or [{}])[0].get("message"))
            msgs.append({"role": "tool", "tool_call_id": tc[0].get("id"),
                         "content": json.dumps({"order_id": "SK-10492",
                                                "status": "entregado",
                                                "total_eur": 40.0})})
            r = call(a.base, msgs, CORE + DISTRACTORS)
        txt = text_of(r)
        low = txt.lower()
        admits = any(k in low for k in (
            "no puedo", "no tengo", "no dispongo", "no existe", "ninguna herramienta",
            "no hay", "no esta disponible", "no está disponible", "cannot", "unable"))
        # El fallo grave seria inventarse que el reembolso se hizo.
        claims = any(k in low for k in (
            "reembolso emitido", "dinero ha salido", "he emitido el reembolso",
            "reembolsado correctamente", "he procesado el reembolso"))
        phase("reject-refund", admits and not claims,
              f"admite={admits} afirma_falsamente={claims}; texto={txt[:130]!r}", R)
    except Exception as e:  # noqa: BLE001
        phase("reject-refund", False, f"excepcion: {e}", R)

    # ── 5 recuperacion tras error ─────────────────────────────────────────────
    # OJO con el diseno de este caso. La primera version pedia la etiqueta "por
    # MRW", esperando que el modelo llamase con un carrier fuera del enum y luego
    # se recuperase del error. El modelo hace algo MEJOR: se niega a llamar con un
    # valor invalido, lista los tres validos y pregunta. El test lo suspendia por
    # acertar. Para medir recuperacion de verdad hay que partir de una llamada
    # VALIDA y que sea la tool la que falle por algo transitorio.
    print("\n=== 5. recuperacion tras error de tool ===")
    try:
        msgs = [{"role": "user", "content":
                 "Crea la etiqueta del pedido SK-55555, 1.2 kg, por SEUR."}]
        r1 = call(a.base, msgs, CORE)
        tc1 = tool_calls(r1)
        if len(tc1) != 1:
            phase("recover", False, f"paso 1 no llamo tool ({len(tc1)} calls)", R)
        else:
            first = parse_args_of(tc1[0])
            msgs.append(r1["choices"][0].get("message"))
            msgs.append({"role": "tool", "tool_call_id": tc1[0].get("id"),
                         "content": json.dumps({
                             "error": "carrier_unavailable",
                             "message": "SEUR no admite recogidas hoy; validos ahora: correos, gls"})})
            r2 = call(a.base, msgs, CORE)
            tc2, txt2 = tool_calls(r2), text_of(r2)
            if tc2:
                ar = parse_args_of(tc2[0])
                c = str(ar.get("carrier", "")).lower()
                # Reintentar con SEUR otra vez seria no haber leido el error.
                ok = (tc2[0]["function"]["name"] == "create_shipping_label"
                      and c in {"correos", "gls"}
                      and ar.get("order_id") == "SK-55555")
                phase("recover", ok,
                      f"reintento con carrier={c!r} args={ar} (primero fue {first.get('carrier')!r})", R)
            else:
                # Preguntar cual de los dos que SIGUEN validos tambien vale: crear
                # una etiqueta tiene efectos, preguntar es defendible.
                low = txt2.lower()
                ok = ("correos" in low or "gls" in low)
                phase("recover", ok, f"no reintento, ofrece alternativa; texto={txt2[:140]!r}", R)
    except Exception as e:  # noqa: BLE001
        phase("recover", False, f"excepcion: {e}", R)

    # ── veredicto ─────────────────────────────────────────────────────────────
    passed = sum(1 for x in R if x["ok"])
    total = len(R)
    print(f"\n=== SCORE TOOL-CALLING: {passed}/{total} ===")
    by = {}
    for x in R:
        fam = x["case"].split("-")[0]
        by.setdefault(fam, [0, 0])
        by[fam][1] += 1
        by[fam][0] += 1 if x["ok"] else 0
    for fam, (p, t) in by.items():
        print(f"  {fam:10} {p}/{t}")
    out = {"model": MODEL, "temperature": TEMP, "top_p": TOP_P,
           "score": f"{passed}/{total}", "passed": passed, "total": total,
           "by_family": {k: f"{v[0]}/{v[1]}" for k, v in by.items()}, "cases": R}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"escrito {a.out}")
    # Fallar en duro si no esta impecable: es la puerta antes de enrutar trafico.
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
