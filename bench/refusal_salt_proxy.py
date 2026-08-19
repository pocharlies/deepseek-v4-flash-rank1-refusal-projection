#!/usr/bin/env python3
"""Proxy que sella cada peticion con `cache_salt: refusal:<lambda>`.

POR QUE EXISTE. Hasta el 18-08 medir calidad por lambda obligaba a mover el dial
GLOBAL (`POST /admin/refusal_lambda`), o sea a poner TODO el servidor en el
lambda del arm mientras duraba la medicion. Por eso el runner viejo serializa los
arms y restaura produccion en un `finally`: no habia otra forma.

Con el lambda por peticion vivo ya la hay. lm-eval solo deja apuntar a un
`base_url`, no tiene hook para meter campos en el body, asi que el sello se
inyecta aqui: lm-eval habla con este proxy y el proxy reenvia al head con el
`cache_salt` puesto. El dial global se queda en 0 toda la medicion.

Consecuencias practicas, que son el motivo de escribirlo:
  - Produccion NO se toca. Se puede medir con el modelo sirviendo trafico real.
  - Si el runner se muere a mitad no deja el servidor en lambda 2.5.
  - El arm queda auditado: /proxy/stats dice cuantas peticiones se sellaron, asi
    que "el sello no llego" es detectable en vez de silencioso — que es
    exactamente el fallo que se acaba de arreglar en el servidor.

Stdlib pura: esto corre dentro del venv de lm-eval y no se le anaden dependencias.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_stats_lock = threading.Lock()
_stats = {"seen": 0, "sealed": 0, "forward_errors": 0, "not_json": 0}


class Handler(BaseHTTPRequestHandler):
    upstream = ""
    salt = ""
    timeout = 1200.0

    def log_message(self, fmt, *a):  # silencio: lm-eval ya es ruidoso
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/proxy/stats":
            with _stats_lock:
                body = json.dumps({**_stats, "salt": self.salt}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # Passthrough de lectura: bench_niah y compania consultan rutas GET del
        # head (p.ej. /admin/refusal_lambda para dejar constancia del dial). Un
        # 404 aqui las rompia sin que el fallo se pareciera a su causa.
        self._forward("GET", None)

    def _forward(self, method: str, out: bytes | None) -> None:
        headers = {"Content-Type": "application/json"} if out else {}
        req = urllib.request.Request(
            self.upstream.rstrip("/") + self.path, data=out,
            headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
                code = resp.getcode()
                ctype = resp.headers.get("Content-Type", "application/json")
        except Exception as exc:  # noqa: BLE001
            with _stats_lock:
                _stats["forward_errors"] += 1
            payload = json.dumps({"error": {"message": str(exc)[:400]}}).encode()
            code, ctype = 502, "application/json"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        with _stats_lock:
            _stats["seen"] += 1

        # El sello se inyecta SIEMPRE que el body sea JSON. Si no lo fuera se
        # reenvia tal cual y se cuenta aparte: preferimos una medicion que se
        # sepa sin sellar a una que lo aparente.
        try:
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                raise ValueError("body no es un objeto")
            body["cache_salt"] = self.salt
            out = json.dumps(body).encode()
            with _stats_lock:
                _stats["sealed"] += 1
        except Exception:  # noqa: BLE001
            out = raw
            with _stats_lock:
                _stats["not_json"] += 1

        headers = {"Content-Type": "application/json"}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        req = urllib.request.Request(
            self.upstream.rstrip("/") + self.path, data=out,
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type",
                                 resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            with _stats_lock:
                _stats["forward_errors"] += 1
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:  # noqa: BLE001
            with _stats_lock:
                _stats["forward_errors"] += 1
            payload = json.dumps({"error": {"message": str(exc)[:400]}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, help="p.ej. http://host:8888")
    ap.add_argument("--lambda", dest="lam", type=float, required=True)
    ap.add_argument("--port", type=int, default=0, help="0 = puerto libre")
    ap.add_argument("--timeout", type=float, default=1200.0)
    args = ap.parse_args()

    Handler.upstream = args.upstream
    Handler.salt = f"refusal:{args.lam}"
    Handler.timeout = args.timeout
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    # El puerto real va a stdout en la PRIMERA linea: el runner lo lee de ahi
    # cuando pide 0, para no chocar con nada que ya escuche.
    print(srv.server_address[1], flush=True)
    print(f"[proxy] {args.upstream} sellando con {Handler.salt}", file=sys.stderr,
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
