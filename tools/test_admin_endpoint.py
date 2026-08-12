#!/usr/bin/env python3
"""Tests del endpoint admin (3.7). Se corre DENTRO de la imagen, con el parche aplicado."""
import os
import sys

os.environ["VLLM_REFUSAL_DIRS"] = "/work/refusal_dirs.safetensors"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    if not cond:
        fails.append(name)


from vllm.entrypoints.serve.refusal import api_router as ar  # noqa: E402

# --- doble del engine: collective_rpc devuelve un valor por rank (TP=2)
class FakeEngine:
    def __init__(self, nranks=2):
        self.lam = 0.0
        self.nranks = nranks
        self.skew = False

    async def collective_rpc(self, method, args=(), **kw):
        if method == "set_refusal_lambda":
            self.lam = float(args[0])
            if self.skew:  # simula un rank que no aplico lo mismo
                return [self.lam] + [0.0] * (self.nranks - 1)
            return [self.lam] * self.nranks
        if method == "get_refusal_lambda":
            return [self.lam] * self.nranks
        raise AssertionError(method)


def build(enabled=True):
    if enabled:
        os.environ["VLLM_REFUSAL_DIRS"] = "/work/refusal_dirs.safetensors"
    else:
        os.environ.pop("VLLM_REFUSAL_DIRS", None)
    app = FastAPI()
    app.state.engine_client = FakeEngine()
    ar.attach_router(app)
    return app


# --- montaje condicional
app_off = build(enabled=False)
paths_off = {r.path for r in app_off.routes}
check("sin VLLM_REFUSAL_DIRS no se monta", "/admin/refusal_lambda" not in paths_off)

app = build(enabled=True)
paths = {r.path for r in app.routes}
check("con VLLM_REFUSAL_DIRS se monta", "/admin/refusal_lambda" in paths)

c = TestClient(app)

# --- GET inicial
r = c.get("/admin/refusal_lambda")
check("GET 200", r.status_code == 200, str(r.json()))
check("GET lambda inicial 0", r.json()["lambda"] == 0.0)
check("GET consistent", r.json()["consistent"] is True)

# --- POST valido
r = c.post("/admin/refusal_lambda", json={"lambda": 1.0})
check("POST lambda=1 -> 200", r.status_code == 200, str(r.json()))
check("POST devuelve ranks", r.json().get("ranks") == 2)
check("GET refleja el cambio", c.get("/admin/refusal_lambda").json()["lambda"] == 1.0)

# --- limites
for bad in (-5.0, 9.0):
    r = c.post("/admin/refusal_lambda", json={"lambda": bad})
    check(f"POST lambda={bad} rechazado", r.status_code == 422, f"status={r.status_code}")

r = c.post("/admin/refusal_lambda", json={"lambda": -1.0})
check("POST lambda=-1 aceptado (amplifica)", r.status_code == 200)

# --- ranks discrepantes -> 500, no 200
app.state.engine_client.skew = True
r = c.post("/admin/refusal_lambda", json={"lambda": 0.5})
check("ranks discrepantes -> 500", r.status_code == 500, f"status={r.status_code}")
check("500 explica el motivo", "no aplicaron el mismo" in str(r.json()))

# --- cuerpo mal formado
r = c.post("/admin/refusal_lambda", json={"lambdaa": 1.0})
check("cuerpo sin 'lambda' rechazado", r.status_code == 422)

print()
print(f"{'TODOS OK' if not fails else 'FALLOS: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
