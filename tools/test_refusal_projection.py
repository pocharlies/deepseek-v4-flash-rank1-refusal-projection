#!/usr/bin/env python3
"""Tests del modulo de proyeccion. Se corre DENTRO de la imagen (necesita vllm.logger)."""
import os
import sys

import torch

sys.path.insert(0, "/work")
os.environ["VLLM_REFUSAL_DIRS"] = "/work/refusal_dirs.safetensors"
import refusal_projection as rp  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    if not cond:
        fails.append(name)


# --- carga
d = rp.get_dirs()
check("carga 46 direcciones", len(d) == 46, f"n={len(d)}")
check("todas [4096] f32", all(t.shape == (4096,) and t.dtype == torch.float32 for t in d.values()))
check("todas norma 1", all(abs(float(t.norm()) - 1.0) < 1e-5 for t in d.values()))

# --- resolucion de prefijos (num_hidden_layers=43)
NHL = 43
cases = [
    ("model.layers.0.attn", "layers.0.attn.wo_b"),
    ("model.layers.42.attn", "layers.42.attn.wo_b"),
    ("model.layers.43.attn", "mtp.0.attn.wo_b"),   # drafter DSpark
    ("model.layers.45.attn", "mtp.2.attn.wo_b"),   # drafter DSpark
    ("model.mtp.1.attn", "mtp.1.attn.wo_b"),       # ruta MTP
]
for prefix, want in cases:
    got = rp.resolve_direction(prefix, NHL)
    ok = got is not None and torch.equal(got, d[want])
    check(f"resolve {prefix} -> {want}", ok)

check("prefijo desconocido -> None", rp.resolve_direction("model.embed_tokens", NHL) is None)

# --- matematica del hook contra referencia float64
torch.manual_seed(0)
H, T = 4096, 64
r = d["layers.0.attn.wo_b"]
mod = rp.RefusalProjection(r)
y = torch.randn(T, H, dtype=torch.bfloat16)

for lam in (0.0, 0.5, 1.0, 2.43):
    rp.set_lambda(lam)
    out = mod(y)
    ref = y.double() - lam * (y.double() @ r.double()).unsqueeze(-1) * r.double()
    err = float(((out.double() - ref).norm(dim=-1) / ref.norm(dim=-1)).median())
    check(f"hook lam={lam}", err < 5e-3, f"err_rel_mediano={err:.3e}")

# --- lam=0 bit-exacto (guardarrail)
rp.set_lambda(0.0)
check("lam=0 bit-exacto", torch.equal(mod(y), y))

# --- r_hat TIENE que acabar en el device de y.
# Esto fallo en produccion el 12-08-2026: r_hat nace en CPU (torch.frombuffer) y
# vLLM construye el modelo en un contexto de device en vez de llamar a .to(), asi
# que el buffer se quedaba en CPU y el forward petaba con "Expected all tensors to
# be on the same device". En CPU esta comprobacion es trivial; con GPU es la de
# verdad. Se deja aqui para fijar el invariante.
mod_dev = rp.RefusalProjection(d["layers.0.attn.wo_b"])
_ = mod_dev(y)
check("r_hat en el device de y", mod_dev.r_hat.device == y.device,
      f"{mod_dev.r_hat.device} vs {y.device}")
check("r_hat en float32 tras el forward", mod_dev.r_hat.dtype == torch.float32)
if torch.cuda.is_available():
    ycu = y.cuda()
    mod_cu = rp.RefusalProjection(d["layers.0.attn.wo_b"])
    out_cu = mod_cu(ycu)
    check("forward con y en CUDA y r_hat en CPU", out_cu.device.type == "cuda")
    check("r_hat migrado a CUDA", mod_cu.r_hat.device.type == "cuda")
else:
    print("SKIP  test de device mixto — no hay GPU en este contenedor")

# --- lam es tensor en device y se muta IN-PLACE (critico para cuda graphs)
rp.set_lambda(1.0)
t1 = mod._lam
addr1 = t1.data_ptr()
rp.set_lambda(0.25)
check("lam es tensor", isinstance(t1, torch.Tensor))
check("lam mutado in-place (mismo data_ptr)", mod._lam.data_ptr() == addr1)
check("lam refleja el valor nuevo", abs(float(mod._lam) - 0.25) < 1e-9)

# --- el tensor de lam se COMPARTE entre capas
mod2 = rp.RefusalProjection(d["layers.5.attn.wo_b"])
mod2(y)
check("lam compartido entre capas", mod2._lam.data_ptr() == addr1)

# --- lam TIENE que sobrevivir a nacer dentro de inference_mode.
# Esto rompio en produccion el 12-08-2026: el tensor se crea perezosamente en el
# primer forward, que corre bajo el torch.inference_mode() de vLLM. Un tensor
# nacido ahi es un *inference tensor* y no admite fill_() despues -> el dial se
# queda clavado y /admin/refusal_lambda devuelve 500. Se reproduce en CPU.
rp._lam_by_device.clear()
rp._lam_value = 0.0
mod3 = rp.RefusalProjection(d["layers.1.attn.wo_b"])
with torch.inference_mode():
    _ = mod3(y)  # aqui nace el tensor de lam
try:
    rp.set_lambda(0.75)
    ok_inf = abs(float(mod3._lam) - 0.75) < 1e-9
    extra = ""
except RuntimeError as e:
    ok_inf, extra = False, str(e)[:80]
check("lam mutable tras nacer en inference_mode", ok_inf, extra)

# y el forward tiene que seguir dando el resultado correcto con ese lam
with torch.inference_mode():
    out_inf = mod3(y)
ref_inf = y.double() - 0.75 * (y.double() @ d["layers.1.attn.wo_b"].double()).unsqueeze(-1) * d["layers.1.attn.wo_b"].double()
err_inf = float(((out_inf.double() - ref_inf).norm(dim=-1) / ref_inf.norm(dim=-1)).median())
check("forward correcto bajo inference_mode", err_inf < 5e-3, f"err={err_inf:.3e}")
rp.set_lambda(0.0)

# --- clave de hash del prefix cache
rp.set_lambda(0.0)
k0 = rp.lambda_hash_key()
rp.set_lambda(1.0)
k1 = rp.lambda_hash_key()
check("hash key distinta por lam", k0 != k1, f"{k0} vs {k1}")
check("hash key entera y estable", isinstance(k1, int) and k1 == 1000, f"k1={k1}")

# --- desactivado por env var: sin hook y sin clave
del os.environ["VLLM_REFUSAL_DIRS"]
rp._dirs = None
check("desactivado -> is_enabled False", not rp.is_enabled())
check("desactivado -> hash key None", rp.lambda_hash_key() is None)
check("desactivado -> resolve None", rp.resolve_direction("model.layers.0.attn", NHL) is None)


# --- seleccion por peticion: parseo del cache_salt
rp.set_lambda(0.0)
casos = [("refusal:1.0",1.0),("refusal:0.75",0.75),("refusal:0",0.0),
         ("refusal:-1",-1.0),(None,None),("",None),("otro-salt",None),
         ("refusal:abc",None),("refusal:",None)]
for salt,esperado in casos:
    got = rp.parse_request_lambda(salt)
    check(f"parse_request_lambda({salt!r}) -> {esperado}", got == esperado, f"got={got}")

# --- lambda POR TOKEN: el hook debe usarlo cuando esta puesto
mod4 = rp.RefusalProjection(d["layers.2.attn.wo_b"])
r4 = d["layers.2.attn.wo_b"].double()
y4 = torch.randn(6, 4096, dtype=torch.bfloat16)
rp.set_lambda(0.0)                      # el global es 0
lam_tok = torch.tensor([0.,1.,0.,1.,0.,1.], dtype=torch.float32)
rp.set_per_token_lambda(lam_tok)
out4 = mod4(y4)
ref4 = y4.double() - lam_tok.double().unsqueeze(-1) * (y4.double() @ r4).unsqueeze(-1) * r4
err4 = float(((out4.double()-ref4).norm(dim=-1)/ref4.norm(dim=-1).clamp_min(1e-30)).max())
check("hook usa lambda POR TOKEN", err4 < 5e-3, f"err_max={err4:.3e}")
# las filas con lambda=0 tienen que salir intactas aunque otras del lote lleven 1
check("filas con lambda=0 intactas en lote mixto",
      torch.equal(out4[0], y4[0]) and torch.equal(out4[2], y4[2]) and torch.equal(out4[4], y4[4]))
check("filas con lambda=1 SI cambian", not torch.equal(out4[1], y4[1]))
# longitud que no cuadra -> se ignora y se cae al global (no debe petar)
rp.set_per_token_lambda(torch.tensor([1.,0.], dtype=torch.float32))
check("longitud incongruente -> cae al global sin petar", torch.equal(mod4(y4), y4))
rp.set_per_token_lambda(None)
check("sin per-token vuelve al global", torch.equal(mod4(y4), y4))

print()
print(f"{'TODOS OK' if not fails else 'FALLOS: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
