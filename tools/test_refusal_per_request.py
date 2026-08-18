#!/usr/bin/env python3
"""Tests de la seleccion de lambda POR PETICION. Se corre DENTRO de la imagen.

Importa de `vllm.*` instalado (no de /work) a proposito: `RefusalState` hace
`from vllm import refusal_projection`, asi que probar la copia suelta no probaria
nada de lo que corre en el pod.

QUE TIENE QUE PODER FALLAR. La version anterior de este mecanismo estaba muerta
en silencio y ningun test lo delataba. Cada bloque de aqui esta escrito para
FALLAR si se revierte una pieza concreta del arreglo:

  bloque C  -> falla si los buffers se crean perezosamente o si el forward deja
               de leer el buffer persistente (el defecto de la captura de grafos:
               en replay no corre Python y quedaba horneado el escalar global).
  bloque D  -> falla si target y drafter comparten buffer (layouts disjuntos:
               multiplos de 1+k frente a multiplos de k).
  bloque B4 -> falla si un lote que desborda el buffer recorta datos en vez de
               caer al global: eso aplicaria el lambda de OTRA peticion.
  bloque B5 -> falla si un dummy run deja el buffer rancio del paso anterior.
"""
import os
import sys

os.environ.setdefault("VLLM_REFUSAL_DIRS", "/opt/refusal/refusal_dirs.safetensors")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vllm.refusal_projection as rp  # noqa: E402
from vllm.v1.worker.gpu.refusal_utils import RefusalState  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    if not cond:
        fails.append(name)


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = rp.get_dirs()
assert d is not None and len(d) == 46, "sin direcciones no hay nada que probar"
R0 = d["layers.0.attn.wo_b"]


def reset_buffers(max_num_tokens=64, max_num_reqs=3):
    """Estado limpio: RefusalState crea los buffers en su __init__."""
    rp._buf[:] = [None] * len(rp._buf)
    rp.set_lambda(0.0)
    return RefusalState(
        max_num_reqs=max_num_reqs, max_num_tokens=max_num_tokens, device=DEV
    )


# --- A. layout de filas por rol ---------------------------------------------
# El defecto 1: un unico vector por token no puede servir al target y al drafter
# porque avanzan en multiplos distintos (1+k frente a k) y son disjuntos.

st = reset_buffers()
st.add_request(0, 1.0)      # peticion con cache_salt refusal:1.0
st.add_request(1, None)     # peticion normal -> global
st.add_request(2, 0.5)

idx = np.array([0, 1, 2], dtype=np.int32)
nst = np.array([4, 2, 1], dtype=np.int32)   # tokens programados por peticion
st.fill_target(idx, nst, global_lambda=0.0)
tgt = rp.view_for(rp.ROLE_TARGET, 7).cpu().numpy()
check(
    "target: fila i <-> token i (repeat por num_scheduled_tokens)",
    np.array_equal(tgt, np.array([1, 1, 1, 1, 0, 0, 0.5], dtype=np.float32)),
    f"got={tgt.tolist()}",
)

st.fill_draft(idx, num_query_per_req=5, global_lambda=0.0)
drf = rp.view_for(rp.ROLE_DRAFT, 15).cpu().numpy()
check(
    "draft: fila i <-> (peticion i//q, paso i%q), repeat CONSTANTE",
    np.array_equal(drf, np.repeat([1.0, 0.0, 0.5], 5).astype(np.float32)),
    f"got={drf.tolist()}",
)

# El target NO se ve tocado por el relleno del drafter: son buffers distintos.
tgt2 = rp.view_for(rp.ROLE_TARGET, 7).cpu().numpy()
check("rellenar draft no pisa el buffer del target", np.array_equal(tgt, tgt2))

# NaN (peticion sin salt) se resuelve al global VIGENTE, no a 0 fijo.
st.fill_target(idx, nst, global_lambda=2.5)
tgt3 = rp.view_for(rp.ROLE_TARGET, 7).cpu().numpy()
check("peticion sin salt toma el lambda global vigente", float(tgt3[4]) == 2.5,
      f"got={tgt3[4]}")

# remove_request deja el slot limpio: una peticion nueva no hereda el modo.
st.remove_request(0)
st.fill_target(np.array([0], dtype=np.int32), np.array([1], dtype=np.int32), 0.0)
check("remove_request limpia el slot (no hereda el lambda anterior)",
      float(rp.view_for(rp.ROLE_TARGET, 1)[0]) == 0.0)


# --- B. semantica del buffer -------------------------------------------------

st = reset_buffers(max_num_tokens=64)
buf_ptr = rp._buf[rp.ROLE_TARGET].data_ptr()

# B1. mutacion IN-PLACE. Es la premisa entera del arreglo: el grafo hornea el
# puntero, asi que reasignar el tensor en vez de mutarlo lo dejaria leyendo
# memoria vieja para siempre.
st.add_request(0, 1.0)
st.fill_target(np.array([0], dtype=np.int32), np.array([3], dtype=np.int32), 0.0)
check("fill muta in-place (mismo data_ptr)",
      rp._buf[rp.ROLE_TARGET].data_ptr() == buf_ptr)

# B2. la vista comparte storage y arranca en offset 0 (invariante para cualquier n)
v = rp.view_for(rp.ROLE_TARGET, 3)
check("view_for comparte storage con el buffer", v.data_ptr() == buf_ptr)
check("view_for tiene la longitud pedida", v.shape == (3,), f"{tuple(v.shape)}")

# B3. el padding lleva el lambda global -> cualquier n >= n_real es correcta.
# Esto es lo que hace que el arreglo sobreviva al padding de los grafos CUDA.
pad = rp.view_for(rp.ROLE_TARGET, 8).cpu().numpy()
check("filas de padding al lambda global",
      np.array_equal(pad, np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)),
      f"got={pad.tolist()}")

# B4. desborde -> fail-safe al global. NUNCA recortar: recortar significaria
# aplicar a una peticion el lambda de otra.
st.add_request(1, 1.0)
st.fill_target(np.array([1], dtype=np.int32), np.array([999], dtype=np.int32), 0.0)
over = rp._buf[rp.ROLE_TARGET].cpu().numpy()
check("desborde -> buffer entero al global (fail-safe)", np.all(over == 0.0))
check("view_for por encima de la capacidad -> None (cae al global)",
      rp.view_for(rp.ROLE_TARGET, 999) is None)

# B5. fill_neutral limpia el estado rancio de un dummy run.
st.fill_target(np.array([1], dtype=np.int32), np.array([3], dtype=np.int32), 0.0)
check("previo a fill_neutral el buffer tiene datos reales",
      float(rp.view_for(rp.ROLE_TARGET, 1)[0]) == 1.0)
st.fill_neutral(global_lambda=0.0)
check("fill_neutral deja los DOS buffers al global",
      float(rp._buf[rp.ROLE_TARGET].max()) == 0.0
      and float(rp._buf[rp.ROLE_DRAFT].max()) == 0.0)


# --- C. supervivencia al replay de grafo CUDA -------------------------------
# ESTE es el bloque que importa. El defecto 2 era invisible: `capture_model` no
# pasa por `execute_model`, asi que en la captura el slot por token era None y lo
# que quedaba trazado DENTRO del grafo era el escalar global. En replay no corre
# ni una linea de Python -> el decode servido por grafo aplicaba el global para
# siempre, sin un solo aviso.

if DEV.type != "cuda":
    print("SKIP  bloque C (replay de grafo CUDA) — no hay GPU en este contenedor")
else:
    st = reset_buffers(max_num_tokens=64)
    st.add_request(0, 0.0)
    N = 6
    mod = rp.RefusalProjection(R0, rp.ROLE_TARGET).to(DEV)
    y = torch.randn(N, 4096, dtype=torch.bfloat16, device=DEV)
    r64 = R0.double().to(DEV)

    def ref(lam_vec):
        lv = torch.tensor(lam_vec, dtype=torch.float64, device=DEV).unsqueeze(-1)
        return y.double() - lv * (y.double() @ r64).unsqueeze(-1) * r64

    def rel_err(out, want):
        return float(
            ((out.double() - want).norm(dim=1) / want.norm(dim=1).clamp_min(1e-30)).max()
        )

    # Captura con el buffer NEUTRO, igual que hara capture_model.
    st.fill_neutral(global_lambda=0.0)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            mod(y)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    static_out = None
    with torch.cuda.graph(g):
        static_out = mod(y)

    # Ahora, SIN volver a capturar, se escribe el lambda por peticion y se
    # reproduce el grafo. Si el arreglo esta puesto, el kernel lee el buffer
    # mutado; si se revierte, lee el escalar horneado y esto falla.
    st.add_request(0, 1.0)
    st.fill_target(np.array([0], dtype=np.int32), np.array([N], dtype=np.int32), 0.0)
    g.replay()
    torch.cuda.synchronize()
    e1 = rel_err(static_out, ref([1.0] * N))
    check("replay ve el lambda por peticion escrito DESPUES de capturar",
          e1 < 5e-3, f"err_max={e1:.3e}")

    # Y el control que convierte esto en una prueba de verdad: cambiar solo el
    # contenido del buffer tiene que cambiar la salida del MISMO grafo.
    out_lam1 = static_out.clone()
    st.add_request(0, 0.0)
    st.fill_target(np.array([0], dtype=np.int32), np.array([N], dtype=np.int32), 0.0)
    g.replay()
    torch.cuda.synchronize()
    check("mismo grafo, lambda distinto -> salida distinta",
          not torch.equal(static_out, out_lam1))
    check("replay con lambda 0 devuelve y intacto", torch.equal(static_out, y))

    # Lote MIXTO bajo replay: dos peticiones en el mismo grafo, lambdas distintos.
    st2 = RefusalState(max_num_reqs=3, max_num_tokens=64, device=DEV)
    st2.add_request(0, 1.0)
    st2.add_request(1, 0.0)
    st2.fill_target(
        np.array([0, 1], dtype=np.int32), np.array([3, 3], dtype=np.int32), 0.0
    )
    g.replay()
    torch.cuda.synchronize()
    e2 = rel_err(static_out, ref([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    check("lote mixto bajo replay: cada peticion con SU lambda",
          e2 < 5e-3, f"err_max={e2:.3e}")
    check("en lote mixto las filas con lambda=0 salen intactas",
          torch.equal(static_out[3:], y[3:]))


# --- D. aislamiento de roles -------------------------------------------------
# Un modulo del drafter no puede leer nunca el buffer del target. El rol se fija
# en la CONSTRUCCION por prefijo, con el mismo criterio que elige la direccion.

NHL = 43
_, role_bb = rp.resolve("model.layers.0.attn", NHL)
_, role_df = rp.resolve("model.layers.43.attn", NHL)   # drafter DSpark
_, role_mtp = rp.resolve("model.mtp.1.attn", NHL)
check("backbone -> ROLE_TARGET", role_bb == rp.ROLE_TARGET)
check("drafter DSpark (layers.43) -> ROLE_DRAFT", role_df == rp.ROLE_DRAFT)
check("ruta MTP -> ROLE_DRAFT", role_mtp == rp.ROLE_DRAFT)

st = reset_buffers(max_num_tokens=64)
st.add_request(0, 1.0)
st.fill_target(np.array([0], dtype=np.int32), np.array([4], dtype=np.int32), 0.0)
st.fill_draft(np.array([0], dtype=np.int32), num_query_per_req=5, global_lambda=0.0)

y_t = torch.randn(4, 4096, dtype=torch.bfloat16, device=DEV)
mod_t = rp.RefusalProjection(R0, rp.ROLE_TARGET).to(DEV)
mod_d = rp.RefusalProjection(R0, rp.ROLE_DRAFT).to(DEV)
check("modulo target lee el buffer del target (proyecta)",
      not torch.equal(mod_t(y_t), y_t))
# El drafter tiene lambda 1.0 tambien: lo que se comprueba es que cada uno lee
# SU buffer, asi que se invierte uno de los dos y se mira que solo cambie ese.
st.fill_draft(np.array([0], dtype=np.int32), num_query_per_req=5, global_lambda=0.0)
st.lambdas[0] = 0.0
st.fill_draft(np.array([0], dtype=np.int32), num_query_per_req=5, global_lambda=0.0)
check("poner el draft a 0 NO cambia lo que ve el target",
      not torch.equal(mod_t(y_t), y_t))
y_d = torch.randn(5, 4096, dtype=torch.bfloat16, device=DEV)
check("el modulo draft ve SU lambda (0) y deja y intacto",
      torch.equal(mod_d(y_d), y_d))

rp.set_lambda(0.0)
print()
print("TODOS OK" if not fails else "FALLOS: " + ", ".join(fails))
sys.exit(1 if fails else 0)
