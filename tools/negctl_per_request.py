"""Control negativo FUERTE: misma API nueva, comportamiento VIEJO.

El control anterior solo probaba que falta un simbolo. Este reproduce el
DEFECTO 2 de verdad -- los buffers no existen cuando se captura el grafo, que
es exactamente lo que pasaba con `capture_model` -- y comprueba que el bloque C
lo DETECTA. Si esto saliera "PASS", el bloque C no valdria para nada.
"""
import os, sys
os.environ.setdefault("VLLM_REFUSAL_DIRS", "/opt/refusal/refusal_dirs.safetensors")
import numpy as np, torch
import vllm.refusal_projection as rp
from vllm.v1.worker.gpu.refusal_utils import RefusalState

if not torch.cuda.is_available():
    print("SIN GPU: este control no vale"); sys.exit(2)

DEV = torch.device("cuda")
R0 = rp.get_dirs()["layers.0.attn.wo_b"]
N = 6

# --- VIEJO: buffers AUSENTES durante la captura (creacion perezosa) ---
rp._buf[:] = [None] * len(rp._buf)
rp.set_lambda(0.0)
mod = rp.RefusalProjection(R0, rp.ROLE_TARGET).to(DEV)
y = torch.randn(N, 4096, dtype=torch.bfloat16, device=DEV)

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): mod(y)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = mod(y)                      # aqui view_for -> None: se hornea el ESCALAR

# ahora si se crean y se escribe lambda=1.0 por peticion
st = RefusalState(max_num_reqs=3, max_num_tokens=64, device=DEV)
st.add_request(0, 1.0)
st.fill_target(np.array([0], dtype=np.int32), np.array([N], dtype=np.int32), 0.0)
g.replay(); torch.cuda.synchronize()

cambio = not torch.equal(out, y)
print(f"replay refleja el lambda por peticion: {cambio}")
if cambio:
    print("FAIL  el bloque C NO discrimina: pasa igual con el defecto puesto")
    sys.exit(1)
print("OK    el bloque C SI detecta el defecto 2 (buffer ausente en captura)")
sys.exit(0)
