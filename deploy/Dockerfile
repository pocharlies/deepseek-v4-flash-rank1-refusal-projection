# Imagen con la proyeccion rank-1 en runtime.
#
# Base intacta: mismo runtime que ya corre en produccion. Lo unico que cambia es
# el parche de vLLM y las 750 KB de direcciones. Los 48 shards del checkpoint NO
# se tocan: siguen viniendo del cache HF por hostPath, byte-identicos al release
# de DeepSeek.
#
# El hook nace INERTE: sin VLLM_REFUSAL_DIRS no se construye el modulo, no hay
# rama extra en el forward y no se monta /admin/refusal_lambda. Para activarlo
# hay que poner la variable en el deployment; para desactivarlo, quitarla. Sin
# rebuild en ninguno de los dos sentidos.
ARG BASE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1
FROM ${BASE}

ARG VLLM_DIR=/usr/local/lib/python3.12/dist-packages/vllm

COPY 0001-rank1-projection.patch /tmp/rank1.patch
COPY refusal_dirs.safetensors /opt/refusal/refusal_dirs.safetensors

RUN cd ${VLLM_DIR} \
 && patch -p1 < /tmp/rank1.patch \
 && rm /tmp/rank1.patch \
 && python3 -m py_compile \
      refusal_projection.py \
      models/deepseek_v4/attention.py \
      v1/core/kv_cache_utils.py \
      v1/worker/gpu_worker.py \
      entrypoints/serve/__init__.py \
      entrypoints/serve/refusal/api_router.py \
 && echo "parche aplicado y compilado"

# Verificacion en build: la cadena de imports tiene que levantar, y con el hook
# DESACTIVADO no debe existir ni la ruta admin. Si esto falla, no hay imagen.
RUN cd /tmp && python3 -c "\
import vllm; \
from vllm.v1.core import kv_cache_utils; \
from vllm.v1.worker.gpu_worker import Worker; \
from vllm.models.deepseek_v4 import attention; \
from vllm.entrypoints.serve import register_vllm_serve_api_routers; \
import vllm.refusal_projection as rp; \
assert callable(Worker.set_refusal_lambda); \
assert callable(Worker.get_refusal_lambda); \
assert not rp.is_enabled(), 'el hook no debe estar activo por defecto'; \
assert rp.lambda_hash_key() is None; \
print('build-check OK', vllm.__version__)"

# Y con el hook ACTIVO: 46 direcciones y la clave de hash viva.
RUN cd /tmp && VLLM_REFUSAL_DIRS=/opt/refusal/refusal_dirs.safetensors python3 -c "\
import vllm.refusal_projection as rp; \
d = rp.get_dirs(); \
assert len(d) == 46, len(d); \
assert all(t.shape == (46*0+4096,) for t in d.values()); \
rp.set_lambda(1.0); \
assert rp.lambda_hash_key() == 1000; \
print('build-check hook OK: 46 direcciones')"

LABEL org.opencontainers.image.base.name="ghcr.io/anemll/dspark-vllm-gx10:0.1.1"
LABEL dgx-infra.rank1.patch="0001-rank1-projection"
LABEL dgx-infra.rank1.dirs="refusal_dirs.safetensors (46 modulos, 757712 bytes)"
