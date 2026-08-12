# SPDX-License-Identifier: Apache-2.0
"""Proyeccion ortogonal rank-1 en runtime sobre la salida de los sublayers de atencion.

Equivalencia de la que parte todo (verificada a 9,3e-16 en la fase 2):

    (W - lam * r r^T W) x   ==   Wx - lam * r (r^T Wx)

o sea, editar el peso y proyectar la salida dan lo mismo -- pero proyectar la salida
no pasa por el round-trip de cuantizacion FP8, asi que lam=1 elimina la componente
limpiamente en vez de dejarla invertida.

MEDIDO sobre cebeuq/DeepSeek-V4-Flash-0731-abliterated (fase 1 y 2, 46 modulos):
  - el checkpoint horneado NO se queda en el 68%: elimina el ~240% de la direccion,
    o sea la invierte y la deja a ~1,4x en sentido contrario (lam_eff = 2,43).
  - lam=1 aqui deja la componente residual en 0 exacto (99,83% con activaciones bf16).
  - el rango 0 < lam <= 1 es territorio que ningun checkpoint horneado alcanza.

ACTIVACION. Sin la variable de entorno VLLM_REFUSAL_DIRS el hook no existe: no se
construye el modulo y el forward no lleva ni una rama extra. Ese es el rollback sin
rebuild que pide el diseno.

CUDA GRAPHS. lam es un TENSOR EN DEVICE que se muta in-place (`lam_.fill_(v)`), nunca
un float de Python. Con `--max-cudagraph-capture-size 48` un escalar de Python queda
horneado en el grafo capturado y cambiarlo en caliente no hace absolutamente nada --
sin error, simplemente no pasa nada.

TP. `wo_b` es RowParallelLinear con reduce_results por defecto, asi que su retorno ya
es post-all-reduce en el rank completo. La proyeccion es lineal y conmuta con el
all-reduce, pero se aplica post-reduce igualmente: es mas simple y no hay forma de
capturar media direccion.
"""

from __future__ import annotations

import json
import os
import struct
import threading

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_DIRS = "VLLM_REFUSAL_DIRS"
ENV_LAMBDA = "VLLM_REFUSAL_LAMBDA_INIT"

# --- seleccion POR PETICION -------------------------------------------------
#
# El dial de /admin/refusal_lambda es GLOBAL: cambia el modo de todo el servidor.
# Para que dos alias de LiteLLM convivan (uno normal y otro abliterado, cada
# peticion con su lambda), lambda tiene que viajar CON la peticion.
#
# Portador: `cache_salt`, que ya existe en la API de vLLM y ya recorre el camino
# completo API -> Request -> clave de hash de bloque. Eso ultimo es clave: el
# aislamiento de prefix cache entre lambdas sale CORRECTO gratis, sin tocar
# kv_cache_utils, porque el salt ya entra en el hash.
#
# Formato: cache_salt = "refusal:<float>"  ->  lambda para esa peticion.
# Cualquier otro salt se trata como salt normal y usa el lambda global.
#
# En LiteLLM se cablea por alias:
#   - model_name: deepseek-v4-flash-0731
#     litellm_params: {model: openai/deepseek-v4-flash-0731, api_base: ...}
#   - model_name: deepseek-v4-flash-0731-abliterated
#     litellm_params: {model: openai/deepseek-v4-flash-0731, api_base: ...,
#                      extra_body: {cache_salt: "refusal:1.0"}}
SALT_PREFIX = "refusal:"

# Tensor [num_tokens] con el lambda de cada token del lote en curso. Lo rellena
# el model runner antes del forward; None = usar el escalar global.
_per_token: torch.Tensor | None = None


def parse_request_lambda(cache_salt: str | None) -> float | None:
    """`refusal:<float>` -> lambda. None si el salt no es nuestro o es invalido."""
    if not cache_salt or not cache_salt.startswith(SALT_PREFIX):
        return None
    try:
        return float(cache_salt[len(SALT_PREFIX) :])
    except ValueError:
        logger.warning("cache_salt %r no parsea como lambda; se ignora", cache_salt)
        return None


def set_per_token_lambda(t: torch.Tensor | None) -> None:
    """Fija el lambda por token del lote en curso. El runner lo llama por paso."""
    global _per_token
    _per_token = t


def get_per_token_lambda() -> torch.Tensor | None:
    return _per_token

# lam se cuantiza a entero para la clave de hash de bloque del prefix cache.
LAMBDA_HASH_SCALE = 1000

_lock = threading.Lock()
_dirs: dict[str, torch.Tensor] | None = None
_lam_by_device: dict[torch.device, torch.Tensor] = {}
_lam_value: float = 0.0


def _load_dirs(path: str) -> dict[str, torch.Tensor]:
    """Lector safetensors minimo: no arrastra la dependencia por 750 KB."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        blob = fh.read()
    out: dict[str, torch.Tensor] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        if meta["dtype"] != "F32":
            raise ValueError(f"{path}: {key} es {meta['dtype']}, se esperaba F32")
        a, b = meta["data_offsets"]
        t = torch.frombuffer(bytearray(blob[a:b]), dtype=torch.float32)
        out[key] = t.clone()
    return out


def get_dirs() -> dict[str, torch.Tensor] | None:
    global _dirs
    path = os.environ.get(ENV_DIRS)
    if not path:
        return None
    with _lock:
        if _dirs is None:
            _dirs = _load_dirs(path)
            logger.info(
                "refusal projection: %d direcciones cargadas de %s", len(_dirs), path
            )
    return _dirs


def is_enabled() -> bool:
    return bool(os.environ.get(ENV_DIRS))


def get_lambda_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Tensor de lam por device, COMPARTIDO por todas las capas y mutado in-place.

    `inference_mode(False)` NO es opcional. Este tensor se crea perezosamente en el
    primer forward, que corre dentro del `torch.inference_mode()` de vLLM. Un tensor
    nacido ahi es un *inference tensor* y PyTorch prohibe mutarlo in-place despues:

        RuntimeError: Inplace update to inference tensor outside InferenceMode is
        not allowed.

    O sea: el dial se queda clavado para siempre y `set_lambda` devuelve 500. Hay que
    crearlo como tensor normal aunque el contexto sea de inferencia; leerlo desde
    inference mode sigue siendo legal.
    """
    global _lam_value
    with _lock:
        t = _lam_by_device.get(device)
        if t is None:
            init = float(os.environ.get(ENV_LAMBDA, "0.0"))
            _lam_value = init
            with torch.inference_mode(False):
                t = torch.tensor(init, device=device, dtype=dtype)
            _lam_by_device[device] = t
        return t


def set_lambda(value: float) -> float:
    """Muta lam IN-PLACE en todos los devices. Seguro con grafos ya capturados.

    El `inference_mode(False)` de aqui es el par del de arriba: el hilo que atiende
    el collective_rpc puede venir con inference mode activo.
    """
    global _lam_value
    with _lock:
        with torch.inference_mode(False):
            for t in _lam_by_device.values():
                t.fill_(value)
        _lam_value = float(value)
        return _lam_value


def get_lambda() -> float:
    return _lam_value


def lambda_hash_key() -> int | None:
    """Clave estable de lam para el hash de bloque del prefix cache.

    El KV DEPENDE de lam: un prefijo cacheado con lam=0 y reusado con lam=1 da
    estados corruptos, en silencio y sin error. Va cuantizado a entero para que la
    clave no dependa de la representacion en coma flotante.
    """
    if not is_enabled():
        return None
    return int(round(_lam_value * LAMBDA_HASH_SCALE))


def resolve_direction(prefix: str, num_hidden_layers: int) -> torch.Tensor | None:
    """Mapea el prefijo del modulo en runtime a la clave del fichero de direcciones.

    Las claves del fichero salen del checkpoint: `layers.0..42.attn.wo_b` y
    `mtp.0..2.attn.wo_b`. En runtime hay tres productores del mismo modulo:

      - backbone           -> `...layers.N.attn`      con N < num_hidden_layers
      - drafter DSpark     -> `...layers.N.attn`      con N >= num_hidden_layers,
                              cargado de los pesos `mtp.(N - num_hidden_layers)`
                              (ver dspark.py: prefix=f"layers.{num_hidden_layers+i}")
      - MTP                -> `...mtp.N...attn`

    Devolver None deja la capa SIN hook (identidad) y se avisa por log: es preferible
    a adivinar y proyectar con la direccion de otra capa.
    """
    dirs = get_dirs()
    if dirs is None:
        return None

    parts = prefix.split(".")
    key = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            n = int(parts[i + 1])
            if n < num_hidden_layers:
                key = f"layers.{n}.attn.wo_b"
            else:
                key = f"mtp.{n - num_hidden_layers}.attn.wo_b"
            break
        if p == "mtp" and i + 1 < len(parts) and parts[i + 1].isdigit():
            key = f"mtp.{int(parts[i + 1])}.attn.wo_b"
            break

    if key is None or key not in dirs:
        logger.warning(
            "refusal projection: sin direccion para prefix=%r (clave probada %r); "
            "esta capa queda SIN proyectar",
            prefix,
            key,
        )
        return None
    return dirs[key]


_warned_shapes: set[tuple[int, int]] = set()


def _warn_shape_mismatch(got: int, want: int) -> None:
    """Avisa UNA vez por par de formas: en el forward, un log por token ahogaria
    el proceso. Pero callarlo del todo es peor — ya paso."""
    key = (got, want)
    if key not in _warned_shapes:
        _warned_shapes.add(key)
        logger.warning(
            "refusal projection: lambda por token con %d elementos pero y tiene %d "
            "filas; se usa el lambda GLOBAL para este lote. La seleccion por "
            "peticion NO esta actuando.",
            got, want,
        )


class RefusalProjection(nn.Module):
    """y <- y - lam * r * (r . y), con lam como tensor en device.

    El producto escalar va en fp32 aunque `y` llegue en bf16 (medido en la fase 2:
    1,66e-3 contra 2,29e-3 haciendolo en bf16 -- mejora un 28%, y es gratis).
    """

    def __init__(self, r_hat: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("r_hat", r_hat.clone(), persistent=False)
        self._lam: torch.Tensor | None = None

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self._lam is None:
            self._lam = get_lambda_tensor(y.device, torch.float32)
        # r_hat nace en CPU (torch.frombuffer sobre el safetensors) y vLLM NO llama
        # a .to(device) sobre el modelo: lo construye dentro de un contexto de
        # device, que crea los parametros ya en la GPU pero NO mueve un tensor que
        # ya existe. Sin esto el forward peta con:
        #   "Expected all tensors to be on the same device, but got mat is on
        #    cuda:0, different from other tensors on cpu"
        # Se mueve UNA vez y se cachea reasignando el buffer. Ocurre en el warmup,
        # antes de la captura de grafos CUDA, asi que no entra en el grafo.
        if self.r_hat.device != y.device or self.r_hat.dtype != torch.float32:
            self.r_hat = self.r_hat.to(device=y.device, dtype=torch.float32)
        r = self.r_hat
        proj = y.to(torch.float32) @ r                       # [num_tokens]

        # lambda POR TOKEN si el runner lo ha puesto para este lote; si no, el
        # escalar global. El broadcast funciona igual con [num_tokens] que con
        # un escalar, asi que el kernel es el mismo en los dos casos.
        lam = get_per_token_lambda()
        if lam is not None and lam.shape[0] != proj.shape[0]:
            # NO silenciar esto. Este fallback existia sin aviso y oculto durante
            # dos ciclos de build que el tensor por token venia con los tokens
            # REALES mientras `y` llega PADEADO para los grafos CUDA: la seleccion
            # por peticion no hacia nada y no habia ni un error que lo delatara.
            _warn_shape_mismatch(lam.shape[0], proj.shape[0])
            lam = None
        if lam is None:
            lam = self._lam
        return y - (lam * proj).unsqueeze(-1).to(y.dtype) * r.to(y.dtype)
