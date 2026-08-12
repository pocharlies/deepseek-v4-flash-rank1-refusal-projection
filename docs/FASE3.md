# Fase 3 — parche de vLLM. 2026-08-12

`patches/0001-rank1-projection.patch`, 460 líneas, 7 ficheros. Aplica limpio sobre
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM `0.25.2.dev0+g752a3a504`), compila, e
importa toda la cadena con el parche puesto. **Los 7 puntos del diseño (3.1–3.7)
implementados.**

**No hay fork de vLLM en disco.** El deployment corre la imagen precompilada, así que
el árbol de referencia (`src/`) se extrajo de la propia imagen — es el código que de
verdad corre, no el upstream.

## Qué toca

| fichero | qué |
|---|---|
| `vllm/refusal_projection.py` | **nuevo.** Carga de direcciones, estado de λ, kernel, resolución de prefijos, clave de hash |
| `vllm/models/deepseek_v4/attention.py` | construcción del hook en `__init__` + aplicación en el retorno de `_o_proj` |
| `vllm/v1/core/kv_cache_utils.py` | λ en la clave de hash de bloque (puntos 411 y 539) |
| `vllm/v1/worker/gpu_worker.py` | `set_refusal_lambda` / `get_refusal_lambda` por `collective_rpc` |
| `vllm/entrypoints/serve/refusal/api_router.py` | **nuevo.** `POST`/`GET /admin/refusal_lambda` |
| `vllm/entrypoints/serve/__init__.py` | registro del router |

## Decisiones de implementación, y por qué

**3.1 — punto de aplicación.** El sitio es [`attention.py:379`](src/models/deepseek_v4/attention.py#L379),
el retorno de `_o_proj`, no [`o_proj.py:70`](src/models/deepseek_v4/nvidia/ops/o_proj.py#L70).
`o_proj.py` es la ruta NVIDIA y es función libre sin identidad de capa; `attention.py`
cubre los cuatro backends (flashmla, flashinfer, amd/rocm, xpu) de una vez y tiene
`self.prefix`. Confirmado que es antes de `hc_post`: en
[`model.py:914`](src/models/deepseek_v4/nvidia/model.py#L914) va `x = self.attn(...)` y
en 918 el `mhc_fused_post_pre_tilelang`.

**3.2 — TP.** `wo_b` es `RowParallelLinear` con `reduce_results` por defecto, así que su
retorno ya viene post-all-reduce en el rank completo. Nada que hacer.

**Bias.** `wo_b` se declara `bias=False, return_bias=False`
([`attention.py:241`](src/models/deepseek_v4/attention.py#L241)), coherente con el
checkpoint (92 claves = 46 × weight+scale, cero bias). La precaución sobre proyectar
antes de sumar el bias no aplica.

**3.4 — λ como tensor.** Un tensor por device, **compartido por todas las capas**, mutado
con `fill_()`. Testeado explícitamente que el `data_ptr` no cambia al reasignar: eso es
lo que garantiza que un grafo CUDA ya capturado vea el valor nuevo.

**3.6 — prefix caching.** El precedente bueno no es `lora_int_id` sino **`cache_salt`**,
que ya existe en `generate_block_hash_extra_keys` y se aplica sólo en
`start_token_idx == 0` — basta, porque el hash encadena por `parent_block_hash` y se
propaga a los bloques siguientes. λ va cuantizado a entero (`×1000`). Al cambiar λ los
bloques viejos dejan de casar y envejecen solos.

**Capa.** La primera versión metía el import de `models/deepseek_v4` dentro de
`v1/core/kv_cache_utils` — código genérico dependiendo de un modelo concreto. Se movió el
módulo a `vllm/refusal_projection.py`, neutro. (Los fallos de import que aparecieron
durante la validación eran artefacto de lanzar python con el cwd dentro del paquete
`vllm`; afectaban igual a la línea base sin parchear.)

## Lo que cambia la Fase 4: probablemente no hay nada que hornear

El prompt daba por hecho que *"las 3 etapas del drafter DSpark corren en su propia ruta y
no las alcanza el hook del target"*. **No es así.** Tanto
[`mtp.py:125`](src/models/deepseek_v4/nvidia/mtp.py#L125) como
[`dspark.py:91`](src/models/deepseek_v4/nvidia/dspark.py#L91) construyen el **mismo
`DeepseekV4DecoderLayer`**, o sea la misma `DeepseekV4Attention` y el mismo `wo_b`. Un
solo hook los alcanza a los tres.

El mapeo lo da [`dspark.py:93`](src/models/deepseek_v4/nvidia/dspark.py#L93): las capas
del drafter se nombran `layers.{num_hidden_layers + i}` pero se cargan de los pesos
`mtp.*` del checkpoint. `resolve_direction` implementa exactamente eso y está testeado:
`layers.43 → mtp.0`, `layers.45 → mtp.2`, `mtp.1 → mtp.1`.

Consecuencia: **el overlay de ~100 MB de la Fase 4 sobra.** Y aparece una capacidad que
el horneado no tenía — λ del target y del drafter son el mismo dial y quedan siempre
alineados. En el checkpoint publicado no lo están (λ_eff 2,44 backbone vs 2,34 MTP), y
esa desalineación es candidata a explicar parte de la caída de acceptance que mediste.

## Tests — 22/22 dentro de la imagen

`test_refusal_projection.py`, corrido en el contenedor con las direcciones reales:

- carga de las 46 direcciones, todas `[4096]` f32 de norma 1
- resolución de prefijos: backbone, drafter DSpark y ruta MTP; desconocido → `None` con aviso
- matemática del hook contra referencia float64 en λ = 0 / 0,5 / 1 / 2,43 → err ≤ 1,6e-3
  (coincide con el suelo de bf16 medido en la Fase 2, 1,66e-3)
- **λ=0 bit-exacto** (`torch.equal`), el guardarraíl del prompt
- λ es tensor, mutado in-place (`data_ptr` estable), compartido entre capas
- clave de hash distinta por λ, entera y estable
- sin `VLLM_REFUSAL_DIRS`: sin hook, sin clave, sin rama en el forward

Más: parche aplica con `patch -p1`, `py_compile` de los 4 ficheros, e import completo de
`vllm` + `kv_cache_utils` + `Worker` + `attention` con el parche puesto.

## 3.7 — endpoint admin

`POST /admin/refusal_lambda {"lambda": 0.0}` y `GET`. Sigue el patrón de
`serve/lora/api_router.py`: **el router sólo se monta si `VLLM_REFUSAL_DIRS` está
definida** — sin hook no hay endpoint que atacar.

`collective_rpc` y no un setter local porque con el executor `mp` el frontend no
comparte proceso con los workers: un setter local no llegaría a ningún rank. Se
comprueba que **todos** los ranks devuelven el mismo valor; si discrepan es **500**, no
200, porque un cluster con λ distinto por rank produce basura en silencio. El `GET`
también va por RPC, por lo mismo: la copia del frontend no es fuente de verdad.

Cota `[-1, 4]`. λ<0 **amplifica** la dirección en vez de eliminarla — se admite a
propósito: es la única forma de pedir un modelo *más* reticente, no menos.

**Lo que vLLM no puede garantizar:** que la ruta no salga a internet. Es una ruta HTTP
más; quien lo decide es el ingress / AgentGateway. `/admin/*` no debería salir de la red
del cluster.

### Tests del endpoint — 15/15

Montaje condicional (con y sin la env var) · `GET` inicial · `POST` válido con eco de
nº de ranks · límites 422 en −5 y 9 · λ=−1 aceptado · **ranks discrepantes → 500 con el
motivo** · cuerpo sin `lambda` → 422.

## Lo que queda

**Nada del diseño.** Falta ejecutar: construir la imagen con el parche y correr la
Fase 5 (A/B λ=0 vs λ=1). Eso implica levantar el deployment TP=2 sobre los dos Sparks,
que son mutuamente excluyentes con el oficial — decisión de operación, no de código.

**Rollback.** La imagen y el launcher actuales quedan intactos; el parche se aplica en
build. En caliente, quitar `VLLM_REFUSAL_DIRS` desactiva el hook por completo sin
rebuild, y λ=0 es bit-exacto al base.
