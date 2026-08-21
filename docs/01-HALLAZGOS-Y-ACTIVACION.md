# DeepSeek-V4-Flash-0731 en 2× DGX Spark — hallazgos, manifiestos y activación

Estado a 2026-08-08. Complementa `00-ESTADO-PREVIO.md` (baseline + rollback).

## 1. Qué se decidió y por qué

### El checkpoint: nativo FP4+FP8, NO el NVFP4 de NVIDIA

Pediste "que esté en NVFP4 todo". Medido con la API de HF:

| | `deepseek-ai/…-0731` ✅ | `nvidia/…-NVFP4` ❌ |
|---|---|---|
| Bytes en disco | 166.9 GB | **168.3 GB** (más grande) |
| Params en FP4 | **296.35B / 304.18B = 97.4 %** | 138.5B empaquetados (U8) |
| Params en FP8 | 6.3B | **23.3B** (4× más) |
| Módulo DSpark | **incluido** (`mtp.*`, 4705 tensores) | ausente |
| Descargas/mes | 785.771 | 911.023 |
| Target | receta GB10 validada | B200 / GB300, vLLM 0.22.1rc1 nightly |

El repo NVFP4 de NVIDIA es **más grande**, tiene **más FP8** en la ruta no-experta y
**no trae drafter**. Sin drafter el acceptance rate es 0 por construcción y tu propio
gate (≥55 %) es inalcanzable. El nativo ya es FP4 en el 97.4 % de los params y aquí
además el KV cache va en `nvfp4_ds_mla`. Es "todo NVFP4" en todo lo que mueve la aguja.

Otros candidatos descartados: `sleepyeldrazi/ds4-nvfp4-spark` y `0xSero/…-162B/180B`
usan **REAP expert pruning** (284B → 162B) para caber en un solo Spark: es un modelo
podado y además reporta ~10 tok/s. `unsloth/…-GGUF` y los `mlx-community` no sirven
para vLLM/TP=2.

### Rendimiento esperado

La receta de 2× Spark reporta **95.9 tok/s single-stream** con grafos CUDA normales,
y 82.4 / 134.6 / ~340 tok/s agregados a concurrencia 1 / 3 / 6. Está **por encima**
de tu objetivo (55-66) y muy por encima del mínimo (45). El ~10 tok/s que se ve en
los hilos de HF es de **un solo Spark**, y encaja con tu propia nota del artefacto
×4 al contar chunks SSE.

## 2. Tres cosas que la receta pública se equivoca en este cluster

**a) Apunta al puerto muerto.** Usa `NCCL_IB_HCA=rocep1s0f1` /
`NCCL_SOCKET_IFNAME=enp1s0f1np1`. En estos Sparks ese puerto **no tiene cable**
(`physical_state DISABLED`, "No cable"). Los enlaces vivos son el puerto `f0` de cada
ConnectX-7 — y hay **dos**, no uno:

```
rocep1s0f0   / enp1s0f0np0     ACTIVE  200000Mb/s DAC   10.0.0.2 <-> 10.0.0.1
roceP2p1s0f0 / enP2p1s0f0np0   ACTIVE  200000Mb/s DAC   10.0.1.2 <-> 10.0.1.1
```

Se usa el primero. El segundo queda libre como margen de tuning (`NCCL_CROSS_NIC=1`
ya está puesto). Copiar la receta tal cual habría colgado NCCL en el init.

**b) `num_speculative_tokens`.** `recipes.vllm.ai` dice 7; ese valor es para las
variantes FP8/NVFP4 con `method=mtp`. Este checkpoint declara
`dspark_block_size: 5` y **k<5 trunca bloques de draft en silencio**. Se usa 5.

**c) `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK`.** El README la lista entre los
defaults; el `.env.example` dice explícitamente que es **solo de la imagen Stage-C**
y que en la Anemll 0.1.1 se ignora con warning. No se pone.

## 3. El bloqueo real: restricciones 1 y 2 son incompatibles

Los pesos son 83.5 GB por nodo con TP=2 (77.7 GiB). Con `util 0.78` el presupuesto
es ~93 GiB por nodo.

| Nodo | Libre ahora | Tras evictar | ¿Cabe? |
|---|---|---|---|
| `gx10-ec3d` | 66 GiB | +38.8 (27B) = **~104 GiB** | sí |
| `nvidia-dgx` | 15 GiB | +68 (35B) = **~83 GiB** | **no** |
| `nvidia-dgx` | | +13.2 (bge/stt/tts) = **~96 GiB** | sí, justo |

Para que quepa en `nvidia-dgx` hay que bajar **también** embeddings, reranker, STT y
TTS. Son los dos únicos nodos GPU del cluster (`sauvage` retirado, el resto amd64).
Así que "el resto del routing sigue sin interrupción" **no se puede cumplir**.

Alias de LiteLLM que se caen: `dense`, `dense-reasoning`, `dense-uncensored`,
`taxonomy`, `bge-m3`, `bge-m3-embedding`, `bge-reranker`, `bge-reranker-v2-m3`,
`stt-turbo`, `whisper-1`, `omnivoice-tts`, `tts-1`. En aquella medicion solo
sobrevivian rutas externas que ya han sido retiradas; hoy no cuentan como
capacidad ni como continuidad del DGX.

Confirmaste "evictar todo". Queda registrado aquí.

## 4. Dos mecanismos de eviction distintos, uno por nodo

Esto no era obvio y cambia el procedimiento:

`/spec/replicas` está en el `ignoreDifferences` de la Application `ai` para **todos**
los deployments de DGX2 — incluido `vllm-qwen36-27b-uncensored-dgx2`. De ahí que en
git esté a `0` y vivo a `1`: el arbitro/Control Nexus manda sobre ese campo.

| Objetivo | Mecanismo | Por qué |
|---|---|---|
| `vllm-qwen36-27b-uncensored-dgx2` (gx10-ec3d) | **`kubectl scale`** | replicas ignorado por Argo → el scale **persiste**, selfHeal no lo revierte |
| Los 5 de `nvidia-dgx` + los 2 ranks de DeepSeek | **GitOps + re-pin** | replicas SÍ sincronizado → `kubectl scale` sería revertido por selfHeal en ~3 min |

Un patch de replicas en el overlay para el 27B habría sido un **no-op silencioso**.

## 5. El gpu_arbiter habría dejado el worker huérfano

`dgx-infra/services/dashboard/gpu_arbiter.py` hace
`kubectl get deploy -A -l gpu.dgx-infra/role=resident` y escala a 0 **todos** los que
caigan en el nodo gb10 que elige. Su `PROTECTED_NODES` es `{"nvidia-dgx"}` —
**`gx10-ec3d` no está protegido**, y ComfyUI/Trellis están pinneados justamente ahí.

Con `role: resident`, cualquier petición de imagen o vídeo habría escalado a 0 **solo
el head**, dejando al worker del otro nodo ocupando 104 GiB esperando un rendezvous
que no llega, y el tier de tooling caído. Su garantía de "≥1 LLM vivo" además filtra
por el prefijo `vllm-`, que estos deployments no tienen.

→ Se usa `gpu.dgx-infra/role: exclusive-2node`, que su selector no matchea. `app.py`
solo propaga la etiqueta al HUD y tolera `None`, así que un tercer valor no rompe nada.

**Consecuencia aceptada:** mientras DeepSeek esté arriba, ComfyUI / Trellis / imagen /
vídeo en DGX2 no consiguen memoria y fallarán al arrancar. Es inherente a un modelo de
dos nodos que se queda los dos Sparks.

## 6. Guard fail-closed del drafter

Tu gate dice: si el acceptance baja del 55 %, es fallo de carga del drafter. Se puede
cazar **antes** de servir. Dato no obvio: los tensores del drafter **no se llaman
`dspark.*`** — un grep de "dspark" sobre el index da **0** y pasaría un checkpoint sin
drafter. Viven bajo `mtp.` y son **4705** de los 72317 (el namespace además es
`layers.N.*`, no `model.layers.N.*`).

El initContainer aborta si no cuadran: 4705 tensores `mtp.*`, 72317 totales, 48 shards,
`total_size` 166878536440 descontando cabeceras safetensors, `dspark_block_size == 5`,
`model_type == deepseek_v4` y presencia de `encoding/encoding_dsv4.py`. Y `launch.sh`
emite `DS4_SPEC_EFFECTIVE method=dspark k=5 …`, para poder distinguir "DSpark activo
sin ganancia" de "DSpark nunca arrancó" sin adivinar desde el tok/s.

## 7. Manifiestos — ya en git, INERTES

Rama `deepseek-v4-flash-2xspark-20260808`, commit **`bb3807e3`**, en
`pocharlies-org/k8s-ai-pocharlies`. Salió de `b5c0a4ef` (el SHA que sirve prod hoy).

- `k8s/deepseek-v4-flash-0731-dspark.yaml` — ConfigMap (verifier + launcher), Deployment
  head (rank 0, `gx10-ec3d`, 10.0.0.2), Deployment worker (rank 1 `--headless`,
  `nvidia-dgx`, 10.0.0.1), Service.
- `k8s/kustomization.yaml` — registra el fichero.
- `overlays/…-20260721/kustomization.yaml` — 35B a 0, los 5 pequeños a 0, los 2 ranks a 1.

Validado: `kubectl kustomize` renderiza 3901 líneas con replicas correctos, y
`kubectl apply --dry-run=server` pasa admisión en los 4 objetos (Kyverno acepta
`ghcr.io`, no-privileged con `IPC_LOCK`, limits declarados).

**No auto-sincroniza**: la app está pinneada a `b5c0a4ef`, así que este commit no
hace nada hasta re-pinear.

Desviaciones frente a la receta, además de las tres del §2: sin `privileged` (policy
Kyverno, y no hay device plugin de RDMA → hostNetwork + hostPath `/dev/infiniband` +
`IPC_LOCK`); sin `ipc: host` y `/dev/shm` a 16Gi en vez de 64 (con TP=2 hay un rank
por nodo, el transporte es NCCL sobre IB, y un `emptyDir medium:Memory` cuenta contra
`limits.memory`); `util 0.78` en vez de 0.80; GID de RoCEv2 resuelto de sysfs en cada
arranque (hoy es 3 en ambos, pero deriva tras un reboot y un literal compartido puede
colgar NCCL); `DEFAULT_THINKING=low` en vez de `max` porque el objetivo de TTFT es <3 s.

## 8. Activación (pendiente de tu OK) y rollback

```bash
# 0. Requisito: descarga completa en AMBOS nodos (166.9 GB cada uno).
#    El initContainer aborta si falta un byte, así que no hay riesgo de servir a medias.

# 1. gx10-ec3d: el 27B. Persiste porque Argo ignora /spec/replicas de este deploy.
kubectl -n llm scale deploy/vllm-qwen36-27b-uncensored-dgx2 --replicas=0

# 2. Re-pinear la app `ai` al commit nuevo -> baja los 5 de nvidia-dgx y sube los 2 ranks
kubectl -n argocd patch app ai --type merge \
  -p '{"spec":{"source":{"targetRevision":"bb3807e35ca28d5a747fc514195d9fc5befda4f9"}}}'
```

### Rollback — un campo, < 5 min

```bash
kubectl -n argocd patch app ai --type merge \
  -p '{"spec":{"source":{"targetRevision":"b5c0a4ef5a741916b69bef8795999fe098a76229"}}}'
kubectl -n llm scale deploy/vllm-qwen36-27b-uncensored-dgx2 --replicas=1
```

selfHeal relevanta los 5 de `nvidia-dgx` y pone los ranks de DeepSeek a 0. No se
borra ningún PVC, ninguna imagen y ningún peso descargado en ningún paso.

## 9. LiteLLM — diff preparado, NO aplicado

Deliberadamente sin aplicar: pediste parar antes de tocar LiteLLM si el acceptance
baja del 55 %. Se aplica **después** de pasar el gate.

Entrada a añadir en `BACKENDS` de `sync.py` (`k8s-litellm-pocharlies`, tronco `main`,
auto-sync). Es una entrada **añadida**, no un reemplazo:

```python
{
    # DeepSeek-V4-Flash-0731 en TP=2 sobre los DOS Sparks. Unico backend del
    # cluster que no vive en un solo nodo: su readiness es el Service del head,
    # pero si el worker de nvidia-dgx cae, el head deja de estar Ready solo.
    "name": "deepseek-v4-flash-tp2",
    "label": "DS4_FLASH_TP2",
    "backend": "dgx1+dgx2",
    "base_model": "openai/deepseek-v4-flash-0731",
    "aliases": TOOLING_RESIDENT_ALIASES,
    "service": os.getenv("K8S_SERVICE_DS4_FLASH", "deepseek-v4-flash-0731"),
    "api_base": os.getenv(
        "DS4_FLASH_API_BASE",
        "http://deepseek-v4-flash-0731.llm.svc.cluster.local:8000/v1",
    ),
    "id_prefix": "ds4-flash-0731-tp2",
    "max_parallel_requests": 6,   # = --max-num-seqs
    "max_tokens": 16384,
    # 262144 y no 1048576 aunque el server sirva 1M: contexto y concurrencia
    # comparten UN pool de KV (~2.49M tokens). Declarar 1M invita a que una sola
    # peticion se lo coma. Cumple el criterio de 256K. Subirlo es una linea.
    "context_window": 262144,
    "supports_function_calling": True,
    "supports_vision": False,   # DeepseekV4ForCausalLM, no multimodal
}
```

Y en `config.yaml` queda un cabo suelto que hay que cortar:

```diff
   fallbacks:
-    - tooling: ["dense"]
+    # `dense` (27B de DGX2) esta a 0 mientras DeepSeek ocupa los dos Sparks, asi
+    # que este fallback apuntaba a un alias muerto. El propio comentario del
+    # fichero avisa: una entrada que nombra un alias inexistente se queda aqui en
+    # silencio. Restaurar junto con el rollback.
```

## 9b. Sampling por cliente — OpenClaw y opencode

Dato del model card que va en contra del instinto: para escenarios **agénticos /
tool-calling** DeepSeek recomienda **`temperature: 1.0`, `top_p: 0.95`**, no
temperatura 0. El modelo está tuneado así; bajarlo a 0 en tool-calling no es
"más determinista y por tanto mejor", es sacarlo de su régimen.

Los benchmarks de velocidad sí van a **temp 0**, porque eso es lo que pide el
criterio de aceptación. Son dos cosas distintas y no hay que mezclarlas.

Entradas a añadir en `config.yaml` (todas **añadidas**, apuntando al mismo backend):

```yaml
# ── OpenClaw: agente, tool-calling, cadenas largas ──
- model_name: deepseek-v4-flash-openclaw
  litellm_params:
    model: openai/deepseek-v4-flash-0731
    api_base: http://deepseek-v4-flash-0731.llm.svc.cluster.local:8000/v1
    api_key: not-used
    temperature: 1.0          # recomendado por el model card para agentico
    top_p: 0.95
    # reasoning_effort low: el tier de tooling prioriza TTFT (<3s). Para una
    # tarea de razonamiento pesado, OpenClaw puede pedir high por peticion.
    extra_body:
      chat_template_kwargs: { thinking: true, reasoning_effort: low }
  model_info:
    mode: chat
    supports_function_calling: true

# ── opencode: edicion de codigo, salidas largas y estructuradas ──
- model_name: deepseek-v4-flash-opencode
  litellm_params:
    model: openai/deepseek-v4-flash-0731
    api_base: http://deepseek-v4-flash-0731.llm.svc.cluster.local:8000/v1
    api_key: not-used
    temperature: 1.0
    top_p: 0.95
    max_tokens: 16384         # los diffs completos se truncan por debajo
    extra_body:
      chat_template_kwargs: { thinking: true, reasoning_effort: low }
  model_info:
    mode: chat
    supports_function_calling: true
```

**Al probar opencode**: una respuesta vacía es un **fallo**, no un éxito — devuelve
0 bytes y exit 0 cuando el backend no está. El test comprueba bytes de salida, no
el código de retorno.

## 9c. Recolocar embedder / reranker / omnivoice al final

Aritmética con la que se decide (a rellenar con medidas reales, no estimaciones):

- DeepSeek a `util 0.78` = 92.8 GiB de GPU por nodo, techo de contenedor 104 GiB.
- Queda por nodo: 119 − ~104 = ~15 GiB, menos daemonsets (~8-10 GiB) = **~5-7 GiB**.
- A recolocar: bge-m3 4.8 GB + bge-reranker 2.9 GB + omnivoice 2.1 GB = **9.8 GB**.

Repartidos entre los dos Sparks son ~4.9 GB por nodo, que entra **justo** en los
~5-7 GiB. Sin margen. Los tres deployments tienen `nodeSelector` solo de
`arch: arm64`, sin pin de hostname, así que el scheduler los reparte solo por
requests de memoria — no hay que forzar nada.

Si no entran, la palanca es bajar `util` de 0.78 a ~0.72 (85.7 GiB), que libera
~7 GiB más por nodo. **Coste:** el KV cache pasa de ~15.1 a ~8 GiB por nodo, y hay
que **re-verificar que los 256K de contexto siguen cabiendo** antes de aceptarlo.
Ese es el orden: medir libre real → probar recolocación a 0.78 → si no entra, bajar
util y re-validar 256K. `stt-turbo` (3.3 GB) queda fuera de esta primera tanda; se
mira después con las cifras delante.

## 9d. Ejecutado — memoria liberada (MEDIDO, 2026-08-09 10:44Z)

Activación ejecutada: `kubectl scale` del 27B + re-pin de `ai` a `bb3807e3`.
Los 7 deployments quedaron a 0 y ningún pod se quedó en Terminating.

| Nodo | Disponible ANTES | Disponible DESPUÉS | Proyectado |
|---|---|---|---|
| `gx10-ec3d` | 59 GiB | **112 GiB** | 104 GiB |
| `nvidia-dgx` | 12 GiB | **114 GiB** | 96 GiB |

**Se liberó más de lo proyectado**, sobre todo en `nvidia-dgx` (114 frente a 96).
Dos motivos: el page cache de la descarga de 166.9 GB se reclamó, y los modelos
pequeños tenían RSS de CPU además de su huella de GPU — la estimación solo contaba
lo que reportaba `nvidia-smi`.

Consecuencia práctica: con `util 0.78` (92.8 GiB) quedan **~20 GiB de aire por nodo**
en vez de los ~5-7 estimados. Eso cambia el cálculo del §9c a mejor: los tres
modelos pequeños (9.8 GB, ~4.9 por nodo repartidos) deberían entrar **sin tocar
`util`** y sin sacrificar KV cache. A confirmar con el modelo ya cargado, que es
cuando la cifra cuenta.

Checkpoint verificado en disco en ambos nodos antes de arrancar: 48 shards y
`encoding/encoding_dsv4.py` presente.

Imagen `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`: 9.0 GiB, pre-descargada en ambos
nodos (~20 min a 7.5-8.3 MiB/s desde ghcr).

## 10. Lo que NO está medido todavía

El modelo no ha arrancado: la descarga iba al 21 % / 26 % al cerrar esta nota. Siguen
pendientes y **no hay cifras que dar**: tok/s por prompt, acceptance rate y mean
acceptance length, TTFT, tiempo de arranque en frío, curva de prefill por profundidad,
memoria libre real antes/después y score de la suite de tool-calling.

Nada de eso se rellena "por estimación": la receta reporta 95.9 tok/s en este mismo
hardware, pero es SU medida, no la nuestra.
