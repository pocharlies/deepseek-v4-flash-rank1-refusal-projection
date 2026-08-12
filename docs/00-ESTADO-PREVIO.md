# Estado previo del cluster — antes de tocar nada

Capturado 2026-08-08. Baseline para rollback de DeepSeek-V4-Flash-0731 en 2× DGX Spark.

## Nodos objetivo

| | `gx10-ec3d` (DGX2) | `nvidia-dgx` (DGX1) |
|---|---|---|
| GPU product | `NVIDIA-GB10-SHARED` | `NVIDIA-GB10` |
| `nvidia.com/gpu` allocatable | **2** (time-slicing, `gpu.count=1`) | **1** |
| compute cap | 12.1 → **sm_121** | 12.1 → **sm_121** |
| gpu.mode | `graphics` | `graphics` |
| Mem unificada total | 119 GiB | 119 GiB |
| Mem **disponible** | **66 GiB** | **15 GiB** |
| GPU mem en uso (procs) | 38.854 MiB (1 proc) | 82.661 MiB (5 procs) |
| Disco libre `/` | 270 G de 916 G (69 % usado) | 272 G de 916 G (69 % usado) |
| Usuario ssh | `<user1>` | **`<user2>`** (ojo, no `<user1>`) |
| HF cache hostPath | `/home/<user>/.cache/huggingface` | `/home/<user>/.cache/huggingface` |
| earlyoom | inactive ✅ | inactive ✅ |

## Fabric 200G — VERIFICADO

Hay **dos** enlaces 200G DAC activos, no uno. Cada Spark tiene dos ConnectX-7 de doble
puerto; sólo el puerto `f0np0` de cada uno está cableado.

| netdev | RDMA dev | estado | speed | gx10-ec3d | nvidia-dgx |
|---|---|---|---|---|---|
| `enp1s0f0np0` | `rocep1s0f0` | **ACTIVE / LINK_UP** | 200000 Mb/s DAC | 10.0.0.2 | 10.0.0.1 |
| `enP2p1s0f0np0` | `roceP2p1s0f0` | **ACTIVE / LINK_UP** | 200000 Mb/s DAC | 10.0.1.2 | 10.0.1.1 |
| `enp1s0f1np1` | `rocep1s0f1` | DOWN / DISABLED | — (No cable) | — | — |
| `enP2p1s0f1np1` | `roceP2p1s0f1` | DOWN / DISABLED | — (No cable) | — | — |

Ping 10.0.0.1 y 10.0.1.1 desde gx10-ec3d: 0 % pérdida, rtt ~0.5 ms.

> **La receta pública apunta al puerto MUERTO.** `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`
> usa `NCCL_IB_HCA=rocep1s0f1` / `NCCL_SOCKET_IFNAME=enp1s0f1np1`. Aquí eso es el puerto sin
> cable. Hay que usar **`rocep1s0f0` / `enp1s0f0np0`**.

## Workloads GPU actuales y a qué alias de LiteLLM sirven

`nvidia-dgx` (DGX1) — 82.661 MiB:

| Deployment | GPU mem | Alias LiteLLM que sirve |
|---|---|---|
| `vllm-nvidia-qwen36-35b-dgx1` | 69.452 MiB | `tooling`, `router`, `auto`, `litellmrouter`, `qwen36-35b*` |
| `bge-m3-embedding` | 4.871 MiB | `bge-m3`, `bge-m3-embedding` |
| `bge-reranker` | 2.891 MiB | `bge-reranker`, `bge-reranker-v2-m3` |
| `stt-turbo` | 3.362 MiB | `stt-turbo`, `whisper-1` |
| `omnivoice-audio` / `omnivoice-tts-dgx2` | 2.085 MiB | `omnivoice-tts`, `tts-1` |

`gx10-ec3d` (DGX2) — 38.854 MiB:

| Deployment | GPU mem | Alias LiteLLM que sirve |
|---|---|---|
| `vllm-qwen36-27b-uncensored-dgx2` | 38.854 MiB | `dense`, `dense-reasoning`, `dense-uncensored`, `taxonomy` |

Deployments ya a 0 réplicas (no consumen, no tocar): `qwen3-embedding-4b-nvfp4-dgx2`,
`qwen3-reranker-4b-nvfp4-dgx2`, `vllm-ornith-35b-nvfp4-mtp-dgx1`,
`vllm-qwen3coder-30b-a3b-nvfp4-dgx2`, `vllm-vision-deep-dgx2`.

## Quién gobierna esto (crítico para el rollback)

| App ArgoCD | Repo | targetRevision | Sync |
|---|---|---|---|
| `ai` (ns `llm`) | `k8s-ai-pocharlies` | **`b5c0a4ef5a741916b69bef8795999fe098a76229`** (SHA pinneado) | **automated, prune:true, selfHeal:true** |
| `litellm` (ns `litellm`) | `k8s-litellm-pocharlies` | `main` | automated, selfHeal |

Estado actual de ambas: `Synced` / `Healthy`.

### Consecuencia operativa

1. **`kubectl scale` NO sirve para liberar memoria.** `selfHeal: true` lo revierte en ~3 min.
2. **Commitear al repo tampoco basta**: `ai` está pinneada a un SHA, así que un commit nuevo
   no se ve hasta cambiar `targetRevision`.
3. Por tanto la vía reversible es: commit con `replicas: 0` → mover `targetRevision` de `ai`
   al SHA nuevo. **Rollback = devolver `targetRevision` a `b5c0a4ef5a...`** (un solo campo,
   selfHeal vuelve a levantar todo). Eso cumple el requisito de < 5 min.

## Restricciones de admisión (Kyverno, verificadas)

| Policy | Efecto sobre este despliegue |
|---|---|
| `restrict-image-registries` | `ghcr.io` **permitido** → `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` pasa ✅ |
| `disallow-privileged-containers` | ns `llm` **no** exento → **prohibido `privileged: true`** |
| `require-resource-limits` | hay que declarar limits |
| `disallow-latest-tag` | pinnear tag (ya lo hacemos) |

**No hay device plugin de RDMA**: `nvidia.com/gpu` es el único recurso extendido; no hay
SR-IOV, ni Multus, ni `k8s-rdma-shared-dev-plugin`. Y no se puede usar `privileged`.
→ Vía viable: `hostNetwork: true` + hostPath `/dev/infiniband` + `capabilities.add: [IPC_LOCK]`
(fijar memoria RDMA). Eso NO es `privileged` y pasa admisión.

## Procedimiento de rollback (a validar antes de tocar nada)

```bash
# 1. Devolver la app `ai` al SHA original  → selfHeal relevanta los 6 deployments
kubectl -n argocd patch app ai --type merge \
  -p '{"spec":{"source":{"targetRevision":"b5c0a4ef5a741916b69bef8795999fe098a76229"}}}'

# 2. Borrar el despliegue de DeepSeek (libera la memoria unificada)
kubectl -n llm delete -f manifests/deepseek-v4-flash-dspark/

# 3. Revertir el config de LiteLLM
cd ~/k8s/k8s-litellm-pocharlies && git revert --no-edit <sha-del-cambio> && git push

# 4. Verificar
kubectl -n llm get deploy   # 6 deployments con réplicas a 1
curl -s $LITELLM/v1/models | jq -r '.data[].id' | sort
```

Los PVCs, imágenes y pesos descargados NO se tocan en ningún paso del rollback.
