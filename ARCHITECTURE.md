# ARCHITECTURE.md — deepseek-v4-flash-rank1-refusal-projection

Experimento publicado: servir DeepSeek-V4-Flash-0731 «sin censura» **sin entregar pesos modificados**, proyectando en runtime una dirección de rechazo de rango 1 (757 KB) con la intensidad λ como dial en caliente (`POST /admin/refusal_lambda`, sin reinicio). Tronco: **`main`**. **Estado: histórico**: DeepSeek fue retirado del clúster el 18-09-2026 (según la página `infra-vigente` del brain; no se ha re-medido en el clúster desde este repo).

## Clientes y versiones
- Sin clientes de producto. Artefactos: parche de vLLM (`patches/0001-rank1-projection.patch`, `vllm/refusal_projection.py`), bundle de compatibilidad `runtime/vllm-0.27.1/` (Dockerfile, DeepGEMM `sm_121`, parches), `tools/` (extracción y verificación), `bench/` (A/B) y `deploy/` (Jobs de build/verificación y Dockerfiles).
- Medido en 2× DGX Spark (GB10), vLLM 0.25.2 y luego 0.27.1, TP=2, DSpark k=5 (benchmarks fechados 2026-08-19/22 en `hf/benchmarks/`; no son el estado actual).

## Dependencias (ambos sentidos)
- **De**: checkpoint DeepSeek-V4-Flash-0731 (no se copia aquí), vLLM, imagen base de los Dockerfiles, `moby/buildkit:v0.31.0-rootless` para los Jobs de build.
- **Dónde viven los datos**: los vectores y la ficha del modelo, en Hugging Face (`pocharlies/deepseek-v4-flash-0731-uncensored-abliterated-refusal-directions`, subidos desde `hf/`); las imágenes salen de los Jobs de `deploy/` (el tag exacto no se copia aquí). Los pesos base los sirve el almacén de pesos del clúster, no este repo.
- **Quién depende**: nadie en código. LiteLLM guarda un alias `deepseek-v4-flash-0731-uncensored` en `k8s-litellm-pocharlies/k8s/manifest.yaml` (comentarios sobre su perfil); no hay manifiesto de despliegue en este repo.

## Stack
Python (parches y herramientas), vLLM, CUDA 13 / PyTorch 2.13 para el bundle 0.27.1, Dockerfiles y Jobs de Kubernetes (`deploy/job-*.yaml`). No se usa: pesos modificados (el dial es proyección en runtime, λ=0 es bit-exacto al original).

## Componentes compartidos (canónicos)
`vllm/refusal_projection.py` (aquí) y su gemelo en `qwen38-27b-rank1-refusal-projection` son el mismo mecanismo. El canónico vigente para el residente actual es el port de `k8s-ai-pocharlies/k8s/qwen38-flash-next-ursucipian-mod/refusal/`. No añadir un tercer fork.

## Cómo se construye aquí
No se construye como servicio. Reproducir: `tools/extract_refusal_dirs.py` → `tools/verify_projection.py` → `bench/compare_full.py`; las imágenes con los Jobs `deploy/job-build*.yaml`. λ es parte de la clave de hash de bloques (la caché de prefijos arranca fría al cambiar λ, a propósito).

## Tests y validaciones
`tools/test_refusal_projection.py`, `tools/test_refusal_per_request.py`, `tools/negctl_per_request.py`, `runtime/vllm-0.27.1/test_minp_spec_decode_hotfix.py`; resultados en `bench/results/` y `docs/*.json`. Sin CI.

## CI/CD y despliegue
Ninguno: sin workflows. **Antes de lanzar cualquier Job o carga en los Sparks (`gx10-ec3d`, `nvidia-dgx`) se lee el ConfigMap `gpu-arbiter-state` (ns `comfyui`)** (`kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}'`): con `llm-tp` efectivo los Sparks están enteros del LLM y los Jobs de build/verificación de `deploy/` no caben; con `phase: switching` no se toca nada.

## Decisiones y trampas
- Resultado (benchmark fechado 2026-08-19, 10 disparadores): λ=0 rechaza 9/10, λ=1.5 rechaza 0/10, sin regresión medida en NIAH 32k/128k (30/30) ni en tools (8/8). El checkpoint horneado equivalente queda en λ_eff≈2,43, que **invierte** la dirección.
- El efecto no es monótono; λ negativo vuelve el modelo más reticente.
- Candidato a archivar (propuesta, C5): modelo retirado y mecanismo duplicado en `qwen38-27b-rank1-refusal-projection` y en `k8s-ai-pocharlies`; los vectores siguen en Hugging Face.
