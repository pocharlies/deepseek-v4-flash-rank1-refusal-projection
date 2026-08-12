# Fase 1 — extracción de direcciones rank-1. Medido 2026-08-12

Salida cruda: `extraction-report.json` (46 módulos), `probe-residual.json` (4 módulos),
`refusal_dirs.safetensors` (757.712 bytes).

Corrido como Job en `nvidia-dgx` (`job-extract.yaml`, `job-probe.yaml`), imagen
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, CPU, límite de 12Gi. **No se descargó nada**:
los dos checkpoints ya estaban en la caché HF del nodo (base 156G @`9e165c30`,
abliterated 157G @`21bd923c`).

## Las cuatro puertas

| # | Métrica | Esperado | Medido | |
|---|---|---|---|---|
| 1 | Módulos editados | 46 | **46** (43 backbone + 3 MTP) | ✅ |
| 2 | Energía rank-1 `S₀²/ΣS²` | ≥ 0,999 | **0,8376 – 0,9448** (mín. `mtp.0`) | ❌ |
| 3 | δ Frobenius medio | 0,0587 | **0,06018** (+2,5 %) | ✅ |
| 4 | λ efectivo | ~1,7 | **2,429** (rango 2,340 – 2,473) | ❌ |

Extra, no pedidas pero decisivas:

- **La edición resta** en los 46 módulos (`⟨s₀v₀, u₀ᵀW_base⟩ < 0`). No hay riesgo de amplificar.
- **`cos(v₀, u₀ᵀW_base) ≥ 0,9873`** en los 46. La componente dominante *es* forma proyección,
  no otra edición rank-1 cualquiera.
- Backbone y MTP se comportan distinto: δ 0,0615 vs 0,0410, λ_eff 2,435 vs 2,343,
  energía 0,848–0,945 vs 0,838–0,848. El MTP está editado **más flojo**.

## Correcciones al planteamiento

**a) El formato de escalas no es el que dice el prompt.** El sufijo es `.scale`, no
`weight_scale_inv`, y el dtype es `F8_E8M0 [32,64]` — exponente puro, potencia de dos,
sin mantisa. Los bloques sí son 128×128. Que la puerta 3 salga en 0,06018 contra el
0,0587 publicado valida que la dirección del producto (`W = w_fp8 * scale`) es la correcta;
si estuviera invertida el número se iría órdenes de magnitud.

**b) `wo_b` no tiene bias.** 92 claves = 46 × (`.weight` + `.scale`), cero `.bias`. La
precaución de la Fase 2 sobre proyectar antes de sumar el bias no aplica aquí.

**c) La convención de signo del prompt no hace falta.** El hook `y − λ·r̂·(r̂·y)` contiene
`r̂` dos veces: es el producto externo `r̂r̂ᵀ`, invariante al signo (`(−r̂)(−r̂)ᵀ = r̂r̂ᵀ`).
Un `r̂` "invertido" **no puede** amplificar el refusal. Lo que sí puede ir al revés es el
signo de λ. La puerta útil es la que se implementó: comprobar que la edición *publicada*
resta — y ese signo es propiedad de ΔW, no una elección del SVD.

## Puerta 2: el residuo NO es una segunda dirección

Tu instrucción para el fallo era "evalúa r=2". **Medido, y r=2 no sirve.**

| módulo | s₁ | s₂ | s₂/s₈ | E(r=1) | E(r=2) |
|---|---:|---:|---:|---:|---:|
| `layers.0` | 8,019 | 0,500 | 2,71 | 0,9014 | 0,9050 |
| `layers.21` | 7,697 | 0,492 | 2,64 | 0,8887 | 0,8923 |
| `layers.42` | 11,32 | 0,490 | 2,51 | 0,9448 | 0,9466 |
| `mtp.0` | 14,92 | 1,157 | 2,66 | 0,8376 | 0,8427 |

s₁/s₂ es de 13–23×, y de s₂ a s₈ la caída total es 2,5–2,7× — cola plana. Pasar a r=2
gana **0,36 puntos** de energía y r=8 no llega a 0,91. No hay segunda dirección: el ~10 %
restante es ruido de banda ancha de la recuantización a E4M3 con escalas potencia de dos.

Es decir: ese 10 % **es justamente el artefacto del que huye tu diseño**. No es algo que el
hook en runtime deba reproducir.

> Salvedad honesta sobre el test B de `probe_residual.py`: el campo `recon_explains` sale
> `false` con `noise_floor = 0.0000`, y ese suelo es basura — recuantizar `W_base` con sus
> propias escalas es idempotente, así que mide 0 por construcción, no mide ruido. El dato
> bueno del test B es `err_recon/‖ΔW‖ = 0,138–0,365`: reconstruir la proyección ideal y
> recuantizar acerca el residuo de 0,314 a 0,223 en `layers.0`, pero no lo cierra. La
> conclusión de "r=1 es correcto" **se sostiene sobre el espectro (test A), que es limpio**,
> no sobre el test B.

## Puerta 4: λ_eff = 2,43, y eso cambia la calibración

Tu hipótesis era λ_nominal 2,5 × techo 68 % ≈ 1,7. La medida en el espacio de pesos dice
**2,429**, o sea el 97 % del nominal. El techo del 68 % no aparece en ΔW.

Lo que importa operativamente: **λ=1 en runtime no equivale al checkpoint horneado.**

- λ=1 → proyección ortogonal exacta, la componente se elimina y se queda en cero.
- λ>1 → sobredisparo: la componente se invierte, con magnitud (λ−1)× la original.
- El checkpoint publicado está en **λ_eff ≈ 2,43**, sobredisparando 1,43×.

Y eso reencuadra tus propias medidas del deployment `unc`: la degradación que ya mediste
(acceptance 51,28 % contra suelo 55 %, código 46,6–55,3 contra 59,32) se produjo a
**λ≈2,43**, no a λ=1. Un λ=1 en runtime es bastante más suave y tiene margen real de caer
dentro del suelo de acceptance. Es el argumento más fuerte a favor de tu diseño que ha
salido de esta fase — y no se podía ver sin medir.

## Predicción para la Fase 2: tu umbral de 1e-3 es inalcanzable

La Fase 2 quiere que `proyectar(W_base·x, λ_eff)` reproduzca `W_abl·x` con error relativo
mediano < ~1e-3. Con los números de arriba eso no puede pasar: el residuo no-rank-1 es
0,314·‖ΔW‖ y ‖ΔW‖/‖W‖ = 0,0602, así que el error relativo esperado en la salida es
**≈ 1,9e-2**, veinte veces el umbral.

No es que el hook esté mal: es que `W_abl` **no es** la proyección ideal, es la proyección
ideal más ruido de recuantización. Pedir que el hook reproduzca `W_abl` es pedirle que
reproduzca el ruido que quieres eliminar. El gate correcto de la Fase 2 es contra la
**proyección ideal en float64**, donde sí debe salir ~1e-7, y reportar la distancia a
`W_abl` como observación, no como puerta.

## Discrepancia en el baseline de acceptance de la Fase 5

El prompt dice "baseline ~48 %, con la config estructurada sube a ~82 %". Tus propias
medidas en `rho/COMPARATIVA.md` dicen baseline limpio **0,5826** con suelo duro **0,55**,
y el candidato A se rechazó por bajar a 0,2551. Con 48 % como referencia, el 51,28 % del
`unc` parecería una mejora cuando en realidad es una regresión bajo tu propio suelo.
Antes de correr la Fase 5 hay que fijar cuál de los dos baselines vale.
