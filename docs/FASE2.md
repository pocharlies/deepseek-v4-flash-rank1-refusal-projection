# Fase 2 — validación del hook fuera de vLLM. Medido 2026-08-12

Harness: `verify_projection.py`, torch en CPU, 4 módulos (`layers.0`, `layers.21`,
`layers.42`, `mtp.0`), 256 tokens de entrada gaussiana, barrido de 14 valores de λ.
Salida cruda: `verify-projection.json`.

## G1 — equivalencia hook ↔ peso editado. **PASA**

| módulo | error relativo máx. |
|---|---:|
| `layers.0` | 8,789e-16 |
| `layers.21` | 8,671e-16 |
| `layers.42` | 9,162e-16 |
| `mtp.0` | 9,262e-16 |

Precisión de máquina en float64. La identidad `(W − λ·r̂r̂ᵀW)x = Wx − λ·r̂(r̂ᵀWx)` se
cumple exactamente. **Ésta es la única puerta dura de la fase y pasa sin discusión.**

## G5 — λ=0 bit-exacto. **PASA** en los 4 módulos

`torch.equal(hook(y, r, 0), y)` → `True`. El guardarraíl del prompt se sostiene.

## G3 — el "techo del 68 %" es falso, y está invertido

Medido exacto sobre los pesos, sin depender de entradas:

| módulo | ⟨r̂ᵀW_abl, r̂ᵀW_base⟩ / ‖r̂ᵀW_base‖² | eliminación real |
|---|---:|---:|
| `layers.0` | **−1,4229** | **+242,3 %** |
| `layers.21` | **−1,4139** | **+241,4 %** |
| `layers.42` | **−1,4708** | **+247,1 %** |
| `mtp.0` | **−1,3104** | **+231,0 %** |

La premisa del proyecto era: el horneado se queda en el 68 %, por eso hay que
sobreproyectar a λ=2,5 para compensar. **Es al revés.** El checkpoint publicado no
se queda corto: elimina el 240 % de la dirección, o sea la **invierte** y la deja a
~1,4× de su magnitud original en sentido contrario. No hay techo que compensar.

Consecuencia directa: λ=2,5 no compensa nada, es el sobredisparo en sí. Y la
degradación que ya mediste en el deployment `unc` — acceptance 51,28 % contra tu suelo
de 55 %, código 46,6–55,3 contra 59,32 — es **el coste de invertir la dirección 1,4×**,
no el coste de eliminarla.

## G2 — curva λ vs distancia a `W_abl` (observación, no puerta)

`layers.0`, error relativo mediano contra `W_abl @ x` y componente residual:

| λ | err vs abl | residual |
|---:|---:|---:|
| 0,00 | 0,0406 | +1,0000 |
| 0,50 | 0,0341 | +0,5000 |
| **1,00** | **0,0287** | **+0,0000** |
| 1,50 | 0,0236 | −0,5000 |
| 2,00 | 0,0204 | −1,0000 |
| **2,43** | **0,0191** | **−1,4300** ← mínimo |
| 3,00 | 0,0210 | −2,0000 |

El mínimo cae en λ*=2,43 en los tres módulos backbone y en 2,25 en `mtp.0`, clavando
el λ_eff de la Fase 1 por una vía independiente. Y confirma la predicción: en λ=1 el
error contra `W_abl` es 0,020–0,039, **veinte veces el umbral de 1e-3** que pedía el
prompt. Ese umbral era inalcanzable por construcción, no por un fallo del hook.

La columna `residual` es la que importa: en λ=1 sale exactamente 0. Eliminación limpia.

## G4 — precisión del kernel (punto 3.3)

| | error relativo mediano |
|---|---:|
| bf16, producto escalar en **fp32** (el diseño) | 1,66e-3 |
| bf16, producto escalar en bf16 (el error fácil) | 2,29e-3 |

El fp32 en el producto escalar mejora un ~28 %. Es real, pero conviene no
sobrevenderlo: los dos están dominados por el almacenamiento de `y` en bf16, no por
el producto. Hazlo en fp32 —es gratis— pero el orden de magnitud no cambia.

**Matiz honesto sobre "elimina al 100 %":** en aritmética exacta sí, el residual en λ=1
es 0. Con activaciones en bf16 el residuo real es ~1,7e-3 relativo. O sea **99,83 %**,
no 100 %. Sigue siendo incomparablemente más limpio que un −143 %, pero el número que
va al documento debería ser 99,83 %.

## Lo que esto cambia

1. **El hook es correcto.** G1 a 9e-16 y λ=0 bit-exacto. La implementación en vLLM
   puede proceder sobre base sólida.
2. **λ=1 es un régimen que el horneado nunca pudo alcanzar.** El baked salta de
   +100 % (sin editar) a −143 %; no existe checkpoint publicado en el punto limpio.
   Todo el rango útil 0 < λ ≤ 1 es territorio que sólo el dial en runtime abre.
3. **La regresión medida del `unc` no predice la del hook a λ=1.** Se midió a 2,43.
   Hay que volver a medir; puede caer dentro del suelo de acceptance del 55 %.
4. **La justificación del proyecto cambia de argumento, no de conclusión.** No es
   "evitamos el round-trip FP8 que nos deja en el 68 %". Es "evitamos un sobredisparo
   de 2,4× que nadie pidió y que es lo que está rompiendo el acceptance".

## Nota de seguridad, que va en la dirección incómoda

Esto refuerza la recomendación de aislamiento de la Fase 1, no la relaja. Un λ=1 limpio
da un modelo **más capaz y a la vez más completamente despojado de su capacidad de
declinar** que el checkpoint horneado — que era torpe, y cuya torpeza limitaba de rebote
lo utilizable que era. Cuanto mejor funcione el dial, más importa que λ>0 no comparta
credenciales con Shopify, Gmail ni Slack.
