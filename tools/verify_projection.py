#!/usr/bin/env python3
"""
Fase 2 — validacion del hook fuera de vLLM. Torch en CPU, un modulo wo_b.

Que se comprueba, y contra que:

  G1  EQUIVALENCIA (puerta dura, ~1e-7).
      El hook   y <- y - lam*r*(r.y)   aplicado a la salida tiene que dar
      exactamente lo mismo que editar el peso  W <- W - lam*r*(r^T W)  y
      multiplicar. Es identidad algebraica pura:
          (W - lam*r r^T W) x  ==  Wx - lam*r*(r^T Wx)
      asi que el error solo puede ser ruido de coma flotante. Si esto no sale
      a nivel de maquina, el hook esta MAL IMPLEMENTADO. Esta es la unica
      puerta real de la fase.

  G2  DISTANCIA AL HORNEADO (observacion, NO puerta).
      Error contra W_abl @ x barriendo lambda. El prompt original pedia <1e-3
      aqui; es inalcanzable y no deberia ser puerta: W_abl no es la proyeccion
      ideal, es la proyeccion ideal MAS ruido de recuantizacion (medido en la
      fase 1: el residuo no-rank-1 es ~0,31 de ||dW||). Pedirle al hook que
      reproduzca W_abl es pedirle que reproduzca el ruido del que huye el
      diseno. Se reporta la curva y el lambda* que la minimiza.

  G3  FRACCION DE ELIMINACION REAL — la que zanja el "techo del 68%".
      Se mide DIRECTO SOBRE LOS PESOS, sin entradas aleatorias:

          ratio = <r^T W_abl , r^T W_base> / ||r^T W_base||^2

      ratio = 0.32  -> el horneado elimino el 68% (hipotesis del model card)
      ratio = -1.43 -> el horneado elimino el 243% (sobredisparo, lo que
                       predice el lambda_eff=2,43 medido en la fase 1)

      Para el hook la fraccion es lambda por construccion y exacta: el
      componente queda en (1-lambda) veces el original.

  G4  PRECISION DEL KERNEL (valida el punto 3.3 del diseno).
      y llega en bf16. Se compara hacer el producto escalar en fp32 (lo que
      pide el diseno) contra hacerlo en bf16 (el error facil), ambos contra
      referencia float64.

  G5  lambda=0 BIT-EXACTO (guardarrail del prompt).
      y - 0*r*(r.y) tiene que ser identico bit a bit a y.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_refusal_dirs import Checkpoint  # noqa: E402


def load_dirs(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
        blob = fh.read()
    out = {}
    for k, meta in h.items():
        if k == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        out[k] = np.frombuffer(blob[a:b], dtype=np.float32).copy()
    return out


def hook(y, r, lam, dot_dtype=torch.float32):
    """El kernel del diseno (3.3). y [T,H], r [H], lam escalar-tensor."""
    proj = (y.to(dot_dtype) @ r.to(dot_dtype))          # [T]
    return y - (lam * proj).unsqueeze(-1) * r.to(y.dtype)


def rel(a, b):
    """Error relativo por fila, mediano y maximo."""
    num = torch.linalg.norm(a.double() - b.double(), dim=-1)
    den = torch.linalg.norm(b.double(), dim=-1).clamp_min(1e-30)
    e = (num / den).cpu().numpy()
    return float(np.median(e)), float(np.max(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--abl", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--modules", default="layers.0,layers.21,layers.42,mtp.0")
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    torch.manual_seed(0)
    base, abl = Checkpoint(args.base), Checkpoint(args.abl)
    dirs = load_dirs(args.dirs)
    lams = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.43, 2.5, 2.75, 3.0]
    out = []

    for tag in args.modules.split(","):
        m = f"{tag}.attn.wo_b"
        Wb = torch.from_numpy(base.dequant(m))           # float64 [4096, 8192]
        Wa = torch.from_numpy(abl.dequant(m))
        r = torch.from_numpy(dirs[m].astype(np.float64))  # [4096]
        H, IN = Wb.shape

        # ---- G3: fraccion de eliminacion real, exacta sobre los pesos
        rowb = r @ Wb                                     # r^T W_base  [8192]
        rowa = r @ Wa
        ratio = float((rowa @ rowb) / (rowb @ rowb))
        removed_baked = 1.0 - ratio

        # ---- entradas. x en el espacio de entrada de wo_b; y = W x
        x = torch.randn(args.tokens, IN, dtype=torch.float64)
        y_base = x @ Wb.T                                 # [T, 4096]
        y_abl = x @ Wa.T

        # ---- G5: lambda=0 bit-exacto
        y32 = y_base.to(torch.float32)
        z0 = hook(y32, r.to(torch.float32), torch.tensor(0.0, dtype=torch.float32))
        bitexact = bool(torch.equal(z0, y32))

        curve = []
        for lam in lams:
            lt = torch.tensor(lam, dtype=torch.float64)
            # ideal: proyeccion horneada en el PESO, float64
            W_ideal = Wb - lt * torch.outer(r, rowb)
            y_ideal = x @ W_ideal.T
            # hook: proyeccion en la SALIDA
            y_hook = hook(y_base, r, lt, dot_dtype=torch.float64)

            g1_med, g1_max = rel(y_hook, y_ideal)         # equivalencia
            g2_med, g2_max = rel(y_hook, y_abl)           # distancia al horneado

            # componente residual en la direccion, medido
            c_hook = float(torch.median((y_hook @ r) / (y_base @ r)))
            curve.append(
                {
                    "lambda": lam,
                    "g1_equiv_median": g1_med,
                    "g1_equiv_max": g1_max,
                    "g2_vs_abl_median": g2_med,
                    "g2_vs_abl_max": g2_max,
                    "residual_component": c_hook,
                }
            )
            del W_ideal, y_ideal, y_hook

        best = min(curve, key=lambda c: c["g2_vs_abl_median"])

        # ---- G4: bf16 con producto escalar en fp32 vs en bf16
        ybf = y_base.to(torch.bfloat16)
        rbf = r.to(torch.bfloat16)
        l1 = torch.tensor(1.0)
        ref = hook(y_base, r, torch.tensor(1.0, dtype=torch.float64))
        g4_fp32 = rel(hook(ybf, rbf, l1.to(torch.float32), torch.float32).double(), ref)
        g4_bf16 = rel(hook(ybf, rbf, l1.to(torch.bfloat16), torch.bfloat16).double(), ref)

        rec = {
            "module": m,
            "shape": [H, IN],
            "g3_ratio_rTWabl_over_rTWbase": ratio,
            "g3_removed_fraction_baked": removed_baked,
            "g5_lambda0_bitexact": bitexact,
            "g4_bf16_dot_fp32_median": g4_fp32[0],
            "g4_bf16_dot_bf16_median": g4_bf16[0],
            "best_fit_lambda": best["lambda"],
            "best_fit_err": best["g2_vs_abl_median"],
            "curve": curve,
        }
        out.append(rec)

        g1w = max(c["g1_equiv_max"] for c in curve)
        print(f"\n=== {m}  {H}x{IN}")
        print(f"  G1 equivalencia hook<->peso : max {g1w:.3e}  (puerta: ruido de fp)")
        print(f"  G5 lambda=0 bit-exacto      : {bitexact}")
        print(f"  G3 r^T W_abl / r^T W_base   : {ratio:+.4f}")
        print(f"     -> el horneado elimino   : {removed_baked*100:+.1f}% de la direccion")
        print(f"  G4 bf16, dot en fp32        : {g4_fp32[0]:.3e}   (diseno 3.3)")
        print(f"     bf16, dot en bf16        : {g4_bf16[0]:.3e}   (el error facil)")
        print(f"  G2 lambda* que mas se acerca a W_abl: {best['lambda']} (err {best['g2_vs_abl_median']:.4f})")
        print("     curva lambda -> err_vs_abl / componente residual:")
        for c in curve:
            mark = " <-" if c is best else ""
            print(
                f"       lam={c['lambda']:.2f}  err={c['g2_vs_abl_median']:.4f}  "
                f"resid={c['residual_component']:+.4f}{mark}"
            )
        del Wb, Wa, x, y_base, y_abl

    json.dump(out, open(args.report, "w"), indent=2)
    print("\nVERIFY-DONE")


if __name__ == "__main__":
    sys.exit(main())
