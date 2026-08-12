#!/usr/bin/env python3
"""
Fase 1b — por que la energia rank-1 sale 0.84-0.94 y no >=0.999.

Dos hipotesis para el ~10% de energia que NO esta en la direccion dominante:

  H1  hay una SEGUNDA direccion real -> tocaria r=2, como dice el prompt.
  H2  es ruido de recuantizacion de banda ancha -> r=1 es correcto y el
      residuo no es representable con mas rangos, solo con mas precision.

Se distinguen con dos medidas:

  A. ESPECTRO. Top-8 valores singulares de DW por deflacion. Si s1 >> s2 hay
     una segunda direccion; si s2 ~ s3 ~ ... ~ s8 la cola es plana = ruido.

  B. RECONSTRUCCION. Se aplica la proyeccion IDEAL a W_base con el lambda_eff
     medido, se recuantiza a E4M3 con los PROPIOS scales E8M0 del checkpoint
     abliterated, y se compara con W_abl. Si el residuo desaparece, el
     checkpoint publicado ES exactamente "proyeccion ideal + recuantizacion",
     y entonces H2 queda probada y r=1 es la respuesta.

B es la prueba dura: no mide la forma del residuo, mide si lo explica.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_refusal_dirs import E4M3, Checkpoint, top_triplet  # noqa: E402

# valores finitos representables en E4M3, ordenados (para redondeo al mas cercano)
_FIN = np.sort(E4M3[np.isfinite(E4M3)])


def quantize_e4m3(x):
    """Redondeo al valor E4M3 finito mas cercano. Satura en +-448."""
    idx = np.searchsorted(_FIN, x)
    idx = np.clip(idx, 1, len(_FIN) - 1)
    lo, hi = _FIN[idx - 1], _FIN[idx]
    return np.where(np.abs(x - lo) <= np.abs(hi - x), lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--abl", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--modules", default="layers.0,layers.21,layers.42,mtp.0")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    base, abl = Checkpoint(args.base), Checkpoint(args.abl)
    out = []

    for tag in args.modules.split(","):
        m = f"{tag}.attn.wo_b"
        Wb, Wa = base.dequant(m), abl.dequant(m)
        dW = Wa - Wb
        fro = float(np.linalg.norm(dW))

        # ---- A. espectro por deflacion
        R = dW.copy()
        svals = []
        for _ in range(args.topk):
            u, s, v, _ = top_triplet(R, iters=300)
            svals.append(float(s))
            R -= s * np.outer(u, v)
        tail = float(np.linalg.norm(R))
        del R

        # ---- B. reconstruccion: proyeccion ideal + recuantizacion con scales del abl
        u0, s0, v0, _ = top_triplet(dW)
        w_row = u0 @ Wb
        lam = float(s0 / np.linalg.norm(w_row))
        W_ideal = Wb - lam * np.outer(u0, w_row)

        sb, sshape, _ = abl.raw(m + ".scale")
        from extract_refusal_dirs import E8M0

        sc = E8M0[sb].reshape(sshape)
        se = np.repeat(np.repeat(sc, 128, axis=0), 128, axis=1)[: Wb.shape[0], : Wb.shape[1]]
        W_recon = quantize_e4m3(W_ideal / se) * se

        err_recon = float(np.linalg.norm(W_recon - Wa))
        # suelo de referencia: recuantizar el propio W_base contra si mismo no da
        # cero porque los scales del abl son distintos; medimos el ruido tipico
        # de una recuantizacion sobre esta misma matriz
        noise_floor = float(np.linalg.norm(quantize_e4m3(Wb / se) * se - Wb))

        rec = {
            "module": m,
            "svals_top8": svals,
            "s1_over_s2": svals[0] / svals[1],
            "s2_over_s8": svals[1] / svals[7],
            "fro_dW": fro,
            "tail_after_top8": tail,
            "energy_top1": svals[0] ** 2 / fro**2,
            "energy_top2": (svals[0] ** 2 + svals[1] ** 2) / fro**2,
            "lambda_eff": lam,
            "err_recon_over_dW": err_recon / fro,
            "noise_floor_over_dW": noise_floor / fro,
            "recon_explains": err_recon <= 1.35 * noise_floor,
        }
        out.append(rec)
        print(
            f"{m:24s} s1/s2={rec['s1_over_s2']:.3f} s2/s8={rec['s2_over_s8']:.3f} "
            f"E1={rec['energy_top1']:.4f} E2={rec['energy_top2']:.4f} "
            f"recon_err={rec['err_recon_over_dW']:.4f} floor={rec['noise_floor_over_dW']:.4f} "
            f"{'EXPLICADO' if rec['recon_explains'] else 'NO-explicado'}"
        )
        print(f"    top8 s: {['%.4g' % s for s in svals]}")
        del Wb, Wa, dW

    json.dump(out, open(args.report, "w"), indent=2)
    print("PROBE-DONE")


if __name__ == "__main__":
    sys.exit(main())
