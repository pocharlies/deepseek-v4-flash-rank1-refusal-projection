#!/usr/bin/env python3
"""
Extraccion de las direcciones rank-1 de refusal para la familia Qwen3.5
(Qwen3.6-27B / Qwen3.8-27B, `Qwen3_5ForConditionalGeneration`).

Hermano de extract_refusal_dirs.py, que es de DeepSeek-V4 y se deja INTACTO: aquel
esta validado byte a byte contra un artefacto publicado y no se toca. Tres cosas
cambian aqui y son las que justifican un fichero aparte.

1) LOS SUBLAYERS SON TRES, NO UNO
   DeepSeek tenia un unico modulo que escribe al residual (`attn.wo_b`) y por eso 46
   direcciones bastaban. Qwen3.5-27B es hibrido 3:1 y tiene tres clases distintas:

     48 x model.language_model.layers.N.linear_attn.out_proj   (Gated DeltaNet)
     16 x model.language_model.layers.N.self_attn.o_proj       (atencion completa, N%4==3)
     64 x model.language_model.layers.N.mlp.down_proj
      + los de mtp.layers.0 (self_attn.o_proj y mlp.down_proj)

   CUALES estan editados es un hecho empirico, no una decision: se descubren mirando
   que ||dW|| != 0. Por eso el descubrimiento va por regex ancha y el filtrado por
   medicion. No se presupone el numero de modulos.

2) EL PAR ES BF16, NO FP8
   `Qwen/Qwen3.8-27B` y `trohrbaugh/Qwen3.8-27B-heretic-ara` son los dos BF16. No hay
   block scales ni round-trip de cuantizacion, asi que el suelo de ruido es MUCHO mas
   bajo que en el caso DeepSeek (donde la energia rank-1 se quedo en 0,84-0,94 por el
   FP8). Aqui la puerta de energia >= 0.999 vuelve a ser una puerta de verdad en vez
   de un fallo esperado.

3) COEFICIENTE POR MODULO
   Heretic NO abla con la misma fuerza en todas las capas: optimiza el peso capa a
   capa. Un lambda unico no reproduce ese perfil — en DeepSeek ya se vio la deriva
   (lam_eff 2,435 en backbone contra 2,343 en MTP) y se senalo como sospechosa de
   parte de la caida de acceptance.

   Asi que se emite, ademas del r_hat unitario, un `coef` por modulo con su lam_eff
   medido. El hook hace

       y  <-  y - lam * coef_m * r_hat (r_hat . y)

   de forma que **lam=1 reproduce EXACTAMENTE el perfil que afino el autor**, lam=0 es
   el base intacto, y lam intermedio es una interpolacion honesta entre los dos. No se
   hereda el lam=1.5 de DeepSeek: aquel numero es de aquel modelo y aquel par.

Convencion de signo: identica a la del hermano. El hook lleva r_hat DOS veces (es el
producto externo r r^T), luego es INVARIANTE al signo. La puerta util no es fijar el
signo de r_hat sino comprobar que la edicion publicada RESTA la componente, que es una
propiedad de dW y no una eleccion del SVD. Se reporta como `subtracts` por modulo.
"""

import argparse
import json
import os
import re
import struct
import sys

import numpy as np

# --------------------------------------------------------------- lectura safetensors

# Modulos candidatos: todo lo que escribe al stream residual. La torre visual
# (`model.visual.*`) queda fuera a proposito — la ablacion es de texto y meterla
# solo anadiria ruido a la busqueda.
DEFAULT_PATTERN = (
    r"(?:model\.language_model\.layers\.\d+|mtp\.layers\.\d+)\."
    r"(?:linear_attn\.out_proj|self_attn\.o_proj|mlp\.down_proj)$"
)


def _bf16_to_f64(raw_u8):
    """BF16 = los 16 bits altos de un float32. Se re-expande sin perder nada."""
    u16 = raw_u8.view(np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32).astype(np.float64)


class Shard:
    """Lector minimo de safetensors: cabecera JSON + offsets. Sin dependencias."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(n))
        self.data_start = 8 + n

    def raw(self, key):
        meta = self.header[key]
        a, b = meta["data_offsets"]
        with open(self.path, "rb") as fh:
            fh.seek(self.data_start + a)
            buf = fh.read(b - a)
        return np.frombuffer(buf, dtype=np.uint8), meta["shape"], meta["dtype"]


class Checkpoint:
    """Checkpoint BF16 sin cuantizar, con o sin index (un solo shard tambien vale)."""

    def __init__(self, root):
        self.root = root
        idx_path = os.path.join(root, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            self.weight_map = json.load(open(idx_path))["weight_map"]
        else:
            single = "model.safetensors"
            sh = Shard(os.path.join(root, single))
            self.weight_map = {k: single for k in sh.header if k != "__metadata__"}
            self._shards = {single: sh}
        if not hasattr(self, "_shards"):
            self._shards = {}

    def shard(self, fname):
        if fname not in self._shards:
            self._shards[fname] = Shard(os.path.join(self.root, fname))
        return self._shards[fname]

    def has(self, key):
        return key in self.weight_map

    def matrix(self, module):
        """Devuelve W float64 [out, in]."""
        rb, shape, dt = self.shard(self.weight_map[module + ".weight"]).raw(
            module + ".weight"
        )
        if dt != "BF16":
            raise RuntimeError(
                f"{module}: dtype {dt}, se esperaba BF16. Este extractor es para el par "
                f"sin cuantizar; un checkpoint cuantizado necesita su decodificador."
            )
        if len(shape) != 2:
            raise RuntimeError(f"{module}: shape {shape}, se esperaba 2D")
        return _bf16_to_f64(rb).reshape(shape)


# ------------------------------------------------------------------ rank-1 top


def top_triplet(dw, iters=500, tol=1e-13, seed=0):
    """Par singular dominante por iteracion de potencia. dw se queda intacto."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dw.shape[1])
    v /= np.linalg.norm(v)
    s_prev = 0.0
    for i in range(iters):
        u = dw @ v
        nu = np.linalg.norm(u)
        if nu == 0:
            raise RuntimeError("dW es cero")
        u /= nu
        v = dw.T @ u
        s = np.linalg.norm(v)
        v /= s
        if abs(s - s_prev) <= tol * max(s, 1.0):
            return u, s, v, i + 1
        s_prev = s
    return u, s, v, iters


def second_singular(dw, u0, v0, s0, iters=200, seed=1):
    """s1 por deflacion. Sirve para el ratio s0/s1, que es la puerta que MANDA cuando
    el par no es BF16 limpio: con ruido de requantizacion la energia rank-1 se hunde
    aunque r_hat siga perfectamente determinado, y el ratio lo delata."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dw.shape[1])
    v -= v0 * np.dot(v0, v)
    v /= np.linalg.norm(v)
    s = 0.0
    for _ in range(iters):
        u = dw @ v - u0 * (s0 * np.dot(v0, v))
        u -= u0 * np.dot(u0, u)
        nu = np.linalg.norm(u)
        if nu == 0:
            return 0.0
        u /= nu
        v = dw.T @ u - v0 * (s0 * np.dot(u0, u))
        v -= v0 * np.dot(v0, v)
        s = np.linalg.norm(v)
        if s == 0:
            return 0.0
        v /= s
    return float(s)


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="checkpoint limpio (BF16)")
    ap.add_argument("--abl", required=True, help="checkpoint ablado (BF16)")
    ap.add_argument("--out", required=True, help="safetensors de direcciones")
    ap.add_argument("--report", required=True, help="informe JSON")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN)
    ap.add_argument(
        "--limit", type=int, default=0, help="solo los N primeros modulos (piloto)"
    )
    ap.add_argument(
        "--energy-gate",
        type=float,
        default=0.999,
        help="puerta de energia rank-1. BF16 vs BF16: 0.999. Par cuantizado: bajala y "
        "mira s0/s1 en su lugar.",
    )
    args = ap.parse_args()

    base = Checkpoint(args.base)
    abl = Checkpoint(args.abl)

    rx = re.compile(args.pattern)
    mods = sorted(
        {
            k[: -len(".weight")]
            for k in base.weight_map
            if k.endswith(".weight") and rx.match(k[: -len(".weight")])
        },
        key=lambda m: (
            0 if ".language_model." in m else 1,
            int(re.search(r"layers\.(\d+)", m).group(1)),
            m.rsplit(".", 1)[-1],
        ),
    )
    if args.limit:
        mods = mods[: args.limit]
    print(f"modulos candidatos: {len(mods)}")

    missing = [m for m in mods if not abl.has(m + ".weight")]
    if missing:
        raise SystemExit(
            f"ABORTA: {len(missing)} modulos del base no estan en el ablado, p.ej. "
            f"{missing[:3]}. Los checkpoints no son comparables — si faltan los `mtp.*` "
            f"el drafter se quedaria SIN ablar y desalineado con el target, que es "
            f"exactamente la caida de acceptance documentada en DeepSeek."
        )

    dirs, coefs, report = {}, {}, []
    for n, m in enumerate(mods):
        Wb = base.matrix(m)
        Wa = abl.matrix(m)
        dW = Wa - Wb

        fro_dw = float(np.linalg.norm(dW))
        fro_wb = float(np.linalg.norm(Wb))
        if fro_dw == 0.0:
            print(f"[{n:3d}/{len(mods)}] {m:62s} SIN EDITAR")
            report.append({"module": m, "edited": False})
            del Wb, Wa, dW
            continue

        u0, s0, v0, it = top_triplet(dW)
        energy = float(s0 * s0 / (fro_dw * fro_dw))
        s1 = second_singular(dW, u0, v0, s0)

        w_row = u0 @ Wb                                  # u0^T W_base, en R^in
        n_wrow = float(np.linalg.norm(w_row))
        lam_eff = float(s0 / n_wrow)
        inner = float(np.dot(s0 * v0, w_row))            # < 0  => la edicion RESTA
        cos_v0 = float(abs(np.dot(v0, w_row)) / n_wrow)  # ~1 => es forma proyeccion

        # signo cosmetico y reproducible (irrelevante para el hook: r r^T)
        nz = np.argmax(np.abs(u0) > 1e-8)
        if u0[nz] < 0:
            u0 = -u0

        dirs[m] = u0.astype(np.float32)
        coefs[m] = lam_eff
        report.append(
            {
                "module": m,
                "edited": True,
                "shape": list(Wb.shape),
                "rank1_energy": energy,
                "s0_over_s1": float(s0 / s1) if s1 else None,
                "delta_frobenius": float(fro_dw / fro_wb),
                "s0": float(s0),
                "lambda_eff": lam_eff,
                "subtracts": bool(inner < 0),
                "cos_v0_wrow": cos_v0,
                "power_iters": it,
            }
        )
        print(
            f"[{n:3d}/{len(mods)}] {m:62s} energy={energy:.6f} "
            f"s0/s1={s0/s1 if s1 else float('inf'):8.2f} delta={fro_dw/fro_wb:.6f} "
            f"lam_eff={lam_eff:.4f} resta={'si' if inner < 0 else 'NO'} cos={cos_v0:.6f}",
            flush=True,
        )
        del Wb, Wa, dW

    ed = [r for r in report if r["edited"]]
    if not ed:
        raise SystemExit(
            "ABORTA: ningun modulo editado. O el par es el mismo checkpoint, o el "
            "--pattern no casa con el naming real."
        )

    # ---- safetensors de salida: r_hat por modulo + el vector de coeficientes.
    # El coeficiente va como tensor propio `__coefs__` con su orden en __metadata__,
    # para que el cargador no tenga que adivinar el emparejamiento.
    order = sorted(dirs)
    header, blob, off = {}, bytearray(), 0
    for m in order:
        b = dirs[m].tobytes()
        header[m] = {
            "dtype": "F32",
            "shape": [dirs[m].shape[0]],
            "data_offsets": [off, off + len(b)],
        }
        blob += b
        off += len(b)
    cv = np.array([coefs[m] for m in order], dtype=np.float32)
    cb = cv.tobytes()
    header["__coefs__"] = {
        "dtype": "F32",
        "shape": [len(order)],
        "data_offsets": [off, off + len(cb)],
    }
    blob += cb
    off += len(cb)

    header["__metadata__"] = {
        "source_base": os.path.basename(args.base.rstrip("/")),
        "source_abl": os.path.basename(args.abl.rstrip("/")),
        "modules": str(len(dirs)),
        "coef_order": json.dumps(order),
        "hook": "y <- y - lam * coef_m * r_hat (r_hat . y)",
        "note": (
            "rank-1 refusal directions, output-space. lam=1 reproduce el perfil por capa "
            "del checkpoint ablado; lam=0 es el base intacto y bit-exacto."
        ),
    }
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)))
        fh.write(hj)
        fh.write(blob)

    by_kind = {}
    for r in ed:
        kind = r["module"].rsplit(".", 1)[-1]
        by_kind.setdefault(kind, []).append(r)

    summary = {
        "modules_total": len(mods),
        "modules_edited": len(ed),
        "edited_by_kind": {k: len(v) for k, v in sorted(by_kind.items())},
        "energy_min": min(r["rank1_energy"] for r in ed),
        "energy_mean": float(np.mean([r["rank1_energy"] for r in ed])),
        "gate_energy": all(r["rank1_energy"] >= args.energy_gate for r in ed),
        "energy_gate_value": args.energy_gate,
        "s0_over_s1_min": min(
            (r["s0_over_s1"] for r in ed if r["s0_over_s1"] is not None), default=None
        ),
        "delta_frobenius_mean": float(np.mean([r["delta_frobenius"] for r in ed])),
        "lambda_eff_mean": float(np.mean([r["lambda_eff"] for r in ed])),
        "lambda_eff_min": min(r["lambda_eff"] for r in ed),
        "lambda_eff_max": max(r["lambda_eff"] for r in ed),
        "lambda_eff_by_kind": {
            k: float(np.mean([r["lambda_eff"] for r in v])) for k, v in sorted(by_kind.items())
        },
        "all_subtract": all(r["subtracts"] for r in ed),
        "cos_v0_wrow_min": min(r["cos_v0_wrow"] for r in ed),
        "out_bytes": 8 + len(hj) + len(blob),
    }
    json.dump({"summary": summary, "modules": report}, open(args.report, "w"), indent=2)

    print("\n================ PUERTAS ================")
    print(f"modulos editados        : {summary['modules_edited']} de {summary['modules_total']}")
    print(f"  por clase             : {summary['edited_by_kind']}")
    print(f"energia rank-1 minima   : {summary['energy_min']:.6f}  (puerta >= {args.energy_gate})")
    print(f"s0/s1 minimo            : {summary['s0_over_s1_min']}")
    print(f"delta Frobenius medio   : {summary['delta_frobenius_mean']:.6f}")
    print(f"lambda_eff medio        : {summary['lambda_eff_mean']:.4f}")
    print(f"  rango                 : {summary['lambda_eff_min']:.4f} .. {summary['lambda_eff_max']:.4f}")
    print(f"  por clase             : {summary['lambda_eff_by_kind']}")
    print(f"la edicion RESTA        : {summary['all_subtract']}")
    print(f"cos(v0, u0^T Wbase) min : {summary['cos_v0_wrow_min']:.6f}  (~1 = forma proyeccion)")
    print(f"salida                  : {summary['out_bytes']} bytes")

    hard = []
    if not summary["all_subtract"]:
        hard.append("algun modulo SUMA la direccion en vez de restarla")
    if summary["cos_v0_wrow_min"] < 0.95:
        hard.append(
            f"cos minimo {summary['cos_v0_wrow_min']:.4f} < 0.95: dW no tiene forma de "
            f"proyeccion, la ablacion no es puramente direccional"
        )
    if hard:
        print("\nPUERTAS DURAS FALLADAS:")
        for h in hard:
            print(f"  - {h}")
        print("EXTRACT-FAILED")
        return 1

    print("EXTRACT-DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
