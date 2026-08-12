#!/usr/bin/env python3
"""
Fase 1 — extraccion de las direcciones rank-1 de refusal de
  cebeuq/DeepSeek-V4-Flash-0731-abliterated  frente a
  deepseek-ai/DeepSeek-V4-Flash-0731

Emite un tensor r_hat de 4096 floats por modulo editado (46 = 43 backbone + 3 MTP)
y un informe con las metricas de las cuatro puertas.

NO descarga nada: los dos checkpoints ya estan en la cache HF del nodo.

Decodificacion FP8 a mano
-------------------------
Los dos dtypes se decodifican desde bytes crudos con tablas de 256 entradas, sin
depender de que este torch soporte float8_e4m3fn / float8_e8m0fnu:

  weight  F8_E4M3  [4096, 8192]   S1 E4 M3, bias 7, variante "fn" (sin inf, max 448)
  scale   F8_E8M0  [32, 64]       exponente puro: valor = 2^(b-127). Bloques 128x128.

OJO: el sufijo es `.scale`, no `weight_scale_inv`, y es potencia de dos pura (E8M0),
no un float32. La puerta de delta Frobenius (0.0587) es la que valida que la
direccion del producto (W = w_fp8 * scale) sea la correcta: si estuviera invertida
(dividir en vez de multiplicar) el numero se va ordenes de magnitud.

Convencion de signo
-------------------
El SVD tiene ambiguedad de signo, PERO el hook que se va a implementar es

    y  <-  y - lam * r_hat * (r_hat . y)

que contiene r_hat DOS veces: es el producto externo (r_hat r_hat^T), y por tanto
es INVARIANTE al signo de r_hat --  (-r)(-r)^T == r r^T. Cambiar el signo de r_hat
NO puede amplificar el refusal. Lo que si puede ir al reves es el signo de lam, o
que la edicion publicada no reste.

Asi que la puerta de signo util no es "fija el signo de r_hat" sino comprobar que
la edicion PUBLICADA resta la componente. Con DW = s0 * u0 * v0^T y
w_row = u0^T W_base, la forma proyeccion exige

    DW = -lam * u0 (u0^T W_base)   =>   s0 * v0  =  -lam * w_row     (lam > 0)

luego el producto <s0*v0, w_row> tiene que ser NEGATIVO. Ese signo es invariante
a como elijas u0/v0 (voltear u0 voltea v0 y w_row a la vez), asi que es una
propiedad de DW, no una eleccion. Se reporta como `subtracts` por modulo.

Se fija ademas r_hat con su primera componente no despreciable positiva, solo para
que el fichero sea reproducible byte a byte entre ejecuciones.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------- FP8 decoders


def _table_e4m3fn():
    """256 entradas float64. S1 E4 M3, bias 7, variante finite (sin inf)."""
    t = np.zeros(256, dtype=np.float64)
    for b in range(256):
        s = -1.0 if (b >> 7) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = (m / 8.0) * (2.0**-6)  # subnormal
        elif e == 0xF and m == 0x7:
            v = np.nan  # unico NaN de la variante fn
        else:
            v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        t[b] = s * v
    return t


def _table_e8m0():
    """256 entradas float64. Exponente puro: 2^(b-127); 255 es NaN."""
    t = np.zeros(256, dtype=np.float64)
    for b in range(256):
        t[b] = np.nan if b == 255 else 2.0 ** (b - 127)
    return t


E4M3 = _table_e4m3fn()
E8M0 = _table_e8m0()

# ------------------------------------------------------- safetensors raw reader


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
    def __init__(self, root):
        self.root = root
        idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
        self.weight_map = idx["weight_map"]
        self._shards = {}

    def shard(self, fname):
        if fname not in self._shards:
            self._shards[fname] = Shard(os.path.join(self.root, fname))
        return self._shards[fname]

    def raw(self, key):
        return self.shard(self.weight_map[key]).raw(key)

    def dequant(self, module):
        """Devuelve W float64 [out, in] con los block scales 128x128 aplicados."""
        wb, wshape, wdt = self.raw(module + ".weight")
        sb, sshape, sdt = self.raw(module + ".scale")
        if wdt != "F8_E4M3" or sdt != "F8_E8M0":
            raise RuntimeError(f"dtypes inesperados en {module}: {wdt} / {sdt}")
        out, inn = wshape
        w = E4M3[wb].reshape(out, inn)
        s = E8M0[sb].reshape(sshape)
        bo = -(-out // s.shape[0])  # ceil, por si algun dia no es multiplo
        bi = -(-inn // s.shape[1])
        if (bo, bi) != (128, 128):
            print(f"  aviso: bloque {bo}x{bi} en {module}, no 128x128")
        # expandir los scales al tamano completo y multiplicar
        se = np.repeat(np.repeat(s, bo, axis=0), bi, axis=1)[:out, :inn]
        return w * se


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
            raise RuntimeError("DW es cero")
        u /= nu
        v = dw.T @ u
        s = np.linalg.norm(v)
        v /= s
        if abs(s - s_prev) <= tol * max(s, 1.0):
            return u, s, v, i + 1
        s_prev = s
    return u, s, v, iters


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--abl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    base = Checkpoint(args.base)
    abl = Checkpoint(args.abl)

    mods = sorted(
        {k.rsplit(".", 1)[0] for k in base.weight_map if ".attn.wo_b." in k},
        key=lambda m: (0 if m.startswith("layers.") else 1, int(m.split(".")[1])),
    )
    print(f"modulos wo_b en el checkpoint base: {len(mods)}")

    dirs, report = {}, []
    for n, m in enumerate(mods):
        Wb = base.dequant(m)
        Wa = abl.dequant(m)
        dW = Wa - Wb

        fro_dw = float(np.linalg.norm(dW))
        fro_wb = float(np.linalg.norm(Wb))
        if fro_dw == 0.0:
            print(f"[{n:2d}/{len(mods)}] {m:28s} SIN EDITAR")
            report.append({"module": m, "edited": False})
            del Wb, Wa, dW
            continue

        u0, s0, v0, it = top_triplet(dW)
        energy = float(s0 * s0 / (fro_dw * fro_dw))

        w_row = u0 @ Wb                      # u0^T W_base, en R^in
        n_wrow = float(np.linalg.norm(w_row))
        lam_eff = float(s0 / n_wrow)
        inner = float(np.dot(s0 * v0, w_row))            # < 0  => la edicion RESTA
        cos_v0 = float(abs(np.dot(v0, w_row)) / n_wrow)  # ~1 => es forma proyeccion

        # signo cosmetico y reproducible (irrelevante para el hook: r r^T)
        nz = np.argmax(np.abs(u0) > 1e-8)
        if u0[nz] < 0:
            u0 = -u0

        dirs[m] = u0.astype(np.float32)
        report.append(
            {
                "module": m,
                "edited": True,
                "shape": list(Wb.shape),
                "rank1_energy": energy,
                "delta_frobenius": float(fro_dw / fro_wb),
                "s0": float(s0),
                "lambda_eff": lam_eff,
                "subtracts": bool(inner < 0),
                "cos_v0_wrow": cos_v0,
                "power_iters": it,
            }
        )
        print(
            f"[{n:2d}/{len(mods)}] {m:28s} energy={energy:.6f} "
            f"delta={fro_dw/fro_wb:.6f} lam_eff={lam_eff:.4f} "
            f"resta={'si' if inner < 0 else 'NO'} cos={cos_v0:.6f}"
        )
        del Wb, Wa, dW

    ed = [r for r in report if r["edited"]]

    # ---- safetensors de salida, escrito a mano (mismo formato que leemos)
    header, blob, off = {}, bytearray(), 0
    for m in sorted(dirs):
        b = dirs[m].tobytes()
        header[m] = {
            "dtype": "F32",
            "shape": [dirs[m].shape[0]],
            "data_offsets": [off, off + len(b)],
        }
        blob += b
        off += len(b)
    header["__metadata__"] = {
        "source_base": os.path.basename(args.base.rstrip("/")),
        "source_abl": os.path.basename(args.abl.rstrip("/")),
        "modules": str(len(dirs)),
        "note": "rank-1 refusal directions, output-space (r_hat in R^4096)",
    }
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)))
        fh.write(hj)
        fh.write(blob)

    summary = {
        "modules_total": len(mods),
        "modules_edited": len(ed),
        "gate_modules_46": len(ed) == 46,
        "energy_min": min(r["rank1_energy"] for r in ed) if ed else None,
        "energy_mean": float(np.mean([r["rank1_energy"] for r in ed])) if ed else None,
        "gate_energy_0999": all(r["rank1_energy"] >= 0.999 for r in ed) if ed else False,
        "delta_frobenius_mean": float(np.mean([r["delta_frobenius"] for r in ed])) if ed else None,
        "lambda_eff_mean": float(np.mean([r["lambda_eff"] for r in ed])) if ed else None,
        "lambda_eff_min": min(r["lambda_eff"] for r in ed) if ed else None,
        "lambda_eff_max": max(r["lambda_eff"] for r in ed) if ed else None,
        "all_subtract": all(r["subtracts"] for r in ed) if ed else False,
        "cos_v0_wrow_min": min(r["cos_v0_wrow"] for r in ed) if ed else None,
        "out_bytes": 8 + len(hj) + len(blob),
    }
    json.dump({"summary": summary, "modules": report}, open(args.report, "w"), indent=2)

    print("\n================ PUERTAS DE LA FASE 1 ================")
    print(f"1. modulos editados      : {summary['modules_edited']}  (esperado 46)")
    print(f"2. energia rank-1 minima : {summary['energy_min']}  (esperado >= 0.999)")
    print(f"3. delta Frobenius medio : {summary['delta_frobenius_mean']}  (esperado 0.0587)")
    print(f"4. lambda_eff medio      : {summary['lambda_eff_mean']}  (esperado ~1.7)")
    print(f"   lambda_eff rango      : {summary['lambda_eff_min']} .. {summary['lambda_eff_max']}")
    print(f"   la edicion resta      : {summary['all_subtract']}")
    print(f"   cos(v0, u0^T Wbase)   : {summary['cos_v0_wrow_min']} (min; ~1 = forma proyeccion)")
    print(f"   salida                : {summary['out_bytes']} bytes")
    print("EXTRACT-DONE")


if __name__ == "__main__":
    sys.exit(main())
