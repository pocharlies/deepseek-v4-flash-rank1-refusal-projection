# SPDX-License-Identifier: Apache-2.0
"""Build DeepGEMM's pybind extension against the active vLLM image's torch.

The DeepGEMM source is pinned by the BuildKit named context.  Compiling the
small host extension in the destination image avoids carrying an extension
linked against the older DSpark image's libtorch ABI.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension


if len(sys.argv) != 4:
    raise SystemExit("usage: build_deepgemm_extension.py <src> <out> <target-python>")

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
target_python = sys.argv[3]
out.mkdir(parents=True, exist_ok=True)

info = json.loads(
    subprocess.check_output(
        [
            target_python,
            "-c",
            "import json,sysconfig; print(json.dumps({k: "
            "sysconfig.get_config_var(k) for k in ('EXT_SUFFIX','INCLUDEPY')}))",
        ]
    ).decode()
)

cuda_home = cpp_extension.CUDA_HOME
if cuda_home is None:
    raise SystemExit("CUDA_HOME not found; cannot build DeepGEMM")
nvrtc_candidates = sorted(
    Path(cuda_home).glob("lib64/libnvrtc.so*"),
    key=lambda path: (path.name != "libnvrtc.so", path.name),
)
if not nvrtc_candidates:
    raise SystemExit(f"NVRTC library not found below {cuda_home}/lib64")
nvrtc_library = nvrtc_candidates[0]

includes = [
    info["INCLUDEPY"],
    f"{cuda_home}/include",
    f"{cuda_home}/include/cccl",
    str(src / "csrc"),
    str(src / "deep_gemm/include"),
    str(src / "third-party/cutlass/include"),
    str(src / "third-party/cutlass/tools/util/include"),
    str(src / "third-party/fmt/include"),
    *cpp_extension.include_paths(device_type="cuda"),
]

command = [
    os.environ.get("CXX", "g++"),
    "-shared",
    "-fPIC",
    "-std=c++20",
    "-O3",
    "-g0",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
    "-DTORCH_EXTENSION_NAME=_C",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
    *(f"-I{path}" for path in includes),
    str(src / "csrc/python_api.cpp"),
    *(f"-L{path}" for path in cpp_extension.library_paths(device_type="cuda")),
    f"-L{cuda_home}/lib64",
    "-ltorch",
    "-ltorch_python",
    "-ltorch_cpu",
    "-ltorch_cuda",
    "-lc10",
    "-lc10_cuda",
    "-lcudart",
    str(nvrtc_library),
    "-o",
    str(out / f"_C{info['EXT_SUFFIX']}"),
]
print("[build_deepgemm_extension] " + " ".join(command), flush=True)
subprocess.check_call(command)
