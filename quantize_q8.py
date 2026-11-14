#!/usr/bin/env python3
"""Quantize a Tev1 SP f16-decision GGUF to Q8_0.

Usage:
  python quantize_q8.py <input.gguf> <output.gguf>
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "llama.cpp/gguf-py"))
import numpy as np  # noqa: E402
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402
from gguf.quants import quantize, GGMLQuantizationType  # noqa: E402
from gguf.quants import quant_shape_to_byte_shape  # noqa: E402
from tqdm import tqdm  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    inp, outp = Path(sys.argv[1]), Path(sys.argv[2])
    qtype = GGMLQuantizationType.Q8_0

    reader = GGUFReader(str(inp), "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    print(f"arch: {arch}")

    writer = GGUFWriter(str(outp), arch=arch, endianess=reader.endianess)

    # Copy all metadata fields
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    # Tensors that must stay float (norms + SSM params).
    # The deployed tev1 GGUF keeps attn_norm / post_attention_norm / ssm_norm /
    # ssm_a / ssm_conv1d / ssm_dt.bias as F32; quantizing them to Q8_0 breaks
    # the RMSNorm / SSM binary ops at runtime (binary_op: unsupported q8_0).
    # NOTE: match on the exact trailing tensor name, NOT substring:
    # "ssm_a" must NOT match "ssm_alpha" (deployed tev1 has ssm_alpha/ssm_beta
    # as Q8_0; a substring match left ssm_alpha as F16 and crashed the SSM
    # binary op with a type mismatch).
    KEEP_FLOAT_NAMES = (
        "attn_norm.weight", "post_attention_norm.weight", "ssm_norm.weight",
        "ssm_a", "ssm_conv1d.weight", "ssm_dt.bias",
        "attn_q_norm.weight", "attn_k_norm.weight",
        "output_norm.weight",
    )

    def keep_float(name: str) -> bool:
        return any(name.endswith(p) for p in KEEP_FLOAT_NAMES)

    # Quantize tensors; keep tensors that can't be Q8_0 (small / non-aligned / SSM) as f32
    # (match the deployed tev1 GGUF which keeps norm/SSM tensors as F32)
    for tensor in reader.tensors:
        arr = np.asarray(tensor.data, dtype=np.float32)
        if keep_float(tensor.name):
            f32 = arr  # keep as f32
            writer.add_tensor_info(tensor.name, tensor.data.shape, np.float32, f32.nbytes)
            writer.tensors[-1][tensor.name].tensor = f32
            continue
        try:
            qbytes = quantize(arr, qtype)
            byte_shape = quant_shape_to_byte_shape(tensor.data.shape, qtype)
            writer.add_tensor_info(tensor.name, byte_shape, np.uint8, qbytes.nbytes, raw_dtype=qtype)
            writer.tensors[-1][tensor.name].tensor = qbytes
        except Exception:
            # keep as f32
            f32 = arr
            writer.add_tensor_info(tensor.name, tensor.data.shape, np.float32, f32.nbytes)
            writer.tensors[-1][tensor.name].tensor = f32

    total_bytes = sum(t.nbytes for t in writer.tensors[-1].values())
    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    items = list(writer.tensors[-1].items())
    for name, ti in items:
        writer.write_tensor_data(ti.tensor, tensor_endianess=reader.endianess)
        bar.update(ti.nbytes)
    writer.close()
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
