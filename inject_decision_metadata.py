#!/usr/bin/env python3
"""Inject pplx-decider decision metadata into a Tev1 SP f16 GGUF.

Adds/overrides:
  - <arch>.decision.type = 'pplx-decider'
  - tokenizer.chat_template.systemone = <deployed systemone template>
  - tokenizer.chat_templates = ['systemone']

Usage:
  python inject_decision_metadata.py <input.gguf> <output.gguf> <systemone-template-file>
"""
import sys
import os
from pathlib import Path

# Load the local gguf package
sys.path.insert(0, str(Path(__file__).parent / "llama.cpp/gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402
from tqdm import tqdm  # noqa: E402


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    inp, outp, tmpl_file = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    template = tmpl_file.read_text(encoding="utf-8")

    reader = GGUFReader(str(inp), "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    print(f"arch: {arch}")

    writer = GGUFWriter(str(outp), arch=arch, endianess=reader.endianess)

    # Copy all existing fields except the ones we override / the main chat template
    override_keys = {
        f"{arch}.decision.type",
        "tokenizer.chat_template.systemone",
        "tokenizer.chat_templates",
    }
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name in override_keys:
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    # Add the decision metadata
    writer.add_key_value(f"{arch}.decision.type", "pplx-decider", gguf.GGUFValueType.STRING)
    writer.add_key_value("tokenizer.chat_template.systemone", template, gguf.GGUFValueType.STRING)
    writer.add_key_value("tokenizer.chat_templates", ["systemone"], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)

    # Copy tensors
    total_bytes = 0
    for tensor in reader.tensors:
        total_bytes += tensor.n_bytes
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        bar.update(tensor.n_bytes)
    writer.close()
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
