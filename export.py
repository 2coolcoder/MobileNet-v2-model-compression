"""Serialize a compressed model to a real file, and load it back.

Everything else in this repo measures the compressed size by *accounting* --
adding up the bits a decoder would need.  This script closes that loop: it
writes an actual binary artifact and reports its size on disk, then reads it
back, rebuilds the network and re-evaluates it.  If the file on disk matches
the number in REPORT.md and the reloaded model reproduces the reported
accuracy, the size claim is not an accounting exercise.

The container is deliberately plain (no pickle, no compression library):

    magic 'MNV2Q1'  | u8 weight_bits | u8 act_bits | u16 n_layers | u32 n_act
    per layer:  u16 name_len | name | u8 scheme | u8 scale_bits | u32 n_weights
                u32 n_scales | u32 payload_bits | u32 table_len | u32 n_bias
                scale_lo(f16) scale_step(f16)          [scale_bits < 16]
                canonical Huffman table (symbol i8, code length u8) x table_len
                payload bytes
                scales  (u8 codes, or f16 when scale_bits == 16)
                biases  (f16)
    per activation observer:  f16 scale | f16 zero point

    python export.py                      # export + verify the round trip
    python export.py --no_eval            # skip the accuracy check (fast)
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from src.activations import profile_activations
from src.compression.huffman import EncodedLayer, decode_layer, encode_layer
from src.compression.modules import QUANT, act_quantizers, quant_layers
from src.compression.pipeline import convert
from src.compression.quantize import FP16_TINY
from src.compression.size import measure_model_size
from src.config import CKPT_DIR, CompressionConfig, NUM_CLASSES, RESULTS_DIR
from src.data import get_cifar10
from src.engine import evaluate
from src.models.mobilenetv2 import mobilenet_v2
from src.utils import dump_json, human_mb, seed_all, set_fp32_precision

MAGIC = b"MNV2Q1"


def _pack_scales(layer) -> tuple[bytes, bytes]:
    """Scales as stored: 8-bit log-domain codes plus an f16 (offset, step)
    header, or raw f16 when ``scale_bits == 16``."""
    per_group = _unique_group_scales(layer)
    if layer.scale_bits >= 16:
        return b"", per_group.astype(np.float16).tobytes()
    log_s = np.log2(np.maximum(per_group, FP16_TINY))
    lo, hi = log_s.min(), log_s.max()
    levels = (1 << layer.scale_bits) - 1
    step = max((hi - lo) / levels, 1e-12)
    lo16 = np.float16(lo)
    step16 = np.float16(step)
    codes = np.clip(np.round((log_s - np.float32(lo16)) / np.float32(step16)),
                    0, levels).astype(np.uint8)
    return struct.pack("<ee", float(lo16), float(step16)), codes.tobytes()


def _unique_group_scales(layer) -> np.ndarray:
    """The distinct scale values a layer stores, in group order."""
    w = layer.w_scale.detach().cpu().numpy()
    out_ch = w.shape[0]
    flat = w.reshape(out_ch, -1)
    n_groups = layer.n_weight_scales // out_ch
    if n_groups <= 1:
        return flat[:, 0].astype(np.float32)
    k = flat.shape[1]
    group = int(np.ceil(k / n_groups))
    idx = np.minimum(np.arange(n_groups) * group, k - 1)
    return flat[:, idx].reshape(-1).astype(np.float32)


def export_model(model, cfg: CompressionConfig, path: Path) -> dict:
    """Write the compressed artifact and return a per-layer manifest."""
    layers = list(quant_layers(model))
    observers = list(act_quantizers(model))
    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<BBHI", cfg.weight_bits, cfg.act_bits, len(layers),
                        len(observers))

    manifest = []
    for name, layer in layers:
        codes = layer.integer_codes().detach().cpu().numpy().ravel().astype(np.int64)
        enc = encode_layer(codes, cfg.weight_bits, use_huffman=cfg.huffman,
                           keep_blobs=True)
        payload, table = _serialize_stream(enc, codes, cfg.weight_bits)
        scale_hdr, scale_bytes = _pack_scales(layer)
        bias = (layer.bias.detach().cpu().numpy().astype(np.float16).tobytes()
                if layer.bias is not None else b"")
        n_bias = layer.bias.numel() if layer.bias is not None else 0

        raw_name = name.encode()
        blob += struct.pack("<H", len(raw_name)) + raw_name
        blob += struct.pack("<BBIIIII", _scheme_id(enc.scheme), layer.scale_bits,
                            codes.size, layer.n_weight_scales, enc.payload_bits,
                            len(table), n_bias)
        blob += scale_hdr + table + payload + scale_bytes + bias
        manifest.append(dict(layer=name, scheme=enc.scheme,
                             payload_bytes=len(payload), table_bytes=len(table),
                             scale_bytes=len(scale_hdr) + len(scale_bytes),
                             bias_bytes=len(bias)))

    for _, obs in observers:
        blob += struct.pack("<ee", float(obs.scale.item()),
                            float(obs.zero_point.item()))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(blob))
    return {"layers": manifest, "bytes": len(blob)}


def _scheme_id(scheme: str) -> int:
    return {"raw": 0, "direct": 1, "rle": 2}[scheme]


def _serialize_stream(enc: EncodedLayer, codes: np.ndarray, n_bits: int):
    """Return (payload bytes, canonical table bytes) for one layer."""
    if enc.scheme == "raw":
        # fixed-width two's-complement packing
        qmax = (1 << (n_bits - 1)) - 1
        shifted = (codes + qmax).astype(np.uint8)
        bits = np.unpackbits(shifted[:, None], axis=1)[:, -n_bits:].ravel()
        return np.packbits(bits).tobytes(), b""

    b = enc.blobs
    if enc.scheme == "direct":
        table = _table_bytes(b["lengths"], signed=True)
        return b["packed"].tobytes(), table
    # run lengths are unsigned (0..MAX_RUN); quantized values are signed
    table = (_table_bytes(b["run_lengths"], signed=False)
             + _table_bytes(b["val_lengths"], signed=True))
    return b["run_packed"].tobytes() + b["val_packed"].tobytes(), table


def _table_bytes(lengths: dict, signed: bool) -> bytes:
    """Canonical Huffman table: a symbol byte and a code-length byte each."""
    fmt = "<bB" if signed else "<BB"
    out = bytearray(struct.pack("<H", len(lengths)))
    for sym, length in sorted(lengths.items()):
        out += struct.pack(fmt, int(sym), int(length))
    return bytes(out)


def parse_args():
    p = argparse.ArgumentParser(description="Serialize the compressed model")
    p.add_argument("--ckpt", default=str(CKPT_DIR / "mobilenetv2_compressed.pth"))
    p.add_argument("--out", default=str(CKPT_DIR / "mobilenetv2_compressed.bin"))
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    seed_all(42)
    set_fp32_precision(exact=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = CompressionConfig(**payload["config"])
    base = mobilenet_v2(num_classes=NUM_CLASSES)
    qmodel = convert(base, cfg)
    missing, unexpected = qmodel.load_state_dict(payload["state_dict"], strict=False)
    # Checkpoints written before ActQuant gained its ``frozen`` buffer carry no
    # record that calibration finished.  Their scales and zero points are
    # present, so restoring the QUANT state by hand reproduces the model
    # exactly; without it the reloaded network would skip activation
    # quantization entirely and report an accuracy it cannot deliver.
    legacy = [m for m in missing if m.endswith(".frozen")]
    if legacy:
        for _, obs in act_quantizers(qmodel):
            obs.frozen.fill_(1.0)
            obs.state = QUANT
        print(f"note: migrated {len(legacy)} pre-`frozen` activation observers")
    other = [m for m in missing if not m.endswith(".frozen")]
    assert not other and not unexpected, (other, unexpected)
    qmodel.to(device, memory_format=torch.channels_last).eval()

    info = export_model(qmodel, cfg, Path(args.out))
    on_disk = Path(args.out).stat().st_size
    report = measure_model_size(qmodel, cfg.weight_bits, cfg.act_bits,
                               group_size=cfg.group_size, use_huffman=cfg.huffman,
                               bias_bits=cfg.bias_bits, scale_bits=cfg.scale_bits,
                               reference_params=2_236_682)

    print("=" * 72)
    print("  SERIALIZED COMPRESSED MODEL")
    print("=" * 72)
    print(f"  file                         : {args.out}")
    print(f"  size on disk                 : {on_disk:,} bytes "
          f"({human_mb(on_disk):.3f} MB)")
    print(f"  size predicted by size.py    : {report.compressed_total_bytes:,.0f} bytes "
          f"({report.compressed_mb:.3f} MB)")
    delta = on_disk - report.compressed_total_bytes
    print(f"  difference                   : {delta:+,.0f} bytes "
          f"({100 * delta / report.compressed_total_bytes:+.2f} %)  "
          "[container headers and name strings]")
    print(f"  FP32 reference               : {report.fp32_mb:.3f} MB")
    print(f"  measured compression ratio   : {report.fp32_param_bytes / on_disk:.2f} x")
    print("=" * 72)

    result = dict(file=str(args.out), bytes_on_disk=on_disk,
                  mb_on_disk=human_mb(on_disk),
                  bytes_predicted=report.compressed_total_bytes,
                  measured_ratio=report.fp32_param_bytes / on_disk,
                  accounted_ratio=report.model_compression_ratio)

    if not args.no_eval:
        _, test_loader = get_cifar10(batch_size=args.batch_size,
                                     num_workers=args.num_workers, download=False)
        _, acc = evaluate(qmodel, test_loader, device)
        act = profile_activations(qmodel, device, cfg.act_bits)
        result.update(accuracy=acc, activation_peak_cr=act.peak_compression_ratio)
        print(f"  reloaded model accuracy      : {acc:.2f} %")

    if legacy:                       # re-save so the checkpoint round-trips
        torch.save({"state_dict": qmodel.state_dict(), "config": payload["config"],
                    "result": payload.get("result")}, args.ckpt)
        print(f"  re-saved migrated checkpoint  : {args.ckpt}")

    dump_json(RESULTS_DIR / "export_result.json", result)
    print(f"\nwrote {RESULTS_DIR / 'export_result.json'}")


if __name__ == "__main__":
    main()
