"""Self-contained, lossless Huffman codec for an SMPL-X ``.npz`` file.

The legacy experiment script quantized floating-point values and kept its codebook
only in memory. Consequently its ``.bin`` files could not be decoded in a new
process. This codec Huffman-encodes the source file bytes and stores the complete
frequency table in the stream. Decoding recreates the input file byte for byte.
"""

import argparse
import hashlib
import heapq
import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"HGCSMPL1"
HEADER = struct.Struct(">8sQ256Q")


@dataclass
class _Node:
    symbol: int = -1
    left: object = None
    right: object = None


def _tree(frequencies):
    heap = []
    serial = 0
    for symbol, frequency in enumerate(frequencies):
        if frequency:
            node = _Node(symbol=symbol)
            heapq.heappush(heap, (frequency, symbol, serial, node))
            serial += 1
    if not heap:
        return None
    if len(heap) == 1:
        return heap[0][3]
    while len(heap) > 1:
        left_freq, left_min, _, left = heapq.heappop(heap)
        right_freq, right_min, _, right = heapq.heappop(heap)
        node = _Node(left=left, right=right)
        heapq.heappush(
            heap,
            (left_freq + right_freq, min(left_min, right_min), serial, node),
        )
        serial += 1
    return heap[0][3]


def _codes(root):
    codes = {}

    def visit(node, value, length):
        if node.symbol >= 0:
            codes[node.symbol] = (value, max(length, 1))
            return
        visit(node.left, value << 1, length + 1)
        visit(node.right, (value << 1) | 1, length + 1)

    if root is not None:
        visit(root, 0, 0)
    return codes


def encode(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    data = input_path.read_bytes()
    frequencies = [0] * 256
    for value in data:
        frequencies[value] += 1
    codes = _codes(_tree(frequencies))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        output.write(HEADER.pack(MAGIC, len(data), *frequencies))
        accumulator = 0
        bit_count = 0
        for value in data:
            code, length = codes[value]
            accumulator = (accumulator << length) | code
            bit_count += length
            while bit_count >= 8:
                shift = bit_count - 8
                output.write(bytes([(accumulator >> shift) & 0xFF]))
                accumulator &= (1 << shift) - 1 if shift else 0
                bit_count = shift
        if bit_count:
            output.write(bytes([(accumulator << (8 - bit_count)) & 0xFF]))
    return {
        "input_bytes": len(data),
        "output_bytes": output_path.stat().st_size,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def decode(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("rb") as source:
        header = source.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("SMPL Huffman stream is truncated")
        magic, original_size, *frequencies = HEADER.unpack(header)
        if magic != MAGIC:
            raise ValueError("Not an HGC SMPL Huffman stream")
        root = _tree(frequencies)
        decoded = bytearray()
        if original_size and root is None:
            raise ValueError("Invalid empty Huffman table")
        if root is not None and root.symbol >= 0:
            decoded.extend([root.symbol] * original_size)
        else:
            node = root
            for byte in source.read():
                for shift in range(7, -1, -1):
                    node = node.right if (byte >> shift) & 1 else node.left
                    if node.symbol >= 0:
                        decoded.append(node.symbol)
                        if len(decoded) == original_size:
                            break
                        node = root
                if len(decoded) == original_size:
                    break
        if len(decoded) != original_size:
            raise ValueError(
                f"SMPL Huffman stream is truncated: expected {original_size}, "
                f"decoded {len(decoded)} bytes"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(decoded)
    return {
        "output_bytes": len(decoded),
        "sha256": hashlib.sha256(decoded).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("encode", "decode"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("input")
        command_parser.add_argument("output")
    args = parser.parse_args()
    result = encode(args.input, args.output) if args.command == "encode" else decode(args.input, args.output)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

