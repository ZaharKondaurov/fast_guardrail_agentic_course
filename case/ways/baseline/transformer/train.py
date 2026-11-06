"""Train a small byte transformer and export dynamic INT8 ONNX inference."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..calibration import SEED, calibrate, split_train
from .solve import CONFIG_PATH, MODEL_PATH, SEQUENCE_LENGTH, encode, load_session


class ByteTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(258, 32, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(SEQUENCE_LENGTH, 32))
        self.encoder = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.1, batch_first=True)
        self.output = nn.Linear(32, 1)

    def forward(self, tokens):
        hidden = self.encoder(self.embedding(tokens) + self.position, src_key_padding_mask=tokens == 0)
        return self.output(hidden[:, 0]).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    texts, heldout, labels, heldout_labels = split_train(args.train)
    data = TensorDataset(torch.from_numpy(encode(texts)), torch.tensor(labels, dtype=torch.float32))
    batches = DataLoader(data, batch_size=64, shuffle=True)
    model = ByteTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    loss = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        for tokens, expected in batches:
            optimizer.zero_grad()
            loss(model(tokens), expected).backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)

    with TemporaryDirectory() as temporary:
        exported = Path(temporary) / "model.fp32.onnx"
        quantized = Path(temporary) / "model.int8.onnx"
        torch.backends.mha.set_fastpath_enabled(False)
        torch.onnx.export(
            model.eval(), data.tensors[0][:1], exported, opset_version=17, dynamo=False,
            input_names=["tokens"], output_names=["scores"],
            dynamic_axes={"tokens": {0: "batch"}, "scores": {0: "batch"}},
        )
        quantize_dynamic(
            exported, quantized, op_types_to_quantize=["MatMul", "Gemm"],
            weight_type=QuantType.QInt8,
        )
        session = load_session(quantized)
        scores = np.concatenate([
            session.run(None, {"tokens": encode(heldout[start:start + 64])})[0]
            for start in range(0, len(heldout), 64)
        ])
        cutoff = calibrate(scores, heldout_labels)
        MODEL_PATH.write_bytes(quantized.read_bytes())
    CONFIG_PATH.write_text(json.dumps({"cutoff": cutoff}) + "\n", encoding="utf-8")
    print(f"{MODEL_PATH}: cutoff {cutoff:.4f}")


if __name__ == "__main__":
    main()
