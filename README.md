# perch-embedding-onnx

The Perch 2.0 bird vocalization model by Google as an ONNX file that computes only the
1536-value embedding. Meant for downstream classifiers trained on these embeddings. The file
is in the releases of this repository, with its SHA-256.

| Release | File | Bytes | SHA-256 |
|---|---|---|---|
| perch-model-1 | `perch_v2_no_dft_flat_emb.onnx` | 49,450,622 | `5fdf18e3f603da1bc6abf019ae075ee4ef6c3a6b696785204e07a44851330817` |

## Original work

Perch 2.0, `google/bird-vocalization-classifier`, variation `perch_v2`:
https://www.kaggle.com/models/google/bird-vocalization-classifier

Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0, see `LICENSE`.

## Changes

1. Converted from TensorFlow to ONNX, with the DFT node rewritten as a MatMul, by justinchuby:
   `perch_v2_no_dft.onnx` from https://huggingface.co/justinchuby/Perch-onnx
   (SHA-256 `4dcf71c18a147198545944bb5149697e89e3ad2e16637fa8f0edf6d13035a017`).
2. In this repository's file, starting from that model:
   - every MatMul of an N-dimensional input (N >= 3) with a 2-dimensional constant weight is
     rewritten as Reshape to [-1, K], MatMul, Reshape back (3 nodes, one of them in the
     classification head), which lets GPU runtimes run batches of more than one window,
   - the graph is cut to the `embedding` output with `onnx.utils.extract_model`, which
     removes the 14,795-class classification head and the spectrogram and spatial embedding
     outputs.

The embeddings stay bit equal to `perch_v2_no_dft.onnx` on the ONNX Runtime CPU provider.

## Use

Input `inputs` [batch, 160000] float32: 5 s of mono audio at 32 kHz. Output `embedding`
[batch, 1536] float32. With the DirectML provider, run every call with the same batch size
(pad the last batch): a session runs the first input shape it sees much faster than others.

## Disclaimer

Distributed on an "AS IS" basis, without warranties or conditions of any kind, as stated in
the Apache License 2.0. This is not an official Google release.
