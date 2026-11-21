# Third-Party Notices

This project includes third-party model weights and runtime components. This file records their origin, license, and checksums.

## StackMFF V3 (v2.0 MIT Release) — Multi-Focus Image Stack Fusion Model

- **Upstream**: https://github.com/Xinzhe99/StackMFF-V3
- **Release**: `v2.0` (published 2026-10-21)
- **License**: MIT License (`Copyright (c) 2026 Xinzhe Xie`).
  - Note: Releases up to and including `v2.0` were published under the MIT License and remain under the MIT License for those distributed snapshots (as confirmed by upstream `v2.0/LICENSE` and the `StackMFF-V3` README). This project derives its bundled ONNX graphs strictly from the MIT-licensed `v2.0` checkpoint and does **not** include any `v2.1` source-available code or retrained V1/V2 checkpoints.
- **Source Checkpoint**: `stackmff_v3_final.pth` from `v2.0` release
  - Size: `658,587` bytes
  - SHA-256: `39b2ea3738e8fea1676c379238d6be81a87988fc9952dbd6ac3825d5d7249f46`
- **Bundled Exported ONNX Models** (`focus_stack_app/fusion/weights/`):
  - `stackmff_v3_final.onnx` (FP32, opset 17, CPU execution): `1,260,491` bytes, SHA-256 `7b92bf31c984eac2b93d6000158b4802b7d62c23140d52e28a47676f58d5c4f7`
  - `stackmff_v3_final_fp16.onnx` (FP16 weights/activations with FP32 I/O, opset 17, DirectML/CUDA GPU execution): `855,160` bytes, SHA-256 `1439c468a537fbb20edb42c39f0ea8759718d7f4f9898e79630a651aadc9cc77`
- **Citation**:
  - Xie, X., Guo, B., Li, B., He, S., Gu, Y., Jiang, Q., & Li, P. (2026). *StackMFF V3: Order-Invariant Focus Stacking via Pixel-Wise Source Classification*.

```text
MIT License

Copyright (c) 2026 Xinzhe Xie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ONNX Runtime / ONNX Runtime DirectML

- **Upstream**: https://github.com/microsoft/onnxruntime
- **License**: MIT License, Copyright (c) Microsoft Corporation.
