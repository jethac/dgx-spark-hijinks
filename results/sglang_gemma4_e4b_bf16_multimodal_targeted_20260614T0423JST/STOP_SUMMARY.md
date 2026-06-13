# SGLang Gemma 4 E4B bf16 multimodal targeted smoke

Status: RED

- Scope: bf16/auto-KV multimodal request-path baseline only; no NVFP4 or capacity claim.
- Model: google/gemma-4-E4B-it
- Image: ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e
- Prompt change vs generic smoke: targeted image question requires `iron`/`ironing` for the banked yellow-cab asset.

## Probe rows
- text: http=True keyword=True deterministic=True keywords=['tokyo']
  - round 1: TOKYO
  - round 2: TOKYO
- image: http=True keyword=False deterministic=True keywords=['iron', 'ironing']
  - round 1: The man is hanging laundry on the back of the yellow cab.
  - round 2: The man is hanging laundry on the back of the yellow cab.
- audio: http=True keyword=True deterministic=True keywords=['quilter', 'apostle', 'middle', 'gospel']
  - round 1: Mr Quilter is the apostle of the middle classes and we are glad to welcome his gospel.
  - round 2: Mr Quilter is the apostle of the middle classes and we are glad to welcome his gospel.
