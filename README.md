# Murmell

Murmell finds the pixels covered by visible watermarks on CPU, for image reviewers and developers who need a binary mask.

<img src="showcase/demo.gif" alt="Gradio demo on two hard scenes and a faint held-out mark" width="560">

```sh
python -m pip install -r benchmarks/requirements.txt -e .
murmell-detect task1_watermark_localisation/examples/good.png -o mask.png
```

**Comparison:** mIoU on 192 synthetic images per split; training seeds 0, 1, 2; 1,600 updates per seed at 128 px.
CPU timing: Intel Xeon, 8 threads, batch 1, PyTorch.

| Detector | Hard mIoU | Held-out mIoU | ms/image |
|---|---:|---:|---:|
| Murmell local contrast (opt-in) | **0.6254** | 0.5243 | 11.49 |
| Murmell residual supervision (experiment) | 0.6015 | 0.5062 | 10.10 |
| Murmell retrained round 1 (default recipe) | 0.5995 | 0.5308 | 9.80 |
| WDNet-style | 0.5874 | 0.5083 | 8.55 |
| Split then Refine-style | 0.5873 | 0.5181 | 13.95 |
| SLBR-style | 0.5816 | 0.5154 | 7.70 |
| Plain RGB U-Net | 0.5784 | 0.5077 | 7.31 |
| All-empty mask | 0.5219 | **0.5415** | n/a |

```text
Image -> 128 px RGB + high-pass detail -> U-Net -> threshold -> mask at input size
```

## Use and reproduce

- [Tutorial](docs/tutorial.md): run the CLI, Python API, Gradio app or ONNX export on your own images.
- [Model card](docs/model_card.md): shipped seed-0 scores, foreground IoU, failure examples and CPU runtime.
- [Benchmark](benchmarks/results.md): all seeds, model changes, failure slices and parity checks. Run `make bench`.
- [Architecture](docs/architecture.md): inference, checkpoint selection and evaluation isolation.
- Run `make install`, `make test` and `make lint` for development; `make showcase` to rebuild the demo assets.

The bundled model ships at threshold 0.900. Use `--variant contrast` to select the local-contrast checkpoint.

## Other tasks

| Task | Run | Guide |
|---|---|---|
| 2. Model equality | `make task2` | [Synthetic endpoint benchmark](docs/task2.md) |
| 3. Unlearning | `make task3` | [Synthetic refit backtest](docs/task3.md) |

Murmell finished first of 44 teams in the original competition. The archived task write-ups describe that work;
the current benchmark is a separate visible-watermark experiment.

## Roadmap and known limitations

- Neither new variant beat round 1 or the all-empty baseline on held-out mIoU. Faint marks and unfamiliar textured
  backgrounds are the main failure modes. Next, evaluate on licensed, manually labelled photographs.
- The paper-style references are small CPU reimplementations, not the authors' full systems or weights.
  CLWD and LVW were not used because reuse and redistribution terms were unclear.
- The detector was not trained to locate invisible watermarks. Tasks 2 and 3 run simulations, not the competition
  endpoints or scoring service.

[MIT licence](LICENSE).
