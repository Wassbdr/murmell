# Murmell

Murmell finds which parts of a spliced, inpainted, or cropped image still carry an invisible provenance watermark.

<img src="showcase/demo.gif" alt="Gradio shows the edited image, decoded regime and predicted watermark mask" width="560">

```sh
python -m pip install -r benchmarks/requirements.txt -e .
make weights
murmell-detect task1_watermark_localisation/examples/edited.png --method auto -o mask.png
```

The CPU rebuild downloads the official TrustMark Q weights and trains the U-Net once.

**Public TrustMark benchmark:** known message, 60 scenes per split, 128 px U-Nets over three training seeds,
Intel Xeon with 8 threads. The product instead decodes the image without being given a message.
See the [protocol and results](benchmarks/results.md) for foreground IoU and per-family tables.

| Model | Hard mIoU | Held-out mIoU |
|---|---:|---:|
| Murmell with decoder, boxes and MobileSAM | 0.6257 | 0.4093 |
| TrustMark decoder U-Net | 0.4934 | 0.3997 |
| ImageNet U-Net | 0.4870 | 0.3402 |
| TrustMark sliding-window | 0.3433 | 0.3712 |
| All-empty | 0.3232 | 0.3026 |
| All-full | 0.1768 | 0.1974 |

```text
                        +-> U-Net -> cell probabilities --------+
Edited RGB -> decode --+                                      +-> 128 x 128 mask
                        +-> message agreement -> box / regime --+
```

**Competition:** team of 3, 1st overall of 44 teams, in the event organised by CISPA and Fraunhofer AISEC with Meta.
The team placed **2nd on Task 1**. The [competition write-up](task1_watermark_localisation/README.md) is separate
from the public benchmark; no competition data were used for the latter.

## Use and reproduce

- [Tutorial](docs/tutorial.md): CLI, Python API, Gradio and ONNX export on your own images.
- [Model card](docs/model_card.md): calibration, known limitations and CPU runtime.
- [Benchmark sources and licences](benchmarks/invisible-sources.md): public images, official embedders and checkpoints.
- [Architecture](docs/architecture.md): regeneration, inputs and evaluation isolation.
- Run `make install`, `make test` and `make lint` for development, or `make showcase` to rebuild the demo.

## Other tasks

| Task | Run | Guide |
|---|---|---|
| 2. Model equality | `make task2` | [Synthetic endpoint benchmark](docs/task2.md) |
| 3. Unlearning | `make task3` | [Synthetic refit backtest](docs/task3.md) |

## Roadmap and known limitations

- The public generator draws from four photos. Split and family scores are not evidence of arbitrary-image accuracy.
- A failed global decode gives an empty mask. Cropping or splicing can hide the message even where marked cells remain.
- Known-message reference scores do not measure product performance. Task 2 and Task 3 use simulations.
- The earlier visible-watermark detector is a different product, preserved in [experiments/visible](experiments/visible/).

[MIT licence](LICENSE).
