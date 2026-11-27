# Public invisible-watermark benchmark

Install the [pinned CPU environment](requirements.txt) with Python 3.10 or 3.11:

```sh
python -m pip install -r benchmarks/requirements.txt -e '.[onnx]'
make bench
```

The second command is the seed-0 ImageNet learnability gate. It caches the fixed corpus under
`task1_watermark_localisation/work/round3/trustmark-bench/`. The following commands train the remaining
TrustMark models and score post-processing from the Task 1 directory. Use the MobileSAM source and
checkpoint listed in [the source notes](invisible-sources.md).

```sh
cd task1_watermark_localisation
work=work/round3/trustmark-bench
python -m src.invisible_compare --work "$work" --sliding
for seed in 0 1 2; do
  if [ "$seed" != 0 ]; then
    python -m src.invisible_bench --work "$work" --initialize imagenet --seed "$seed"
  fi
  python -m src.invisible_bench --work "$work" --initialize decoder --seed "$seed"
  PYTHONPATH=/path/to/MobileSAM python -m src.invisible_compare --work "$work" \
    --sam /path/to/MobileSAM/weights/mobile_sam.pt --maps "$work/trustmark-decoder-128-$seed-1600.npz"
done
```

Sliding-window decode scores are cached per scene. Each of the four window scales broadcasts window bit
accuracy to covered cells, retaining the maximum where windows overlap. MobileSAM is optional for training
and for the product. The known-message benchmark scores it as a separate post-processing stage.

For WAM and PixelSeal, install each official source checkout's inference dependencies and put its root and
the absolute Task 1 directory on `PYTHONPATH`. Run from the respective source checkout so its checkpoint
configuration paths resolve. Download only the MIT checkpoints listed in the [source notes](invisible-sources.md).
Install the additional versions in [meta-requirements.txt](meta-requirements.txt) for the optional Meta runs.

```sh
python -m src.invisible_meta_bench wam /path/to/wam_mit.pth --work /path/to/work/round3/wam-bench \
  --trustmark-checkpoint /path/to/work/round3/trustmark-bench/trustmark-decoder-128-0-1600.pt
python -m src.invisible_meta_bench pixelseal /path/to/pixelseal.pth --work /path/to/work/round3/pixelseal-bench
```

For the product on the same CPU, run `make weights` from the repository root. Measure warmed batch-1
ONNX and PyTorch inference from the Task 1 directory, then render the result tables from the repository root:

```sh
python -m src.invisible_onnx_bench work/round3/trustmark-bench/trustmark-hard-202612-60.npz work/round3/localizer.onnx
cd ..
python benchmarks/report_invisible.py task1_watermark_localisation/work/round3 \
  --timing task1_watermark_localisation/work/round3/localizer.json
```

Run CPU comparisons serially. See [the results](results.md) and the [model card](../docs/model_card.md)
for the distinction between known-message metrics and unknown-message product abstention.
