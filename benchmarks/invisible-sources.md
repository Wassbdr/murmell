# Public invisible-watermark sources

The public benchmark needs no competition data, metadata, checkpoints or starter kit.
It is separate from the [visible-watermark experiment](../experiments/visible/).

## Images and labels

The main generator uses [grass](https://www.deviantart.com/linolafett/art/Grass-01-434853879)
(CC0) and the [Hubble eXtreme Deep Field](https://science.nasa.gov/mission/hubble/science/explore-the-night-sky/hubble-deep-fields/)
(NASA, public domain). The held-out generator uses gravel and brick
(CC0Textures, now ambientCG, assets Gravel04 and Bricks25).
The per-image licence and preparation notes are in the
[scikit-image data documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html).
The same four bundled images as the visible experiment are used. The held-out generator
also changes the crop, JPEG quality, patch size, irregular outline and resampling ranges.
This is a generator shift check, not a representative sample of photographs.

The generator applies crop and resize to both image and pixel support. The latter uses
nearest-neighbor resampling and is not JPEG compressed. Each output cell is positive if
at least 8 of its 16 support pixels are marked. The label tracks edit geometry, not
whether the watermark remains decodable after compression. An irregular pasted region is a
polygon cutout of a different licensed photo, not a semantic object annotation.

## Third-party models

| Model | Code and checkpoint terms | Public image and mask outputs |
| --- | --- | --- |
| Adobe TrustMark Q | MIT for both, as stated in the [README](https://github.com/adobe/trustmark#license) and [licence](https://github.com/adobe/trustmark/blob/main/LICENSE). [PyPI](https://pypi.org/project/trustmark/0.9.2/) | Allowed with the source image terms above. |
| StegaStamp | [Code MIT](https://github.com/tancik/StegaStamp/blob/master/LICENSE); the [README](https://github.com/tancik/StegaStamp#encoding-a-message) expects an off-tree SavedModel and does not give that artifact's terms. | Not included: no verified official checkpoint grant. |
| Meta PixelSeal | [Video Seal MIT](https://github.com/facebookresearch/videoseal/blob/f12875a9b5e711e0c6e2dc3a22aca18231ce4a1a/LICENSE); the [README](https://github.com/facebookresearch/videoseal/blob/f12875a9b5e711e0c6e2dc3a22aca18231ce4a1a/README.md) explicitly includes models. [PixelSeal checkpoint](https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth) | Allowed with the source image terms above. |
| Meta WAM | [Code MIT](https://github.com/facebookresearch/watermark-anything/blob/2c08af04d037d5667c02f6ddebbda9ff04581c3e/LICENSE). Use [wam_mit.pth](https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth) as explicitly licensed in the [README](https://github.com/facebookresearch/watermark-anything/blob/2c08af04d037d5667c02f6ddebbda9ff04581c3e/README.md), not the noncommercial COCO checkpoint. | Allowed with the source image terms above. |
| MobileSAM | [Apache 2.0](https://github.com/ChaoningZhang/MobileSAM/blob/f706ad9c4eb7f219c00d9050e46328518ffb65d2/LICENSE); the [checkpoint is tracked in the licensed source tree](https://github.com/ChaoningZhang/MobileSAM/tree/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights). | May generate masks for the licensed inputs. |
| SAM ViT-B | [Apache 2.0 code and model](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/README.md#license), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth). | May generate masks for the licensed inputs. |

No external weights are shipped by the package. When redistributing model code or weights,
include the applicable third-party notices.

## Pilot commands

In the repository CPU environment, install `trustmark==0.9.2`, then run
`python -m src.invisible_smoke` from `task1_watermark_localisation/`.
The first run downloads Q encoder and decoder weights from Adobe.
This is a timing check, not a trained-model comparison. It uses the provided
100-bit message for its sliding-window reference, just like the competition.

For the optional Meta probes, put the cited official source checkout on
`PYTHONPATH` alongside the absolute path to `task1_watermark_localisation/`.
From the official source checkout, run
`python -m src.invisible_meta_smoke wam /path/to/wam_mit.pth`
or `python -m src.invisible_meta_smoke pixelseal /path/to/pixelseal.pth`.
The WAM command also scores the native foreground head on edited WAM images.
