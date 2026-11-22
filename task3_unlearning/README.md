# Task 3: unlearning

Make a ViT-B/16 classifier (100 classes) behave on a forget set like a model that never saw it, while keeping its other
predictions.

Best public score: 0.4782.

## Method ([unlearn.py](unlearn.py))

For each forget image we estimate what a model that never saw it would predict: the label comes from the original model,
the difficulty from the original loss under strong alterations, the other classes from altered views. The model is
distilled towards these targets on exact and altered images, and kept close to the original on mosaics of other images.
[interp.py](interp.py) interpolates between the original and the unlearned weights to set the strength.
