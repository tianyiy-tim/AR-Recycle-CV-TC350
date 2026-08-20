# Earlier versions

Kept because the path to `../detector.py` is most of the point. Neither of these is
maintained.

**`v1_boxes_only.py`** is YOLO-World alone, drawing bounding boxes. Enough to prove
open-vocabulary detection worked on a waste stream, and enough to show why a box is
the wrong shape: over a crowded belt a rectangle covers the item and everything around
it.

**`v2_fastsam.py`** adds segmentation the expensive way round. FastSAM segments the
whole frame unprompted, every third frame, and then the code compares every returned
mask against each detection box and keeps whichever overlaps best. It works, and the
scoring in `get_mask_for_bbox` is where most of the effort went, but the model is doing
far more work than the problem needs. Everything not on the belt gets segmented too.

The move to `detector.py` inverts that. MobileSAM is prompted with the boxes the
detector already produced, so it only segments the objects that matter, and all the
boxes go over in one call so the frame is encoded once. That plus staggering the two
stages is the difference between roughly 500 ms and roughly 8 ms a frame.
