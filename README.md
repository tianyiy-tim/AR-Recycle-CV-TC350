# AR Recyclable Detection

Draws a contour around recyclables in a live camera feed, for the people sorting
them. The pipeline is short. The work was in the frame budget.

Team project, four people, January to April 2026. This repository is the vision
pipeline. The design report that came with it is not here.

## The problem

Sorters at a recycling facility stand at a moving belt and decide, item by item, what
each thing is made of. The idea is to put that decision in front of them: outline the
object through AR glasses and label which bin it belongs in.

Two things follow from that setting, and they drive the whole design.

**It has to be a contour, not a box.** A rectangle over a moving belt covers the item
and its neighbors. To tell someone *this one*, the outline has to follow the object.

**A fixed class list is useless here.** The set of things on a belt is long, and it
changes. A conventional detector has to be retrained for a container it has not seen.
So detection is open-vocabulary: `CLASSES` in `detector.py` is a list of plain English
strings, read at inference time. Adding an item is editing that list.

## The pipeline

Two stages, kept separable on purpose.

**YOLO-World** does detection and returns boxes for whatever is in `CLASSES`.
**MobileSAM** takes each box as a prompt and returns a mask, and the mask is what gets
drawn. Prompting a segmentation model with a box the detector already found is far
cheaper than asking it to segment the whole frame and then working out which mask
matters.

`CLASS_INFO` maps each class to a bin and a color, so the overlay says both what the
thing is and where it goes.

## The frame budget

The first working version ran full SAM at roughly 500 ms a frame. Two frames a second
is not an overlay, it is a slideshow, and for something strapped to someone's head it
is worse than nothing: by the time the outline is drawn, the wearer and the belt have
both moved and it is sitting over the wrong object.

Two changes brought that to about 8 ms.

**MobileSAM instead of SAM.** A distilled image encoder with the same prompt
interface, so it is close to a drop-in. It gives up some mask fidelity, which is a real
tradeoff and not a free win. At this object scale, for an outline someone glances at
rather than measures, a slightly looser contour that arrives in time beats a tighter one
that arrives late. That is defensible here and would not be somewhere else.

**Stagger the stages across frames.** This is the more interesting one, because it is
not a faster model, it is a scheduling decision. Nothing on a belt changes identity
between consecutive frames, so neither stage has to run on every tick:

```python
yolo_interval = 2   # detection every 2nd camera frame
seg_interval  = 3   # segmentation every 3rd detection frame
```

Masks are therefore recomputed about every sixth frame and reused in between, and the
pipeline never pays for both stages on the same frame. Segmentation is also batched:
all boxes go to MobileSAM in one call, so the image is encoded once rather than once
per object, and at most three detections are segmented in a pass.

This only works because the two stages are separable, which is why they were built
that way.

## Running it

Needs a webcam. Weights download on first run.

```bash
pip install -r requirements.txt
python detector.py        # q to quit
```

Tuned against a 640x480 feed. `camera_diagnostic.py` is there for when OpenCV cannot
open the camera or the window comes up black, which on macOS is usually permissions
rather than code.

## Layout

```
detector.py             the pipeline: YOLO-World, MobileSAM, staggering
camera_diagnostic.py    OpenCV capture and display checks
prior/                  the two earlier versions, and what each one cost
```

## Notes

The eyewear side is the part a report has to carry rather than code. Anything worn over
the eyes in a facility is safety equipment first and a display second, which means
ANSI/ISEA Z87.1: impact resistance, coverage, optical clarity. An overlay that
compromises the lens it is printed on does not ship no matter how good the model is.
