from .extract_patches import PatchType
_3X3 = PatchType._3X3
_2X2 = PatchType._2X2
OVERLAP = PatchType.OVERLAP

# 2-patch configurations
# See: Carl Doersch, Abhinav Gupta, and Alexei A. Efros. "Unsupervised
#   visual representation learning by context prediction." In
#   Proceedings of the IEEE International Conference on Computer
#   Vision, pp. 1422-1430. 2015.
patterns_2patch = (
    ((_3X3, 4), (_3X3, 0)),
    ((_3X3, 4), (_3X3, 1)),
    ((_3X3, 4), (_3X3, 2)),
    ((_3X3, 4), (_3X3, 5)),
    ((_3X3, 4), (_3X3, 8)),
    ((_3X3, 4), (_3X3, 7)),
    ((_3X3, 4), (_3X3, 6)),
    ((_3X3, 4), (_3X3, 3))
)

# 3-patch configurations
# See: T. Nathan Mundhenk, Daniel Ho, and Barry Y. Chen. "Improvements
#   to Context Based Self-Supervised Learning." In CVPR, pp.
#   9339-9348. 2018.
patterns_3patch = (

    # Line
    ((_3X3, 4), (_3X3, 0), (_3X3, 8)),
    ((_3X3, 4), (_3X3, 1), (_3X3, 7)),
    ((_3X3, 4), (_3X3, 2), (_3X3, 6)),
    ((_3X3, 4), (_3X3, 5), (_3X3, 3)),

    # L-shape
    ((_2X2, 0), (_2X2, 1), (_2X2, 3)),
    ((_2X2, 1), (_2X2, 3), (_2X2, 2)),
    ((_2X2, 3), (_2X2, 2), (_2X2, 0)),
    ((_2X2, 2), (_2X2, 0), (_2X2, 1)),

    # Hybrid scale patches
    ((OVERLAP, 0), (_3X3, 2), (_3X3, 5)),
    ((OVERLAP, 0), (_3X3, 6), (_3X3, 7)),
    ((OVERLAP, 1), (_3X3, 8), (_3X3, 7)),
    ((OVERLAP, 1), (_3X3, 0), (_3X3, 3)),
    ((OVERLAP, 3), (_3X3, 6), (_3X3, 3)),
    ((OVERLAP, 3), (_3X3, 2), (_3X3, 1)),
    ((OVERLAP, 2), (_3X3, 0), (_3X3, 1)),
    ((OVERLAP, 2), (_3X3, 8), (_3X3, 5))

)

# 4-patch configurations
patterns_4patch = (

    # T-shape
    ((_3X3, 4), (_3X3, 1), (_3X3, 5), (_3X3, 7)),
    ((_3X3, 4), (_3X3, 5), (_3X3, 7), (_3X3, 3)),
    ((_3X3, 4), (_3X3, 7), (_3X3, 3), (_3X3, 1)),
    ((_3X3, 4), (_3X3, 3), (_3X3, 1), (_3X3, 5)),

    # Z-shape
    ((_3X3, 4), (_3X3, 2), (_3X3, 5), (_3X3, 7)),
    ((_3X3, 4), (_3X3, 8), (_3X3, 7), (_3X3, 3)),
    ((_3X3, 4), (_3X3, 6), (_3X3, 3), (_3X3, 1)),
    ((_3X3, 4), (_3X3, 0), (_3X3, 1), (_3X3, 5)),
    ((_3X3, 4), (_3X3, 0), (_3X3, 3), (_3X3, 7)),
    ((_3X3, 4), (_3X3, 2), (_3X3, 1), (_3X3, 3)),
    ((_3X3, 4), (_3X3, 8), (_3X3, 5), (_3X3, 1)),
    ((_3X3, 4), (_3X3, 6), (_3X3, 7), (_3X3, 5)),

    # L-shape
    ((_3X3, 4), (_3X3, 2), (_3X3, 1), (_3X3, 7)),
    ((_3X3, 4), (_3X3, 8), (_3X3, 5), (_3X3, 3)),
    ((_3X3, 4), (_3X3, 6), (_3X3, 7), (_3X3, 1)),
    ((_3X3, 4), (_3X3, 0), (_3X3, 3), (_3X3, 5)),
    ((_3X3, 4), (_3X3, 0), (_3X3, 1), (_3X3, 7)),
    ((_3X3, 4), (_3X3, 2), (_3X3, 5), (_3X3, 3)),
    ((_3X3, 4), (_3X3, 8), (_3X3, 7), (_3X3, 1)),
    ((_3X3, 4), (_3X3, 6), (_3X3, 3), (_3X3, 5)),

    # Square
    ((_2X2, 0), (_2X2, 1), (_2X2, 3), (_2X2, 2)),

    # Hybrid scale
    ((OVERLAP, 0), (_3X3, 2), (_3X3, 5), (_3X3, 8)),
    ((OVERLAP, 1), (_3X3, 8), (_3X3, 7), (_3X3, 6)),
    ((OVERLAP, 3), (_3X3, 6), (_3X3, 3), (_3X3, 0)),
    ((OVERLAP, 2), (_3X3, 0), (_3X3, 1), (_3X3, 2)),
    ((OVERLAP, 0), (_3X3, 6), (_3X3, 7), (_3X3, 8)),
    ((OVERLAP, 1), (_3X3, 0), (_3X3, 3), (_3X3, 6)),
    ((OVERLAP, 3), (_3X3, 2), (_3X3, 1), (_3X3, 0)),
    ((OVERLAP, 2), (_3X3, 8), (_3X3, 5), (_3X3, 2)),
    ((OVERLAP, 0), (_3X3, 5), (_3X3, 8), (_3X3, 7)),
    ((OVERLAP, 1), (_3X3, 7), (_3X3, 6), (_3X3, 3)),
    ((OVERLAP, 3), (_3X3, 3), (_3X3, 0), (_3X3, 1)),
    ((OVERLAP, 2), (_3X3, 1), (_3X3, 2), (_3X3, 5)),

)

# 5-patch configurations
patterns_5patch = (

    # Cross
    ((_3X3, 4), (_3X3, 1), (_3X3, 5), (_3X3, 7), (_3X3, 3)),

    # X-shape
    ((_3X3, 4), (_3X3, 0), (_3X3, 2), (_3X3, 8), (_3X3, 6)),

    # T-shape
    ((_3X3, 4), (_3X3, 0), (_3X3, 1), (_3X3, 2), (_3X3, 7)),
    ((_3X3, 4), (_3X3, 2), (_3X3, 5), (_3X3, 8), (_3X3, 3)),
    ((_3X3, 4), (_3X3, 8), (_3X3, 7), (_3X3, 6), (_3X3, 1)),
    ((_3X3, 4), (_3X3, 6), (_3X3, 3), (_3X3, 0), (_3X3, 5)),

    # Z-shape
    ((_3X3, 4), (_3X3, 0), (_3X3, 1), (_3X3, 7), (_3X3, 8)),
    ((_3X3, 4), (_3X3, 2), (_3X3, 5), (_3X3, 3), (_3X3, 6)),
    ((_3X3, 4), (_3X3, 8), (_3X3, 7), (_3X3, 1), (_3X3, 0)),
    ((_3X3, 4), (_3X3, 6), (_3X3, 3), (_3X3, 5), (_3X3, 2)),
    ((_3X3, 4), (_3X3, 0), (_3X3, 3), (_3X3, 5), (_3X3, 8)),
    ((_3X3, 4), (_3X3, 2), (_3X3, 1), (_3X3, 7), (_3X3, 6)),
    ((_3X3, 4), (_3X3, 8), (_3X3, 5), (_3X3, 3), (_3X3, 0)),
    ((_3X3, 4), (_3X3, 6), (_3X3, 7), (_3X3, 1), (_3X3, 2)),

    # U-shape
    ((_3X3, 4), (_3X3, 0), (_3X3, 3), (_3X3, 5), (_3X3, 2)),
    ((_3X3, 4), (_3X3, 2), (_3X3, 1), (_3X3, 7), (_3X3, 8)),
    ((_3X3, 4), (_3X3, 8), (_3X3, 5), (_3X3, 3), (_3X3, 6)),
    ((_3X3, 4), (_3X3, 6), (_3X3, 7), (_3X3, 1), (_3X3, 0)),

    # V-shape
    ((_3X3, 0), (_3X3, 4), (_3X3, 8), (_3X3, 7), (_3X3, 6)),
    ((_3X3, 2), (_3X3, 4), (_3X3, 6), (_3X3, 3), (_3X3, 0)),
    ((_3X3, 8), (_3X3, 4), (_3X3, 0), (_3X3, 1), (_3X3, 2)),
    ((_3X3, 6), (_3X3, 4), (_3X3, 2), (_3X3, 5), (_3X3, 8)),
    ((_3X3, 0), (_3X3, 4), (_3X3, 8), (_3X3, 5), (_3X3, 2)),
    ((_3X3, 2), (_3X3, 4), (_3X3, 6), (_3X3, 7), (_3X3, 8)),
    ((_3X3, 8), (_3X3, 4), (_3X3, 0), (_3X3, 3), (_3X3, 6)),
    ((_3X3, 6), (_3X3, 4), (_3X3, 2), (_3X3, 1), (_3X3, 0)),

    # Hybrid scale
    ((OVERLAP, 0), (_3X3, 2), (_3X3, 5), (_3X3, 6), (_3X3, 7)),
    ((OVERLAP, 1), (_3X3, 8), (_3X3, 7), (_3X3, 0), (_3X3, 3)),
    ((OVERLAP, 3), (_3X3, 6), (_3X3, 3), (_3X3, 2), (_3X3, 1)),
    ((OVERLAP, 2), (_3X3, 0), (_3X3, 1), (_3X3, 8), (_3X3, 5))

)
