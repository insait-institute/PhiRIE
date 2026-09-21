import numpy as np
import pytest
from robo.roundtrip.scorer_bounds import corners,distance_bounds,sink_bounds,combine_bounds


def box(lo,hi):return corners([lo,hi,lo,hi])[0]


def test_distance_bounds_include_every_point_and_exact_boundaries():
    v=box([-1,-2,-3],[1,2,3]);lo,hi=distance_bounds(v,[2,0,0])
    assert lo==1 and hi==np.sqrt(22)
    assert distance_bounds(v,[0,0,0])[0]==0
    assert distance_bounds(v,[2,0,999],2)==(1,np.sqrt(13))


def test_native_sink_region_thresholds_unchanged_and_ambiguous_preserved():
    r={'inside':[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]}
    assert sink_bounds(box([0,0,0],[1,1,1]),r)=='always_true'
    assert sink_bounds(box([1.01,0,0],[2,1,1]),r)=='always_false'
    assert sink_bounds(box([.9,0,0],[1.1,1,1]),r)=='ambiguous'
    assert combine_bounds(['always_true','ambiguous']).startswith('origin_sensitive')


def test_disjoint_regions_do_not_make_false_union_certification():
    regions={'a':[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],'b':[[2,0,0],[3,0,0],[2,1,0],[2,0,1]]}
    assert sink_bounds(box([.5,.1,.1],[2.5,.9,.9]),regions)=='ambiguous'
    with pytest.raises(ValueError,match='missing'):sink_bounds(box([0,0,0],[1,1,1]),{})
