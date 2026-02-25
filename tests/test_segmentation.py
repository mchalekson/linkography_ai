from linkography_ai.segmentation import segment_thirds


def test_segment_thirds_counts():
    labels = segment_thirds(9)
    assert labels.count("beginning") == 3
    assert labels.count("middle") == 3
    assert labels.count("end") == 3


def test_segment_thirds_small():
    labels = segment_thirds(2)
    assert labels == ["middle", "end"]
