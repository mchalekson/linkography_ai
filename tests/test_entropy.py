from linkography_ai.entropy import shannon_entropy_from_counts


def test_entropy_zero_when_single_category_normalized():
    assert shannon_entropy_from_counts([10, 0], normalize=True) == 0.0


def test_entropy_binary_half_half_normalized():
    val = shannon_entropy_from_counts([5, 5], normalize=True)
    assert abs(val - 1.0) < 1e-6


def test_entropy_nan_when_empty():
    val = shannon_entropy_from_counts([], normalize=True)
    assert val != val  # NaN check
