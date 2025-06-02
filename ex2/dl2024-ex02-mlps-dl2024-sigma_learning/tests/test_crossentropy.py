import numpy as np

from lib.losses import CrossEntropyLoss


def test_crossentropy_two_classes():
    """Test cross-entropy loss on a binary classification example."""
    preds = np.array([[-2.3, 2.], [3.15, -4]])
    labels = np.array([[0, 1], [1, 0]])
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(preds, labels)
    assert isinstance(
        loss, float
    ), f"Returned Cross-Entropy-Loss is not a float, but a {type(loss)}"
    np.testing.assert_allclose(
        loss,
        0.007130943326295574,
        err_msg="Cross-Entropy-Loss is not implemented correctly for "
        "the binary case.",
    )


def test_crossentropy_many_classes():
    """Test cross-entropy loss on a multiclass example."""
    preds = np.array(
        [[1.5, 2.3, -3.4, -4.1], [1.5, -3.1, 2.7, 0.8], [-2.3, 1.5, 2.3, -3.0]]
    )
    labels = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(preds, labels)
    assert isinstance(
        loss, float
    ), f"Returned Cross-Entropy-Loss is not a float, but a {type(loss)}"
    np.testing.assert_allclose(
        loss,
        1.910050243100022,
        err_msg="Cross-Entropy-Loss is not implemented correctly for "
        "the general case of more than 2 classes.",
    )


if __name__ == "__main__":
    test_crossentropy_two_classes()
    test_crossentropy_many_classes()
    print("Test complete.")
