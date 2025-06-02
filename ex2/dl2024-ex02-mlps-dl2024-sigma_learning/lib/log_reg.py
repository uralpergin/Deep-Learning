"""Logistic regression."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Tuple


def logistic_regression(inputs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
    """Implement a logistic regression model for binary classification.

    Args:
        inputs: np.ndarray with shape (nr_examples, nr_features). Input examples.
        labels: np.ndarray with shape (nr_examples). True labels.

    Returns:
    Tuple[prediction, score]:
        prediction: np.ndarray with shape (nr_examples). Predicted labels.
        score: float. Accuracy of the model on the input data.
    """
    # START TODO #################
    lgs = LogisticRegression()

    lgs.fit(inputs, labels)

    prediction = lgs.predict(inputs)

    score = lgs.score(inputs, labels)

    # END TODO ##################
    print(f"Prediction: {prediction}")
    print(f"Score: {score}")
    return prediction, score
