import jax
import jax.numpy as jnp
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.metrics import root_mean_squared_error

from genepax.supervised_learning.metrics import (
    categorical_cross_entropy,
    classification_accuracy,
    mse,
    r2_score,
    rmse,
    rrmse_per_target,
)


def test_r2_score_perfect():
    # Perfect prediction → R² should be 1.0
    y = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.0, 2.0, 3.0])
    assert r2_score(y, y_pred) == 1.0


def test_r2_score_zero_variance_perfect():
    # Zero variance in y and perfect prediction → defined as 1.0
    y = jnp.array([5.0, 5.0, 5.0])
    y_pred = jnp.array([5.0, 5.0, 5.0])
    assert r2_score(y, y_pred) == 1.0


def test_r2_score_zero_variance_imperfect():
    # Zero variance in y but prediction is not perfect → defined as 0.0
    y = jnp.array([5.0, 5.0, 5.0])
    y_pred = jnp.array([4.0, 6.0, 5.0])
    assert r2_score(y, y_pred) == 0.0


def test_r2_score_known_value():
    # Known example used in scikit-learn documentation
    # R² should match sklearn's output closely
    y = jnp.array([3.0, -0.5, 2.0, 7.0])
    y_pred = jnp.array([2.5, 0.0, 2.0, 8.0])
    result = r2_score(y, y_pred)
    expected = sklearn_r2_score(y, y_pred)
    assert jnp.isclose(result, expected)


def test_mse_perfect():
    # Perfect prediction → MSE should be 0
    y = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.0, 2.0, 3.0])
    assert mse(y, y_pred) == 0.0


def test_mse_known_value():
    # MSE for a known simple example:
    # Errors: [1, 0, 1] → MSE = 2/3
    y = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([2.0, 2.0, 2.0])
    result = mse(y, y_pred)
    expected = mean_squared_error(y, y_pred)
    assert jnp.isclose(result, expected)


def test_rmse_perfect():
    # Perfect prediction → RMSE should be 0
    y = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.0, 2.0, 3.0])
    assert rmse(y, y_pred) == 0.0


def test_rmse_known_value():
    # RMSE for a known simple example:
    # Errors: [1, 0, 1] → MSE = 2/3 → RMSE = sqrt(2/3)
    y = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([2.0, 2.0, 2.0])
    result = rmse(y, y_pred)
    expected = root_mean_squared_error(y, y_pred)
    assert jnp.isclose(result, expected)


def test_perfect_prediction():
    """RRMSE should be zero when predictions are perfect."""
    y_train = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_test = jnp.array([[2.0, 3.0], [1.0, 2.0]])
    y_pred = y_test.copy()

    rrmse = rrmse_per_target(y_test, y_pred, y_train)

    assert jnp.allclose(rrmse, jnp.zeros_like(rrmse)), rrmse


def test_mean_predictor_rrmse_is_one():
    """
    Predicting the training mean for each target should yield RRMSE = 1.
    """
    y_train = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_test = jnp.array([[2.0, 3.0], [4.0, 5.0]])

    y_train_mean = jnp.mean(y_train, axis=0)
    y_pred = jnp.tile(y_train_mean, (y_test.shape[0], 1))

    rrmse = rrmse_per_target(y_test, y_pred, y_train)

    assert jnp.allclose(rrmse, jnp.ones_like(rrmse)), rrmse


def test_output_shape():
    """Output shape must be (T,)"""
    y_train = jnp.ones((10, 5))
    y_test = jnp.ones((4, 5))
    y_pred = jnp.ones((4, 5))

    rrmse = rrmse_per_target(y_test, y_pred, y_train)

    assert rrmse.shape == (5,), rrmse.shape


def test_zero_variance_target():
    """
    If a target has zero variance in test data,
    RRMSE should be finite (due to epsilon).
    """
    y_train = jnp.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
    y_test = jnp.array([[1.0, 5.0], [1.0, 5.0]])
    y_pred = jnp.array([[1.0, 4.0], [1.0, 6.0]])

    rrmse = rrmse_per_target(y_test, y_pred, y_train)

    assert jnp.all(jnp.isfinite(rrmse)), rrmse


def test_categorical_cross_entropy_output_shape_and_type():
    # Random logits and one-hot labels (3 samples, 4 classes)
    logits = jnp.array(
        [[1.0, 2.0, 0.5, 0.1], [0.2, 0.3, 0.5, 0.0], [1.0, 2.0, 1.0, 0.0]]
    )
    y_true = jnp.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])

    loss = categorical_cross_entropy(y_true, logits)

    assert isinstance(loss, jnp.ndarray), "Loss should be a JAX array"
    assert loss.shape == (), "Loss should be a scalar"


def test_categorical_cross_entropy_perfect_predictions_have_low_loss():
    # Perfect predictions (logits align with one-hot labels)
    y_true = jnp.array([[1, 0], [0, 1]])
    logits = jnp.array([[10.0, 0.0], [0.0, 10.0]])

    loss = categorical_cross_entropy(y_true, logits)

    assert loss < 1e-4, "Loss should be near zero for perfect predictions"


def test_categorical_cross_entropy_wrong_predictions_have_high_loss():
    # Completely wrong predictions
    y_true = jnp.array([[1, 0], [0, 1]])
    logits = jnp.array([[0.0, 10.0], [10.0, 0.0]])

    loss = categorical_cross_entropy(y_true, logits)

    assert loss > 9.0, "Loss should be large for totally wrong predictions"


def test_categorical_cross_entropy_binary_classification_one_hot():
    # Binary case: one-hot encoded labels
    y_true = jnp.array([[1, 0], [0, 1], [1, 0]])
    logits = jnp.array([[2.0, -1.0], [-1.0, 2.0], [2.0, -1.0]])

    loss = categorical_cross_entropy(y_true, logits)

    assert loss > 0, "Loss should be positive"
    assert loss < 1.0, "Loss should be small for mostly correct predictions"


def test_categorical_cross_entropy_multiclass_classification():
    # Multi-class case: 5 classes
    y_true = jnp.eye(5)  # identity matrix, perfect one-hot
    logits = jnp.array(
        [
            [5, 1, 0, 0, 0],
            [0, 5, 1, 0, 0],
            [0, 0, 5, 1, 0],
            [0, 0, 0, 5, 1],
            [1, 0, 0, 0, 5],
        ]
    )

    loss = categorical_cross_entropy(y_true, logits)

    assert loss > 0, "Loss should be positive"
    assert loss < 1.0, "Loss should be small for mostly correct predictions"


def test_classification_accuracy_binary_perfect():
    """Binary classification, perfect predictions"""
    y_true = jnp.array([[1, 0], [0, 1], [1, 0]])
    logits = jnp.array([[2.0, -1.0], [-1.0, 2.0], [2.0, -1.0]])

    acc = classification_accuracy(y_true, logits)
    assert acc == 1.0, "Accuracy should be 1.0 for perfect binary predictions"


def test_classification_accuracy_binary_wrong():
    """Binary classification, completely wrong predictions"""
    y_true = jnp.array([[1, 0], [0, 1]])
    logits = jnp.array([[-1.0, 2.0], [2.0, -1.0]])

    acc = classification_accuracy(y_true, logits)
    assert acc == 0.0, "Accuracy should be 0.0 for totally wrong binary predictions"


def test_classification_accuracy_multiclass_perfect():
    """Multi-class classification, perfect predictions"""
    y_true = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    logits = jnp.array([[5, 1, 0], [0, 5, 1], [1, 0, 5]])

    acc = classification_accuracy(y_true, logits)
    assert acc == 1.0, "Accuracy should be 1.0 for perfect multi-class predictions"


def test_classification_accuracy_multiclass_partial():
    """Multi-class classification, some wrong predictions"""
    y_true = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    logits = jnp.array([[0, 5, 0], [0, 5, 1], [1, 0, 5]])

    acc = classification_accuracy(y_true, logits)
    # First prediction wrong, next two correct
    expected_acc = 2 / 3
    assert jnp.isclose(
        acc, expected_acc
    ), "Accuracy should match partially correct predictions"


def test_classification_accuracy_softmax_inputs():
    """Check that function works when inputs are already softmaxed"""
    y_true = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    logits = jnp.array([[5, 1, 0], [0, 5, 1], [1, 0, 5]])
    probs = jax.nn.softmax(logits, axis=-1)

    acc_logits = classification_accuracy(y_true, logits)
    acc_probs = classification_accuracy(y_true, probs)

    assert (
        acc_logits == acc_probs
    ), "Accuracy should be same for logits or softmaxed probabilities"


def test_classification_accuracy_zero_batch():
    """Edge case: empty batch"""
    y_true = jnp.empty((0, 2))
    logits = jnp.empty((0, 2))

    # Expect nan due to mean over empty array
    acc = classification_accuracy(y_true, logits)
    assert jnp.isnan(acc), "Accuracy should be nan for empty batch"
