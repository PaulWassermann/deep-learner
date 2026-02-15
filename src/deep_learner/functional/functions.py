import deep_learner._core.utils as utils
import deep_learner._tensor as t
import deep_learner.functional._backward_functions as B
import deep_learner.functional.utils as f


# TODO : add input validation (shapes, types, ...)
# TODO : much later, add fused operations and their corresponding backward functions
# ---------- Unary functions ----------
def dropout(a: t.Tensor, drop_proba: float) -> t.Tensor:
    backend = utils.get_backend(a.device)
    mask = backend.random.rand(*a.data.shape) > drop_proba
    return t.Tensor(
        data=mask * a.data / (1 - drop_proba),
        grad_func=B.DropoutBackward(a, mask=mask, drop_proba=drop_proba),
        device=a.device,
    )


def exponential(a: t.Tensor) -> t.Tensor:
    backend = utils.get_backend(a.device)
    return t.Tensor(
        data=backend.exp(a.data), grad_func=B.ExponentialBackward(a), device=a.device
    )


def relu(a: t.Tensor) -> t.Tensor:
    backend = utils.get_backend(a.device)
    return t.Tensor(
        data=backend.maximum(a.data, 0, a.data),
        grad_func=B.ReluBackward(a),
        device=a.device,
    )


def sigmoid(a: t.Tensor) -> t.Tensor:
    backend = utils.get_special_backend(a.device)
    return t.Tensor(
        data=backend.expit(a.data), grad_func=B.SigmoidBackward(a), device=a.device
    )


def softmax(a: t.Tensor) -> t.Tensor:
    backend = utils.get_special_backend(a.device)
    result = backend.softmax(a.data, axis=-1)
    return t.Tensor(
        data=result, grad_func=B.SoftmaxBackward(a, cache=result), device=a.device
    )


def softsign(a: t.Tensor) -> t.Tensor:
    backend = utils.get_backend(a.device)
    return t.Tensor(
        data=(a.data / (1 + backend.abs(a.data))),
        grad_func=B.SoftsignBackward(a),
        device=a.device,
    )


# TODO : add axis and keepdim parameters
def sum(a: t.Tensor) -> t.Tensor:
    return t.Tensor(data=a.data.sum(), grad_func=B.SumBackward(a), device=a.device)


def tanh(a: t.Tensor) -> t.Tensor:
    backend = utils.get_backend(a.device)
    return t.Tensor(
        data=backend.tanh(a.data), grad_func=B.TanhBackward(a), device=a.device
    )


# ---------- Binary functions ----------
def add(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    return t.Tensor(
        data=a.data + b.data, grad_func=B.AddBackward(a, b), device=a.device
    )


def binary_cross_entropy(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    return t.Tensor(
        data=-(
            y.data * f.log_clamp(x.data) + (1 - y.data) * f.log_clamp(1 - x.data)
        ).mean(),
        grad_func=B.BinaryCrossEntropyBackward(x, y),
        device=x.device,
    )


def cross_entropy(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    return t.Tensor(
        data=-(y.data * f.log_clamp(x.data)).sum() / x.data.shape[0],
        grad_func=B.CrossEntropyBackward(x, y),
        device=x.device,
    )


def divide(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    return t.Tensor(
        data=a.data / b.data, grad_func=B.DivideBackward(a, b), device=a.device
    )


def matmul(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    return t.Tensor(
        data=a.data @ b.data, grad_func=B.MatmulBackward(a, b), device=a.device
    )


def mean_squared_error(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    return t.Tensor(
        data=((y.data - x.data) ** 2).mean() / 2,
        grad_func=B.MeanSquaredErrorBackward(x, y),
        device=x.device,
    )


def multiply(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    return t.Tensor(
        data=a.data * b.data, grad_func=B.MutliplyBackward(a, b), device=a.device
    )


def subtract(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    return t.Tensor(
        data=a.data - b.data, grad_func=B.SubtractBackward(a, b), device=a.device
    )
