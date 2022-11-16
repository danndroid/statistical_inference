from .exeptions import NotFittedError


def is_fitted(estimator):

    msg = (
        " The %(name)s estimator is not fitted. Call 'fit' method"
    )

    attributes = [m for m in vars(estimator) if m.endswith("_") and not m.startswith("__")]
    print(attributes)

    if not attributes:
        raise NotFittedError(msg % {'name':type(estimator).__name__})

    
    
