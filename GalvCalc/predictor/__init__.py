"""
GalvCalc.predictor
Machine-learning predictors for surface and hydrogen-adsorption properties.

The heavy dependencies (torch, scikit-learn, tabpfn) are imported lazily so that
``import GalvCalc`` stays fast and does not require the optional ``ml`` extras
when the predictors are not used.
"""

__all__ = ["predict_cgcnn", "predict"]


def predict_cgcnn(*args, **kwargs):
    """Predict surface energies / work functions with the CGCNN model.

    Lazy wrapper around
    :func:`GalvCalc.predictor.surface_properties.predict_surf.predict_cgcnn`.
    """
    from GalvCalc.predictor.surface_properties.predict_surf import predict_cgcnn as _predict_cgcnn

    return _predict_cgcnn(*args, **kwargs)


def predict(*args, **kwargs):
    """Predict hydrogen adsorption energies with the TabPFN model.

    Lazy wrapper around
    :func:`GalvCalc.predictor.Eads.utils.ActiveLearningPred.predict`.
    """
    from GalvCalc.predictor.Eads.utils.ActiveLearningPred import predict as _predict

    return _predict(*args, **kwargs)
