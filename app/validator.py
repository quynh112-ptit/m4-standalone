import numpy as np
def validate(fv,meta):
    if fv["feature_set_version"] != meta["feature_set_version"]:
        raise ValueError("FEATURE_SET_VERSION_MISMATCH")
    fs=fv["features"]
    missing=[k for k in meta["feature_order"] if k not in fs]
    if missing:
        raise ValueError(f"MISSING_FEATURES:{missing}")
    x=np.array([float(fs[k]) for k in meta["feature_order"]])
    if not np.isfinite(x).all():
        raise ValueError("NON_FINITE_FEATURE")
    if not 0 <= float(fs["dst_port"]) <= 65535:
        raise ValueError("INVALID_DST_PORT")
    return x.reshape(1,-1)
