from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
import joblib

@dataclass
class KMeansConfig:
    k: int = 4
    random_state: int = 42
    n_init: str | int = "auto"   # use 10 if older sklearn
    minibatch: bool = False

def prepare_features(
    feat: pd.DataFrame,
    feature_cols: tuple[str, ...] = ("log_total_purchase_eur",),
    scaler: str = "standard",
) -> tuple[np.ndarray, object, list[str]]:
    X = feat.loc[:, list(feature_cols)].astype("float64").to_numpy()
    if scaler == "standard":
        sc = StandardScaler()
    elif scaler == "robust":
        sc = RobustScaler()
    else:
        raise ValueError("scaler must be 'standard' or 'robust'")
    Xz = sc.fit_transform(X)
    return Xz, sc, list(feature_cols)

def scan_k(
    Xz: np.ndarray,
    k_range: range = range(2, 9),
) -> pd.DataFrame:
    out = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(Xz)
        sil = silhouette_score(Xz, labels) if len(np.unique(labels)) > 1 else np.nan
        out.append((k, sil, km.inertia_))
    return pd.DataFrame(out, columns=["k", "silhouette", "inertia"])

def fit_kmeans(
    Xz: np.ndarray, cfg: KMeansConfig
) -> KMeans:
    if cfg.minibatch:
        model = MiniBatchKMeans(n_clusters=cfg.k, random_state=cfg.random_state, n_init=cfg.n_init)
    else:
        model = KMeans(n_clusters=cfg.k, random_state=cfg.random_state, n_init=cfg.n_init)
    model.fit(Xz)
    return model

def label_dataframe(
    feat: pd.DataFrame, model: KMeans, scaler, feature_cols: list[str], label_col: str = "cluster"
) -> pd.DataFrame:
    Xz = scaler.transform(feat.loc[:, feature_cols].to_numpy())
    labels = model.predict(Xz)
    out = feat.copy()
    out[label_col] = labels
    return out

def centroids_df(model: KMeans, scaler, feature_cols: list[str]) -> pd.DataFrame:
    centers_z = model.cluster_centers_
    centers = scaler.inverse_transform(centers_z)
    dfc = pd.DataFrame(centers, columns=feature_cols)
    dfc.insert(0, "cluster", range(len(dfc)))
    return dfc

def save_artifacts(model: KMeans, scaler, meta: dict, path: str | Path) -> None:
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "kmeans.joblib")
    joblib.dump(scaler, path / "scaler.joblib")
    (path / "meta.json").write_text(json.dumps(meta, indent=2))

def load_artifacts(path: str | Path) -> tuple[KMeans, object, dict]:
    path = Path(path)
    model = joblib.load(path / "kmeans.joblib")
    scaler = joblib.load(path / "scaler.joblib")
    meta = json.loads((path / "meta.json").read_text())
    return model, scaler, meta
