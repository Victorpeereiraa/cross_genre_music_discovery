from sklearn.mixture import GaussianMixture
import pandas as pd

GMM_CONFIGS = {
    "pca3_tied15": {
        "n_components": 15,
        "covariance_type": "tied"
    },
    "pca5_diag30": {
        "n_components": 30,
        "covariance_type": "diag"
    }
}

def train_gmm(
    X,
    n_components=15,
    covariance_type="tied",
    random_state=42,
    n_init=5
):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init
    )

    model.fit(X)

    return model

def predict_gmm(model, scaler, pca, X_new, features):
    """Classifica músicas novas usando os mesmos objetos ajustados no treino."""
    X_scaled = scaler.transform(X_new[features])
    X_pca = pca.transform(X_scaled)

    probs = model.predict_proba(X_pca)

    resultado = X_new.copy()
    resultado["cluster"] = probs.argmax(axis=1)
    resultado["confianca"] = probs.max(axis=1)
    resultado["log_verossimilhanca"] = model.score_samples(X_pca)
    return resultado
