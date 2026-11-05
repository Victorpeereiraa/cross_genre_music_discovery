from sklearn.mixture import GaussianMixture

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
