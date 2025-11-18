import numpy as np
import pandas as pd
import statsmodels.api as sm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def manual_irls(X, y, max_iter=30, tol=1e-6):
    beta = np.zeros(X.shape[1])

    for i in range(max_iter):
        eta = X @ beta
        mu = sigmoid(eta)

        # Clip mu for stability?
        # mu = np.clip(mu, 1e-10, 1 - 1e-10)

        W = np.diag(mu * (1 - mu))

        score = X.T @ (y - mu)
        info = X.T @ W @ X

        try:
            delta = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            print(f"Singular matrix at iteration {i}")
            break

        beta = beta + delta

        if np.linalg.norm(delta) < tol:
            print(f"Converged at iteration {i}")
            break

    return beta


def manual_irls_clipped(X, y, max_iter=30, tol=1e-6):
    beta = np.zeros(X.shape[1])

    for i in range(max_iter):
        eta = X @ beta
        mu = sigmoid(eta)

        # Clip mu for stability
        mu = np.clip(mu, 1e-10, 1 - 1e-10)

        W = np.diag(mu * (1 - mu))

        score = X.T @ (y - mu)
        info = X.T @ W @ X

        try:
            delta = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            print(f"Singular matrix at iteration {i}")
            break

        beta = beta + delta

        if np.linalg.norm(delta) < tol:
            print(f"Converged at iteration {i}")
            break

    return beta


def manual_irls_robust(X, y, max_iter=30, tol=1e-6):
    beta = np.zeros(X.shape[1])

    for i in range(max_iter):
        eta = X @ beta
        mu = sigmoid(eta)

        # Clip mu for W calculation only
        mu_clipped = np.clip(mu, 1e-15, 1 - 1e-15)
        W = np.diag(mu_clipped * (1 - mu_clipped))

        # Use unclipped mu for score? Or clipped?
        # Statsmodels likely uses unclipped for score (gradient)
        score = X.T @ (y - mu_clipped)
        info = X.T @ W @ X

        try:
            # Use pinv
            delta = np.linalg.pinv(info) @ score
        except np.linalg.LinAlgError:
            print(f"Singular matrix at iteration {i}")
            break

        beta = beta + delta

        if np.linalg.norm(delta) < tol:
            print(f"Converged at iteration {i}")
            break

    return beta


# Generate data with separation
np.random.seed(42)
n = 100
x = np.random.randn(n)
# Perfect separation
y = (x > 0).astype(float)
# Add constant
X = np.column_stack([np.ones(n), x])

print("--- Statsmodels ---")
model = sm.GLM(y, X, family=sm.families.Binomial())
res = model.fit(maxiter=30, tol=1e-6)
print(res.params)
# print(f"Iterations: {res.n_iter}")

print("\n--- Manual IRLS (Unclipped) ---")
beta_manual = manual_irls(X, y)
print(beta_manual)

print("\n--- Manual IRLS (Clipped) ---")
beta_manual_clipped = manual_irls_clipped(X, y)
print(beta_manual_clipped)

print("\n--- Manual IRLS (Robust) ---")
beta_manual_robust = manual_irls_robust(X, y)
print(beta_manual_robust)
