import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             classification_report, confusion_matrix)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("  MODEL RISK VALIDATION FRAMEWORK")
print("  Stress Testing & Validation of Credit Risk Models")
print("=" * 55 + "\n")

# ── Generate realistic credit data ────────────────────────────────────────────
np.random.seed(42)
n = 2000

data = pd.DataFrame({
    "credit_score":     np.random.normal(650, 80, n).clip(300, 900),
    "debt_to_income":   np.random.uniform(0.1, 0.65, n),
    "loan_amount":      np.random.uniform(50000, 800000, n),
    "employment_years": np.random.uniform(0, 25, n),
    "missed_payments":  np.random.poisson(0.5, n).clip(0, 6),
    "loan_to_value":    np.random.uniform(0.3, 0.95, n),
    "age":              np.random.normal(38, 10, n).clip(21, 70),
})

log_odds = (
    -2.5
    + (-0.004 * data["credit_score"])
    + (2.5 * data["debt_to_income"])
    + (0.4 * data["missed_payments"])
    + (-0.04 * data["employment_years"])
    + (1.2 * data["loan_to_value"])
)
prob = 1 / (1 + np.exp(-log_odds))
data["default"] = (prob > prob.median()).astype(int)

X = data.drop("default", axis=1)
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Dataset: {n} borrowers | {y.mean():.1%} default rate")
print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

# ── Train 3 competing models ───────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    cv = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()

    results[name] = {
        "model": model, "y_prob": y_prob, "y_pred": y_pred,
        "auc": auc, "brier": brier, "cv_auc": cv
    }
    print(f"  {name:<25} AUC: {auc:.4f} | Brier: {brier:.4f} | CV AUC: {cv:.4f}")

# ── Model Risk Tests ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  MODEL RISK VALIDATION TESTS")
print("=" * 55)

# Test 1: Stability — does AUC hold across folds?
print("\n① Stability Test (K-Fold Cross Validation)")
for name, res in results.items():
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(res["model"], X, y, cv=kf, scoring="roc_auc")
    stability = scores.std()
    flag = "⚠ UNSTABLE" if stability > 0.03 else "✓ STABLE"
    print(f"  {name:<25} Std: {stability:.4f}  {flag}")

# Test 2: Population Stability Index
print("\n② Population Stability Index (PSI)")


def compute_psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    exp_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_counts = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
    exp_counts = np.where(exp_counts == 0, 0.0001, exp_counts)
    act_counts = np.where(act_counts == 0, 0.0001, act_counts)
    psi = np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts))
    return psi


for name, res in results.items():
    train_prob = res["model"].predict_proba(X_train)[:, 1]
    test_prob = res["y_prob"]
    psi = compute_psi(train_prob, test_prob)
    flag = "⚠ HIGH DRIFT" if psi > 0.2 else (
        "⚠ MODERATE" if psi > 0.1 else "✓ STABLE")
    print(f"  {name:<25} PSI: {psi:.4f}  {flag}")

# Test 3: Stress Testing
print("\n③ Stress Test (Economic Downturn Scenario)")
X_stress = X_test.copy()
X_stress["credit_score"] = X_stress["credit_score"] * 0.85
X_stress["debt_to_income"] = X_stress["debt_to_income"] * 1.40
X_stress["missed_payments"] = X_stress["missed_payments"] + 2
X_stress["employment_years"] = X_stress["employment_years"] * 0.70

for name, res in results.items():
    base_auc = res["auc"]
    stress_prob = res["model"].predict_proba(X_stress)[:, 1]
    stress_auc = roc_auc_score(y_test, stress_prob)
    degradation = base_auc - stress_auc
    flag = "⚠ FRAGILE" if degradation > 0.05 else "✓ ROBUST"
    print(f"  {name:<25} Base: {base_auc:.4f} → Stress: {stress_auc:.4f} | Drop: {degradation:.4f}  {flag}")

# Test 4: Gini coefficient
print("\n④ Gini Coefficient (Discriminatory Power)")
for name, res in results.items():
    gini = 2 * res["auc"] - 1
    flag = "✓ STRONG" if gini > 0.6 else (
        "⚠ MODERATE" if gini > 0.4 else "⚠ WEAK")
    print(f"  {name:<25} Gini: {gini:.4f}  {flag}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

colors = {"Logistic Regression": "#1D9E75",
          "Random Forest": "#7F77DD", "Gradient Boosting": "#D85A30"}

# Panel 1: ROC Curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax1.plot(fpr, tpr, color=colors[name], linewidth=2,
             label=f"{name} (AUC={res['auc']:.3f})")
ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
ax1.set_title("ROC Curves — Model Comparison", fontweight="bold")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend(fontsize=8)

# Panel 2: Stress test comparison
names_list = list(results.keys())
base_aucs = [results[n]["auc"] for n in names_list]
stress_aucs = []
for name, res in results.items():
    stress_prob = res["model"].predict_proba(X_stress)[:, 1]
    stress_aucs.append(roc_auc_score(y_test, stress_prob))

x = np.arange(len(names_list))
w = 0.35
bars1 = ax2.bar(x - w/2, base_aucs,   w, label="Base",
                color="#1D9E75", edgecolor="white")
bars2 = ax2.bar(x + w/2, stress_aucs, w, label="Stress",
                color="#D85A30", edgecolor="white", alpha=0.8)
ax2.set_title("Stress Test — Base vs Downturn Scenario", fontweight="bold")
ax2.set_ylabel("AUC Score")
ax2.set_xticks(x)
ax2.set_xticklabels([n.replace(" ", "\n") for n in names_list], fontsize=8)
ax2.set_ylim(0.5, 1.0)
ax2.legend()
for bar in bars1:
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

# Panel 3: Calibration curves
for name, res in results.items():
    fraction, mean_pred = calibration_curve(y_test, res["y_prob"], n_bins=10)
    ax3.plot(mean_pred, fraction, color=colors[name], linewidth=2,
             marker="o", markersize=4, label=name)
ax3.plot([0, 1], [0, 1], "k--", linewidth=0.8,
         alpha=0.5, label="Perfect calibration")
ax3.set_title("Calibration Plot — Predicted vs Actual", fontweight="bold")
ax3.set_xlabel("Mean Predicted Probability")
ax3.set_ylabel("Fraction of Positives")
ax3.legend(fontsize=8)

# Panel 4: CV stability boxplot
cv_scores = {}
for name, res in results.items():
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores[name] = cross_val_score(
        res["model"], X, y, cv=kf, scoring="roc_auc")

bp = ax4.boxplot(cv_scores.values(), patch_artist=True, notch=False)
for patch, color in zip(bp["boxes"], colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_xticklabels([n.replace(" ", "\n")
                    for n in cv_scores.keys()], fontsize=8)
ax4.set_title("Model Stability — 10-Fold CV Distribution", fontweight="bold")
ax4.set_ylabel("AUC Score")
ax4.axhline(0.75, color="#888780", linewidth=0.8, linestyle="--", alpha=0.7)

plt.suptitle("Model Risk Validation Framework — Credit Risk Models",
             fontsize=13, fontweight="bold")

save_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "model_risk_output.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved as model_risk_output.png")
