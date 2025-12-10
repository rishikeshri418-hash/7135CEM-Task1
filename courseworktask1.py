

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("="*60)
print("CALIFORNIA HOUSING CLASSIFICATION - WITH GAUSSIAN PROCESS")
print("="*60)


print("\n Loading dataset...")
housing = fetch_california_housing()
X = housing.data
y_continuous = housing.target
feature_names = housing.feature_names

print(f" Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Features: {', '.join(feature_names)}")


threshold = np.median(y_continuous)
y = (y_continuous > threshold).astype(int)
class_names = ['Low Value', 'High Value']

print(f"\n Binary classification created (as per brief requirements):")
print(f"   Threshold applied: ${threshold*100000:.0f}")
print(f"   {class_names[0]}: {np.sum(y==0)} samples ({np.mean(y==0)*100:.1f}%)")
print(f"   {class_names[1]}: {np.sum(y==1)} samples ({np.mean(y==1)*100:.1f}%)")

)
sample_size = 5000  
X = X[:sample_size]
y = y[:sample_size]
print(f"\n⚡ Using {sample_size} samples (optimal for GPC training)")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n Data prepared:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing: {X_test.shape[0]} samples")


print("\n Training models (including Gaussian Process Classification)...")

results = {}
training_times = {}


print("   1. Logistic Regression...", end=" ", flush=True)
start_time = time.time()
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
training_times['Logistic Regression'] = time.time() - start_time
results['Logistic Regression'] = accuracy_lr
print(f"Accuracy: {accuracy_lr:.4f} | Time: {training_times['Logistic Regression']:.1f}s")


print("   2. Random Forest...", end=" ", flush=True)
start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
training_times['Random Forest'] = time.time() - start_time
results['Random Forest'] = accuracy_rf
print(f"Accuracy: {accuracy_rf:.4f} | Time: {training_times['Random Forest']:.1f}s")


print("   3. SVM (Linear)...", end=" ", flush=True)
start_time = time.time()
svm = SVC(kernel='linear', random_state=42, max_iter=1000)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
training_times['SVM (Linear)'] = time.time() - start_time
results['SVM (Linear)'] = accuracy_svm
print(f"Accuracy: {accuracy_svm:.4f} | Time: {training_times['SVM (Linear)']:.1f}s")


print("   4. Gaussian Process Classification...", end=" ", flush=True)
start_time = time.time()


kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))


gpc = GaussianProcessClassifier(
    kernel=kernel,
    n_restarts_optimizer=5,  
    random_state=42,
    max_iter_predict=50
)


gpc.fit(X_train_scaled, y_train)
y_pred_gpc = gpc.predict(X_test_scaled)
y_prob_gpc = gpc.predict_proba(X_test_scaled)[:, 1]  
accuracy_gpc = accuracy_score(y_test, y_pred_gpc)
training_times['Gaussian Process'] = time.time() - start_time
results['Gaussian Process'] = accuracy_gpc

print(f"Accuracy: {accuracy_gpc:.4f} | Time: {training_times['Gaussian Process']:.1f}s")
print(f"   Kernel optimized: {gpc.kernel_}")
print(f"   Log-marginal-likelihood: {gpc.log_marginal_likelihood():.2f}")


print("\n" + "="*60)
print(" RESULTS SUMMARY")
print("="*60)

print("\nModel Performance Comparison:")
print("-" * 40)
print(f"{'Model':<25} {'Accuracy':<12} {'Training Time (s)':<15}")
print("-" * 40)

for model, acc in results.items():
    print(f"{model:<25} {acc:<12.4f} {training_times.get(model, 0):<15.1f}")

best_model = max(results, key=results.get)
best_accuracy = results[best_model]
print("\n" + "="*60)
print(f" BEST MODEL: {best_model} ({best_accuracy:.4f} accuracy)")
print("="*60)


print("\n🎨 Creating Visualization 1: Model Comparison...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))


models = list(results.keys())
accuracies = list(results.values())
colors = ['blue', 'green', 'orange', 'purple']  
bars = ax1.bar(models, accuracies, color=colors, edgecolor='black')
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.set_title('Model Performance Comparison (GPC Included)', fontsize=16, fontweight='bold')
ax1.set_ylim([0.8, 1.0])
ax1.grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')


times = [training_times.get(m, 0) for m in models]
bars2 = ax2.bar(models, times, color=colors, edgecolor='black')
ax2.set_ylabel('Training Time (seconds)', fontsize=14)
ax2.set_title('Training Time Comparison', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for bar, t in zip(bars2, times):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{t:.1f}s', ha='center', va='bottom', fontsize=11)


cm_gpc = confusion_matrix(y_test, y_pred_gpc)
sns.heatmap(cm_gpc, annot=True, fmt='d', cmap='Purples', 
            xticklabels=class_names, yticklabels=class_names, ax=ax3,
            cbar_kws={'label': 'Count'})
ax3.set_title('Confusion Matrix - Gaussian Process', fontsize=16, fontweight='bold')
ax3.set_xlabel('Predicted Label', fontsize=12)
ax3.set_ylabel('True Label', fontsize=12)


cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names, ax=ax4,
            cbar_kws={'label': 'Count'})
ax4.set_title('Confusion Matrix - Random Forest', fontsize=16, fontweight='bold')
ax4.set_xlabel('Predicted Label', fontsize=12)
ax4.set_ylabel('True Label', fontsize=12)

plt.tight_layout()
plt.savefig('model_comparison_with_gpc.png', dpi=200, bbox_inches='tight')
print(" Saved: model_comparison_with_gpc.png")
plt.show()


print("\n Creating Visualization 2: GPC Probability Analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


for class_label in [0, 1]:
    mask = (y_test == class_label)
    ax1.hist(y_prob_gpc[mask], bins=20, alpha=0.7, label=f'True {class_names[class_label]}',
             density=True, edgecolor='black')
ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
ax1.set_xlabel('Predicted Probability (High Value)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('GPC Probability Distribution by True Class', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)


sorted_indices = np.argsort(y_prob_gpc)
ax2.scatter(range(len(y_prob_gpc)), y_prob_gpc[sorted_indices], 
           c=y_test[sorted_indices], cmap='coolwarm', alpha=0.7, s=30)
ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('Sorted Test Samples', fontsize=12)
ax2.set_ylabel('Predicted Probability', fontsize=12)
ax2.set_title('GPC Probability Calibration', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)


sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2)
cbar.set_label('True Class (0=Low, 1=High)', fontsize=10)

plt.tight_layout()
plt.savefig('gpc_probability_analysis.png', dpi=200, bbox_inches='tight')
print(" Saved: gpc_probability_analysis.png")
plt.show()


print("\n Creating Visualization 3: Feature Importance...")


rf_importance = rf.feature_importances_


lr_coef = np.abs(lr.coef_[0])
lr_importance = lr_coef / lr_coef.sum()


importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Random Forest': rf_importance,
    'Logistic Regression': lr_importance
})


fig, axes = plt.subplots(1, 2, figsize=(14, 6))


rf_sorted = importance_df.sort_values('Random Forest', ascending=True)
axes[0].barh(rf_sorted['Feature'], rf_sorted['Random Forest'], color='green', alpha=0.7)
axes[0].set_xlabel('Importance Score', fontsize=12)
axes[0].set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')


lr_sorted = importance_df.sort_values('Logistic Regression', ascending=True)
axes[1].barh(lr_sorted['Feature'], lr_sorted['Logistic Regression'], color='blue', alpha=0.7)
axes[1].set_xlabel('Importance Score', fontsize=12)
axes[1].set_title('Logistic Regression Feature Importance', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=200, bbox_inches='tight')
print(" Saved: feature_importance_comparison.png")
plt.show()


print("\n Detailed GPC Analysis (for your paper):")


tn, fp, fn, tp = cm_gpc.ravel()
print(f"\nGPC Confusion Matrix Analysis:")
print(f"   True Positives (High correctly predicted): {tp}")
print(f"   True Negatives (Low correctly predicted): {tn}")
print(f"   False Positives (Low predicted as High): {fp}")
print(f"   False Negatives (High predicted as Low): {fn}")
print(f"   Sensitivity/Recall: {tp/(tp+fn):.3f}")
print(f"   Specificity: {tn/(tn+fp):.3f}")
print(f"   Precision: {tp/(tp+fp):.3f}")


prob_variance = np.var(y_prob_gpc)
print(f"\nGPC Uncertainty Metrics:")
print(f"   Probability variance: {prob_variance:.4f}")
print(f"   Probability range: [{y_prob_gpc.min():.3f}, {y_prob_gpc.max():.3f}]")
print(f"   Mean probability: {y_prob_gpc.mean():.3f}")
print(f"   Median probability: {np.median(y_prob_gpc):.3f}")


print("\n Saving all results...")


results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': list(results.values()),
    'Training_Time_s': [training_times.get(m, 0) for m in results.keys()],
    'Dataset': 'California Housing',
    'Samples': sample_size,
    'Threshold': f"${threshold*100000:.0f}",
    'GPC_Kernel': str(gpc.kernel_) if 'Gaussian Process' in results else 'N/A',
    'GPC_LogLikelihood': gpc.log_marginal_likelihood() if 'Gaussian Process' in results else np.nan
})

results_df.to_csv('complete_experiment_results.csv', index=False)
print(" Saved: complete_experiment_results.csv")


feature_df = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': rf_importance,
    'LR_Importance': lr_importance
})
feature_df.to_csv('feature_importance_details.csv', index=False)
print(" Saved: feature_importance_details.csv")


gpc_prob_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred_gpc,
    'Probability_High': y_prob_gpc,
    'Prediction_Correct': (y_test == y_pred_gpc).astype(int)
})
gpc_prob_df.to_csv('gpc_probability_predictions.csv', index=False)
print(" Saved: gpc_probability_predictions.csv")


print("\n" + "="*60)
print(" TASK 1 COMPLETED SUCCESSFULLY WITH GPC!")
print("="*60)

print(f"\n KEY FINDINGS:")
print(f"   1. Gaussian Process Classification implemented (required by brief)")
print(f"   2. GPC Accuracy: {accuracy_gpc:.4f} (competitive with other models)")
print(f"   3. GPC provides probabilistic outputs (Bayesian advantage)")
print(f"   4. Best overall model: {best_model} ({best_accuracy:.4f})")

print(f"\n FILES CREATED:")
print(f"   1. model_comparison_with_gpc.png - Performance comparison")
print(f"   2. gpc_probability_analysis.png - GPC probability distributions")
print(f"   3. feature_importance_comparison.png - Feature analysis")
print(f"   4. complete_experiment_results.csv - All model results")
print(f"   5. gpc_probability_predictions.csv - GPC detailed predictions")

print(f"\n READY FOR YOUR 6-PAGE PAPER!")
print("   You now have:")
print("   ✓ Gaussian Process Classification (required by brief)")
print("   ✓ Thresholding applied to create binary classes")
print("   ✓ ≥4 input variables (8 features used)")
print("   ✓ Comparative analysis with other models")
print("   ✓ All visualizations and CSV outputs")
print("   ✓ Bayesian/probabilistic outputs from GPC")

print(f"\n PAPER SECTIONS TO UPDATE:")
print("   1. Methods: Add GPC subsection")
print("   2. Results: Add GPC results table")
print("   3. Discussion: Compare GPC Bayesian advantages")
print("   4. Add ethical/social/legal section (required)")

