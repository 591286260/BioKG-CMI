from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

X = np.loadtxt('../../06Multi-source feature fusion/2匹配/亚细胞定位/9589/SampleFeature(KG_9589_亚细胞).csv', delimiter=',')

y = np.concatenate((np.ones(len(X)//2), np.zeros(len(X)//2)))

clf = AdaBoostClassifier()

skf = StratifiedKFold(n_splits=5)

mean_fpr = np.linspace(0, 1, 100)
with open("5-fold data.txt", "w") as f:
    f.write("")
    f.write(f"\t\tAccuracy Precision\t Recall\t\tspecificity\t  F1-score\t\tMCC\n")

fold_aucs = []
fold_auprs = []

for train_idx, test_idx in skf.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    clf.fit(X_train, y_train)

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred = np.where(y_pred_prob > threshold, 1, 0)

    print(classification_report(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fold_aucs.append(roc_auc)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    aupr = average_precision_score(y_test, y_pred_prob)
    fold_auprs.append(aupr)

    np.save(f"Y_pre{len()}.npy", y_pred_prob)
    np.save(f"Y_test{len()}.npy", y_test)

    fold_aucs.append(roc_auc)
    fold_auprs.append(aupr)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)



    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"SPEC: {specificity:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    print(f"MCC: {mcc:.4f}")
    with open("5-fold data.txt", "a") as f:
        f.write(f"\t\t{accuracy:.4f}\t  {precision:.4f}\t  {recall:.4f}\t  {specificity:.4f}\t  {f1_score:.4f}\t  {mcc:.4f}\n")




