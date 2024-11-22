import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve, auc


def plot_roc_curve(model, X_test, y_test):
    # calculate the fpr and tpr for all thresholds of the classification
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def generic_classif_metrics(model, X_test, y_test, visualize=False):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {'accuracy': accuracy_score(y_test, y_pred),
               'f1-score': f1_score(y_test, y_pred),
               'confusion_matrix': confusion_matrix(y_test, y_pred),
               'auc': roc_auc_score(y_test, y_proba)}
    
    if visualize:
        print('Accuracy: ', metrics['accuracy'])
        print('F1 score: ', metrics['f1-score'])
        print('Confusion matrix: ', metrics['confusion_matrix'])
        
        print('AUC: ', metrics['auc'])
        plot_roc_curve(model, X_test, y_test)
