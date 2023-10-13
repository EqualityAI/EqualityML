import pandas as pd
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from equalityml.fair import FAIR
from equalityml.stats import paired_ttest
from equalityml.threshold import discrimination_threshold

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))

if __name__ == "__main__":

    # First train a Machine Learning estimator with the training data
    random_state = 42

    # Read training and testing data.
    target_var = "HOS"
    protected_variable = 'RACERETH'
    train_path = os.path.join(PACKAGE_PATH, 'data', 'data_train.csv')
    training_data = pd.read_csv(train_path)
    X_train = training_data.drop(columns=target_var)
    y_train = training_data[target_var]
    test_path = os.path.join(PACKAGE_PATH, 'data', 'data_test.csv')
    testing_data = pd.read_csv(test_path)
    X_test = testing_data.drop(columns=target_var)
    y_test = testing_data[target_var]

    # Train a machine learning estimator
    mdl_clf_1 = RandomForestClassifier(random_state=random_state)
    mdl_clf_1.fit(X_train, y_train)

    # Compute Fairness score for "statistical_parity_ratio"
    fair_object = FAIR(ml_model=mdl_clf_1,
                       training_data=training_data,
                       testing_data=testing_data,
                       target_variable=target_var,
                       protected_variable=protected_variable,
                       privileged_class=1,
                       random_seed=random_state)

    fair_object.print_fairness_metrics()
    for metric_name in fair_object.fairness_metrics_list:
        print(f"{metric_name} - {fair_object.fairness_metric(metric_name)}")

    metric_name = "statistical_parity_ratio"
    prev_fairness_metric = fair_object.fairness_metric(metric_name)

    # Compare bias mitigation results
    comparison_df = fair_object.compare_mitigation_methods(show=False)
    print(comparison_df)

    fair_object.print_bias_mitigation_methods()
    mitigation_method = "resampling-preferential"
    # "resampling-uniform", "resampling", "resampling-preferential", "correlation-remover", "reweighing",
    # "disparate-impact-remover"

    # mitigation_res = fair_object.bias_mitigation(mitigation_method=mitigation_method)
    mdl_clf_2 = fair_object.model_mitigation(mitigation_method)

    # Estimate prediction probability and predicted class of training data (Put empty dataframe for testing in order to
    # estimate this)
    pred_class = mdl_clf_1.predict(X_test)
    pred_prob = mdl_clf_1.predict_proba(X_test)
    pred_prob = pred_prob[:, 1]  # keep probabilities for positive outcomes only

    # Evaluate some scores
    prev_auc = roc_auc_score(y_test, pred_prob)  # Area under a curve
    prev_accuracy = accuracy_score(y_test, pred_class)  # classification accuracy

    pred_class = mdl_clf_2.predict(X_test)
    pred_prob = mdl_clf_2.predict_proba(X_test)
    pred_prob = pred_prob[:, 1]  # keep probabilities for positive outcomes only

    # re-evaluate the scores
    new_auc = roc_auc_score(y_test, pred_prob)  # Area under a curve
    print(f"Previous AUC = {prev_auc} and New AUC = {new_auc}")

    new_accuracy = accuracy_score(y_test, pred_class)  # classification accuracy
    print(f"Previous accuracy = {prev_accuracy} and New accuracy = {new_accuracy}")

    fair_object.update_classifier(mdl_clf_2)
    new_fairness_metric = fair_object.fairness_metric(metric_name)

    print(
        f"Previous Fairness Score = {prev_fairness_metric:.2f} and New Fairness Score = {new_fairness_metric:.2f}")

    data = pd.concat([training_data, testing_data])
    X = data.drop(columns=target_var)
    y = data[target_var]
    X_test = testing_data.drop(columns=target_var)
    y_test = testing_data[target_var]

    # Paired ttest
    results = paired_ttest(mdl_clf_1,
                           X_test,
                           y_test,
                           model_2=mdl_clf_2,
                           method="mcnemar",
                           threshold=0.5)
    print("Mcnemar result (chi2, p) = ", results)

    results = paired_ttest(mdl_clf_1,
                           X,
                           y,
                           model_2=mdl_clf_2,
                           method="5x2cv",
                           random_seed=random_state)
    print("5x2cv result (chi2, p) = ", results)

    # Paired t test based on Fairness metric
    results = paired_ttest(mdl_clf_1,
                           X,
                           y,
                           model_2=mdl_clf_2,
                           method="5x2cv",
                           fair_object=fair_object,
                           mitigation_method=mitigation_method,
                           scoring=metric_name,
                           random_seed=random_state)
    print("5x2cv fairness result by scoring fairness metric (chi2, p) = ", results)

    results = paired_ttest(mdl_clf_1,
                           X,
                           y,
                           method="5x2cv",
                           fair_object=fair_object,
                           mitigation_method=mitigation_method,
                           scoring=metric_name,
                           compute_discrimination_threshold=True,
                           decision_maker=(metric_name, 'max'),
                           random_seed=random_state)
    print("5x2cv fairness result by scoring fairness metric (chi2, p) = ", results)

    # Paired t test based on accuracy
    results = paired_ttest(mdl_clf_1,
                           X,
                           y,
                           method="5x2cv",
                           fair_object=fair_object,
                           mitigation_method=mitigation_method,
                           scoring="accuracy",
                           compute_discrimination_threshold=True,
                           decision_maker=("accuracy", 'max'),
                           random_seed=random_state)
    print("5x2cv fairness result by scoring accuracy (chi2, p) = ", results)

    # Paired t test based on accuracy
    results = paired_ttest(mdl_clf_1,
                           X,
                           y,
                           method="5x2cv",
                           fair_object=fair_object,
                           mitigation_method=mitigation_method,
                           scoring="accuracy",
                           random_seed=random_state)
    print("5x2cv fairness result by scoring accuracy (chi2, p) = ", results)

    # Discrimination Threshold
    dt = discrimination_threshold(mdl_clf_1,
                                  X,
                                  y,
                                  fair_object=fair_object,
                                  decision_maker=['f1', 'max'],
                                  metrics=['f1', 'cost', metric_name],
                                  utility_costs=[1, -1, -0.1, 0.1],
                                  show=True)
    print(f"Discrimination Threshold {dt}")

    X_test = testing_data.drop(columns=target_var)
    y_Test = testing_data[target_var]
    # Discrimination Threshold
    dt = discrimination_threshold(mdl_clf_2,
                                  X_test,
                                  y_test,
                                  fair_object=fair_object,
                                  decision_maker=['f1', 'max'],
                                  metrics=['f1', 'cost', metric_name],
                                  utility_costs=[1, -1, -0.1, 0.1],
                                  show=True,
                                  model_training=False)
    print(f"Discrimination Threshold {dt}")
