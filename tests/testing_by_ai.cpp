#include "decision_tree_classifier.hpp"
#include "decision_tree_regressor.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <numeric>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

class Timer {
    Clock::time_point start_time;
public:
    Timer() : start_time(Clock::now()) {}
    double elapsed() const {
        return Duration(Clock::now() - start_time).count();
    }
};

std::mt19937& get_rng() {
    static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    return rng;
}

double random_uniform(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(get_rng());
}

int random_int(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(get_rng());
}

double compute_accuracy(const std::vector<int>& pred, const std::vector<int>& true_labels) {
    if (pred.empty()) return 0.0;
    size_t correct = 0;
    for (size_t i = 0; i < pred.size(); ++i) {
        if (pred[i] == true_labels[i]) ++correct;
    }
    return static_cast<double>(correct) / pred.size();
}

double compute_rmse(const std::vector<double>& pred, const std::vector<double>& true_vals) {
    if (pred.empty()) return 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < pred.size(); ++i) {
        double diff = pred[i] - true_vals[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / pred.size());
}

double compute_mae(const std::vector<double>& pred, const std::vector<double>& true_vals) {
    if (pred.empty()) return 0.0;
    double sum_abs = 0.0;
    for (size_t i = 0; i < pred.size(); ++i) {
        sum_abs += std::abs(pred[i] - true_vals[i]);
    }
    return sum_abs / pred.size();
}

void print_separator() {
    std::cout << "------------------------------------------------------------\n";
}

// ======================= CLASSIFICATION TESTS =======================

void test_classification_binary() {
    std::cout << "[CLASSIFICATION 1] Binary classification - linearly separable\n";
    Timer timer;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    for (int i = 0; i < 500; ++i) {
        double x1 = random_uniform(-3.0, 3.0);
        double x2 = random_uniform(-3.0, 3.0);
        int label = (x1 + x2 > 0.0) ? 1 : 0;
        X_train.push_back({x1, x2});
        y_train.push_back(label);
    }

    for (int i = 0; i < 200; ++i) {
        double x1 = random_uniform(-3.0, 3.0);
        double x2 = random_uniform(-3.0, 3.0);
        int label = (x1 + x2 > 0.0) ? 1 : 0;
        X_test.push_back({x1, x2});
        y_test.push_back(label);
    }

    DecisionTreeClassifier clf(8, 5, 2, "gini", 0.0);
    clf.fit(X_train, y_train);

    std::vector<int> predictions;
    for (const auto& x : X_test) {
        predictions.push_back(clf.predict(x));
    }

    double accuracy = compute_accuracy(predictions, y_test);
    std::cout << "  Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100 << "%\n";
    std::cout << "  Number of leaves: " << clf.get_n_leaves() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_classification_multiclass() {
    std::cout << "[CLASSIFICATION 2] Multi-class classification (3 classes)\n";
    Timer timer;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    for (int i = 0; i < 600; ++i) {
        double x1 = random_uniform(-2.0, 2.0);
        double x2 = random_uniform(-2.0, 2.0);
        int label;
        if (x1 < -0.5) label = 0;
        else if (x1 < 0.5) label = 1;
        else label = 2;
        X_train.push_back({x1, x2});
        y_train.push_back(label);
    }

    for (int i = 0; i < 200; ++i) {
        double x1 = random_uniform(-2.0, 2.0);
        double x2 = random_uniform(-2.0, 2.0);
        int label;
        if (x1 < -0.5) label = 0;
        else if (x1 < 0.5) label = 1;
        else label = 2;
        X_test.push_back({x1, x2});
        y_test.push_back(label);
    }

    DecisionTreeClassifier clf(8, 5, 2, "gini", 0.0);
    clf.fit(X_train, y_train);

    std::vector<int> predictions;
    for (const auto& x : X_test) {
        predictions.push_back(clf.predict(x));
    }

    double accuracy = compute_accuracy(predictions, y_test);
    std::cout << "  Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100 << "%\n";
    std::cout << "  Number of leaves: " << clf.get_n_leaves() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_classification_noisy_data() {
    std::cout << "[CLASSIFICATION 3] Noisy data with pruning effect\n";
    Timer timer;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    for (int i = 0; i < 300; ++i) {
        double x = random_uniform(-3.0, 3.0);
        int true_label = (x > 0.0) ? 1 : 0;
        int noisy_label = (random_uniform(0.0, 1.0) < 0.15) ? (1 - true_label) : true_label;
        X_train.push_back({x});
        y_train.push_back(noisy_label);
    }

    for (int i = 0; i < 500; ++i) {
        double x = random_uniform(-3.0, 3.0);
        int label = (x > 0.0) ? 1 : 0;
        X_test.push_back({x});
        y_test.push_back(label);
    }

    DecisionTreeClassifier clf_no_prune(15, 2, 1, "gini", 0.0);
    DecisionTreeClassifier clf_prune(15, 2, 1, "gini", 0.02);

    clf_no_prune.fit(X_train, y_train);
    clf_prune.fit(X_train, y_train);

    std::vector<int> pred_no_prune, pred_prune;
    for (const auto& x : X_test) {
        pred_no_prune.push_back(clf_no_prune.predict(x));
        pred_prune.push_back(clf_prune.predict(x));
    }

    double acc_no_prune = compute_accuracy(pred_no_prune, y_test);
    double acc_prune = compute_accuracy(pred_prune, y_test);

    std::cout << "  Without pruning: " << std::fixed << std::setprecision(4) << acc_no_prune * 100 << "% (" << clf_no_prune.get_n_leaves() << " leaves)\n";
    std::cout << "  With pruning: " << std::fixed << std::setprecision(4) << acc_prune * 100 << "% (" << clf_prune.get_n_leaves() << " leaves)\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_classification_feature_importance() {
    std::cout << "[CLASSIFICATION 4] Feature importance\n";
    Timer timer;

    std::vector<std::vector<double>> X;
    std::vector<int> y;

    for (int i = 0; i < 1000; ++i) {
        double f0 = random_uniform(-5.0, 5.0);
        double f1 = random_uniform(-5.0, 5.0);
        double f2 = random_uniform(-5.0, 5.0);
        double f3 = random_uniform(-5.0, 5.0);

        int label = (f0 + 0.5 * f1 > 0.0) ? 1 : 0;
        X.push_back({f0, f1, f2, f3});
        y.push_back(label);
    }

    DecisionTreeClassifier clf(10, 10, 5, "gini", 0.0);
    clf.fit(X, y);

    auto importances = clf.get_feature_importances();
    std::cout << "  Feature importances:\n";
    for (size_t i = 0; i < importances.size(); ++i) {
        std::cout << "    F" << i << ": " << std::fixed << std::setprecision(4) << importances[i] << "\n";
    }
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_classification_predict_proba() {
    std::cout << "[CLASSIFICATION 5] Predict probability consistency\n";
    Timer timer;

    std::vector<std::vector<double>> X = {{1.0}, {1.0}, {1.0}, {2.0}, {2.0}, {3.0}, {3.0}, {3.0}};
    std::vector<int> y = {0, 0, 0, 1, 1, 2, 2, 2};

    DecisionTreeClassifier clf(5, 2, 1, "gini", 0.0);
    clf.fit(X, y);

    double test_x = 1.5;
    auto proba = clf.predict_proba({test_x});

    std::cout << "  Predict proba for x=" << test_x << ":\n";
    for (const auto& [cls, p] : proba) {
        std::cout << "    Class " << cls << ": " << std::fixed << std::setprecision(4) << p << "\n";
    }

    int prediction = clf.predict({test_x});
    std::cout << "  Prediction: " << prediction << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

// ======================= REGRESSION TESTS =======================

void test_regression_linear() {
    std::cout << "[REGRESSION 1] Linear relationship y = 2*x + 3\n";
    Timer timer;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    for (int i = 0; i < 400; ++i) {
        double x = random_uniform(-5.0, 5.0);
        double noise = random_uniform(-0.2, 0.2);
        X_train.push_back({x});
        y_train.push_back(2.0 * x + 3.0 + noise);
    }

    for (int i = 0; i < 200; ++i) {
        double x = random_uniform(-5.0, 5.0);
        X_test.push_back({x});
        y_test.push_back(2.0 * x + 3.0);
    }

    DecisionTreeRegressor reg(8, 5, 2, "mse", 0.0);
    reg.fit(X_train, y_train);

    std::vector<double> predictions;
    for (const auto& x : X_test) {
        predictions.push_back(reg.predict(x));
    }

    double rmse = compute_rmse(predictions, y_test);
    double mae = compute_mae(predictions, y_test);

    std::cout << "  RMSE: " << std::fixed << std::setprecision(4) << rmse << "\n";
    std::cout << "  MAE: " << std::fixed << std::setprecision(4) << mae << "\n";
    std::cout << "  Number of leaves: " << reg.get_n_leaves() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_regression_quadratic() {
    std::cout << "[REGRESSION 2] Quadratic relationship y = x^2\n";
    Timer timer;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    for (int i = 0; i < 600; ++i) {
        double x = random_uniform(-3.0, 3.0);
        double noise = random_uniform(-0.3, 0.3);
        X_train.push_back({x});
        y_train.push_back(x * x + noise);
    }

    for (int i = 0; i < 200; ++i) {
        double x = random_uniform(-3.0, 3.0);
        X_test.push_back({x});
        y_test.push_back(x * x);
    }

    DecisionTreeRegressor reg(10, 5, 2, "mse", 0.0);
    reg.fit(X_train, y_train);

    std::vector<double> predictions;
    for (const auto& x : X_test) {
        predictions.push_back(reg.predict(x));
    }

    double rmse = compute_rmse(predictions, y_test);
    double mae = compute_mae(predictions, y_test);

    std::cout << "  RMSE: " << std::fixed << std::setprecision(4) << rmse << "\n";
    std::cout << "  MAE: " << std::fixed << std::setprecision(4) << mae << "\n";
    std::cout << "  Number of leaves: " << reg.get_n_leaves() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_regression_multidimensional() {
    std::cout << "[REGRESSION 3] Multidimensional linear y = 3*x1 - 2*x2 + x3\n";
    Timer timer;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    for (int i = 0; i < 800; ++i) {
        double x1 = random_uniform(-2.0, 2.0);
        double x2 = random_uniform(-2.0, 2.0);
        double x3 = random_uniform(-2.0, 2.0);
        double noise = random_uniform(-0.15, 0.15);
        X_train.push_back({x1, x2, x3});
        y_train.push_back(3.0 * x1 - 2.0 * x2 + x3 + noise);
    }

    for (int i = 0; i < 200; ++i) {
        double x1 = random_uniform(-2.0, 2.0);
        double x2 = random_uniform(-2.0, 2.0);
        double x3 = random_uniform(-2.0, 2.0);
        X_test.push_back({x1, x2, x3});
        y_test.push_back(3.0 * x1 - 2.0 * x2 + x3);
    }

    DecisionTreeRegressor reg(12, 10, 5, "mse", 0.0);
    reg.fit(X_train, y_train);

    std::vector<double> predictions;
    for (const auto& x : X_test) {
        predictions.push_back(reg.predict(x));
    }

    double rmse = compute_rmse(predictions, y_test);
    double mae = compute_mae(predictions, y_test);

    auto importances = reg.get_feature_importances();

    std::cout << "  RMSE: " << std::fixed << std::setprecision(4) << rmse << "\n";
    std::cout << "  MAE: " << std::fixed << std::setprecision(4) << mae << "\n";
    std::cout << "  Feature importances: ";
    for (size_t i = 0; i < importances.size(); ++i) {
        std::cout << "F" << i << "=" << std::fixed << std::setprecision(3) << importances[i] << " ";
    }
    std::cout << "\n";
    std::cout << "  Number of leaves: " << reg.get_n_leaves() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_regression_mse_vs_mae() {
    std::cout << "[REGRESSION 4] MSE vs MAE with outliers\n";
    Timer timer;

    std::vector<std::vector<double>> X;
    std::vector<double> y_mse, y_mae;

    for (int i = 0; i < 500; ++i) {
        double x = random_uniform(-5.0, 5.0);
        X.push_back({x});
        double true_val = 2.0 * x + 1.0;
        if (random_uniform(0.0, 1.0) < 0.1) {
            true_val += 15.0;
        }
        y_mse.push_back(true_val);
        y_mae.push_back(true_val);
    }

    DecisionTreeRegressor reg_mse(10, 5, 2, "mse", 0.0);
    DecisionTreeRegressor reg_mae(10, 5, 2, "mae", 0.0);

    reg_mse.fit(X, y_mse);
    reg_mae.fit(X, y_mae);

    std::vector<std::vector<double>> X_test;
    std::vector<double> y_test;
    for (int i = 0; i < 200; ++i) {
        double x = random_uniform(-5.0, 5.0);
        X_test.push_back({x});
        y_test.push_back(2.0 * x + 1.0);
    }

    std::vector<double> pred_mse, pred_mae;
    for (const auto& x : X_test) {
        pred_mse.push_back(reg_mse.predict(x));
        pred_mae.push_back(reg_mae.predict(x));
    }

    double mae_mse = compute_mae(pred_mse, y_test);
    double mae_mae = compute_mae(pred_mae, y_test);

    std::cout << "  MAE using MSE criterion: " << std::fixed << std::setprecision(4) << mae_mse << "\n";
    std::cout << "  MAE using MAE criterion: " << std::fixed << std::setprecision(4) << mae_mae << "\n";
    std::cout << "  MSE leaves: " << reg_mse.get_n_leaves() << ", MAE leaves: " << reg_mae.get_n_leaves() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_regression_pruning() {
    std::cout << "[REGRESSION 5] Cost complexity pruning\n";
    Timer timer;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    for (int i = 0; i < 200; ++i) {
        double x = random_uniform(-3.0, 3.0);
        double noise = random_uniform(-0.8, 0.8);
        X_train.push_back({x});
        y_train.push_back(std::sin(x) + noise);
    }

    for (int i = 0; i < 500; ++i) {
        double x = random_uniform(-3.0, 3.0);
        X_test.push_back({x});
        y_test.push_back(std::sin(x));
    }

    DecisionTreeRegressor reg_no_prune(20, 2, 1, "mse", 0.0);
    DecisionTreeRegressor reg_prune(20, 2, 1, "mse", 0.1);

    reg_no_prune.fit(X_train, y_train);
    reg_prune.fit(X_train, y_train);

    std::vector<double> pred_no_prune, pred_prune;
    for (const auto& x : X_test) {
        pred_no_prune.push_back(reg_no_prune.predict(x));
        pred_prune.push_back(reg_prune.predict(x));
    }

    double mae_no_prune = compute_mae(pred_no_prune, y_test);
    double mae_prune = compute_mae(pred_prune, y_test);

    std::cout << "  Without pruning: MAE=" << std::fixed << std::setprecision(4) << mae_no_prune << " (" << reg_no_prune.get_n_leaves() << " leaves)\n";
    std::cout << "  With pruning: MAE=" << std::fixed << std::setprecision(4) << mae_prune << " (" << reg_prune.get_n_leaves() << " leaves)\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_regression_constant() {
    std::cout << "[REGRESSION 6] Constant target value\n";
    Timer timer;

    std::vector<std::vector<double>> X(200, {1.0, 2.0, 3.0});
    std::vector<double> y(200, 42.0);

    DecisionTreeRegressor reg(10, 5, 2, "mse", 0.0);
    reg.fit(X, y);

    double pred1 = reg.predict({1.0, 2.0, 3.0});
    double pred2 = reg.predict({5.0, 6.0, 7.0});

    std::cout << "  Prediction for training point: " << std::fixed << std::setprecision(4) << pred1 << "\n";
    std::cout << "  Prediction for new point: " << std::fixed << std::setprecision(4) << pred2 << "\n";
    std::cout << "  Number of leaves: " << reg.get_n_leaves() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_regression_depth_effect() {
    std::cout << "[REGRESSION 7] Effect of tree depth\n";
    Timer timer;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    for (int i = 0; i < 300; ++i) {
        double x = random_uniform(-4.0, 4.0);
        X_train.push_back({x});
        y_train.push_back(std::exp(-x * x) + random_uniform(-0.05, 0.05));
    }

    for (int i = 0; i < 300; ++i) {
        double x = random_uniform(-4.0, 4.0);
        X_test.push_back({x});
        y_test.push_back(std::exp(-x * x));
    }

    std::vector<int> depths = {2, 5, 10, 15};
    std::cout << "  Depth vs Performance:\n";
    for (int depth : depths) {
        DecisionTreeRegressor reg(depth, 5, 2, "mse", 0.0);
        reg.fit(X_train, y_train);

        std::vector<double> predictions;
        for (const auto& x : X_test) {
            predictions.push_back(reg.predict(x));
        }
        double mae = compute_mae(predictions, y_test);
        std::cout << "    Depth=" << depth << ": MAE=" << std::fixed << std::setprecision(4) << mae << ", leaves=" << reg.get_n_leaves() << "\n";
    }
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_regression_feature_importance() {
    std::cout << "[REGRESSION 8] Feature importance detection\n";
    Timer timer;

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for (int i = 0; i < 1000; ++i) {
        double f0 = random_uniform(-3.0, 3.0);
        double f1 = random_uniform(-3.0, 3.0);
        double f2 = random_uniform(-3.0, 3.0);
        double f3 = random_uniform(-3.0, 3.0);

        double target = 5.0 * f0 + 2.0 * f1 + random_uniform(-0.2, 0.2);
        X.push_back({f0, f1, f2, f3});
        y.push_back(target);
    }

    DecisionTreeRegressor reg(10, 10, 5, "mse", 0.0);
    reg.fit(X, y);

    auto importances = reg.get_feature_importances();
    std::cout << "  Feature importances:\n";
    for (size_t i = 0; i < importances.size(); ++i) {
        std::cout << "    F" << i << ": " << std::fixed << std::setprecision(4) << importances[i] << "\n";
    }
    std::cout << "  (F0 and F1 should dominate, F2 and F3 should be near zero)\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_regression_sample_splits() {
    std::cout << "[REGRESSION 9] Min samples split constraint\n";
    Timer timer;

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for (int i = 0; i < 50; ++i) {
        double x = static_cast<double>(i);
        X.push_back({x});
        y.push_back(2.0 * x + 1.0);
    }

    DecisionTreeRegressor reg_low(10, 2, 1, "mse", 0.0);
    DecisionTreeRegressor reg_high(10, 20, 1, "mse", 0.0);

    reg_low.fit(X, y);
    reg_high.fit(X, y);

    std::cout << "  min_samples_split=2: " << reg_low.get_n_leaves() << " leaves\n";
    std::cout << "  min_samples_split=20: " << reg_high.get_n_leaves() << " leaves\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

void test_regression_interaction() {
    std::cout << "[REGRESSION 10] Interaction term x1 * x2\n";
    Timer timer;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    for (int i = 0; i < 600; ++i) {
        double x1 = random_uniform(-2.0, 2.0);
        double x2 = random_uniform(-2.0, 2.0);
        double noise = random_uniform(-0.1, 0.1);
        X_train.push_back({x1, x2});
        y_train.push_back(x1 * x2 + noise);
    }

    for (int i = 0; i < 200; ++i) {
        double x1 = random_uniform(-2.0, 2.0);
        double x2 = random_uniform(-2.0, 2.0);
        X_test.push_back({x1, x2});
        y_test.push_back(x1 * x2);
    }

    DecisionTreeRegressor reg(12, 5, 2, "mse", 0.0);
    reg.fit(X_train, y_train);

    std::vector<double> predictions;
    for (const auto& x : X_test) {
        predictions.push_back(reg.predict(x));
    }

    double rmse = compute_rmse(predictions, y_test);
    std::cout << "  RMSE on interaction term: " << std::fixed << std::setprecision(4) << rmse << "\n";
    std::cout << "  Leaves: " << reg.get_n_leaves() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << timer.elapsed() << " sec\n\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "==================================================\n";
    std::cout << "DECISION TREE COMPREHENSIVE TEST SUITE\n";
    std::cout << "==================================================\n\n";

    Timer total_timer;

    // std::cout << "================ CLASSIFICATION TESTS ================\n\n";
    // test_classification_binary();
    // test_classification_multiclass();
    // test_classification_noisy_data();
    // test_classification_feature_importance();
    // test_classification_predict_proba();

    std::cout << "================= REGRESSION TESTS ==================\n\n";
    test_regression_linear();
    test_regression_quadratic();
    test_regression_multidimensional();
    test_regression_mse_vs_mae();
    test_regression_pruning();
    test_regression_constant();
    test_regression_depth_effect();
    test_regression_feature_importance();
    test_regression_sample_splits();
    test_regression_interaction();

    std::cout << "==================================================\n";
    std::cout << "ALL TESTS COMPLETED IN " << std::fixed << std::setprecision(2)
              << total_timer.elapsed() << " SECONDS\n";
    std::cout << "==================================================\n";

    return 0;
}