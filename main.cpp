#include "include/decision_tree_classifier.hpp"
#include "include/decision_tree_regressor.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <map>
#include <numeric>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

class Timer {
    Clock::time_point start;
public:
    Timer() : start(Clock::now()) {}
    double elapsed() const {
        return Duration(Clock::now() - start).count();
    }
};

// ======================= CLASSIFICATION TESTS =======================

void test_classification_linear_separable() {
    std::cout << "1. LINEARLY SEPARABLE DATA\n";

    std::vector<std::vector<double>> X;
    std::vector<int> y;

    // Class 0: x1 + x2 < 0
    // Class 1: x1 + x2 > 0
    for (int i = 0; i < 200; ++i) {
        double x1 = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        double x2 = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        X.push_back({x1, x2});
        y.push_back((x1 + x2 > 0) ? 1 : 0);
    }

    DecisionTreeClassifier clf(5, 10, 5, "gini", 0.0);
    clf.fit(X, y);

    int correct = 0;
    for (int i = 0; i < 100; ++i) {
        double x1 = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        double x2 = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        int true_label = (x1 + x2 > 0) ? 1 : 0;
        int pred_label = clf.predict({x1, x2});
        if (pred_label == true_label) correct++;
    }

    double accuracy = static_cast<double>(correct) / 100;
    std::cout << "   Accuracy: " << accuracy * 100 << "% (expected > 95%)\n";
    std::cout << "   Leaves: " << clf.get_n_leaves() << "\n\n";
}

void test_classification_multiclass() {
    std::cout << "2. MULTI-CLASS (3 classes)\n";

    std::vector<std::vector<double>> X;
    std::vector<int> y;

    // 3 classes in different quadrants
    for (int i = 0; i < 300; ++i) {
        double x1 = (static_cast<double>(rand()) / RAND_MAX * 6) - 3;
        double x2 = (static_cast<double>(rand()) / RAND_MAX * 6) - 3;
        X.push_back({x1, x2});

        if (x1 < -1.0) y.push_back(0);
        else if (x1 < 1.0) y.push_back(1);
        else y.push_back(2);
    }

    DecisionTreeClassifier clf(7, 10, 5, "gini", 0.0);
    clf.fit(X, y);

    int correct = 0;
    for (int i = 0; i < 100; ++i) {
        double x1 = (static_cast<double>(rand()) / RAND_MAX * 6) - 3;
        double x2 = (static_cast<double>(rand()) / RAND_MAX * 6) - 3;

        int true_label;
        if (x1 < -1.0) true_label = 0;
        else if (x1 < 1.0) true_label = 1;
        else true_label = 2;

        if (clf.predict({x1, x2}) == true_label) correct++;
    }

    double accuracy = static_cast<double>(correct) / 100;
    std::cout << "   Accuracy: " << accuracy * 100 << "% (expected > 90%)\n";
    std::cout << "   Leaves: " << clf.get_n_leaves() << "\n\n";
}

void test_classification_pruning_effect() {
    std::cout << "3. PRUNING EFFECT ON OVERFITTING\n";

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    // Generate train (small, noisy) and test (large, clean)
    for (int i = 0; i < 100; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        X_train.push_back({x});
        // Add noise: 10% wrong labels
        if (rand() % 10 == 0) {
            y_train.push_back((x > 0) ? 0 : 1);  // Wrong label
        } else {
            y_train.push_back((x > 0) ? 1 : 0);  // Correct label
        }
    }

    for (int i = 0; i < 500; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        X_test.push_back({x});
        y_test.push_back((x > 0) ? 1 : 0);
    }

    DecisionTreeClassifier clf_no_prune(20, 2, 1, "gini", 0.0);
    DecisionTreeClassifier clf_prune(20, 2, 1, "gini", 0.05);

    clf_no_prune.fit(X_train, y_train);
    clf_prune.fit(X_train, y_train);

    double acc_no_prune = 0.0, acc_prune = 0.0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        if (clf_no_prune.predict(X_test[i]) == y_test[i]) acc_no_prune += 1;
        if (clf_prune.predict(X_test[i]) == y_test[i]) acc_prune += 1;
    }
    acc_no_prune /= X_test.size();
    acc_prune /= X_test.size();

    std::cout << "   No prune: " << acc_no_prune * 100 << "% ("
              << clf_no_prune.get_n_leaves() << " leaves)\n";
    std::cout << "   With prune: " << acc_prune * 100 << "% ("
              << clf_prune.get_n_leaves() << " leaves)\n";
    std::cout << "   Pruning should improve generalization\n\n";
}

void test_classification_feature_importance() {
    std::cout << "4. FEATURE IMPORTANCE\n";

    std::vector<std::vector<double>> X;
    std::vector<int> y;

    // Feature 0: very important
    // Feature 1: somewhat important
    // Feature 2: irrelevant
    for (int i = 0; i < 500; ++i) {
        double f0 = (static_cast<double>(rand()) / RAND_MAX * 10) - 5;
        double f1 = (static_cast<double>(rand()) / RAND_MAX * 10) - 5;
        double f2 = (static_cast<double>(rand()) / RAND_MAX * 10) - 5;

        X.push_back({f0, f1, f2});
        // Decision based mostly on f0, somewhat on f1
        if (f0 > 1.5 * f1 + 0.5) {
            y.push_back(1);
        } else {
            y.push_back(0);
        }
    }

    DecisionTreeClassifier clf(8, 10, 5, "gini", 0.0);
    clf.fit(X, y);

    auto importances = clf.get_feature_importances();
    std::cout << "   F0: " << importances[0] << " (should be highest)\n";
    std::cout << "   F1: " << importances[1] << " (should be medium)\n";
    std::cout << "   F2: " << importances[2] << " (should be lowest)\n";

    if (importances[0] > importances[1] && importances[1] > importances[2]) {
        std::cout << "   ✓ Importance ordering correct\n";
    } else {
        std::cout << "   ⚠️ Importance ordering may be wrong\n";
    }
    std::cout << "\n";
}

void test_classification_predict_proba() {
    std::cout << "5. PREDICT_PROBA CONSISTENCY\n";

    std::vector<std::vector<double>> X = {{1.0}, {1.0}, {1.0}, {2.0}, {2.0}, {3.0}};
    std::vector<int> y = {0, 0, 0, 1, 1, 2};  // 3 classes

    DecisionTreeClassifier clf(5, 2, 1, "gini", 0.0);
    clf.fit(X, y);

    // Test point similar to class 0
    auto proba1 = clf.predict_proba({1.1});
    std::cout << "   For x=1.1 (near class 0): ";
    for (const auto& [cls, prob] : proba1) {
        std::cout << "class " << cls << "=" << std::fixed << std::setprecision(2) << prob << " ";
    }
    std::cout << "\n";

    // Test predict vs predict_proba consistency
    int pred = clf.predict({1.1});
    if (!proba1.empty()) {
        auto max_it = std::max_element(proba1.begin(), proba1.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        if (max_it != proba1.end() && max_it->first == pred) {
            std::cout << "   ✓ predict and predict_proba consistent\n";
        }
    }
    std::cout << "\n";
}

void test_classification_entropy_vs_gini() {
    std::cout << "6. ENTROPY VS GINI COMPARISON\n";

    std::vector<std::vector<double>> X;
    std::vector<int> y;

    for (int i = 0; i < 300; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 10) - 5;
        X.push_back({x});
        y.push_back((x > 0) ? 1 : 0);
    }

    DecisionTreeClassifier clf_gini(5, 10, 5, "gini", 0.0);
    DecisionTreeClassifier clf_entropy(5, 10, 5, "entropy", 0.0);

    clf_gini.fit(X, y);
    clf_entropy.fit(X, y);

    // They should give similar results
    int same_predictions = 0;
    for (int i = 0; i < 100; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 10) - 5;
        if (clf_gini.predict({x}) == clf_entropy.predict({x})) {
            same_predictions++;
        }
    }

    double agreement = static_cast<double>(same_predictions) / 100;
    std::cout << "   Agreement between Gini and Entropy: " << agreement * 100 << "%\n";
    std::cout << "   Gini leaves: " << clf_gini.get_n_leaves()
              << ", Entropy leaves: " << clf_entropy.get_n_leaves() << "\n\n";
}

void test_classification_single_class() {
    std::cout << "7. SINGLE CLASS DATA\n";

    std::vector<std::vector<double>> X(100, {1.0, 2.0, 3.0});
    std::vector<int> y(100, 0);  // All same class

    DecisionTreeClassifier clf(5, 10, 5, "gini", 0.0);
    clf.fit(X, y);

    // Should always predict class 0
    bool all_zero = true;
    for (int i = 0; i < 10; ++i) {
        if (clf.predict({1.5, 2.5, 3.5}) != 0) all_zero = false;
    }

    std::cout << "   Always predicts class 0: " << (all_zero ? "YES" : "NO") << "\n";
    std::cout << "   Leaves: " << clf.get_n_leaves() << " (should be 1)\n\n";
}

void test_classification_small_dataset() {
    std::cout << "8. SMALL DATASET (min_samples_split test)\n";

    std::vector<std::vector<double>> X = {{1}, {2}, {3}, {4}};
    std::vector<int> y = {0, 0, 1, 1};

    // With min_samples_split=4, should not split at all
    DecisionTreeClassifier clf(5, 4, 2, "gini", 0.0);
    clf.fit(X, y);

    std::cout << "   Leaves with min_samples_split=4: " << clf.get_n_leaves()
              << " (should be 1)\n\n";
}

void test_classification_depth_limit() {
    std::cout << "9. DEPTH LIMIT\n";

    std::vector<std::vector<double>> X;
    std::vector<int> y;

    // Perfectly alternating pattern
    for (int i = 0; i < 100; ++i) {
        X.push_back({static_cast<double>(i)});
        y.push_back(i % 2);
    }

    DecisionTreeClassifier clf_shallow(3, 2, 1, "gini", 0.0);
    DecisionTreeClassifier clf_deep(10, 2, 1, "gini", 0.0);

    clf_shallow.fit(X, y);
    clf_deep.fit(X, y);

    std::cout << "   Shallow (depth=3) leaves: " << clf_shallow.get_n_leaves() << "\n";
    std::cout << "   Deep (depth=10) leaves: " << clf_deep.get_n_leaves() << "\n";
    std::cout << "   Deep tree should have more leaves\n\n";
}

void test_classification_pruning_stability() {
    std::cout << "10. PRUNING STABILITY\n";

    std::vector<std::vector<double>> X;
    std::vector<int> y;

    for (int i = 0; i < 200; ++i) {
        double x1 = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        double x2 = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        X.push_back({x1, x2});
        y.push_back((x1 * x1 + x2 * x2 < 2.0) ? 1 : 0);
    }

    // Same data, same parameters - should get same tree size
    std::vector<int> leaf_counts;
    for (int run = 0; run < 5; ++run) {
        DecisionTreeClassifier clf(8, 10, 5, "gini", 0.02);
        clf.fit(X, y);
        leaf_counts.push_back(clf.get_n_leaves());
    }

    bool stable = true;
    for (size_t i = 1; i < leaf_counts.size(); ++i) {
        if (leaf_counts[i] != leaf_counts[0]) stable = false;
    }

    std::cout << "   Leaf counts over 5 runs: ";
    for (int count : leaf_counts) std::cout << count << " ";
    std::cout << "\n   Stable: " << (stable ? "YES" : "NO") << "\n\n";
}

// ======================= REGRESSION TESTS =======================

void test_regression_linear() {
    std::cout << "1. LINEAR RELATIONSHIP\n";

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    // y = 2*x + 3 + noise
    for (int i = 0; i < 200; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 10) - 5;
        double noise = (static_cast<double>(rand()) / RAND_MAX * 0.4) - 0.2;
        X.push_back({x});
        y.push_back(2.0 * x + 3.0 + noise);
    }

    DecisionTreeRegressor reg(5, 10, 5, "mse", 0.0);
    reg.fit(X, y);

    double total_error = 0.0;
    for (int i = 0; i < 50; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 10) - 5;
        double true_val = 2.0 * x + 3.0;
        double pred = reg.predict({x});
        total_error += std::abs(pred - true_val);
    }

    double avg_error = total_error / 50;
    std::cout << "   Avg absolute error: " << avg_error << "\n";
    std::cout << "   Expected: < 0.5 for linear relationship\n";
    std::cout << "   Leaves: " << reg.get_n_leaves() << "\n\n";
}

void test_regression_quadratic() {
    std::cout << "2. QUADRATIC RELATIONSHIP\n";

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    // y = x^2 + noise
    for (int i = 0; i < 300; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 8) - 4;
        double noise = (static_cast<double>(rand()) / RAND_MAX * 1.0) - 0.5;
        X.push_back({x});
        y.push_back(x * x + noise);
    }

    DecisionTreeRegressor reg(7, 10, 5, "mse", 0.0);
    reg.fit(X, y);

    double total_error = 0.0;
    for (int i = 0; i < 50; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 8) - 4;
        double true_val = x * x;
        double pred = reg.predict({x});
        total_error += std::abs(pred - true_val);
    }

    double avg_error = total_error / 50;
    std::cout << "   Avg absolute error: " << avg_error << "\n";
    std::cout << "   Expected: < 1.0 for quadratic\n";
    std::cout << "   Leaves: " << reg.get_n_leaves() << "\n\n";
}

void test_regression_mse_vs_mae() {
    std::cout << "3. MSE VS MAE WITH OUTLIERS\n";

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    // y = x + noise, with outliers
    for (int i = 0; i < 200; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 10);
        double noise = (static_cast<double>(rand()) / RAND_MAX * 0.3) - 0.15;
        double y_val = x + noise;

        // Add outliers to 10% of points
        if (rand() % 10 == 0) {
            y_val += 10.0;  // Big outlier
        }

        X.push_back({x});
        y.push_back(y_val);
    }

    DecisionTreeRegressor reg_mse(5, 10, 5, "mse", 0.0);
    DecisionTreeRegressor reg_mae(5, 10, 5, "mae", 0.0);

    reg_mse.fit(X, y);
    reg_mae.fit(X, y);

    // Test on clean data (without outliers)
    double error_mse = 0.0, error_mae = 0.0;
    for (int i = 0; i < 50; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 10);
        double true_val = x;  // No outliers in test

        error_mse += std::abs(reg_mse.predict({x}) - true_val);
        error_mae += std::abs(reg_mae.predict({x}) - true_val);
    }

    error_mse /= 50;
    error_mae /= 50;

    std::cout << "   MSE error: " << error_mse << "\n";
    std::cout << "   MAE error: " << error_mae << "\n";
    std::cout << "   MAE should be lower (more robust to outliers)\n";
    std::cout << "   MSE leaves: " << reg_mse.get_n_leaves()
              << ", MAE leaves: " << reg_mae.get_n_leaves() << "\n\n";
}

void test_regression_multidimensional() {
    std::cout << "4. MULTIDIMENSIONAL REGRESSION\n";

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    // y = 2*x1 + 3*x2 - 1.5*x3 + noise
    for (int i = 0; i < 400; ++i) {
        double x1 = (static_cast<double>(rand()) / RAND_MAX * 5);
        double x2 = (static_cast<double>(rand()) / RAND_MAX * 5);
        double x3 = (static_cast<double>(rand()) / RAND_MAX * 5);
        double noise = (static_cast<double>(rand()) / RAND_MAX * 0.5) - 0.25;

        X.push_back({x1, x2, x3});
        y.push_back(2.0*x1 + 3.0*x2 - 1.5*x3 + noise);
    }

    DecisionTreeRegressor reg(8, 10, 5, "mse", 0.0);
    reg.fit(X, y);

    // Check feature importances
    auto importances = reg.get_feature_importances();
    std::cout << "   Feature importances: ";
    for (size_t i = 0; i < importances.size(); ++i) {
        std::cout << "F" << i << "=" << std::fixed << std::setprecision(3)
                  << importances[i] << " ";
    }
    std::cout << "\n";

    // x2 should be most important (coefficient 3.0)
    // x1 next (coefficient 2.0)
    // x3 least (coefficient -1.5)
    if (importances.size() >= 3) {
        if (importances[1] > importances[0] && importances[0] > importances[2]) {
            std::cout << "   ✓ Importance ordering correct\n";
        }
    }

    // Test predictions
    double total_error = 0.0;
    for (int i = 0; i < 50; ++i) {
        double x1 = (static_cast<double>(rand()) / RAND_MAX * 5);
        double x2 = (static_cast<double>(rand()) / RAND_MAX * 5);
        double x3 = (static_cast<double>(rand()) / RAND_MAX * 5);
        double true_val = 2.0*x1 + 3.0*x2 - 1.5*x3;

        total_error += std::abs(reg.predict({x1, x2, x3}) - true_val);
    }

    std::cout << "   Avg error: " << total_error / 50 << "\n";
    std::cout << "   Leaves: " << reg.get_n_leaves() << "\n\n";
}

void test_regression_pruning_effect() {
    std::cout << "5. REGRESSION PRUNING EFFECT\n";

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    // Train on small noisy data, test on large clean data
    for (int i = 0; i < 100; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 10);
        double noise = (static_cast<double>(rand()) / RAND_MAX * 2.0) - 1.0;
        X_train.push_back({x});
        y_train.push_back(std::sin(x) + noise);
    }

    for (int i = 0; i < 500; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 10);
        X_test.push_back({x});
        y_test.push_back(std::sin(x));
    }

    DecisionTreeRegressor reg_no_prune(15, 2, 1, "mse", 0.0);
    DecisionTreeRegressor reg_prune(15, 2, 1, "mse", 0.03);

    reg_no_prune.fit(X_train, y_train);
    reg_prune.fit(X_train, y_train);

    double error_no_prune = 0.0, error_prune = 0.0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        error_no_prune += std::abs(reg_no_prune.predict(X_test[i]) - y_test[i]);
        error_prune += std::abs(reg_prune.predict(X_test[i]) - y_test[i]);
    }

    error_no_prune /= X_test.size();
    error_prune /= X_test.size();

    std::cout << "   No prune: " << error_no_prune << " ("
              << reg_no_prune.get_n_leaves() << " leaves)\n";
    std::cout << "   With prune: " << error_prune << " ("
              << reg_prune.get_n_leaves() << " leaves)\n";
    std::cout << "   Pruning should reduce overfitting\n\n";
}

void test_regression_constant_data() {
    std::cout << "6. CONSTANT DATA\n";

    std::vector<std::vector<double>> X(100, {1.0, 2.0, 3.0});
    std::vector<double> y(100, 5.0);  // All same value

    DecisionTreeRegressor reg(5, 10, 5, "mse", 0.0);
    reg.fit(X, y);

    // Should always predict 5.0
    bool all_same = true;
    double first_pred = reg.predict({1.0, 2.0, 3.0});
    for (int i = 0; i < 5; ++i) {
        if (std::abs(reg.predict({1.5, 2.5, 3.5}) - first_pred) > 1e-6) {
            all_same = false;
        }
    }

    std::cout << "   Constant predictions: " << (all_same ? "YES" : "NO") << "\n";
    std::cout << "   Leaves: " << reg.get_n_leaves() << " (should be 1)\n";
    std::cout << "   Prediction: " << first_pred << " (should be ~5.0)\n\n";
}

void test_regression_interaction_terms() {
    std::cout << "7. INTERACTION TERMS\n";

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    // y = x1 * x2 + noise (multiplicative interaction)
    for (int i = 0; i < 300; ++i) {
        double x1 = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        double x2 = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        double noise = (static_cast<double>(rand()) / RAND_MAX * 0.3) - 0.15;

        X.push_back({x1, x2});
        y.push_back(x1 * x2 + noise);
    }

    DecisionTreeRegressor reg(10, 10, 5, "mse", 0.0);
    reg.fit(X, y);

    double total_error = 0.0;
    for (int i = 0; i < 50; ++i) {
        double x1 = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        double x2 = (static_cast<double>(rand()) / RAND_MAX * 4) - 2;
        double true_val = x1 * x2;

        total_error += std::abs(reg.predict({x1, x2}) - true_val);
    }

    std::cout << "   Avg error for x1*x2: " << total_error / 50 << "\n";
    std::cout << "   Tree should learn interaction\n";
    std::cout << "   Leaves: " << reg.get_n_leaves() << "\n\n";
}

void test_regression_min_samples_leaf() {
    std::cout << "8. MIN_SAMPLES_LEAF CONSTRAINT\n";

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for (int i = 0; i < 50; ++i) {
        double x = static_cast<double>(i) / 50.0 * 10.0;
        X.push_back({x});
        y.push_back(std::sin(x));
    }

    // With min_samples_leaf=10, each leaf should have at least 10 samples
    DecisionTreeRegressor reg(10, 5, 10, "mse", 0.0);
    reg.fit(X, y);

    // Count approximate minimum samples per leaf
    int leaves = reg.get_n_leaves();
    int min_samples_per_leaf = 50 / leaves;  // Rough estimate

    std::cout << "   Total samples: 50\n";
    std::cout << "   Leaves: " << leaves << "\n";
    std::cout << "   Avg samples per leaf: ~" << min_samples_per_leaf << "\n";
    std::cout << "   Should be >= min_samples_leaf (10)\n\n";
}

void test_regression_large_depth() {
    std::cout << "9. LARGE DEPTH (Potential overfitting)\n";

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    // Simple linear relationship
    for (int i = 0; i < 100; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 10);
        double noise = (static_cast<double>(rand()) / RAND_MAX * 0.2) - 0.1;
        X.push_back({x});
        y.push_back(2.0 * x + 3.0 + noise);
    }

    DecisionTreeRegressor reg_shallow(3, 10, 5, "mse", 0.0);
    DecisionTreeRegressor reg_deep(20, 10, 5, "mse", 0.0);

    reg_shallow.fit(X, y);
    reg_deep.fit(X, y);

    std::cout << "   Shallow (depth=3) leaves: " << reg_shallow.get_n_leaves() << "\n";
    std::cout << "   Deep (depth=20) leaves: " << reg_deep.get_n_leaves() << "\n";
    std::cout << "   Deep tree may overfit but should have lower train error\n";

    // Check train error
    double error_shallow = 0.0, error_deep = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        error_shallow += std::abs(reg_shallow.predict(X[i]) - y[i]);
        error_deep += std::abs(reg_deep.predict(X[i]) - y[i]);
    }

    std::cout << "   Shallow train error: " << error_shallow / X.size() << "\n";
    std::cout << "   Deep train error: " << error_deep / X.size() << "\n\n";
}

void test_regression_pruning_stability() {
    std::cout << "10. REGRESSION PRUNING STABILITY\n";

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for (int i = 0; i < 300; ++i) {
        double x = (static_cast<double>(rand()) / RAND_MAX * 8) - 4;
        double noise = (static_cast<double>(rand()) / RAND_MAX * 0.5) - 0.25;
        X.push_back({x});
        y.push_back(x * x + noise);
    }

    // Same data, same parameters - should get same tree size
    std::vector<int> leaf_counts;
    for (int run = 0; run < 5; ++run) {
        DecisionTreeRegressor reg(10, 10, 5, "mse", 0.02);
        reg.fit(X, y);
        leaf_counts.push_back(reg.get_n_leaves());
    }

    bool stable = true;
    for (size_t i = 1; i < leaf_counts.size(); ++i) {
        if (leaf_counts[i] != leaf_counts[0]) stable = false;
    }

    std::cout << "   Leaf counts over 5 runs: ";
    for (int count : leaf_counts) std::cout << count << " ";
    std::cout << "\n   Stable: " << (stable ? "YES" : "NO") << "\n\n";
}

int main() {
    std::cout << "==================================================\n";
    std::cout << "CLASSIFICATION TESTS (10 tests)\n";
    std::cout << "==================================================\n\n";

    Timer total_timer;

    test_classification_linear_separable();
    test_classification_multiclass();
    test_classification_pruning_effect();
    test_classification_feature_importance();
    test_classification_predict_proba();
    test_classification_entropy_vs_gini();
    test_classification_single_class();
    test_classification_small_dataset();
    test_classification_depth_limit();
    test_classification_pruning_stability();

    std::cout << "==================================================\n";
    std::cout << "REGRESSION TESTS (10 tests)\n";
    std::cout << "==================================================\n\n";

    test_regression_linear();
    test_regression_quadratic();
    test_regression_mse_vs_mae();
    test_regression_multidimensional();
    test_regression_pruning_effect();
    test_regression_constant_data();
    test_regression_interaction_terms();
    test_regression_min_samples_leaf();
    test_regression_large_depth();
    test_regression_pruning_stability();

    double total_time = total_timer.elapsed();
    std::cout << "==================================================\n";
    std::cout << "ALL 20 TESTS COMPLETED IN " << std::fixed << std::setprecision(2)
              << total_time << " SECONDS\n";
    std::cout << "==================================================\n";

    return 0;
}