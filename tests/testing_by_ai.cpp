#include "decision_tree_regressor.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <numeric>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

std::mt19937& get_rng() {
    static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    return rng;
}

double random_uniform(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(get_rng());
}

void print_separator() {
    std::cout << "------------------------------------------------------------\n";
}

double compute_rmse(const std::vector<double>& pred, const std::vector<double>& true_vals) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < pred.size(); ++i) {
        double diff = pred[i] - true_vals[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / pred.size());
}

// ======================= BENCHMARK TESTS =======================

void benchmark_large_dataset() {
    std::cout << "[BENCHMARK 1] Large dataset - 5000 samples, depth=15\n";

    const size_t n_samples = 5000;
    const size_t n_test = 1000;
    const int max_depth = 15;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    // y = 2*x1 + 3*x2 - 1.5*x3 + x4*0.5 + noise
    for (size_t i = 0; i < n_samples; ++i) {
        double x1 = random_uniform(-3.0, 3.0);
        double x2 = random_uniform(-3.0, 3.0);
        double x3 = random_uniform(-3.0, 3.0);
        double x4 = random_uniform(-3.0, 3.0);
        double noise = random_uniform(-0.2, 0.2);
        double y = 2.0*x1 + 3.0*x2 - 1.5*x3 + 0.5*x4 + noise;
        X_train.push_back({x1, x2, x3, x4});
        y_train.push_back(y);
    }

    for (size_t i = 0; i < n_test; ++i) {
        double x1 = random_uniform(-3.0, 3.0);
        double x2 = random_uniform(-3.0, 3.0);
        double x3 = random_uniform(-3.0, 3.0);
        double x4 = random_uniform(-3.0, 3.0);
        double y = 2.0*x1 + 3.0*x2 - 1.5*x3 + 0.5*x4;
        X_test.push_back({x1, x2, x3, x4});
        y_test.push_back(y);
    }

    DecisionTreeRegressor reg(max_depth, 5, 2, "mse", 0.0);

    auto start = Clock::now();
    reg.fit(X_train, y_train);
    auto end = Clock::now();
    double fit_time = Duration(end - start).count();

    std::vector<double> predictions;
    for (const auto& x : X_test) {
        predictions.push_back(reg.predict(x));
    }

    double rmse = compute_rmse(predictions, y_test);

    std::cout << "  Samples: " << n_samples << "\n";
    std::cout << "  Features: 4\n";
    std::cout << "  Max depth: " << max_depth << "\n";
    std::cout << "  Fit time: " << std::fixed << std::setprecision(2) << fit_time << " ms\n";
    std::cout << "  Leaves: " << reg.get_n_leaves() << "\n";
    std::cout << "  RMSE: " << std::fixed << std::setprecision(4) << rmse << "\n";
    std::cout << "  Time per sample: " << std::fixed << std::setprecision(3) << (fit_time / n_samples) << " ms\n\n";
}

void benchmark_deep_tree() {
    std::cout << "[BENCHMARK 2] Deep tree - 2000 samples, depth=25\n";

    const size_t n_samples = 2000;
    const size_t n_test = 500;
    const int max_depth = 25;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    // Sinusoidal function with 3 features
    for (size_t i = 0; i < n_samples; ++i) {
        double x1 = random_uniform(-5.0, 5.0);
        double x2 = random_uniform(-5.0, 5.0);
        double x3 = random_uniform(-5.0, 5.0);
        double noise = random_uniform(-0.15, 0.15);
        double y = std::sin(x1) + 0.5 * std::cos(x2) + 0.3 * std::sin(x3) + noise;
        X_train.push_back({x1, x2, x3});
        y_train.push_back(y);
    }

    for (size_t i = 0; i < n_test; ++i) {
        double x1 = random_uniform(-5.0, 5.0);
        double x2 = random_uniform(-5.0, 5.0);
        double x3 = random_uniform(-5.0, 5.0);
        double y = std::sin(x1) + 0.5 * std::cos(x2) + 0.3 * std::sin(x3);
        X_test.push_back({x1, x2, x3});
        y_test.push_back(y);
    }

    DecisionTreeRegressor reg(max_depth, 2, 1, "mse", 0.0);

    auto start = Clock::now();
    reg.fit(X_train, y_train);
    auto end = Clock::now();
    double fit_time = Duration(end - start).count();

    std::vector<double> predictions;
    for (const auto& x : X_test) {
        predictions.push_back(reg.predict(x));
    }

    double rmse = compute_rmse(predictions, y_test);

    std::cout << "  Samples: " << n_samples << "\n";
    std::cout << "  Features: 3\n";
    std::cout << "  Max depth: " << max_depth << "\n";
    std::cout << "  Fit time: " << std::fixed << std::setprecision(2) << fit_time << " ms\n";
    std::cout << "  Leaves: " << reg.get_n_leaves() << "\n";
    std::cout << "  RMSE: " << std::fixed << std::setprecision(4) << rmse << "\n";
    std::cout << "  Time per sample: " << std::fixed << std::setprecision(3) << (fit_time / n_samples) << " ms\n\n";
}

void benchmark_with_pruning() {
    std::cout << "[BENCHMARK 3] With pruning - 3000 samples, depth=20, ccp_alpha=0.01\n";

    const size_t n_samples = 3000;
    const int max_depth = 20;
    const double ccp_alpha = 0.01;

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    // Complex non-linear function
    for (size_t i = 0; i < n_samples; ++i) {
        double x1 = random_uniform(-4.0, 4.0);
        double x2 = random_uniform(-4.0, 4.0);
        double x3 = random_uniform(-4.0, 4.0);
        double noise = random_uniform(-0.25, 0.25);
        double y_val = x1 * x2 + 2.0 * std::sin(x3) + noise;
        X.push_back({x1, x2, x3});
        y.push_back(y_val);
    }

    DecisionTreeRegressor reg_no_prune(max_depth, 2, 1, "mse", 0.0);
    DecisionTreeRegressor reg_prune(max_depth, 2, 1, "mse", ccp_alpha);

    auto start = Clock::now();
    reg_no_prune.fit(X, y);
    auto end = Clock::now();
    double time_no_prune = Duration(end - start).count();

    start = Clock::now();
    reg_prune.fit(X, y);
    end = Clock::now();
    double time_prune = Duration(end - start).count();

    std::cout << "  Samples: " << n_samples << "\n";
    std::cout << "  Max depth: " << max_depth << "\n";
    std::cout << "  ccp_alpha: " << ccp_alpha << "\n";
    std::cout << "  Without pruning: " << std::fixed << std::setprecision(2) << time_no_prune
              << " ms, leaves=" << reg_no_prune.get_n_leaves() << "\n";
    std::cout << "  With pruning: " << std::fixed << std::setprecision(2) << time_prune
              << " ms, leaves=" << reg_prune.get_n_leaves() << "\n";
    std::cout << "  Pruning overhead: " << std::fixed << std::setprecision(2)
              << (time_prune - time_no_prune) << " ms\n\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "==================================================\n";
    std::cout << "REGRESSION BENCHMARK SUITE\n";
    std::cout << "==================================================\n\n";

    benchmark_large_dataset();
    benchmark_deep_tree();
    benchmark_with_pruning();

    std::cout << "==================================================\n";
    std::cout << "BENCHMARK COMPLETED\n";
    std::cout << "==================================================\n";

    return 0;
}