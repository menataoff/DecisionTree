#include "decision_tree_regressor.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <numeric>

DecisionTreeRegressor::DecisionTreeRegressor(int max_depth,
                 int min_samples_split,
                 int min_samples_leaf,
                 const std::string& string_criterion,
                 double ccp_alpha)
        : DecisionTree<double>(max_depth, min_samples_split, min_samples_leaf, ccp_alpha) {
    if (string_criterion == "mse") {
        criterion = RegressionSplitCriterion::MSE;
    } else if (string_criterion == "mae") {
        criterion = RegressionSplitCriterion::MAE;
    } else {
        throw std::invalid_argument("Criterion must be 'mse' or 'mae'");
    }
}

double DecisionTreeRegressor::calculate_mean_value(const std::vector<double>& targets) const {
    if (targets.empty()) return 0.0;

    double result = 0.0;
    for (double target : targets) {
        result += target;
    }
    return result / targets.size();
}

double DecisionTreeRegressor::calculate_mean_value_of_squares(const std::vector<double>& targets) const {
    if (targets.empty()) return 0.0;

    double sum_sq = 0.0;
    for (double target : targets) {
        sum_sq += target * target;
    }
    return sum_sq / targets.size();
}


double DecisionTreeRegressor::calculate_variance(const std::vector<double>& targets) const {
    if (targets.empty()) return 0.0;

    return calculate_mean_value_of_squares(targets) - calculate_mean_value(targets)*calculate_mean_value(targets);
}

double DecisionTreeRegressor::calculate_median(const std::vector<double>& targets) {
    if (targets.empty()) return 0.0;

    size_t n = targets.size();
    std::vector<double> sorted = targets;
    std::sort(sorted.begin(), sorted.end());

    double median = 0.0;
    if (n % 2 == 0) {
        median = (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
    } else {
        median = sorted[n/2];
    }
    return median;
}

double DecisionTreeRegressor::calculate_mae(const std::vector<double>& targets) const {
    if (targets.empty()) return 0.0;

    size_t n = targets.size();
    double median = calculate_median(targets);

    double result = 0.0;
    for (double target : targets) {
        result += std::abs(target - median);
    }
    return result / n;
}

std::pair<std::vector<int>, std::vector<int>> DecisionTreeRegressor::split_data(
    const std::vector<DataPoint<double>>& data,
    const std::vector<int>& indices, int feature_index, double threshold) {

    std::vector<int> left_idx;
    std::vector<int> right_idx;
    left_idx.reserve(indices.size());
    right_idx.reserve(indices.size());

    for (int idx : indices) {
        if (data[idx].features[feature_index] <= threshold) {
            left_idx.push_back(idx);
        } else {
            right_idx.push_back(idx);
        }
    }

    return {std::move(left_idx), std::move(right_idx)};
}

std::vector<double> DecisionTreeRegressor::extract_targets(
    const std::vector<DataPoint<double> > &data,
    const std::vector<int> &indices) {
    std::vector<double> targets;
    targets.reserve(indices.size());
    for (const size_t idx : indices) {
        targets.push_back(data[idx].target);
    }
    return targets;
}

double DecisionTreeRegressor::calculate_node_quality(const std::vector<double> &targets) const {
    if (criterion == RegressionSplitCriterion::MSE) {
        return calculate_variance(targets);
    } else {
        return calculate_mae(targets);
    }
}

double DecisionTreeRegressor::calculate_split_quality(
    const std::vector<double> &left_targets,
    const std::vector<double> &right_targets,
    double parent_quality) const {
    int n_left = static_cast<int>(left_targets.size());
    int n_right = static_cast<int>(right_targets.size());
    if (n_left == 0 || n_right == 0) {
        return 0.0;
    }
    int n_total = n_left + n_right;

    if (criterion == RegressionSplitCriterion::MSE) {
        return parent_quality - (static_cast<double>(n_left) / n_total)*calculate_variance(left_targets) - (static_cast<double>(n_right) / n_total)*calculate_variance(right_targets);
    } else {
        return parent_quality - (static_cast<double>(n_left) / n_total)*calculate_mae(left_targets) - (static_cast<double>(n_right) / n_total)*calculate_mae(right_targets);
    }
}

SplitInfo DecisionTreeRegressor::find_best_split(const std::vector<DataPoint<double> > &data, const std::vector<int> &indices) const {
    SplitInfo best_split;

    if (indices.size() < 2) return best_split;
    size_t num_features = data[0].features.size();

    std::vector<double> parent_targets = extract_targets(data, indices);
    double parent_quality = (criterion == RegressionSplitCriterion::MSE)
                          ? calculate_variance(parent_targets)
                          : calculate_mae(parent_targets);

    for (size_t feature_idx = 0; feature_idx < num_features; ++feature_idx) {
        std::vector<int> sorted_indices = indices;


        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t i, size_t j) {
            return (data[i].features[feature_idx] < data[j].features[feature_idx]);
        });

        double min_val = data[sorted_indices.front()].features[feature_idx];
        double max_val = data[sorted_indices.back()].features[feature_idx];
        if (min_val == max_val) continue;

        double left_sums = 0.0;
        double right_sums = 0.0;
        double left_sqsums = 0.0;
        double right_sqsums = 0.0;


        if (criterion == RegressionSplitCriterion::MSE) {
            for (int idx : sorted_indices) {
                double target = data[idx].target;
                right_sums += target;
                right_sqsums += target * target;
            }
        }

        int left_total = 0;
        int right_total = static_cast<int>(sorted_indices.size());

        for (size_t i = 0; i < sorted_indices.size()-1; ++i) {
            size_t current_idx = sorted_indices[i];
            size_t next_idx = sorted_indices[i+1];
            double value = data[current_idx].target;
            //TODO: Оптимизация для MAE
            if (criterion == RegressionSplitCriterion::MSE) {
                right_sums -= value;
                right_sqsums -= value * value;
                left_sums += value;
                left_sqsums += value * value;
            }
            left_total++;
            right_total--;

            if ((data[current_idx].features[feature_idx] == data[next_idx].features[feature_idx])) {
                continue;
            }

            if (left_total < min_samples_leaf || right_total < min_samples_leaf) {
                continue;
            }

            double threshold = (data[current_idx].features[feature_idx] + data[next_idx].features[feature_idx])/2.0;

            double left_quality = 0.0;
            double right_quality = 0.0;

            if (criterion == RegressionSplitCriterion::MSE) {
                double left_mean = left_sums / left_total;
                left_quality = (left_sqsums / left_total) - (left_mean * left_mean);

                double right_mean = right_sums / right_total;
                right_quality = (right_sqsums / right_total) - (right_mean * right_mean);
            } else {
                std::vector<int> left_indices(sorted_indices.begin(), sorted_indices.begin() + i + 1);
                std::vector<int> right_indices(sorted_indices.begin() + i + 1, sorted_indices.end());
                std::vector<double> left_targets = extract_targets(data, left_indices);
                std::vector<double> right_targets = extract_targets(data, right_indices);

                left_quality = calculate_mae(left_targets);
                right_quality = calculate_mae(right_targets);
            }
            int total = left_total + right_total;
            double gain = parent_quality - (static_cast<double>(left_total) / total)*left_quality - (static_cast<double>(right_total) / total)*right_quality;

            if (gain > best_split.information_gain) {
                best_split.feature_index = static_cast<int>(feature_idx);
                best_split.threshold = threshold;
                best_split.information_gain = gain;

                best_split.split_position = i;
                best_split.sorted_indices = (sorted_indices);
            }
        }
    }
    return best_split;
}

std::unique_ptr<Node<double>> DecisionTreeRegressor::build_tree(const std::vector<DataPoint<double>>& data,
        const std::vector<int>& indices,
        int depth, int total_samples) {

    std::vector<double> targets = extract_targets(data, indices);
    double node_value;
    if (criterion == RegressionSplitCriterion::MSE) {
        node_value = calculate_mean_value(targets);  // Среднее для MSE
    } else {
        node_value = calculate_median(targets);  // Медиана для MAE
    }
    double node_quality = calculate_node_quality(targets);
    double node_error = (static_cast<double>(indices.size()) / total_samples) * node_quality;
    if (node_quality < 1e-10) {  // почти нулевая дисперсия/MAE
        return std::make_unique<RegressionLeafNode>(node_value, node_quality, indices.size(), node_error);
    }

    if (indices.size() < min_samples_split ||
    indices.empty() ||
    depth >= max_depth) {
        return std::make_unique<RegressionLeafNode>(node_value, node_quality, indices.size(), node_error);
    }

    auto best_split = find_best_split(data, indices);

    if (best_split.feature_index < 0 || best_split.information_gain <= 0.0) {
        return std::make_unique<RegressionLeafNode>(node_value, node_quality, indices.size(), node_error);
    }

    auto left_indices = best_split.get_left_indices();
    auto right_indices = best_split.get_right_indices();

    if (left_indices.empty() || right_indices.empty() || left_indices.size() < min_samples_leaf || right_indices.size() < min_samples_leaf || best_split.information_gain <= 0.0) {
        return std::make_unique<RegressionLeafNode>(node_value, node_quality, indices.size(), node_error);
    }

    int feature_index = best_split.feature_index;
    double threshold = best_split.threshold;

    if (feature_importances.empty() && total_samples > 0) {
        feature_importances.resize(data[0].features.size(), 0.0);
    }

    if (!feature_importances.empty()) {
        double weight = static_cast<double>(indices.size()) / total_samples;

        // Для feature importances всегда используем снижение дисперсии
        std::vector<double> left_targets = extract_targets(data, left_indices);
        std::vector<double> right_targets = extract_targets(data, right_indices);
        std::vector<double> parent_targets = extract_targets(data, indices);

        double parent_var = calculate_variance(parent_targets);
        double left_var = calculate_variance(left_targets);
        double right_var = calculate_variance(right_targets);

        double total_size = static_cast<double>(indices.size());
        double variance_reduction = parent_var -
                                   (static_cast<double>(left_targets.size())/total_size)*left_var -
                                   (static_cast<double>(right_targets.size())/total_size)*right_var;

        if (variance_reduction > 0.0) {
            feature_importances[feature_index] += weight * variance_reduction;
        }
    }

    auto left_child_ptr = build_tree(data, std::move(left_indices), depth + 1, total_samples);
    auto right_child_ptr = build_tree(data, std::move(right_indices), depth + 1, total_samples);

    auto left_child = std::unique_ptr<RegressionNode>(
        dynamic_cast<RegressionNode*>(left_child_ptr.release()));

    auto right_child = std::unique_ptr<RegressionNode>(
        dynamic_cast<RegressionNode*>(right_child_ptr.release()));

    if (!left_child || !right_child) {
        throw std::runtime_error("Failed to cast node to RegressionNode");
    }
    return std::make_unique<RegressionInternalNode>(
        feature_index,
        threshold,
        std::move(left_child),
        std::move(right_child),
        node_value,
        node_quality,
        indices.size(),
        node_error
    );
}

int DecisionTreeRegressor::count_subtree_leaves(Node<double>* node) const {
    if (node == nullptr) {
        return 0;
    }
    if (dynamic_cast<RegressionLeafNode*>(node) != nullptr) {
        return 1;
    }

    auto internal_node = dynamic_cast<RegressionInternalNode*>(node);

    if (internal_node != nullptr) {
        return count_subtree_leaves(internal_node->get_left_child()) + count_subtree_leaves(internal_node->get_right_child());
    }
    return 0;
}

std::pair<double, int> DecisionTreeRegressor::calculate_tree_error(Node<double>* node) const {
    if (!node) return {0.0, 0};

    if (dynamic_cast<RegressionLeafNode*>(node) != nullptr) {
        return {node->get_node_error(), 1};
    }

    if (dynamic_cast<RegressionInternalNode*>(node) != nullptr) {
        RegressionInternalNode* internal_node = dynamic_cast<RegressionInternalNode*>(node);

        auto left_error = calculate_tree_error(internal_node->get_left_child());
        auto right_error = calculate_tree_error(internal_node->get_right_child());

        return {left_error.first + right_error.first, left_error.second + right_error.second};
    }
    return {0.0, 0};
}

std::pair<Node<double>*, double> DecisionTreeRegressor::find_global_weakest_link(
    Node<double>* node,
    Node<double>* current_best_node,
    double current_min_alpha) const {


    if (node == nullptr || dynamic_cast<RegressionLeafNode*>(node) != nullptr) {
        return {current_best_node, current_min_alpha};
    }

    if (dynamic_cast<RegressionInternalNode*>(node) != nullptr) {
        RegressionInternalNode* internal_node = dynamic_cast<RegressionInternalNode*>(node);

        double R_t = node->get_node_error();
        auto [R_Tt, T_t] = calculate_tree_error(node);

        if (T_t <= 1) {
        } else {
            double alpha = (R_t - R_Tt) / (T_t - 1);

            if (alpha >= 0 && alpha < current_min_alpha) {
                current_min_alpha = alpha;
                current_best_node = node;
            }
        }

        auto [left_candidate, left_alpha] = find_global_weakest_link(
            internal_node->get_left_child(), current_best_node, current_min_alpha);

        auto [right_candidate, right_alpha] = find_global_weakest_link(
            internal_node->get_right_child(), current_best_node, current_min_alpha);

        if (left_alpha < current_min_alpha && left_alpha <= right_alpha) {
            return {left_candidate, left_alpha};
        } else if (right_alpha < current_min_alpha && right_alpha <= left_alpha) {
            return {right_candidate, right_alpha};
        } else {
            return {current_best_node, current_min_alpha};
        }
    }

    return {current_best_node, current_min_alpha};
}

bool DecisionTreeRegressor::is_leaf_node(Node<double>* node) const {
    return (dynamic_cast<RegressionLeafNode*>(node) != nullptr);
}

void DecisionTreeRegressor::prune_node_to_leaf(RegressionInternalNode* node_to_prune,
        Node<double>* parent,
        bool is_left_child) {

    if (!node_to_prune) return;

    int sample_count = node_to_prune->get_sample_count();
    double node_error = node_to_prune->get_node_error();
    double median_or_mean_value = node_to_prune->get_mean_value();
    double variance = node_to_prune->get_variance();
    //RegressionLeafNode(double mean_value, double variance, int sample_count, double node_error)
    auto new_leaf = std::make_unique<RegressionLeafNode>(median_or_mean_value, variance, sample_count, node_error);

    if (parent != nullptr) {
        RegressionInternalNode* parent_internal = static_cast<RegressionInternalNode*>(parent);

        if (is_left_child) {
            parent_internal->set_left_child(std::move(new_leaf));
        } else {
            parent_internal->set_right_child(std::move(new_leaf));
        }
    } else {
        root = std::move(new_leaf);
    }
}

std::pair<Node<double>*, bool> DecisionTreeRegressor::find_parent(Node<double>* root, Node<double>* target) const {
    if (root == nullptr || target == nullptr) return {nullptr, false};

    if (dynamic_cast<RegressionInternalNode*>(root) != nullptr) {
        RegressionInternalNode* internal = static_cast<RegressionInternalNode*>(root);

        if (internal->get_left_child() == target) return {internal, true};
        if (internal->get_right_child() == target) return {internal, false};

        auto left_result = find_parent(internal->get_left_child(), target);
        if (left_result.first != nullptr) return left_result;

        auto right_result = find_parent(internal->get_right_child(), target);
        if (right_result.first != nullptr) return right_result;
    }

    return {nullptr, false};
}

void DecisionTreeRegressor::cost_complexity_prune() {
    int iteration = 0;
    const int MAX_ITERATIONS = 1024;

    while (!is_leaf_node(root.get()) && iteration < MAX_ITERATIONS) {
        iteration++;
        auto [weakest_node, alpha] = find_global_weakest_link(root.get());

        if (weakest_node == nullptr || alpha == std::numeric_limits<double>::infinity() || alpha > ccp_alpha) {
            break;
        }

        auto [parent, is_left] = find_parent(root.get(), weakest_node);

        prune_node_to_leaf(static_cast<RegressionInternalNode*>(weakest_node), parent, is_left);
    }
}

// void DecisionTreeRegressor::cost_complexity_prune() {
//     std::cout << "DEBUG: Starting pruning, root=" << root.get() << std::endl;
//
//     int iteration = 0;
//     const int MAX_ITERATIONS = 1024;
//
//     while (!is_leaf_node(root.get()) && iteration < MAX_ITERATIONS) {
//         iteration++;
//         std::cout << "DEBUG: Iteration " << iteration << std::endl;
//
//         auto [weakest_node, alpha] = find_global_weakest_link(root.get());
//         std::cout << "DEBUG: Found weakest node=" << weakest_node
//                   << ", alpha=" << alpha << std::endl;
//
//         if (weakest_node == nullptr ||
//             alpha == std::numeric_limits<double>::infinity() ||
//             alpha > ccp_alpha) {
//             std::cout << "DEBUG: Stopping condition met" << std::endl;
//             break;
//             }
//
//         auto [parent, is_left] = find_parent(root.get(), weakest_node);
//         std::cout << "DEBUG: Parent=" << parent << ", is_left=" << is_left << std::endl;
//
//         if (parent == nullptr && weakest_node != root.get()) {
//             std::cout << "ERROR: Found node without parent that is not root!" << std::endl;
//             break;
//         }
//
//         std::cout << "DEBUG: Pruning node..." << std::endl;
//         prune_node_to_leaf(static_cast<RegressionInternalNode*>(weakest_node),
//                           parent, is_left);
//         std::cout << "DEBUG: After pruning, root=" << root.get() << std::endl;
//     }
//     std::cout << "DEBUG: Pruning finished after " << iteration << " iterations" << std::endl;
// }

void DecisionTreeRegressor::fit(const std::vector<DataPoint<double>>& data) {

    if (data.empty()) {
        root = nullptr;
        feature_importances.clear();
        return;
    }

    std::vector<int> indices(data.size());
    for (int i = 0; i < data.size(); ++i) {
        indices[i] = i;
    }

    feature_importances = std::vector<double>(data[0].features.size(), 0.0);
    root = build_tree(data, indices, 0, data.size());

    double summary_importance = 0.0;
    for (const auto& importance : feature_importances) {
        summary_importance += importance;
    }

    if (summary_importance > 0.0) {
        for (size_t i = 0; i < feature_importances.size(); ++i) {
            feature_importances[i] /= summary_importance;
        }
    }

    if (ccp_alpha > 0.0) {
        cost_complexity_prune();
    }
}

void DecisionTreeRegressor::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    if (X.size() != y.size() || X.empty()) {
        throw std::invalid_argument("X and y must have same size and be non-empty");
    }

    std::vector<DataPoint<double>> data;
    data.reserve(X.size());

    for (size_t i = 0; i < X.size(); ++i) {
        data.emplace_back(X[i], y[i]);
    }

    fit(data);
}

int DecisionTreeRegressor::get_n_leaves() const {
    return count_subtree_leaves(root.get());
}

void DecisionTreeRegressor::set_ccp_alpha(double new_alpha) {
    ccp_alpha = new_alpha;
}

//TODO: перенести fit'ы, get_n_leaves, set_ccp_alpha и другое в общий класс

