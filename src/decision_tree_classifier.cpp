#include "../include/decision_tree_classifier.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>


DecisionTreeClassifier::DecisionTreeClassifier(int max_depth,
                 int min_samples_split,
                 int min_samples_leaf,
                 const std::string& string_criterion,
                 double ccp_alpha)
        : DecisionTree<int>(max_depth, min_samples_split, min_samples_leaf, ccp_alpha) {
    if (string_criterion == "entropy") {
        criterion = ClassificationSplitCriterion::ENTROPY;
    } else if (string_criterion == "gini") {
        criterion = ClassificationSplitCriterion::GINI;
    } else {
        throw std::invalid_argument("Criterion must be 'entropy' or 'gini'");
    }
}

double DecisionTreeClassifier::calculate_impurity(
    const std::unordered_map<int, int>& class_counts, int total) const
{
    if (total == 0) return 0.0;

    if (criterion == ClassificationSplitCriterion::ENTROPY) {
        double entropy = 0.0;
        for (const auto& [label, count] : class_counts) {
            double p = static_cast<double>(count) / total;
            if (p > 0.0) {
                entropy -= p * std::log2(p);
            }
        }
        return entropy;
    } else {
        double gini = 1.0;
        for (const auto& [label, count] : class_counts) {
            double p = static_cast<double>(count) / total;
            gini -= p*p;
        }
        return gini;
    }
}

std::unordered_map<int, int> DecisionTreeClassifier::merge_to_parent(
        const std::unordered_map<int, int>& left_counts,
        const std::unordered_map<int, int>& right_counts) const {
    std::unordered_map<int, int> parent_counts;
    for (const auto& [label, count] : left_counts) {
        parent_counts[label] += count;
    }
    for (const auto& [label, count] : right_counts) {
        parent_counts[label] += count;
    }
    return parent_counts;
}

std::pair<std::vector<int>, std::vector<int>> DecisionTreeClassifier::split_data(
    const std::vector<DataPoint<int>>& data,
    const std::vector<int>& indices, int feature_index, double threshold) {

    std::vector<int> left_idx;
    std::vector<int> right_idx;

    for (int idx : indices) {
        if (data[idx].features[feature_index] <= threshold) {
            left_idx.push_back(idx);
        } else {
            right_idx.push_back(idx);
        }
    }

    return {std::move(left_idx), std::move(right_idx)};
}

std::unordered_map<int, int> DecisionTreeClassifier::calculate_class_counts(
        const std::vector<DataPoint<int>>& data,
        const std::vector<int>& indices) const {

    std::unordered_map<int, int> counts;
    for (int idx : indices) {
        counts[data[idx].target]++;
    }
    return counts;
}

std::unordered_map<int, double> DecisionTreeClassifier::calculate_probabilities(
    const std::vector<DataPoint<int>>& data, const std::vector<int>& indices) const {

    if (indices.empty()) return {};

    int total = static_cast<int>(indices.size());
    auto class_counts = calculate_class_counts(data, indices);

    std::unordered_map<int, double> class_probabilities;
    for (const auto& [class_label, count] : class_counts) {
        class_probabilities[class_label] = (static_cast<double>(count)/total);
    }
    return class_probabilities;
}

int DecisionTreeClassifier::get_majority_class_in_node(const std::vector<DataPoint<int>>& data, const std::vector<int>& indices) const {
    if (indices.empty()) return 0;

    std::unordered_map<int, int> class_counts;

    for (int idx : indices) {
        int label = data[idx].target;
        class_counts[label]++;
    }

    int majority_class = 0;
    int max_count = 0;
    for (const auto& [class_label, count] : class_counts) {
        if (count > max_count) {
            max_count = count;
            majority_class = class_label;
        }
    }
    return majority_class;
}

bool DecisionTreeClassifier::all_same_class(const std::vector<DataPoint<int>>& data, const std::vector<int>& indices) const {
    if (indices.empty()) return true;
    int first_label = data[indices[0]].target;
    for (int idx : indices) {
        if (data[idx].target != first_label) {
            return false;
        }
    }
    return true;
}

SplitInfo DecisionTreeClassifier::find_best_split(const std::vector<DataPoint<int>>& data, const std::vector<int>& indices) const {
    SplitInfo best_split;

    if (indices.size() < 2) return best_split;

    int num_features = static_cast<int>(data[0].features.size());

    for (size_t feature_idx = 0; feature_idx < num_features; ++feature_idx) {
        std::vector<int> sorted_indices = indices;


        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t i, size_t j) {
            return (data[i].features[feature_idx] < data[j].features[feature_idx]);
        });


        std::unordered_map<int, int> left_counts;
        std::unordered_map<int, int> right_counts;

        for (const int idx : sorted_indices) {
            right_counts[data[idx].target]++;
        }

        int left_total = 0;
        int right_total = sorted_indices.size();

        for (size_t i = 0; i < sorted_indices.size()-1; ++i) {
            size_t current_idx = sorted_indices[i];
            size_t next_idx = sorted_indices[i+1];
            int label = data[current_idx].target;

            left_counts[label]++;
            right_counts[label]--;
            left_total++;
            right_total--;

            if (right_counts[label] == 0) {
                right_counts.erase(label);
            }

            if (data[current_idx].features[feature_idx] == data[next_idx].features[feature_idx]) {
                continue;
            }

            double threshold = (data[current_idx].features[feature_idx] + data[next_idx].features[feature_idx])/2.0;

            double left_impurity = calculate_impurity(left_counts, left_total);
            double right_impurity = calculate_impurity(right_counts, right_total);
            auto parent_counts = merge_to_parent(left_counts, right_counts);
            int total = left_total + right_total;

            double gain = calculate_impurity(parent_counts, total) - ((static_cast<double>(left_total)/total)*left_impurity + (static_cast<double>(right_total)/total)*right_impurity);

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

std::unique_ptr<Node<int>> DecisionTreeClassifier::build_tree(const std::vector<DataPoint<int>>& data,
        const std::vector<int>& indices,
        int depth, int total_samples) {

    auto probabilities = calculate_probabilities(data, indices);
    double impurity = calculate_impurity(calculate_class_counts(data, indices), indices.size());
    double node_error = (static_cast<double>(indices.size()) / total_samples) * impurity;
    int majority_class = get_majority_class_in_node(data, indices);

    if (indices.size() < min_samples_split ||
    indices.empty() ||
    all_same_class(data, indices) ||
    depth >= max_depth) {

        return std::make_unique<ClassificationLeafNode>(probabilities, indices.size(), node_error);
    }

    auto best_split = find_best_split(data, indices);
    auto left_indices = best_split.get_left_indices();    // создаем один раз
    auto right_indices = best_split.get_right_indices();

    if (left_indices.size() < min_samples_leaf || right_indices.size() < min_samples_leaf || best_split.information_gain <= 0.0) {
        return std::make_unique<ClassificationLeafNode>(probabilities, indices.size(), node_error);
    }

    int feature_index = best_split.feature_index;
    double threshold = best_split.threshold;

    if (feature_importances.empty() && total_samples > 0) {
        feature_importances.resize(data[0].features.size(), 0.0);
    }

    if  (!feature_importances.empty()) {
        double weight = static_cast<double>(indices.size()) / total_samples;
        feature_importances[feature_index] += weight * best_split.information_gain;
    }

    auto left_child_ptr = build_tree(data, std::move(left_indices), depth + 1, total_samples);
    auto right_child_ptr = build_tree(data, std::move(right_indices), depth + 1, total_samples);

    auto left_child = std::unique_ptr<ClassificationNode>(
        dynamic_cast<ClassificationNode*>(left_child_ptr.release()));

    auto right_child = std::unique_ptr<ClassificationNode>(
        dynamic_cast<ClassificationNode*>(right_child_ptr.release()));

    if (!left_child || !right_child) {
        throw std::runtime_error("Failed to cast node to ClassificationNode");
    }

    return std::make_unique<ClassificationInternalNode>(
        feature_index,
        threshold,
        std::move(left_child),
        std::move(right_child),
        majority_class,
        probabilities,
        indices.size(),
        node_error
    );
}

std::pair<Node<int>*, bool> DecisionTreeClassifier::find_parent(Node<int>* root, Node<int>* target) const {
    if (root == nullptr || target == nullptr) return {nullptr, false};

    if (dynamic_cast<ClassificationInternalNode*>(root) != nullptr) {
        ClassificationInternalNode* internal = static_cast<ClassificationInternalNode*>(root);

        if (internal->get_left_child() == target) return {internal, true};
        if (internal->get_right_child() == target) return {internal, false};

        auto left_result = find_parent(internal->get_left_child(), target);
        if (left_result.first != nullptr) return left_result;

        auto right_result = find_parent(internal->get_right_child(), target);
        if (right_result.first != nullptr) return right_result;
    }

    return {nullptr, false};
}

int DecisionTreeClassifier::count_subtree_leaves(Node<int>* node) const {
    if (node == nullptr) {
        return 0;
    }
    if (dynamic_cast<ClassificationLeafNode*>(node) != nullptr) {
        return 1;
    }

    auto internal_node = dynamic_cast<ClassificationInternalNode*>(node);

    if (internal_node != nullptr) {
        return count_subtree_leaves(internal_node->get_left_child()) + count_subtree_leaves(internal_node->get_right_child());
    }
    return 0;
}

void DecisionTreeClassifier::prune_node_to_leaf(ClassificationInternalNode* node_to_prune,
        Node<int>* parent,
        bool is_left_child) {

    if (!node_to_prune) return;

    int sample_count = node_to_prune->get_sample_count();
    double node_error = node_to_prune->get_node_error();
    auto class_probabilities = node_to_prune->get_class_probabilities();
    auto new_leaf = std::make_unique<ClassificationLeafNode>(class_probabilities, sample_count, node_error);

    if (parent != nullptr) {
        ClassificationInternalNode* parent_internal = static_cast<ClassificationInternalNode*>(parent);

        if (is_left_child) {
            parent_internal->set_left_child(std::move(new_leaf));
        } else {
            parent_internal->set_right_child(std::move(new_leaf));
        }
    } else {
        root = std::move(new_leaf);
    }
}

bool DecisionTreeClassifier::is_leaf_node(Node<int>* node) const {
    return (dynamic_cast<ClassificationLeafNode*>(node) != nullptr);
}

std::pair<Node<int>*, double> DecisionTreeClassifier::find_global_weakest_link(
    Node<int>* node,
    Node<int>* current_best_node,
    double current_min_alpha) const {

    if (node == nullptr || dynamic_cast<ClassificationLeafNode*>(node) != nullptr) {
        return {current_best_node, current_min_alpha};
    }

    if (dynamic_cast<ClassificationInternalNode*>(node) != nullptr) {
        ClassificationInternalNode* internal_node = dynamic_cast<ClassificationInternalNode*>(node);

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

std::pair<double, int> DecisionTreeClassifier::calculate_tree_error(Node<int>* node) const {
    if (dynamic_cast<ClassificationLeafNode*>(node) != nullptr) {
        return {node->get_node_error(), 1};
    }

    if (dynamic_cast<ClassificationInternalNode*>(node) != nullptr) {
        ClassificationInternalNode* internal_node = dynamic_cast<ClassificationInternalNode*>(node);

        auto left_error = calculate_tree_error(internal_node->get_left_child());
        auto right_error = calculate_tree_error(internal_node->get_right_child());

        return {left_error.first + right_error.first, left_error.second + right_error.second};
    }
    return {0.0, 0};
}

void DecisionTreeClassifier::cost_complexity_prune() {
    int iteration = 0;
    const int MAX_ITERATIONS = 1024;

    while (!is_leaf_node(root.get()) && iteration < MAX_ITERATIONS) {
        iteration++;
        auto [weakest_node, alpha] = find_global_weakest_link(root.get());

        if (weakest_node == nullptr || alpha == std::numeric_limits<double>::infinity() || alpha > ccp_alpha) {
            break;
        }

        auto [parent, is_left] = find_parent(root.get(), weakest_node);

        prune_node_to_leaf(static_cast<ClassificationInternalNode*>(weakest_node), parent, is_left);
    }
}


void DecisionTreeClassifier::fit(const std::vector<DataPoint<int>>& data) {

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

void DecisionTreeClassifier::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    if (X.size() != y.size() || X.empty()) {
        std::cout << "Wrong size of data\n";
    }

    std::vector<DataPoint<int>> data;
    data.reserve(X.size());

    for (size_t i = 0; i < X.size(); ++i) {
        data.emplace_back(X[i], y[i]);
    }

    fit(data);
}

int DecisionTreeClassifier::get_n_leaves() const {
    return count_subtree_leaves(root.get());
}

void DecisionTreeClassifier::set_ccp_alpha(double new_alpha) {
    ccp_alpha = new_alpha;
}




