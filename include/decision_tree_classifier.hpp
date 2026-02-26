#pragma once

#include "decision_tree.hpp"
#include <ranges>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <limits>

enum class ClassificationSplitCriterion { ENTROPY, GINI };

class ClassificationNode : public Node<int> {
protected:
    int majority_class;
    std::unordered_map<int, double> class_probabilities;
public:
    ClassificationNode(const std::unordered_map<int, double>& class_probabilities,
                      int sample_count, double node_error)
        : Node<int>(node_error, sample_count),
          class_probabilities(class_probabilities)
    {
        // ТВОЯ логика нахождения majority_class
        double max_prob = 0.0;
        for (const auto& [class_label, prob] : class_probabilities) {
            if (prob > max_prob) {
                majority_class = class_label;
                max_prob = prob;
            }
        }
    }

    ClassificationNode(int majority_class,
                      const std::unordered_map<int, double>& class_probabilities,
                      int sample_count, double node_error)
        : Node<int>(node_error, sample_count),
          majority_class(majority_class),
          class_probabilities(class_probabilities) {}

    virtual ~ClassificationNode() = default;

    virtual std::unordered_map<int, double> predict_proba(
        const std::vector<double>& features) const = 0;

    // Геттеры
    int get_majority_class() const { return majority_class; }
    const std::unordered_map<int, double>& get_class_probabilities() const {
        return class_probabilities;
    }

    // Базовый predict (можно переопределить)
    int predict(const std::vector<double>& features) const override {
        return majority_class;
    }
};

class ClassificationLeafNode : public ClassificationNode {
public:
    ClassificationLeafNode(const std::unordered_map<int, double>& class_probabilities,
                          int sample_count, double node_error)
        : ClassificationNode(class_probabilities, sample_count, node_error) {}

    std::unordered_map<int, double> predict_proba(
        const std::vector<double>& features) const override {
        return class_probabilities;
    }
};

class ClassificationInternalNode : public ClassificationNode {
private:
    int feature_index;
    double threshold;
    std::unique_ptr<ClassificationNode> left_child;
    std::unique_ptr<ClassificationNode> right_child;
public:
    ClassificationInternalNode(int feature_index, double threshold,
        std::unique_ptr<ClassificationNode> left_child, std::unique_ptr<ClassificationNode> right_child,
        int majority_class, const std::unordered_map<int, double>& class_probabilities,
        int sample_count, double node_error) :
    ClassificationNode(majority_class, class_probabilities, sample_count, node_error),
    feature_index(feature_index), threshold(threshold),
    left_child(std::move(left_child)), right_child(std::move(right_child)) {}

    void set_left_child(std::unique_ptr<ClassificationNode> new_child) {
        left_child = std::move(new_child);
    }

    void set_right_child(std::unique_ptr<ClassificationNode> new_child) {
        right_child = std::move(new_child);
    }

    ClassificationNode* get_left_child() const { return left_child.get(); }
    ClassificationNode* get_right_child() const { return right_child.get(); }
    int get_feature_index() const {return feature_index;}
    double get_threshold() const {return threshold;}

    int predict(const std::vector<double>& features) const override {
        if (features[feature_index] <= threshold) {
            return left_child->predict(features);
        } else {
            return right_child->predict(features);
        }
    }

    std::unordered_map<int, double> predict_proba(const std::vector<double>& features) const override {
        if (features[feature_index] <= threshold) {
            return left_child->predict_proba(features);
        } else {
            return right_child->predict_proba(features);
        }
    }
};

class DecisionTreeClassifier : public DecisionTree<int> {
private:
    ClassificationSplitCriterion criterion;

    double calculate_impurity(const std::unordered_map<int, int>& class_counts, int total) const;

    std::unordered_map<int, int> merge_to_parent(const std::unordered_map<int, int>& left_counts,
        const std::unordered_map<int, int>& right_counts) const;

    std::unordered_map<int, int> calculate_class_counts(
        const std::vector<DataPoint<int>>& data,
        const std::vector<int>& indices) const;

    SplitInfo find_best_split(const std::vector<DataPoint<int>>& data, const std::vector<int>& indices) const;

    static std::pair<std::vector<int>, std::vector<int>> split_data(const std::vector<DataPoint<int>>& data,
            const std::vector<int>& indices, int feature_index, double threshold);

    std::unordered_map<int, double> calculate_probabilities(const std::vector<DataPoint<int>>& data,
        const std::vector<int>& indices) const;

    bool all_same_class(const std::vector<DataPoint<int>>& data, const std::vector<int>& indices) const;

    int get_majority_class_in_node(const std::vector<DataPoint<int>>& data, const std::vector<int>& indices) const;

    std::pair<double, int> calculate_tree_error(Node<int>* node) const;

    std::pair<Node<int>*, double> find_global_weakest_link(
    Node<int>* node,
    Node<int>* current_best_node = nullptr,
    double current_min_alpha = std::numeric_limits<double>::infinity()) const;

    bool is_leaf_node(Node<int>* node) const;

    void prune_node_to_leaf(ClassificationInternalNode* node_to_prune,
        Node<int>* parent,
        bool is_left_child);

    int count_subtree_leaves(Node<int>* node) const;

    std::pair<Node<int>*, bool> find_parent(Node<int>* root, Node<int>* target) const;

    std::unique_ptr<Node<int>> build_tree(
        const std::vector<DataPoint<int>>& data,
        const std::vector<int>& indices,
        int depth, int total_samples);

    void cost_complexity_prune();
public:
    DecisionTreeClassifier(int max_depth = 32,
                          int min_samples_split = 5,
                          int min_samples_leaf = 2,
                          const std::string& string_criterion = "entropy",
                          double ccp_alpha = 0.0);

    void fit(const std::vector<DataPoint<int>>& data);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) override;
    int get_n_leaves() const;
    void set_ccp_alpha(double new_alpha);
};