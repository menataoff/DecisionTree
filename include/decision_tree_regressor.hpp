#pragma once

#include <algorithm>

#include "decision_tree.hpp"
#include <vector>
#include <limits>

enum class RegressionSplitCriterion { MSE, MAE };

class RegressionNode : public Node<double> {
protected:
    double mean_value;
    double variance;
public:
    RegressionNode(double node_error, int sample_count, double mean_value, double variance = 0.0) :
    Node<double>(node_error, sample_count), mean_value(mean_value), variance(variance) {}

    ~RegressionNode() override = default;

    [[nodiscard("Should be used to get mean value")]] double get_mean_value() const { return mean_value; }
    [[nodiscard("Should be used to get variance")]] double get_variance() const { return variance; }

    [[nodiscard("Should be used to predict value")]]
    double predict(const std::vector<double>& features) const override {
        return mean_value;
    };
};

class RegressionLeafNode : public RegressionNode {
public:
    RegressionLeafNode(double node_error, int sample_count, double mean_value, double variance) :
    RegressionNode(node_error, sample_count, mean_value, variance) {}
};

class RegressionInternalNode : public RegressionNode {
private:
    int feature_index;
    double threshold;
    std::unique_ptr<RegressionNode> left_child;
    std::unique_ptr<RegressionNode> right_child;
public:
    RegressionInternalNode(double node_error, int sample_count,
        double mean_value, double variance,
        int feature_index, double threshold,
        std::unique_ptr<RegressionNode> left_child, std::unique_ptr<RegressionNode> right_child) :
    RegressionNode(node_error, sample_count, mean_value, variance),
    feature_index(feature_index), threshold(threshold),
    left_child(std::move(left_child)), right_child(std::move(right_child)) {}

    void set_left_child(std::unique_ptr<RegressionNode> new_child) {
        left_child = std::move(new_child);
    }

    void set_right_child(std::unique_ptr<RegressionNode> new_child) {
        right_child = std::move(new_child);
    }

    [[nodiscard("Should be used to get left child")]]
    RegressionNode* get_left_child() const { return left_child.get(); }

    [[nodiscard("Should be used to get right child")]]
    RegressionNode* get_right_child() const { return right_child.get(); }

    [[nodiscard("Should be used to get feature index")]]
    int get_feature_index() const { return feature_index; }

    [[nodiscard("Should be used to get threshold")]]
    double get_threshold() const { return threshold; }

    [[nodiscard("Should be used to predict")]]
    double predict(const std::vector<double>& features) const override {
        if (features[feature_index] <= threshold) {
            return left_child->predict(features);
        } else {
            return right_child->predict(features);
        }
    }
};

class DecisionTreeRegressor : public DecisionTree<double> {
private:
    RegressionSplitCriterion criterion;

    [[nodiscard("Should be used to calculate mean value")]]
    static double calculate_mean_value(const std::vector<double>& targets);

    [[nodiscard("Should be used to calculate mean for squares")]]
    static double calculate_mean_value_of_squares(const std::vector<double>& targets);

    [[nodiscard("Should be used to calculate variance")]]
    static double calculate_variance(const std::vector<double>& targets);

    [[nodiscard("Should be used to calculate median")]]
    static double calculate_median(const std::vector<double>& targets);

    [[nodiscard("Should be used to get threshold")]]
    static double calculate_mae(const std::vector<double>& targets);

    [[nodiscard("Should be used to find best split")]]
    SplitInfo find_best_split(const std::vector<DataPoint<double>>& data, const std::vector<size_t>& indices) const;

    static std::pair<std::vector<size_t>, std::vector<size_t>> split_data(
        const std::vector<DataPoint<double>>& data,
        const std::vector<size_t>& indices,
        int feature_index,
        double threshold);

    static std::vector<double> extract_targets(
        const std::vector<DataPoint<double>>& data,
        const std::vector<size_t>& indices);

    [[nodiscard("Should be used to calculate split quality")]]
    double calculate_split_quality(
       const std::vector<double>& left_targets,
       const std::vector<double>& right_targets,
       double parent_quality) const;

    [[nodiscard("Should be used to calculate node quality")]]
    double calculate_node_quality(const std::vector<double>& targets) const;

    static std::pair<double, int> calculate_tree_error(Node<double>* node);

    static std::pair<Node<double>*, double> find_global_weakest_link(
    Node<double>* node,
    Node<double>* current_best_node = nullptr,
    double current_min_alpha = std::numeric_limits<double>::infinity());

    static bool is_leaf_node(Node<double>* node);

    void prune_node_to_leaf(const RegressionInternalNode* node_to_prune,
        Node<double>* parent,
        bool is_left_child);

    static int count_subtree_leaves(Node<double>* node);

    static std::pair<Node<double>*, bool> find_parent(Node<double>* root, Node<double>* target);

    std::unique_ptr<Node<double>> build_tree(
        const std::vector<DataPoint<double>>& data,
        const std::vector<size_t>& indices,
        int depth, size_t total_samples);

    void cost_complexity_prune();
public:
    explicit DecisionTreeRegressor(int max_depth = 32,
                          int min_samples_split = 5,
                          int min_samples_leaf = 2,
                          const std::string& string_criterion = "mse",
                          double ccp_alpha = 0.0);
    void fit(const std::vector<DataPoint<double>>& data);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) override;

    [[nodiscard("Should be used to get number of leaves")]]
    int get_n_leaves() const;

    void set_ccp_alpha(double new_alpha);
};