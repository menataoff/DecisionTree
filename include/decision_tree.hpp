#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

template<typename TargetType>
struct DataPoint {
    std::vector<double> features;
    TargetType target;

    DataPoint(const std::vector<double>& features, TargetType target)
        : features(features), target(target) {}
};

template<typename TargetType>
class Node {
protected:
    double node_error;
    int sample_count;

public:
    Node(double node_error, int sample_count) : node_error(node_error), sample_count(sample_count) {}

    virtual ~Node() = default;

    [[nodiscard("Should be used to get proba")]] virtual std::unordered_map<int, double> predict_proba(
        const std::vector<double>& features) const
    {
        throw std::runtime_error("predict_proba not supported for this node type");
    }

    virtual TargetType predict(const std::vector<double>& features) const = 0;

    [[nodiscard("Should be uded to get node error")]] double get_node_error() const { return node_error; }
    [[nodiscard("Should be uded to get sample count")]] int get_sample_count() const { return sample_count; }
};

struct SplitInfo {
    int feature_index;
    double threshold;
    double information_gain;
    size_t split_position;
    std::vector<size_t> sorted_indices;

    [[nodiscard("Should be uded to get left indices")]] std::vector<size_t> get_left_indices() const {
        return {
        sorted_indices.begin(),
        sorted_indices.begin() + static_cast<ptrdiff_t>(split_position) + 1};
    }

    [[nodiscard("Should be uded to get right indices")]] std::vector<size_t> get_right_indices() const {
        return {
        sorted_indices.begin() + static_cast<ptrdiff_t>(split_position) + 1,
        sorted_indices.end()};
    }

    SplitInfo() : feature_index(-1), threshold(0.0), information_gain(-1.0), split_position(0) {}
};

template<typename TargetType>
class DecisionTree {
protected:
    std::unique_ptr<Node<TargetType>> root;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    std::vector<double> feature_importances;
    double ccp_alpha;
public:
    explicit DecisionTree(int max_depth = 32,
                 int min_samples_split = 5,
                 int min_samples_leaf = 2,
                 double ccp_alpha = 0.0) :
    max_depth(max_depth),
    min_samples_split(min_samples_split),
    min_samples_leaf(min_samples_leaf),
    ccp_alpha(ccp_alpha) {}

    virtual ~DecisionTree() = default;

    virtual void fit(const std::vector<std::vector<double>>& X, const std::vector<TargetType>& y) = 0;

    TargetType predict(const std::vector<double>& features) const {
        if (!root) {
            return TargetType{};
        }
        return root->predict(features);
    }

    [[nodiscard("Should be used to predict proba")]]
    std::unordered_map<int, double> predict_proba(const std::vector<double>& features) const {
        if (!root) {
            return {};
        }
        try {
            return root->predict_proba(features);
        } catch (const std::runtime_error&) {
            return {};
        }
    }

    [[nodiscard("Should be used to get feature importances")]]
    const std::vector<double>& get_feature_importances() const {
        return feature_importances;
    }

    Node<TargetType>* get_root() {
        return root.get();
    }

    DecisionTree(const DecisionTree&) = delete;
    DecisionTree& operator=(const DecisionTree&) = delete;

    DecisionTree(DecisionTree&&) = default;
    DecisionTree& operator=(DecisionTree&&) = default;
};
