# Decision Tree Implementation (C++17/20)

A from‑scratch implementation of Decision Tree algorithm for classification and regression. Built as a learning project to understand how tree‑based models work under the hood.

## Overview

This project implements a fully functional Decision Tree that supports both classification and regression tasks. It includes cost‑complexity pruning (CCP), feature importance calculation, and predict probabilities for classification. The code is written in modern C++ (C++17/20) with focus on performance, memory safety (using `std::unique_ptr`), and clean object‑oriented design.

Key features:
- Classification (binary and multiclass) with Gini impurity and Entropy criteria
- Regression with MSE (Mean Squared Error) and MAE (Mean Absolute Error)
- Cost complexity pruning to reduce overfitting
- Feature importance based on impurity reduction
- Predict probabilities for classification tasks
- Configurable hyperparameters: max depth, min samples split, min samples leaf, ccp_alpha

## How It Works

The tree is built recursively. At each node, the algorithm scans all features and possible split thresholds to find the split that maximizes impurity reduction (for classification) or variance reduction (for regression). The process stops when a stopping criterion is met (max depth, min samples split, or pure node).

After the initial tree is built, cost‑complexity pruning removes branches that do not improve generalization. The pruning parameter `ccp_alpha` controls the trade‑off between tree size and accuracy.

## Implementation Highlights

- **Polymorphic node hierarchy** – abstract `Node<T>` base class with `LeafNode` and `InternalNode` specializations
- **Memory management** – `std::unique_ptr` for automatic node cleanup, no manual `new`/`delete`
- **Template‑based design** – same code works for `int` (classification) and `double` (regression)
- **No external dependencies** – only standard library
- **Comprehensive test suite** – 19+ tests covering all features

## Build & Run

```bash
git clone <repo>
cd DecisionTree
mkdir build && cd build
cmake ..
cmake --build .
```

Run the full test suite:
```bash
./TestingAI
```

## Usage Examples

### Classification
```c++
#include "decision_tree_classifier.hpp"

// Train data: 500 samples, 4 features
DecisionTreeClassifier clf(10, 5, 2, "gini", 0.0);
clf.fit(X_train, y_train);

// Predict
int label = clf.predict(x_test);

// Get class probabilities
auto probs = clf.predict_proba(x_test);

// Feature importance
auto importance = clf.get_feature_importances();
```

### Regression
```c++
#include "decision_tree_regressor.hpp"

DecisionTreeRegressor reg(10, 5, 2, "mse", 0.0);
reg.fit(X_train, y_train);

double value = reg.predict(x_test);
```

## Test Results

Example output from the test suite:

```txt
[CLASSIFICATION 1] Binary classification
  Accuracy: 96.00%
  Leaves: 19
  Fit time: 3.90 ms

[REGRESSION 5] Cost complexity pruning
  No prune: MAE=0.4162 (200 leaves)
  With prune: MAE=0.1809 (5 leaves)
  Improvement: 0.2353 MAE reduction
```

All 19 tests pass, including:

- Binary and multiclass classification
- Linear and quadratic regression
- Pruning validation
- Feature importance checks
- Predict probability consistency 
## Performance
The tree is fast enough for educational use and small to medium datasets:

- 500 samples, depth=5 → ~2‑3 ms training time
- 2000 samples, depth=5 → ~10‑20 ms
- 10000 samples, depth=5 → ~60‑130 ms

For comparison with scikit‑learn, run the benchmark suite (separate Python script included).

## Project Structure
```text
DecisionTree/
├── include/
│   ├── decision_tree.hpp              # base template class
│   ├── decision_tree_classifier.hpp   # classification
│   └── decision_tree_regressor.hpp    # regression
├── src/
│   ├── decision_tree_classifier.cpp
│   └── decision_tree_regressor.cpp
├── tests/
│   └── testing_by_ai.cpp              # test suite
└── CMakeLists.txt
```

## Requirements

- C++17 or C++20 compatible compiler (GCC, Clang, MSVC)

- CMake 3.10 or higher

## What I Learned
Building this project helped me understand:

- How Decision Trees actually work under the hood (not just using sklearn)
- Recursive algorithms with optimal complexity
Memory management with smart pointers in polymorphic hierarchies
- Template programming and type safety
- Writing tests and benchmarks for C++ code
- Performance analysis and comparison with industrial implementations
## Future Plans

- Gradient Boosting ensemble built on top of this tree
- Support for categorical features
