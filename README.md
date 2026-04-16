# Decision tree implementation in C++

A C++ implementation of Decision Tree algorithms for both classification and regression tasks, with support for pruning and feature importance calculation.

## Features

- Classification (Gini impurity, Entropy)
- Regression (MSE, MAE)
- Cost complexity pruning (CCP)
- Feature importance analysis
- Prediction probabilities for classification
- Configurable tree parameters:
    - `max_depth` - Maximum tree depth
    - `min_samples_split` - Minimum samples required to split a node
    - `min_samples_leaf` - Minimum samples required in a leaf node
    - `ccp_alpha` - Cost complexity pruning parameter

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Usage
### Classification
```c++
DecisionTreeClassifier clf(10, 5, 2, "gini", 0.0);
clf.fit(X_train, y_train);
int prediction = clf.predict(x_test);
auto probabilities = clf.predict_proba(x_test);
```

## Testing
### Run the test suite:
```bash
./TestingAI
```

### Current Status
- Classification: Fully functional
- Regression: Functional, pruning under investigation

### Requirements
- C++20 compatible compiler
- CMake 3.10+