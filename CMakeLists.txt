cmake_minimum_required(VERSION 3.10)
project(DecisionTree)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(decision_tree
        src/decision_tree_classifier.cpp
        src/decision_tree_regressor.cpp
)

target_include_directories(decision_tree PUBLIC include)

add_executable(DecisionTree main.cpp)

target_link_libraries(DecisionTree PRIVATE decision_tree)