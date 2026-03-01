cmake_minimum_required(VERSION 3.28)
project(DecisionTree)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(decision_tree_core
        src/decision_tree_classifier.cpp
        src/decision_tree_regressor.cpp
)

target_include_directories(decision_tree_core PUBLIC include)

add_executable(DecisionTree main.cpp)
target_link_libraries(DecisionTree PRIVATE decision_tree_core)

add_executable(TestingAI tests/testing_by_ai.cpp)
target_link_libraries(TestingAI PRIVATE decision_tree_core)