cmake_minimum_required(VERSION 3.10)
project(NumberRecognitionNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

set(CMAKE_AUTOMOC ON)

set(SOURCES
    src/main.cpp
    src/neural_network/layers.cpp
    src/neural_network/network.cpp
    src/screen/screen.cpp
)

set(HEADERS
    src/screen/screen.hpp
    src/neural_network/layers.hpp
    src/neural_network/network.hpp
)

add_executable(number_recognition ${SOURCES} ${HEADERS})

target_link_libraries(number_recognition Qt6::Core Qt6::Widgets)

set_target_properties(number_recognition PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)