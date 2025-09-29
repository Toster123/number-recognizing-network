cmake_minimum_required(VERSION 3.10)
project(NumberRecognitionNetwork)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Qt6
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

# Enable automatic MOC (Meta-Object Compiler)
set(CMAKE_AUTOMOC ON)

# Source files
set(SOURCES
    src/main.cpp
    src/neural_network/layers.cpp
    src/neural_network/network.cpp
    src/screen/screen.cpp
)

# Header files that need MOC
set(HEADERS
    src/screen/screen.hpp
    src/neural_network/layers.hpp
    src/neural_network/network.hpp
)

# Create executable
add_executable(number_recognition ${SOURCES} ${HEADERS})

# Link Qt libraries
target_link_libraries(number_recognition Qt6::Core Qt6::Widgets)

# Set output directory
set_target_properties(number_recognition PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
