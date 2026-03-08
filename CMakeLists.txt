cmake_minimum_required(VERSION 3.16)
project(VisionMatch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OpenCV ──────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ── Executable ──────────────────────────────────────────────
add_executable(vision_match main.cpp)

target_link_libraries(vision_match
    ${OpenCV_LIBS}
    pthread
)

# ── Optional: enable optimisations on Raspberry Pi ─────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    target_compile_options(vision_match PRIVATE -O2 -march=native)
    message(STATUS "ARM detected – enabling -O2 -march=native")
endif()