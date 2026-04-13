cmake_minimum_required(VERSION 3.16)

project(fight_detection_cpp_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

option(FIGHT_DEMO_ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)
option(FIGHT_DEMO_ENABLE_MINDSPORE "Enable MindSpore Lite backend (Windows only)" OFF)

if (WIN32)
    set(DEFAULT_ONNXRUNTIME_INCLUDE_DIR "D:/project/lxh/YOLOv5-Lite/cpp_demo/ort/include")
    set(DEFAULT_ONNXRUNTIME_LIBRARY "D:/App/Anaconda3/envs/pytorch/Lib/site-packages/onnxruntime/capi/onnxruntime.dll")
elseif(UNIX)
    set(ONNXRUNTIME_LOCAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.23.2")
    if (EXISTS "${ONNXRUNTIME_LOCAL_ROOT}/include/onnxruntime_c_api.h" AND EXISTS "${ONNXRUNTIME_LOCAL_ROOT}/lib/libonnxruntime.so")
        set(DEFAULT_ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_LOCAL_ROOT}/include")
        set(DEFAULT_ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_LOCAL_ROOT}/lib/libonnxruntime.so")
    else()
        set(DEFAULT_ONNXRUNTIME_INCLUDE_DIR "/usr/local/include/onnxruntime")
        set(DEFAULT_ONNXRUNTIME_LIBRARY "/usr/local/lib/libonnxruntime.so")
    endif()
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${DEFAULT_ONNXRUNTIME_INCLUDE_DIR}" CACHE PATH "ONNX Runtime headers directory (contains onnxruntime_c_api.h)")
set(ONNXRUNTIME_LIBRARY "${DEFAULT_ONNXRUNTIME_LIBRARY}" CACHE FILEPATH "ONNX Runtime shared library path (.dll/.so)")

set(MINDSPORE_ROOT "D:/project/mindspore-lite-2.7.0-win-x64/mindspore-lite-2.7.0-win-x64/runtime" CACHE PATH "MindSpore Lite runtime root containing include/ and lib/libmindspore-lite.dll")

set(FIGHT_DEMO_HAS_MINDSPORE OFF)
if (
    FIGHT_DEMO_ENABLE_MINDSPORE
    AND WIN32
    AND EXISTS "${MINDSPORE_ROOT}/include/c_api/model_c.h"
    AND EXISTS "${MINDSPORE_ROOT}/lib/libmindspore-lite.dll"
)
    set(MINDSPORE_LIBRARY "${MINDSPORE_ROOT}/lib/libmindspore-lite.dll")
    set(FIGHT_DEMO_HAS_MINDSPORE ON)
elseif(FIGHT_DEMO_ENABLE_MINDSPORE)
    message(WARNING "MindSpore backend requested but runtime was not found or platform is not Windows. MindSpore backend will be disabled.")
endif()

set(FIGHT_DEMO_HAS_ONNXRUNTIME OFF)
if (
    FIGHT_DEMO_ENABLE_ONNXRUNTIME
    AND EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h"
    AND EXISTS "${ONNXRUNTIME_LIBRARY}"
)
    set(FIGHT_DEMO_HAS_ONNXRUNTIME ON)
elseif(FIGHT_DEMO_ENABLE_ONNXRUNTIME)
    message(WARNING "ONNX Runtime backend requested but ONNXRUNTIME_INCLUDE_DIR/ONNXRUNTIME_LIBRARY is invalid. ONNX backend will be disabled.")
endif()

add_executable(fight_detection_demo
    src/inference_core.cpp
    src/main.cpp
)

target_include_directories(fight_detection_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fight_detection_demo PRIVATE ${OpenCV_LIBS})

if (FIGHT_DEMO_HAS_MINDSPORE)
    target_compile_definitions(
        fight_detection_demo PRIVATE
        FIGHT_DEMO_HAS_MINDSPORE=1
        "FIGHT_DEMO_MINDSPORE_DLL_PATH=\"${MINDSPORE_LIBRARY}\""
    )
    target_include_directories(fight_detection_demo PRIVATE "${MINDSPORE_ROOT}")
endif()

if (FIGHT_DEMO_HAS_ONNXRUNTIME)
    target_compile_definitions(
        fight_detection_demo PRIVATE
        FIGHT_DEMO_HAS_ONNXRUNTIME=1
        "FIGHT_DEMO_ONNXRUNTIME_LIBRARY_PATH=\"${ONNXRUNTIME_LIBRARY}\""
    )
    target_include_directories(fight_detection_demo PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    if(UNIX AND NOT APPLE)
        target_link_libraries(fight_detection_demo PRIVATE dl)
    endif()
endif()
