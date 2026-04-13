#include "inference_core.h"
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <deque>
#include <filesystem>

#ifdef FIGHT_DEMO_HAS_MINDSPORE
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <include/c_api/context_c.h>
#include <include/c_api/data_type_c.h>
#include <include/c_api/model_c.h>
#include <include/c_api/status_c.h>
#include <include/c_api/tensor_c.h>
#endif

#ifdef FIGHT_DEMO_HAS_ONNXRUNTIME
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#else
#include <dlfcn.h>
#endif
#include <onnxruntime_c_api.h>
#endif

namespace fs = std::filesystem;

namespace {

std::optional<std::string> getArgValue(const std::vector<std::string>& args, const std::string& key) {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) {
            return args[i + 1];
        }
    }
    return std::nullopt;
}

bool hasArg(const std::vector<std::string>& args, const std::string& key) {
    return std::find(args.begin(), args.end(), key) != args.end();
}

void validateModelPath(const std::string& backend, const std::string& modelPath);

float clampFloat(float value, float minValue, float maxValue) {
    return std::max(minValue, std::min(value, maxValue));
}

std::vector<float> generateSyntheticFrameFeatures(int frameIndex, int featureDim) {
    std::vector<float> features(static_cast<size_t>(featureDim), 0.0f);
    const float base = static_cast<float>(frameIndex % 17) / 16.0f;
    for (int index = 0; index < featureDim; ++index) {
        features[static_cast<size_t>(index)] = base;
    }
    return features;
}

std::vector<std::vector<float>> loadFeatureCsv(const std::string& path, int expectedFeatureDim) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open feature csv: " + path);
    }

    std::vector<std::vector<float>> rows;
    std::string line;
    bool isFirstLine = true;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        if (isFirstLine) {
            isFirstLine = false;
            if (line.rfind("frame_index,feature_dim", 0) == 0) {
                continue;
            }
        }

        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(lineStream, cell, ',')) {
            cells.push_back(cell);
        }

        if (cells.size() < 2) {
            throw std::runtime_error("invalid feature csv row: " + line);
        }

        const int featureDim = std::stoi(cells[1]);
        if (featureDim != expectedFeatureDim) {
            throw std::runtime_error(
                "feature csv dimension mismatch. expected=" + std::to_string(expectedFeatureDim) +
                " actual=" + std::to_string(featureDim));
        }
        if (cells.size() != static_cast<size_t>(featureDim + 2)) {
            throw std::runtime_error("feature csv row has unexpected column count: " + line);
        }

        std::vector<float> row(static_cast<size_t>(featureDim), 0.0f);
        for (int index = 0; index < featureDim; ++index) {
            row[static_cast<size_t>(index)] = std::stof(cells[static_cast<size_t>(index + 2)]);
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

void writeFeatureCsvHeader(std::ofstream& output, int featureDim) {
    output << "frame_index,feature_dim";
    for (int index = 0; index < featureDim; ++index) {
        output << ",f" << index;
    }
    output << "\n";
}

void writeFeatureCsvRow(std::ofstream& output, int frameIndex, const std::vector<float>& features) {
    output << frameIndex << "," << features.size();
    output << std::fixed << std::setprecision(6);
    for (float value : features) {
        output << "," << value;
    }
    output << "\n";
}

#ifdef FIGHT_DEMO_HAS_ONNXRUNTIME
#ifdef _WIN32
using RuntimeLibraryHandle = HMODULE;
#else
using RuntimeLibraryHandle = void*;
#endif

RuntimeLibraryHandle loadRuntimeLibrary(const char* path) {
#ifdef _WIN32
    return LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

void unloadRuntimeLibrary(RuntimeLibraryHandle handle) {
    if (handle == nullptr) {
        return;
    }
#ifdef _WIN32
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
}

template <typename T>
T resolveOnnxRuntimeSymbol(RuntimeLibraryHandle module, const char* symbolName) {
#ifdef _WIN32
    auto symbol = reinterpret_cast<T>(GetProcAddress(module, symbolName));
#else
    auto symbol = reinterpret_cast<T>(dlsym(module, symbolName));
#endif
    if (symbol == nullptr) {
        throw std::runtime_error(std::string("failed to resolve ONNX Runtime symbol: ") + symbolName);
    }
    return symbol;
}

template <typename T>
T tryResolveOnnxRuntimeSymbol(RuntimeLibraryHandle module, const char* symbolName) {
#ifdef _WIN32
    return reinterpret_cast<T>(GetProcAddress(module, symbolName));
#else
    return reinterpret_cast<T>(dlsym(module, symbolName));
#endif
}

void throwOnnxStatus(const OrtApi* api, OrtStatus* status, const std::string& contextMessage) {
    if (status == nullptr) {
        return;
    }

    std::string message = contextMessage;
    if (api != nullptr) {
        const char* errorMessage = api->GetErrorMessage(status);
        if (errorMessage != nullptr) {
            message += ": ";
            message += errorMessage;
        }
        api->ReleaseStatus(status);
    }
    throw std::runtime_error(message);
}

#ifdef _WIN32
std::wstring toWideString(const std::string& value);
#endif

struct OrtSessionBundle {
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* sessionOptions = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memoryInfo = nullptr;
    OrtAllocator* allocator = nullptr;
    std::string inputName;
    std::string outputName;
};

void releaseOrtSessionBundle(OrtSessionBundle& bundle) {
    if (bundle.api != nullptr) {
        if (bundle.memoryInfo != nullptr) {
            bundle.api->ReleaseMemoryInfo(bundle.memoryInfo);
            bundle.memoryInfo = nullptr;
        }
        if (bundle.session != nullptr) {
            bundle.api->ReleaseSession(bundle.session);
            bundle.session = nullptr;
        }
        if (bundle.sessionOptions != nullptr) {
            bundle.api->ReleaseSessionOptions(bundle.sessionOptions);
            bundle.sessionOptions = nullptr;
        }
        if (bundle.env != nullptr) {
            bundle.api->ReleaseEnv(bundle.env);
            bundle.env = nullptr;
        }
    }
}

void appendExecutionProviderIfNeeded(
    RuntimeLibraryHandle runtimeDll,
    const OrtApi* api,
    OrtSessionOptions* sessionOptions,
    const std::string& onnxEp,
    int onnxDeviceId) {
    if (onnxEp == "cpu") {
        return;
    }

    if (onnxEp != "cann") {
        throw std::runtime_error("unsupported onnx execution provider: " + onnxEp + ". expected cpu or cann");
    }

    (void)runtimeDll;

    if (api == nullptr ||
        api->CreateCANNProviderOptions == nullptr ||
        api->UpdateCANNProviderOptions == nullptr ||
        api->SessionOptionsAppendExecutionProvider_CANN == nullptr ||
        api->ReleaseCANNProviderOptions == nullptr) {
        throw std::runtime_error(
            "onnx-ep=cann requested, but current ONNX Runtime API table does not expose CANN EP entry points. "
            "Please replace ONNX Runtime with a CANN-EP build and ensure libonnxruntime_providers_cann.so is present.");
    }

    OrtCANNProviderOptions* cannOptions = nullptr;
    throwOnnxStatus(api, api->CreateCANNProviderOptions(&cannOptions), "CreateCANNProviderOptions failed");

    const std::string deviceIdValue = std::to_string(onnxDeviceId);
    const char* optionKeys[] = {"device_id"};
    const char* optionValues[] = {deviceIdValue.c_str()};
    try {
        throwOnnxStatus(
            api,
            api->UpdateCANNProviderOptions(cannOptions, optionKeys, optionValues, 1),
            "UpdateCANNProviderOptions failed");
        throwOnnxStatus(
            api,
            api->SessionOptionsAppendExecutionProvider_CANN(sessionOptions, cannOptions),
            "SessionOptionsAppendExecutionProvider_CANN failed");
    } catch (...) {
        api->ReleaseCANNProviderOptions(cannOptions);
        throw;
    }
    api->ReleaseCANNProviderOptions(cannOptions);
    std::cout << "[info] ONNX Runtime EP: cann (device_id=" << onnxDeviceId << ")\n";
}

void initializeOrtSessionBundle(
    RuntimeLibraryHandle runtimeDll,
    const std::string& modelPath,
    const char* logId,
    const std::string& onnxEp,
    int onnxDeviceId,
    OrtSessionBundle& bundle) {
    using GetApiBaseFn = decltype(&OrtGetApiBase);
    const auto getApiBase = resolveOnnxRuntimeSymbol<GetApiBaseFn>(runtimeDll, "OrtGetApiBase");
    const OrtApiBase* apiBase = getApiBase();
    if (apiBase == nullptr) {
        throw std::runtime_error("OrtGetApiBase returned nullptr.");
    }

    bundle.api = apiBase->GetApi(ORT_API_VERSION);
    if (bundle.api == nullptr) {
        throw std::runtime_error("failed to get ONNX Runtime API table.");
    }

    throwOnnxStatus(bundle.api, bundle.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, logId, &bundle.env), "CreateEnv failed");
    throwOnnxStatus(bundle.api, bundle.api->CreateSessionOptions(&bundle.sessionOptions), "CreateSessionOptions failed");
    throwOnnxStatus(bundle.api, bundle.api->SetIntraOpNumThreads(bundle.sessionOptions, 1), "SetIntraOpNumThreads failed");
    throwOnnxStatus(
        bundle.api,
        bundle.api->SetSessionGraphOptimizationLevel(bundle.sessionOptions, ORT_ENABLE_EXTENDED),
        "SetSessionGraphOptimizationLevel failed");
    appendExecutionProviderIfNeeded(runtimeDll, bundle.api, bundle.sessionOptions, onnxEp, onnxDeviceId);

#ifdef _WIN32
    const std::wstring wideModelPath = toWideString(modelPath);
    throwOnnxStatus(bundle.api, bundle.api->CreateSession(bundle.env, wideModelPath.c_str(), bundle.sessionOptions, &bundle.session), "CreateSession failed");
#else
    throwOnnxStatus(bundle.api, bundle.api->CreateSession(bundle.env, modelPath.c_str(), bundle.sessionOptions, &bundle.session), "CreateSession failed");
#endif

    throwOnnxStatus(bundle.api, bundle.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &bundle.memoryInfo), "CreateCpuMemoryInfo failed");
    throwOnnxStatus(bundle.api, bundle.api->GetAllocatorWithDefaultOptions(&bundle.allocator), "GetAllocatorWithDefaultOptions failed");

    char* inputName = nullptr;
    char* outputName = nullptr;
    throwOnnxStatus(bundle.api, bundle.api->SessionGetInputName(bundle.session, 0, bundle.allocator, &inputName), "SessionGetInputName failed");
    throwOnnxStatus(bundle.api, bundle.api->SessionGetOutputName(bundle.session, 0, bundle.allocator, &outputName), "SessionGetOutputName failed");
    bundle.inputName = inputName != nullptr ? inputName : "input";
    bundle.outputName = outputName != nullptr ? outputName : "output0";
    if (inputName != nullptr) {
        bundle.allocator->Free(bundle.allocator, inputName);
    }
    if (outputName != nullptr) {
        bundle.allocator->Free(bundle.allocator, outputName);
    }
}

struct PoseDetection {
    cv::Rect2f box;
    float score = 0.0f;
    std::vector<float> keypoints;
};

struct PoseFrameResult {
    std::vector<float> frameFeatures;
    std::vector<PoseDetection> detections;
};

void writePoseDebugJson(
    const std::string& path,
    int frameIndex,
    const cv::Mat& frame,
    const std::vector<PoseDetection>& detections) {
    if (path.empty()) {
        return;
    }

    fs::create_directories(fs::path(path).parent_path());
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write pose debug json: " + path);
    }

    output << "{\n";
    output << "  \"frame_index\": " << frameIndex << ",\n";
    output << "  \"image_width\": " << frame.cols << ",\n";
    output << "  \"image_height\": " << frame.rows << ",\n";
    output << "  \"detection_count\": " << detections.size() << ",\n";
    output << "  \"detections\": [\n";
    for (size_t detectionIndex = 0; detectionIndex < detections.size(); ++detectionIndex) {
        const auto& detection = detections[detectionIndex];
        output << "    {\n";
        output << "      \"score\": " << std::fixed << std::setprecision(6) << detection.score << ",\n";
        output << "      \"box_xyxy\": ["
               << detection.box.x << ", "
               << detection.box.y << ", "
               << (detection.box.x + detection.box.width) << ", "
               << (detection.box.y + detection.box.height) << "],\n";
        output << "      \"keypoints\": [\n";
        for (int kp = 0; kp < 17; ++kp) {
            const size_t base = static_cast<size_t>(kp * 3);
            output << "        ["
                   << detection.keypoints[base + 0] << ", "
                   << detection.keypoints[base + 1] << ", "
                   << detection.keypoints[base + 2] << "]";
            output << (kp < 16 ? ",\n" : "\n");
        }
        output << "      ]\n";
        output << "    }" << (detectionIndex + 1 < detections.size() ? ",\n" : "\n");
    }
    output << "  ]\n";
    output << "}\n";
}

class YoloPoseOnnxExtractor {
public:
    explicit YoloPoseOnnxExtractor(const std::string& modelPath, std::string onnxEp, int onnxDeviceId) {
        validateModelPath("yolo-onnx", modelPath);
        std::transform(
            onnxEp.begin(),
            onnxEp.end(),
            onnxEp.begin(),
            [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
        runtimeDll_ = loadRuntimeLibrary(FIGHT_DEMO_ONNXRUNTIME_LIBRARY_PATH);
        if (runtimeDll_ == nullptr) {
            throw std::runtime_error(std::string("failed to load ONNX Runtime shared library: ") + FIGHT_DEMO_ONNXRUNTIME_LIBRARY_PATH);
        }
        initializeOrtSessionBundle(runtimeDll_, modelPath, "fight_detection_pose", onnxEp, onnxDeviceId, session_);
    }

    ~YoloPoseOnnxExtractor() {
        releaseOrtSessionBundle(session_);
        if (runtimeDll_ != nullptr) {
            unloadRuntimeLibrary(runtimeDll_);
        }
    }

    PoseFrameResult extractFrameFeatures(const cv::Mat& frame, int expectedFeatureDim) {
        if (frame.empty()) {
            throw std::runtime_error("pose extractor received empty frame.");
        }

        const int inputSize = 640;
        const float scale = std::min(
            static_cast<float>(inputSize) / static_cast<float>(frame.cols),
            static_cast<float>(inputSize) / static_cast<float>(frame.rows));
        const int resizedWidth = static_cast<int>(std::round(frame.cols * scale));
        const int resizedHeight = static_cast<int>(std::round(frame.rows * scale));
        const int padX = (inputSize - resizedWidth) / 2;
        const int padY = (inputSize - resizedHeight) / 2;

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(resizedWidth, resizedHeight));
        cv::Mat letterboxed(inputSize, inputSize, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(letterboxed(cv::Rect(padX, padY, resizedWidth, resizedHeight)));

        cv::Mat rgb;
        cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

        std::vector<float> inputBuffer(static_cast<size_t>(3 * inputSize * inputSize), 0.0f);
        for (int y = 0; y < inputSize; ++y) {
            for (int x = 0; x < inputSize; ++x) {
                const cv::Vec3f pixel = rgb.at<cv::Vec3f>(y, x);
                inputBuffer[static_cast<size_t>(0 * inputSize * inputSize + y * inputSize + x)] = pixel[0];
                inputBuffer[static_cast<size_t>(1 * inputSize * inputSize + y * inputSize + x)] = pixel[1];
                inputBuffer[static_cast<size_t>(2 * inputSize * inputSize + y * inputSize + x)] = pixel[2];
            }
        }

        std::array<int64_t, 4> inputShape{1, 3, inputSize, inputSize};
        OrtValue* inputTensor = nullptr;
        throwOnnxStatus(
            session_.api,
            session_.api->CreateTensorWithDataAsOrtValue(
                session_.memoryInfo,
                inputBuffer.data(),
                inputBuffer.size() * sizeof(float),
                inputShape.data(),
                inputShape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &inputTensor),
            "CreateTensorWithDataAsOrtValue failed for pose model");

        const char* inputNames[] = {session_.inputName.c_str()};
        const char* outputNames[] = {session_.outputName.c_str()};
        OrtValue* outputTensor = nullptr;
        throwOnnxStatus(
            session_.api,
            session_.api->Run(
                session_.session,
                nullptr,
                inputNames,
                reinterpret_cast<const OrtValue* const*>(&inputTensor),
                1,
                outputNames,
                1,
                &outputTensor),
            "ONNX Runtime Run failed for pose model");

        float* outputData = nullptr;
        throwOnnxStatus(session_.api, session_.api->GetTensorMutableData(outputTensor, reinterpret_cast<void**>(&outputData)), "GetTensorMutableData failed for pose model");
        if (outputData == nullptr) {
            session_.api->ReleaseValue(outputTensor);
            session_.api->ReleaseValue(inputTensor);
            throw std::runtime_error("pose model output tensor had no data.");
        }

        constexpr int numChannels = 56;
        constexpr int numPredictions = 8400;
        std::vector<cv::Rect> nmsBoxes;
        std::vector<float> scores;
        std::vector<PoseDetection> candidates;
        candidates.reserve(32);

        for (int predictionIndex = 0; predictionIndex < numPredictions; ++predictionIndex) {
            const float score = outputData[4 * numPredictions + predictionIndex];
            if (score < 0.35f) {
                continue;
            }

            const float cx = outputData[0 * numPredictions + predictionIndex];
            const float cy = outputData[1 * numPredictions + predictionIndex];
            const float width = outputData[2 * numPredictions + predictionIndex];
            const float height = outputData[3 * numPredictions + predictionIndex];
            const float x1 = (cx - width * 0.5f - static_cast<float>(padX)) / scale;
            const float y1 = (cy - height * 0.5f - static_cast<float>(padY)) / scale;
            const float x2 = (cx + width * 0.5f - static_cast<float>(padX)) / scale;
            const float y2 = (cy + height * 0.5f - static_cast<float>(padY)) / scale;

            PoseDetection detection;
            detection.score = score;
            detection.box = cv::Rect2f(
                clampFloat(x1, 0.0f, static_cast<float>(frame.cols - 1)),
                clampFloat(y1, 0.0f, static_cast<float>(frame.rows - 1)),
                clampFloat(x2 - x1, 0.0f, static_cast<float>(frame.cols)),
                clampFloat(y2 - y1, 0.0f, static_cast<float>(frame.rows)));
            detection.keypoints.reserve(51);
            for (int keypointIndex = 0; keypointIndex < 17; ++keypointIndex) {
                const int baseChannel = 5 + keypointIndex * 3;
                const float keypointX = (outputData[(baseChannel + 0) * numPredictions + predictionIndex] - static_cast<float>(padX)) / scale;
                const float keypointY = (outputData[(baseChannel + 1) * numPredictions + predictionIndex] - static_cast<float>(padY)) / scale;
                const float keypointConf = outputData[(baseChannel + 2) * numPredictions + predictionIndex];
                detection.keypoints.push_back(clampFloat(keypointX, 0.0f, static_cast<float>(frame.cols - 1)));
                detection.keypoints.push_back(clampFloat(keypointY, 0.0f, static_cast<float>(frame.rows - 1)));
                detection.keypoints.push_back(clampFloat(keypointConf, 0.0f, 1.0f));
            }
            candidates.push_back(std::move(detection));
            nmsBoxes.emplace_back(
                static_cast<int>(std::round(candidates.back().box.x)),
                static_cast<int>(std::round(candidates.back().box.y)),
                static_cast<int>(std::round(candidates.back().box.width)),
                static_cast<int>(std::round(candidates.back().box.height)));
            scores.push_back(score);
        }

        session_.api->ReleaseValue(outputTensor);
        session_.api->ReleaseValue(inputTensor);

        std::vector<int> keptIndices;
        cv::dnn::NMSBoxes(nmsBoxes, scores, 0.35f, 0.45f, keptIndices);
        std::sort(keptIndices.begin(), keptIndices.end(), [&](int left, int right) {
            return candidates[static_cast<size_t>(left)].score > candidates[static_cast<size_t>(right)].score;
        });

        constexpr int maxPersons = 3;
        constexpr int featuresPerPerson = 51;
        std::vector<float> frameFeatures(static_cast<size_t>(expectedFeatureDim), 0.0f);
        const int keptCount = std::min(static_cast<int>(keptIndices.size()), maxPersons);
        std::vector<PoseDetection> keptDetections;
        keptDetections.reserve(static_cast<size_t>(keptCount));
        for (int personIndex = 0; personIndex < keptCount; ++personIndex) {
            const auto& detection = candidates[static_cast<size_t>(keptIndices[static_cast<size_t>(personIndex)])];
            const auto& keypoints = detection.keypoints;
            std::copy(
                keypoints.begin(),
                keypoints.begin() + std::min(static_cast<int>(keypoints.size()), featuresPerPerson),
                frameFeatures.begin() + static_cast<ptrdiff_t>(personIndex * featuresPerPerson));
            keptDetections.push_back(detection);
        }
        return PoseFrameResult{std::move(frameFeatures), std::move(keptDetections)};
    }

private:
    RuntimeLibraryHandle runtimeDll_ = nullptr;
    OrtSessionBundle session_;
};
#endif

cv::VideoCapture openCapture(const Config& config, std::string& sourceDesc) {
    if (config.sourceType == "camera") {
        std::string input = config.cameraInput;
        // Fallback to integer index if string is empty
        if (input.empty()) input = std::to_string(config.cameraIndex);

        // Check if input is a numeric string (traditional camera index)
        bool isNumeric = !input.empty() && std::all_of(input.begin(), input.end(), ::isdigit);

        cv::VideoCapture cap;
        if (isNumeric) {
            int idx = std::stoi(input);
            sourceDesc = "camera:" + input;
#ifdef _WIN32
            cap.open(idx, cv::CAP_DSHOW);
#else
            cap.open(idx, cv::CAP_ANY);
#endif
        } else {
            // Treat as network stream or file path
            sourceDesc = "stream:" + input;
            cap.open(input, cv::CAP_FFMPEG);
        }

        // Apply buffer optimization for streams/cameras
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_BUFFERSIZE, 2);
        }
        return cap;
    }

    // Video File Logic
    sourceDesc = config.videoPath;
    cv::VideoCapture cap(config.videoPath, cv::CAP_FFMPEG);
    if (config.videoPath.find("rtsp://") == 0 || config.videoPath.find("rtmp://") == 0 || config.videoPath.find("http://") == 0) {
        cap.set(cv::CAP_PROP_BUFFERSIZE, 2);
    }
    if (!cap.isOpened()) {
        cap.open(config.videoPath);
    }
    return cap;
}

class VideoCaptureThread {
public:
    VideoCaptureThread() : running_(false), hasNewFrame_(false) {}

    ~VideoCaptureThread() {
        stop();
    }

    bool open(const Config& config, std::string& sourceDesc) {
        stop();
        cap_ = openCapture(config, sourceDesc);
        if (cap_.isOpened()) {
            running_ = true;
            hasNewFrame_ = false;
            thread_ = std::thread(&VideoCaptureThread::loop, this);
            return true;
        }
        return false;
    }

    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
        if (cap_.isOpened()) {
            cap_.release();
        }
    }

    bool isOpened() const {
        return running_;
    }
    
    void release() {
        stop();
    }

    double get(int propId) const {
        return const_cast<cv::VideoCapture&>(cap_).get(propId);
    }

    bool read(cv::Mat& output) {
        std::unique_lock<std::mutex> lock(mutex_);
        if(cond_.wait_for(lock, std::chrono::milliseconds(2000), [this]{ return hasNewFrame_ || !running_; })) {
            if (!running_ && !hasNewFrame_) return false;
            latestFrame_.copyTo(output);
            hasNewFrame_ = false;
            return true;
        }
        return false;
    }

private:
    void loop() {
        cv::Mat temp;
        while (running_) {
            if (cap_.read(temp)) {
                if (!temp.empty()) {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        temp.copyTo(latestFrame_);
                        hasNewFrame_ = true;
                    }
                    cond_.notify_one();
                }
            } else {
                 running_ = false;
                 cond_.notify_all();
            }
        }
    }

    cv::VideoCapture cap_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cond_;
    cv::Mat latestFrame_;
    std::atomic<bool> running_;
    bool hasNewFrame_;
};

cv::VideoWriter buildWriter(const Config& config, const VideoCaptureThread& cap, const cv::Mat& frame) {
    if (config.outputVideo.empty()) {
        return cv::VideoWriter();
    }

    fs::path outputPath(config.outputVideo);
    if (outputPath.has_parent_path()) {
        fs::create_directories(outputPath.parent_path());
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) {
        fps = 20.0;
    }

    std::string outputPathString = config.outputVideo;
    std::string extension = outputPath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

    if (extension.empty()) {
        outputPathString += ".mp4";
        extension = ".mp4";
        std::cout << "[warn] output-video has no extension, fallback to: " << outputPathString << "\n";
    }

    cv::VideoWriter writer;
    auto tryOpen = [&](int apiPreference, int fourcc, const char* tag) {
        if (writer.open(outputPathString, apiPreference, fourcc, fps, frame.size())) {
            std::cout << "[info] video writer opened with " << tag << " -> " << outputPathString << "\n";
            return true;
        }
        return false;
    };

    bool opened = false;
    if (extension == ".avi") {
        opened =
            tryOpen(cv::CAP_ANY, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), "CAP_ANY + MJPG") ||
            tryOpen(cv::CAP_ANY, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), "CAP_ANY + XVID");
    } else {
        opened =
            tryOpen(cv::CAP_FFMPEG, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), "FFMPEG + mp4v") ||
            tryOpen(cv::CAP_ANY, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), "CAP_ANY + mp4v") ||
            tryOpen(cv::CAP_ANY, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), "CAP_ANY + avc1") ||
            tryOpen(cv::CAP_ANY, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), "CAP_ANY + MJPG");
    }

    if (!opened) {
        std::cout << "[warn] failed to open video writer for output: " << outputPathString
                  << ". Detection continues without output video.\n";
        return cv::VideoWriter();
    }

    return writer;
}

double pseudoProbability(int frameIndex) {
    const double value = 0.5 + 0.45 * std::sin(static_cast<double>(frameIndex) * 0.18);
    return std::clamp(value, 0.01, 0.99);
}

double meanDeque(const std::deque<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double updateSmoothedProbability(std::deque<double>& probBuffer, int smoothK, double rawProb) {
    probBuffer.push_back(rawProb);
    if (static_cast<int>(probBuffer.size()) > smoothK) {
        probBuffer.pop_front();
    }
    return meanDeque(probBuffer);
}

void validateModelPath(const std::string& backend, const std::string& modelPath) {
    if (modelPath.empty()) {
        throw std::runtime_error(backend + " backend requires --model-path");
    }
    if (!fs::exists(modelPath)) {
        throw std::runtime_error("model file not found for backend '" + backend + "': " + modelPath);
    }
}

std::unique_ptr<ISequenceClassifier> createClassifier(const Config& config) {
    if (config.backend == "pseudo") {
        return std::make_unique<PseudoClassifier>(config.smoothK, config.threshold);
    }
    if (config.backend == "mindir") {
        return std::make_unique<MindIRClassifier>(config.modelPath, config.smoothK, config.threshold);
    }
    if (config.backend == "onnx") {
        return std::make_unique<OnnxRuntimeClassifier>(
            config.modelPath, config.smoothK, config.threshold, config.onnxEp, config.onnxDeviceId);
    }

    throw std::runtime_error(
        "unsupported backend: " + config.backend + ". expected one of: pseudo, mindir, onnx");
}

std::string escapeJson(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped += character;
            break;
        }
    }
    return escaped;
}

void writeSummaryJson(const std::string& path, const Summary& summary) {
    if (path.empty()) {
        return;
    }

    fs::create_directories(fs::path(path).parent_path());
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write summary json: " + path);
    }

    output << "{\n";
    output << "  \"source\": \"" << escapeJson(summary.source) << "\",\n";
    output << "  \"processed_frames\": " << summary.processedFrames << ",\n";
    output << "  \"incident_count\": " << summary.incidentCount << ",\n";
    output << "  \"elapsed_seconds\": " << std::fixed << std::setprecision(6) << summary.elapsedSeconds << ",\n";
    output << "  \"avg_fps\": " << std::fixed << std::setprecision(6) << summary.avgFps << ",\n";
    if (summary.outputVideo.empty()) {
        output << "  \"output_video\": null,\n";
    } else {
        output << "  \"output_video\": \"" << escapeJson(summary.outputVideo) << "\",\n";
    }
    if (summary.eventLog.empty()) {
        output << "  \"event_log\": null,\n";
    } else {
        output << "  \"event_log\": \"" << escapeJson(summary.eventLog) << "\",\n";
    }
    output << "  \"incidents\": [\n";
    for (size_t index = 0; index < summary.incidents.size(); ++index) {
        const Incident& incident = summary.incidents[index];
        output << "    {\n";
        output << "      \"index\": " << incident.index << ",\n";
        output << "      \"frame\": " << incident.frame << ",\n";
        output << "      \"smoothed_prob\": " << std::fixed << std::setprecision(4) << incident.smoothedProb << "\n";
        output << "    }";
        output << (index + 1 < summary.incidents.size() ? ",\n" : "\n");
    }
    output << "  ]\n";
    output << "}\n";
}

void writeIncidentLog(const std::string& path, const std::string& source, const std::vector<Incident>& incidents) {
    if (path.empty()) {
        return;
    }

    fs::create_directories(fs::path(path).parent_path());
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write incident log: " + path);
    }

    output << "{\n";
    output << "  \"source\": \"" << escapeJson(source) << "\",\n";
    output << "  \"incidents\": [\n";
    for (size_t index = 0; index < incidents.size(); ++index) {
        const Incident& incident = incidents[index];
        output << "    {\n";
        output << "      \"index\": " << incident.index << ",\n";
        output << "      \"frame\": " << incident.frame << ",\n";
        output << "      \"smoothed_prob\": " << std::fixed << std::setprecision(4) << incident.smoothedProb << "\n";
        output << "    }";
        output << (index + 1 < incidents.size() ? ",\n" : "\n");
    }
    output << "  ]\n";
    output << "}\n";
}

void drawOverlay(cv::Mat& frame, const std::string& label, double smoothedProb, double threshold) {
    const cv::Scalar color = label == "fight" ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    std::ostringstream stream;
    stream << label << " (p=" << std::fixed << std::setprecision(2) << smoothedProb
           << ", thr=" << threshold << ")";
    cv::putText(frame, stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
}

#ifdef _WIN32
std::wstring toWideString(const std::string& value) {
    if (value.empty()) {
        return std::wstring();
    }

    const int requiredSize = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
    if (requiredSize <= 0) {
        throw std::runtime_error("failed to convert UTF-8 path to wide string: " + value);
    }

    std::wstring wide(static_cast<size_t>(requiredSize), L'\0');
    const int converted = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, wide.data(), requiredSize);
    if (converted <= 0) {
        throw std::runtime_error("failed to convert UTF-8 path to wide string: " + value);
    }
    if (!wide.empty() && wide.back() == L'\0') {
        wide.pop_back();
    }
    return wide;
}
#endif

}  // namespace

#ifdef FIGHT_DEMO_HAS_MINDSPORE
struct MindIRClassifier::Impl {
    using ContextCreateFn = decltype(&MSContextCreate);
    using ContextDestroyFn = decltype(&MSContextDestroy);
    using ContextSetThreadNumFn = decltype(&MSContextSetThreadNum);
    using ContextAddDeviceInfoFn = decltype(&MSContextAddDeviceInfo);
    using DeviceInfoCreateFn = decltype(&MSDeviceInfoCreate);
    using DeviceInfoDestroyFn = decltype(&MSDeviceInfoDestroy);
    using ModelCreateFn = decltype(&MSModelCreate);
    using ModelDestroyFn = decltype(&MSModelDestroy);
    using ModelBuildFromFileFn = decltype(&MSModelBuildFromFile);
    using ModelGetInputsFn = decltype(&MSModelGetInputs);
    using ModelPredictFn = decltype(&MSModelPredict);
    using TensorCreateFn = decltype(&MSTensorCreate);
    using TensorDestroyFn = decltype(&MSTensorDestroy);
    using TensorGetNameFn = decltype(&MSTensorGetName);
    using TensorGetDataFn = decltype(&MSTensorGetData);
    using TensorGetDataSizeFn = decltype(&MSTensorGetDataSize);

    HMODULE runtimeDll = nullptr;
    MSContextHandle context = nullptr;
    MSDeviceInfoHandle deviceInfo = nullptr;
    MSModelHandle model = nullptr;
    std::string inputName;

    ContextCreateFn contextCreate = nullptr;
    ContextDestroyFn contextDestroy = nullptr;
    ContextSetThreadNumFn contextSetThreadNum = nullptr;
    ContextAddDeviceInfoFn contextAddDeviceInfo = nullptr;
    DeviceInfoCreateFn deviceInfoCreate = nullptr;
    DeviceInfoDestroyFn deviceInfoDestroy = nullptr;
    ModelCreateFn modelCreate = nullptr;
    ModelDestroyFn modelDestroy = nullptr;
    ModelBuildFromFileFn modelBuildFromFile = nullptr;
    ModelGetInputsFn modelGetInputs = nullptr;
    ModelPredictFn modelPredict = nullptr;
    TensorCreateFn tensorCreate = nullptr;
    TensorDestroyFn tensorDestroy = nullptr;
    TensorGetNameFn tensorGetName = nullptr;
    TensorGetDataFn tensorGetData = nullptr;
    TensorGetDataSizeFn tensorGetDataSize = nullptr;

    ~Impl() {
        if (model != nullptr && modelDestroy != nullptr) {
            modelDestroy(&model);
        }
        if (deviceInfo != nullptr && deviceInfoDestroy != nullptr) {
            deviceInfoDestroy(&deviceInfo);
        }
        if (context != nullptr && contextDestroy != nullptr) {
            contextDestroy(&context);
        }
        if (runtimeDll != nullptr) {
            FreeLibrary(runtimeDll);
        }
    }
};

template <typename T>
T resolveMindSporeSymbol(HMODULE module, const char* symbolName) {
    auto symbol = reinterpret_cast<T>(GetProcAddress(module, symbolName));
    if (symbol == nullptr) {
        throw std::runtime_error(std::string("failed to resolve MindSpore symbol: ") + symbolName);
    }
    return symbol;
}

std::string mindsporeStatusToString(MSStatus status) {
    switch (status) {
    case kMSStatusSuccess:
        return "kMSStatusSuccess";
    case kMSStatusLiteGraphFileError:
        return "kMSStatusLiteGraphFileError";
    case kMSStatusLiteInvalidOpAttr:
        return "kMSStatusLiteInvalidOpAttr";
    case kMSStatusLiteInferError:
        return "kMSStatusLiteInferError";
    case kMSStatusLiteInputTensorError:
        return "kMSStatusLiteInputTensorError";
    default:
        return "MSStatus(" + std::to_string(static_cast<int>(status)) + ")";
    }
}
#endif

#ifdef FIGHT_DEMO_HAS_ONNXRUNTIME
struct OnnxRuntimeClassifier::Impl {
    RuntimeLibraryHandle runtimeDll = nullptr;
    OrtSessionBundle session;

    ~Impl() {
        releaseOrtSessionBundle(session);
        if (runtimeDll != nullptr) {
            unloadRuntimeLibrary(runtimeDll);
        }
    }
};
#endif

PseudoClassifier::PseudoClassifier(int smoothK, double threshold)
    : smoothK_(smoothK), threshold_(threshold) {}

std::string PseudoClassifier::backendName() const {
    return "pseudo";
}

ClassificationResult PseudoClassifier::classify(const SequenceWindow& window) {
    const double rawProb = pseudoProbability(window.lastFrameIndex);
    const double smoothedProb = updateSmoothedProbability(probBuffer_, smoothK_, rawProb);
    const bool isFight = smoothedProb > threshold_;

    return ClassificationResult{
        rawProb,
        smoothedProb,
        isFight,
        isFight ? "fight" : "no-fight",
    };
}

MindIRClassifier::MindIRClassifier(std::string modelPath, int smoothK, double threshold)
    : modelPath_(std::move(modelPath)), smoothK_(smoothK), threshold_(threshold) {
    validateModelPath("mindir", modelPath_);

#ifdef FIGHT_DEMO_HAS_MINDSPORE
    impl_ = std::make_shared<Impl>();
    impl_->runtimeDll = LoadLibraryA(FIGHT_DEMO_MINDSPORE_DLL_PATH);
    if (impl_->runtimeDll == nullptr) {
        throw std::runtime_error(
            std::string("failed to load MindSpore Lite runtime DLL: ") + FIGHT_DEMO_MINDSPORE_DLL_PATH);
    }

    impl_->contextCreate = resolveMindSporeSymbol<Impl::ContextCreateFn>(impl_->runtimeDll, "MSContextCreate");
    impl_->contextDestroy = resolveMindSporeSymbol<Impl::ContextDestroyFn>(impl_->runtimeDll, "MSContextDestroy");
    impl_->contextSetThreadNum =
        resolveMindSporeSymbol<Impl::ContextSetThreadNumFn>(impl_->runtimeDll, "MSContextSetThreadNum");
    impl_->contextAddDeviceInfo =
        resolveMindSporeSymbol<Impl::ContextAddDeviceInfoFn>(impl_->runtimeDll, "MSContextAddDeviceInfo");
    impl_->deviceInfoCreate = resolveMindSporeSymbol<Impl::DeviceInfoCreateFn>(impl_->runtimeDll, "MSDeviceInfoCreate");
    impl_->deviceInfoDestroy =
        resolveMindSporeSymbol<Impl::DeviceInfoDestroyFn>(impl_->runtimeDll, "MSDeviceInfoDestroy");
    impl_->modelCreate = resolveMindSporeSymbol<Impl::ModelCreateFn>(impl_->runtimeDll, "MSModelCreate");
    impl_->modelDestroy = resolveMindSporeSymbol<Impl::ModelDestroyFn>(impl_->runtimeDll, "MSModelDestroy");
    impl_->modelBuildFromFile =
        resolveMindSporeSymbol<Impl::ModelBuildFromFileFn>(impl_->runtimeDll, "MSModelBuildFromFile");
    impl_->modelGetInputs = resolveMindSporeSymbol<Impl::ModelGetInputsFn>(impl_->runtimeDll, "MSModelGetInputs");
    impl_->modelPredict = resolveMindSporeSymbol<Impl::ModelPredictFn>(impl_->runtimeDll, "MSModelPredict");
    impl_->tensorCreate = resolveMindSporeSymbol<Impl::TensorCreateFn>(impl_->runtimeDll, "MSTensorCreate");
    impl_->tensorDestroy = resolveMindSporeSymbol<Impl::TensorDestroyFn>(impl_->runtimeDll, "MSTensorDestroy");
    impl_->tensorGetName = resolveMindSporeSymbol<Impl::TensorGetNameFn>(impl_->runtimeDll, "MSTensorGetName");
    impl_->tensorGetData = resolveMindSporeSymbol<Impl::TensorGetDataFn>(impl_->runtimeDll, "MSTensorGetData");
    impl_->tensorGetDataSize =
        resolveMindSporeSymbol<Impl::TensorGetDataSizeFn>(impl_->runtimeDll, "MSTensorGetDataSize");

    impl_->context = impl_->contextCreate();
    impl_->deviceInfo = impl_->deviceInfoCreate(kMSDeviceTypeCPU);
    impl_->model = impl_->modelCreate();
    if (impl_->context == nullptr || impl_->deviceInfo == nullptr || impl_->model == nullptr) {
        throw std::runtime_error("failed to initialize MindSpore Lite runtime handles.");
    }

    impl_->contextSetThreadNum(impl_->context, 1);
    impl_->contextAddDeviceInfo(impl_->context, impl_->deviceInfo);

    const MSStatus buildStatus = impl_->modelBuildFromFile(
        impl_->model,
        modelPath_.c_str(),
        kMSModelTypeMindIR,
        impl_->context);
    if (buildStatus != kMSStatusSuccess) {
        throw std::runtime_error("failed to build MindIR model: " + mindsporeStatusToString(buildStatus));
    }

    const MSTensorHandleArray inputs = impl_->modelGetInputs(impl_->model);
    if (inputs.handle_num == 0 || inputs.handle_list == nullptr) {
        throw std::runtime_error("MindIR model exposes no input tensors: " + modelPath_);
    }
    const char* inputName = impl_->tensorGetName(inputs.handle_list[0]);
    if (inputName == nullptr) {
        throw std::runtime_error("MindIR model input tensor has no name.");
    }
    impl_->inputName = inputName;
#endif
}

MindIRClassifier::~MindIRClassifier() = default;

std::string MindIRClassifier::backendName() const {
    return "mindir";
}

ClassificationResult MindIRClassifier::classify(const SequenceWindow& window) {
#ifdef FIGHT_DEMO_HAS_MINDSPORE
    if (!impl_) {
        throw std::runtime_error("MindIR runtime state was not initialized.");
    }
    if (window.flattenedFeatures.empty()) {
        throw std::runtime_error("MindIR backend received empty SequenceWindow::flattenedFeatures.");
    }

    const size_t expectedSize = static_cast<size_t>(window.sequenceLength) * static_cast<size_t>(window.featureDim);
    if (window.flattenedFeatures.size() != expectedSize) {
        throw std::runtime_error(
            "MindIR input feature size mismatch. expected " + std::to_string(expectedSize) +
            ", got " + std::to_string(window.flattenedFeatures.size()));
    }

    const std::vector<int64_t> shape = {1, window.sequenceLength, window.featureDim};
    const size_t dataBytes = window.flattenedFeatures.size() * sizeof(float);
    MSTensorHandle inputTensor = impl_->tensorCreate(
        impl_->inputName.c_str(),
        kMSDataTypeNumberTypeFloat32,
        shape.data(),
        shape.size(),
        window.flattenedFeatures.data(),
        dataBytes);
    if (inputTensor == nullptr) {
        throw std::runtime_error("failed to create MindSpore input tensor.");
    }

    MSTensorHandle inputHandles[] = {inputTensor};
    const MSTensorHandleArray inputArray{1, inputHandles};
    MSTensorHandleArray outputs{0, nullptr};
    const MSStatus predictStatus = impl_->modelPredict(impl_->model, inputArray, &outputs, nullptr, nullptr);
    impl_->tensorDestroy(&inputTensor);
    if (predictStatus != kMSStatusSuccess) {
        throw std::runtime_error("MindIR predict failed: " + mindsporeStatusToString(predictStatus));
    }
    if (outputs.handle_num == 0 || outputs.handle_list == nullptr) {
        throw std::runtime_error("MindIR predict returned no outputs.");
    }

    const size_t outputBytes = impl_->tensorGetDataSize(outputs.handle_list[0]);
    const void* outputData = impl_->tensorGetData(outputs.handle_list[0]);
    if (outputData == nullptr || outputBytes < sizeof(float)) {
        for (size_t index = 0; index < outputs.handle_num; ++index) {
            if (outputs.handle_list[index] != nullptr) {
                impl_->tensorDestroy(&outputs.handle_list[index]);
            }
        }
        throw std::runtime_error("MindIR output tensor has no accessible host data.");
    }

    const float* rawPtr = static_cast<const float*>(outputData);
    const double rawProb = static_cast<double>(rawPtr[0]);
    for (size_t index = 0; index < outputs.handle_num; ++index) {
        if (outputs.handle_list[index] != nullptr) {
            impl_->tensorDestroy(&outputs.handle_list[index]);
        }
    }

    const double smoothedProb = updateSmoothedProbability(probBuffer_, smoothK_, rawProb);
    const bool isFight = smoothedProb > threshold_;

    return ClassificationResult{
        rawProb,
        smoothedProb,
        isFight,
        isFight ? "fight" : "no-fight",
    };
#else
    (void)window;
    throw std::runtime_error(
        "mindir backend requested, but this binary was built without MindSpore C++ support. "
        "Reconfigure cpp_demo with a valid MINDSPORE_ROOT so CMake can generate and link the import library.");
#endif
}

OnnxRuntimeClassifier::OnnxRuntimeClassifier(
    std::string modelPath,
    int smoothK,
    double threshold,
    std::string onnxEp,
    int onnxDeviceId)
    : modelPath_(std::move(modelPath)),
      smoothK_(smoothK),
      threshold_(threshold),
      onnxEp_(std::move(onnxEp)),
      onnxDeviceId_(onnxDeviceId) {
    validateModelPath("onnx", modelPath_);

    std::transform(
        onnxEp_.begin(),
        onnxEp_.end(),
        onnxEp_.begin(),
        [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

#ifdef FIGHT_DEMO_HAS_ONNXRUNTIME
    impl_ = std::make_shared<Impl>();
    impl_->runtimeDll = loadRuntimeLibrary(FIGHT_DEMO_ONNXRUNTIME_LIBRARY_PATH);
    if (impl_->runtimeDll == nullptr) {
        throw std::runtime_error(
            std::string("failed to load ONNX Runtime shared library: ") + FIGHT_DEMO_ONNXRUNTIME_LIBRARY_PATH);
    }
    initializeOrtSessionBundle(impl_->runtimeDll, modelPath_, "fight_detection_demo", onnxEp_, onnxDeviceId_, impl_->session);
#endif
}

OnnxRuntimeClassifier::~OnnxRuntimeClassifier() = default;

std::string OnnxRuntimeClassifier::backendName() const {
    return "onnx";
}

ClassificationResult OnnxRuntimeClassifier::classify(const SequenceWindow& window) {
#ifdef FIGHT_DEMO_HAS_ONNXRUNTIME
    if (!impl_) {
        throw std::runtime_error("ONNX Runtime state was not initialized.");
    }
    if (window.flattenedFeatures.empty()) {
        throw std::runtime_error("ONNX backend received empty SequenceWindow::flattenedFeatures.");
    }

    const size_t expectedSize = static_cast<size_t>(window.sequenceLength) * static_cast<size_t>(window.featureDim);
    if (window.flattenedFeatures.size() != expectedSize) {
        throw std::runtime_error(
            "ONNX backend received mismatched flattened feature size. expected=" + std::to_string(expectedSize) +
            " actual=" + std::to_string(window.flattenedFeatures.size()));
    }

    std::array<int64_t, 3> inputShape{
        1,
        static_cast<int64_t>(window.sequenceLength),
        static_cast<int64_t>(window.featureDim),
    };

    std::vector<float> inputBuffer = window.flattenedFeatures;
    // --- Normalization Patch Start ---
    const int seqLen = window.sequenceLength;
    const int featDim = window.featureDim;
    if (seqLen > 0 && featDim > 0 && inputBuffer.size() == static_cast<size_t>(seqLen * featDim)) {
        for (int d = 0; d < featDim; ++d) {
            float minVal = 1e30f; // Use simple large val
            float maxVal = -1e30f;
            for (int t = 0; t < seqLen; ++t) {
                float val = inputBuffer[t * featDim + d];
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }
            float range = maxVal - minVal;
            if (range < 1e-6f) range = 1.0f;
            for (int t = 0; t < seqLen; ++t) {
                inputBuffer[t * featDim + d] = (inputBuffer[t * featDim + d] - minVal) / range;
            }
        }
    }
    // --- Normalization Patch End ---
    OrtValue* inputTensor = nullptr;
    throwOnnxStatus(
        impl_->session.api,
        impl_->session.api->CreateTensorWithDataAsOrtValue(
            impl_->session.memoryInfo,
            inputBuffer.data(),
            inputBuffer.size() * sizeof(float),
            inputShape.data(),
            inputShape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &inputTensor),
        "CreateTensorWithDataAsOrtValue failed");

    const char* inputNames[] = {impl_->session.inputName.c_str()};
    const char* outputNames[] = {impl_->session.outputName.c_str()};
    OrtValue* outputTensor = nullptr;
    throwOnnxStatus(
        impl_->session.api,
        impl_->session.api->Run(
            impl_->session.session,
            nullptr,
            inputNames,
            reinterpret_cast<const OrtValue* const*>(&inputTensor),
            1,
            outputNames,
            1,
            &outputTensor),
        "ONNX Runtime Run failed");

    float* rawOutput = nullptr;
    throwOnnxStatus(
        impl_->session.api,
        impl_->session.api->GetTensorMutableData(outputTensor, reinterpret_cast<void**>(&rawOutput)),
        "GetTensorMutableData failed");
    if (rawOutput == nullptr) {
        impl_->session.api->ReleaseValue(outputTensor);
        impl_->session.api->ReleaseValue(inputTensor);
        throw std::runtime_error("ONNX Runtime output tensor had no data.");
    }

    const double rawProb = static_cast<double>(rawOutput[0]);
    impl_->session.api->ReleaseValue(outputTensor);
    impl_->session.api->ReleaseValue(inputTensor);

    const double smoothedProb = updateSmoothedProbability(probBuffer_, smoothK_, rawProb);
    const bool isFight = smoothedProb > threshold_;
    return ClassificationResult{
        rawProb,
        smoothedProb,
        isFight,
        isFight ? "fight" : "no-fight",
    };
#else
    (void)window;
    throw std::runtime_error(
        "onnx backend requested, but this binary was built without ONNX Runtime support. "
        "Reconfigure cpp_demo with valid ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY paths.");
#endif
}

void printUsage() {
    std::cout
        << "Minimal C++ + OpenCV demo for the fight-detection pipeline\n\n"
        << "Usage:\n"
        << "  fight_detection_demo --video-path <path> [options]\n"
        << "  fight_detection_demo --source-type camera --camera-index 0 [options]\n\n"
        << "Options:\n"
        << "  --source-type <video|camera>\n"
        << "  --video-path <path>\n"
        << "  --backend <pseudo|mindir|onnx>\n"
        << "  --model-path <path>\n"
        << "  --feature-source <synthetic|csv|yolo-onnx>\n"
        << "  --feature-file <path>\n"
        << "  --feature-dump <path>\n"
        << "  --pose-model-path <path>\n"
        << "  --pose-debug-frame <int>\n"
        << "  --pose-debug-json <path>\n"
        << "  --camera-index <int>\n"
        << "  --max-frames <int>\n"
        << "  --log-every <int>\n"
        << "  --sequence-length <int>\n"
        << "  --feature-dim <int>\n"
        << "  --smooth-k <int>\n"
        << "  --threshold <float>\n"
        << "  --output-video <path>\n"
        << "  --show-window\n"
        << "  --event-log <path>\n"
        << "  --summary-json <path>\n"
        << "  --onnx-ep <cpu|cann>\n"
        << "  --onnx-device-id <int>\n";
}

Config parseArgs(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (hasArg(args, "--help") || hasArg(args, "-h")) {
        printUsage();
        std::exit(0);
    }

    Config config;
    if (auto value = getArgValue(args, "--source-type")) {
        config.sourceType = *value;
    }
    if (auto value = getArgValue(args, "--video-path")) {
        config.videoPath = *value;
    }
    if (auto value = getArgValue(args, "--backend")) {
        config.backend = *value;
    }
    if (auto value = getArgValue(args, "--model-path")) {
        config.modelPath = *value;
    }
    if (auto value = getArgValue(args, "--feature-source")) {
        config.featureSource = *value;
    }
    if (auto value = getArgValue(args, "--feature-file")) {
        config.featureFile = *value;
    }
    if (auto value = getArgValue(args, "--feature-dump")) {
        config.featureDump = *value;
    }
    if (auto value = getArgValue(args, "--pose-model-path")) {
        config.poseModelPath = *value;
    }
    if (auto value = getArgValue(args, "--pose-debug-frame")) {
        config.poseDebugFrame = std::stoi(*value);
    }
    if (auto value = getArgValue(args, "--pose-debug-json")) {
        config.poseDebugJson = *value;
    }
    if (auto value = getArgValue(args, "--camera-index")) {
        config.cameraInput = *value;
    }
    if (auto value = getArgValue(args, "--max-frames")) {
        config.maxFrames = std::stoi(*value);
    }
    if (auto value = getArgValue(args, "--log-every")) {
        config.logEvery = std::stoi(*value);
    }
    if (auto value = getArgValue(args, "--sequence-length")) {
        config.sequenceLength = std::stoi(*value);
    }
    if (auto value = getArgValue(args, "--feature-dim")) {
        config.featureDim = std::stoi(*value);
    }
    if (auto value = getArgValue(args, "--smooth-k")) {
        config.smoothK = std::stoi(*value);
    }
    if (auto value = getArgValue(args, "--threshold")) {
        config.threshold = std::stod(*value);
    }
    if (auto value = getArgValue(args, "--output-video")) {
        config.outputVideo = *value;
    }
    if (auto value = getArgValue(args, "--event-log")) {
        config.eventLog = *value;
    }
    if (auto value = getArgValue(args, "--summary-json")) {
        config.summaryJson = *value;
    }
    if (auto value = getArgValue(args, "--onnx-ep")) {
        config.onnxEp = *value;
    }
    if (auto value = getArgValue(args, "--onnx-device-id")) {
        config.onnxDeviceId = std::stoi(*value);
    }
    if (hasArg(args, "--show-window")) {
        config.showWindow = true;
    }

    if (config.sourceType == "video" && config.videoPath.empty()) {
        throw std::runtime_error("source-type=video requires --video-path");
    }
    if (config.featureSource != "synthetic" && config.featureSource != "csv" && config.featureSource != "yolo-onnx") {
        throw std::runtime_error("feature-source must be one of: synthetic, csv, yolo-onnx");
    }
    if (config.featureSource == "csv" && config.featureFile.empty()) {
        throw std::runtime_error("feature-source=csv requires --feature-file");
    }
    if (config.featureSource == "yolo-onnx" && config.poseModelPath.empty()) {
        throw std::runtime_error("feature-source=yolo-onnx requires --pose-model-path");
    }
    if (!config.poseDebugJson.empty() && config.poseDebugFrame <= 0) {
        throw std::runtime_error("pose-debug-json requires pose-debug-frame > 0");
    }
    if (config.onnxEp != "cpu" && config.onnxEp != "cann") {
        throw std::runtime_error("onnx-ep must be one of: cpu, cann");
    }

    return config;
}

Summary runDemo(const Config& config) {
    std::string sourceDesc;
    VideoCaptureThread cap;
    if (!cap.open(config, sourceDesc)) {
        throw std::runtime_error("failed to open source: " + sourceDesc);
    }

    std::deque<std::vector<float>> frameBuffer;
    std::vector<Incident> incidents;
    cv::VideoWriter writer;
    bool writerInitialized = false;
    std::unique_ptr<ISequenceClassifier> classifier = createClassifier(config);
    std::vector<std::vector<float>> featureRows;
    std::ofstream featureDumpStream;
#ifdef FIGHT_DEMO_HAS_ONNXRUNTIME
    std::unique_ptr<YoloPoseOnnxExtractor> poseExtractor;
#endif
    if (config.featureSource == "csv") {
        featureRows = loadFeatureCsv(config.featureFile, config.featureDim);
    }
    if (!config.featureDump.empty()) {
        fs::create_directories(fs::path(config.featureDump).parent_path());
        featureDumpStream.open(config.featureDump, std::ios::out | std::ios::trunc);
        if (!featureDumpStream.is_open()) {
            throw std::runtime_error("failed to open feature dump csv: " + config.featureDump);
        }
        writeFeatureCsvHeader(featureDumpStream, config.featureDim);
    }
#ifdef FIGHT_DEMO_HAS_ONNXRUNTIME
    if (config.featureSource == "yolo-onnx") {
        poseExtractor = std::make_unique<YoloPoseOnnxExtractor>(config.poseModelPath, config.onnxEp, config.onnxDeviceId);
    }
#else
    if (config.featureSource == "yolo-onnx") {
        throw std::runtime_error(
            "feature-source=yolo-onnx requested, but this binary was built without ONNX Runtime support.");
    }
#endif

    int processedFrames = 0;
    int incidentCount = 0;
    auto start = std::chrono::steady_clock::now();
    auto lastIncident = start - std::chrono::seconds(10);

    cv::Mat frame;
    // --- Smart Recording & Visualization --- 
    std::deque<cv::Mat> preEventBuffer;
    const int preEventFrames = 60; // 3 seconds @ 20fps
    const int postEventFrames = 100; // 5 seconds @ 20fps
    int framesSinceLastFight = 99999;
    bool isRecording = false;
    cv::VideoWriter eventWriter;
    int currentEventId = 0;
    // ----------------------------------------
    while (true) {
        bool success = cap.read(frame);
        if (!success) {
            bool isStream = (config.sourceType == "video" && (config.videoPath.find("rtsp://") == 0 || config.videoPath.find("rtmp://") == 0 || config.videoPath.find("http://") == 0));
            if (isStream) {
                std::cout << "[warn] Stream disconnected. Reconnecting in 3s..." << std::endl;
                cap.release();
                std::this_thread::sleep_for(std::chrono::seconds(3));
                std::string tempDesc;
                cap.open(config, tempDesc);
                continue;
            } else {
                break;
            }
        }
        ++processedFrames;

        std::vector<float> frameFeatures;
        if (config.featureSource == "csv") {
            const size_t featureIndex = static_cast<size_t>(processedFrames - 1);
            if (featureIndex >= featureRows.size()) {
                throw std::runtime_error(
                    "feature csv does not have enough rows for frame " + std::to_string(processedFrames));
            }
            frameFeatures = featureRows[featureIndex];
        } else if (config.featureSource == "yolo-onnx") {
#ifdef FIGHT_DEMO_HAS_ONNXRUNTIME
            const PoseFrameResult poseResult = poseExtractor->extractFrameFeatures(frame, config.featureDim);
            frameFeatures = poseResult.frameFeatures;
            if (!config.poseDebugJson.empty() && processedFrames == config.poseDebugFrame) {
                writePoseDebugJson(config.poseDebugJson, processedFrames, frame, poseResult.detections);
            }
#else
            throw std::runtime_error(
                "feature-source=yolo-onnx requested, but this binary was built without ONNX Runtime support.");
#endif
        } else {
            frameFeatures = generateSyntheticFrameFeatures(processedFrames, config.featureDim);
        }

        if (featureDumpStream.is_open()) {
            writeFeatureCsvRow(featureDumpStream, processedFrames, frameFeatures);
        }

        frameBuffer.push_back(std::move(frameFeatures));
        if (static_cast<int>(frameBuffer.size()) > config.sequenceLength) {
            frameBuffer.pop_front();
        }

        std::string label = "warming-up";
        double rawProb = -1.0;
        double smoothedProb = -1.0;

        if (static_cast<int>(frameBuffer.size()) == config.sequenceLength) {
            SequenceWindow window;
            window.lastFrameIndex = processedFrames;
            window.sequenceLength = config.sequenceLength;
            window.featureDim = config.featureDim;
            window.flattenedFeatures.reserve(static_cast<size_t>(config.sequenceLength * config.featureDim));
            for (const auto& frameFeature : frameBuffer) {
                window.flattenedFeatures.insert(
                    window.flattenedFeatures.end(),
                    frameFeature.begin(),
                    frameFeature.end());
            }

            const ClassificationResult result = classifier->classify(window);
            rawProb = result.rawProb;
            smoothedProb = result.smoothedProb;
            const bool isFight = result.isFight;
            label = result.label;
            drawOverlay(frame, label, smoothedProb, config.threshold);
            // --- Smart Recording Logic --- 
            if (isFight) {
                framesSinceLastFight = 0;
                if (!isRecording) {
                    isRecording = true;
                    currentEventId++;
                    std::string filename = "event_" + std::to_string(currentEventId) + ".mp4";
                    if (!config.eventLog.empty()) {
                         std::filesystem::path logPath(config.eventLog);
                         if (logPath.has_parent_path()) {
                             filename = (logPath.parent_path() / ("event_" + std::to_string(currentEventId) + ".mp4")).string();
                         }
                    }
                    std::cout << "[info] Starting recording: " << filename << std::endl;
                    eventWriter.open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20.0, frame.size());
                    for (const auto& bufFrame : preEventBuffer) {
                        if (eventWriter.isOpened()) eventWriter.write(bufFrame);
                    }
                    preEventBuffer.clear();
                }
            } else {
                framesSinceLastFight++;
            }
            if (isRecording) {
                if (eventWriter.isOpened()) eventWriter.write(frame);
                if (framesSinceLastFight > postEventFrames) {
                    std::cout << "[info] Stopping recording (event finished)." << std::endl;
                    isRecording = false;
                    if (eventWriter.isOpened()) eventWriter.release();
                }
            } else {
                preEventBuffer.push_back(frame.clone());
                if (preEventBuffer.size() > preEventFrames) preEventBuffer.pop_front();
            }
            // --- Visualization Logic (Shared Memory Export) ---
            if (processedFrames % 2 == 0) {
                cv::Mat smallFrame; cv::resize(frame, smallFrame, cv::Size(640, 360)); cv::imwrite("/dev/shm/preview.jpg", smallFrame);
                std::ofstream statusFile("/dev/shm/status.json");
                statusFile << "{\"isFight\": " << (isFight ? "true" : "false") << ", \"prob\": " << smoothedProb << ", \"label\": \"" << label << "\"}";
            }

            const auto now = std::chrono::steady_clock::now();
            const auto cooldown = std::chrono::seconds(5);
            if (isFight && (now - lastIncident) > cooldown) {
                ++incidentCount;
                lastIncident = now;
                incidents.push_back(Incident{incidentCount, processedFrames, smoothedProb});
                std::cout << "[incident] #" << incidentCount
                          << " frame=" << processedFrames
                          << " smoothed_prob=" << std::fixed << std::setprecision(4) << smoothedProb
                          << "\n";
            }
        }

        if (!writerInitialized && !config.outputVideo.empty()) {
            writer = buildWriter(config, cap, frame);
            writerInitialized = writer.isOpened();
        }
        if (writerInitialized) {
            writer.write(frame);
        }

        if (config.showWindow) {
            cv::imshow("Fight Detection Demo", frame);
            if (cv::waitKey(1) == 27) { // ESC
                break;
            }
        }

        if (processedFrames == 1 || (config.logEvery > 0 && processedFrames % config.logEvery == 0)) {
            if (rawProb < 0.0) {
                std::cout << "[frame " << processedFrames << "] collecting sequence buffer: "
                          << frameBuffer.size() << "/" << config.sequenceLength << "\n";
            } else {
                std::cout << "[frame " << processedFrames << "] label=" << label
                          << " prob=" << std::fixed << std::setprecision(4) << rawProb
                          << " smoothed=" << smoothedProb
                          << " incidents=" << incidentCount << "\n";
            }
        }

        if (config.maxFrames > 0 && processedFrames >= config.maxFrames) {
            break;
        }
    }

    cap.release();
    if (writerInitialized) {
        writer.release();
    }

    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    const double fps = elapsed > 0.0 ? static_cast<double>(processedFrames) / elapsed : 0.0;

    std::cout << "\nInference summary\n";
    std::cout << "source: " << sourceDesc << "\n";
    std::cout << "backend: " << classifier->backendName() << "\n";
    std::cout << "processed_frames: " << processedFrames << "\n";
    std::cout << "incident_count: " << incidentCount << "\n";
    std::cout << "elapsed_seconds: " << std::fixed << std::setprecision(2) << elapsed << "\n";
    std::cout << "avg_fps: " << std::fixed << std::setprecision(2) << fps << "\n";
    if (!config.outputVideo.empty()) {
        std::cout << "output_video: " << config.outputVideo << "\n";
    }
    if (!config.eventLog.empty()) {
        std::cout << "event_log: " << config.eventLog << "\n";
    }
    if (!config.summaryJson.empty()) {
        std::cout << "summary_json: " << config.summaryJson << "\n";
    }

    Summary summary{sourceDesc, processedFrames, incidentCount, elapsed, fps, config.outputVideo, config.eventLog, incidents};
    writeIncidentLog(config.eventLog, sourceDesc, incidents);
    writeSummaryJson(config.summaryJson, summary);
    return summary;
}
