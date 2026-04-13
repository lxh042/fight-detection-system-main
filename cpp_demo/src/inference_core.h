#pragma once

#include <deque>
#include <memory>
#include <string>
#include <vector>

struct Config {
    std::string sourceType = "video";
    std::string videoPath;
    std::string backend = "pseudo";
    std::string modelPath;
    std::string featureSource = "synthetic";
    std::string featureFile;
    std::string featureDump;
    std::string poseModelPath;
    std::string poseDebugJson;
    int cameraIndex = 0;
    int maxFrames = -1;
    int poseDebugFrame = -1;
    int logEvery = 30;
    int sequenceLength = 41;
    int featureDim = 153;
    int smoothK = 5;
    double threshold = 0.5;
    std::string outputVideo;
    std::string eventLog;
    std::string summaryJson;
    std::string onnxEp = "cpu";
    int onnxDeviceId = 0;
};

struct Incident {
    int index = 0;
    int frame = 0;
    double smoothedProb = 0.0;
};

struct ClassificationResult {
    double rawProb = 0.0;
    double smoothedProb = 0.0;
    bool isFight = false;
    std::string label = "warming-up";
};

struct SequenceWindow {
    int lastFrameIndex = 0;
    int sequenceLength = 0;
    int featureDim = 0;
    std::vector<float> flattenedFeatures;
};

struct Summary {
    std::string source;
    int processedFrames = 0;
    int incidentCount = 0;
    double elapsedSeconds = 0.0;
    double avgFps = 0.0;
    std::string outputVideo;
    std::string eventLog;
    std::vector<Incident> incidents;
};

class ISequenceClassifier {
public:
    virtual ~ISequenceClassifier() = default;
    virtual std::string backendName() const = 0;
    virtual ClassificationResult classify(const SequenceWindow& window) = 0;
};

class PseudoClassifier : public ISequenceClassifier {
public:
    PseudoClassifier(int smoothK, double threshold);
    std::string backendName() const override;
    ClassificationResult classify(const SequenceWindow& window) override;

private:
    int smoothK_;
    double threshold_;
    std::deque<double> probBuffer_;
};

class MindIRClassifier : public ISequenceClassifier {
public:
    MindIRClassifier(std::string modelPath, int smoothK, double threshold);
    ~MindIRClassifier() override;
    std::string backendName() const override;
    ClassificationResult classify(const SequenceWindow& window) override;

private:
    struct Impl;
    std::string modelPath_;
    int smoothK_;
    double threshold_;
    std::deque<double> probBuffer_;
    std::shared_ptr<Impl> impl_;
};

class OnnxRuntimeClassifier : public ISequenceClassifier {
public:
    OnnxRuntimeClassifier(std::string modelPath, int smoothK, double threshold, std::string onnxEp, int onnxDeviceId);
    ~OnnxRuntimeClassifier() override;
    std::string backendName() const override;
    ClassificationResult classify(const SequenceWindow& window) override;

private:
    struct Impl;
    std::string modelPath_;
    int smoothK_;
    double threshold_;
    std::string onnxEp_;
    int onnxDeviceId_;
    std::deque<double> probBuffer_;
    std::shared_ptr<Impl> impl_;
};

Config parseArgs(int argc, char** argv);
void printUsage();
Summary runDemo(const Config& config);
