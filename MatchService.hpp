#ifndef MatchService_hpp
#define MatchService_hpp

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <atomic>
#include <thread>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────
//  Paths  – edit these to match your layout
// ─────────────────────────────────────────────
static const std::string REFERENCE_IMAGE  = "../test/input/test1.png";
static const std::string OUTPUT_IMAGE_DIR = "../test/output/frames";
static const std::string LIBRARY_DIR      = "../test/output/library";
static const std::string CSV_PATH         = "../test/output/results.csv";
static const int         CAPTURE_INTERVAL_SEC = 2;   // seconds between captures
// ─────────────────────────────────────────────

struct MatchResult {
    std::string frameName;
    std::string savedFramePath;
    std::string bestMatchFile;
    double      confidence;
    int         goodMatches;
    int         totalKeypoints;
    std::string timestamp;
};

// ── helpers ──────────────────────────────────

static std::string currentTimestamp() {
    auto now   = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

static void ensureDir(const std::string& path) {
    fs::create_directories(path);
}

static void writeCsvHeader(const std::string& path) {
    // Only write header if file doesn't exist yet
    if (fs::exists(path)) return;
    std::ofstream f(path);
    f << "frame_name,timestamp,best_match_file,confidence_%,"
         "good_keypoint_matches,total_keypoints_in_frame,saved_frame_path\n";
}

static void appendCsvRow(const std::string& path, const MatchResult& r) {
    std::ofstream f(path, std::ios::app);
    f << r.frameName          << ","
      << r.timestamp          << ","
      << r.bestMatchFile      << ","
      << std::fixed << std::setprecision(2) << r.confidence << ","
      << r.goodMatches        << ","
      << r.totalKeypoints     << ","
      << r.savedFramePath     << "\n";
}

// ─────────────────────────────────────────────
//  MatchService
// ─────────────────────────────────────────────

class MatchService {
public:
    MatchService() : m_frameIndex(0), m_running(false) {
        ensureDir(OUTPUT_IMAGE_DIR);
        ensureDir(LIBRARY_DIR);
        writeCsvHeader(CSV_PATH);

        // Pre-load reference image once
        cv::Mat ref = cv::imread(REFERENCE_IMAGE, cv::IMREAD_COLOR);
        if (ref.empty()) {
            std::cerr << "[ERROR] Reference image not found: " << REFERENCE_IMAGE << "\n";
        } else {
            cv::resize(ref, m_reference, cv::Size(), 0.5, 0.5);
            m_orb = cv::ORB::create(1500);
            m_orb->detectAndCompute(m_reference, cv::noArray(), m_refKp, m_refDesc);
            std::cout << "[LOG] Reference image loaded. Keypoints: " << m_refKp.size() << "\n";
        }
    }

    // ── Main blocking loop ────────────────────
    void run() {
        cv::VideoCapture cap(0, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        if (!cap.isOpened()) {
            std::cerr << "[ERROR] Cannot open camera.\n";
            return;
        }
        std::cout << "[LOG] Camera opened. Starting capture loop...\n";
        m_running = true;

        while (m_running) {
            cv::Mat frame = captureFrame(cap);
            if (frame.empty()) {
                std::cerr << "[WARN] Empty frame – retrying...\n";
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            ++m_frameIndex;
            std::string frameName = "frame" + std::to_string(m_frameIndex);
            std::string savePath  = OUTPUT_IMAGE_DIR + "/" + frameName + ".jpg";

            // Save the raw captured frame
            cv::imwrite(savePath, frame);
            std::cout << "\n[LOG] Captured " << frameName << " → " << savePath << "\n";

            // Run ORB matching against library
            MatchResult result = matchFrame(frame, frameName, savePath);

            // Print summary
            std::cout << "[RESULT] Best match: " << result.bestMatchFile
                      << "  Confidence: "        << result.confidence << "%"
                      << "  Good matches: "      << result.goodMatches
                      << "  Keypoints: "         << result.totalKeypoints << "\n";

            // Append to CSV
            appendCsvRow(CSV_PATH, result);
            std::cout << "[LOG] CSV updated → " << CSV_PATH << "\n";

            // Wait before next capture
            std::this_thread::sleep_for(std::chrono::seconds(CAPTURE_INTERVAL_SEC));
        }
    }

    void stop() { m_running = false; }

private:
    // ── Flush V4L2 buffer & grab fresh frame ──
    static cv::Mat captureFrame(cv::VideoCapture& cap) {
        for (int i = 0; i < 5; ++i) cap.grab();   // discard stale buffered frames
        cv::Mat frame;
        cap.retrieve(frame);
        return frame;
    }

    // ── ORB match one captured frame ──────────
    MatchResult matchFrame(const cv::Mat& rawFrame,
                           const std::string& frameName,
                           const std::string& savePath) {
        MatchResult result;
        result.frameName      = frameName;
        result.savedFramePath = savePath;
        result.timestamp      = currentTimestamp();
        result.bestMatchFile  = "None";
        result.confidence     = 0.0;
        result.goodMatches    = 0;
        result.totalKeypoints = 0;

        if (m_refDesc.empty()) {
            result.bestMatchFile = "Error: reference not loaded";
            return result;
        }

        // Resize captured frame to match reference scale
        cv::Mat frame;
        cv::resize(rawFrame, frame, cv::Size(), 0.5, 0.5);

        // Detect keypoints in the live frame
        std::vector<cv::KeyPoint> frameKp;
        cv::Mat frameDesc;
        m_orb->detectAndCompute(frame, cv::noArray(), frameKp, frameDesc);
        result.totalKeypoints = static_cast<int>(frameKp.size());

        if (frameDesc.empty()) {
            result.bestMatchFile = "Error: no keypoints in frame";
            return result;
        }

        // Compare frame against every image in the library
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        double maxConfidence = 0.0;

        for (const auto& entry : fs::directory_iterator(LIBRARY_DIR)) {
            if (!entry.is_regular_file()) continue;

            cv::Mat libImg = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
            if (libImg.empty()) continue;

            cv::Mat libResized;
            cv::resize(libImg, libResized, cv::Size(), 0.5, 0.5);

            std::vector<cv::KeyPoint> libKp;
            cv::Mat libDesc;
            m_orb->detectAndCompute(libResized, cv::noArray(), libKp, libDesc);
            if (libDesc.empty()) continue;

            // KNN match: frame vs library image
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher.knnMatch(frameDesc, libDesc, knnMatches, 2);

            // Lowe's ratio test
            std::vector<cv::DMatch> good;
            for (auto& m : knnMatches) {
                if (m.size() < 2) continue;
                if (m[0].distance < 0.75f * m[1].distance)
                    good.push_back(m[0]);
            }

            double confidence = 0.0;
            if (good.size() >= 10) {
                std::vector<cv::Point2f> pts1, pts2;
                for (auto& m : good) {
                    pts1.push_back(frameKp[m.queryIdx].pt);
                    pts2.push_back(libKp[m.trainIdx].pt);
                }
                cv::Mat mask;
                cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask);
                if (!H.empty()) {
                    double inliers = cv::countNonZero(mask);
                    confidence = (inliers / static_cast<double>(pts1.size())) * 100.0;
                }
            }

            std::string libName = entry.path().filename().string();
            std::cout << "  [SCAN] " << std::left << std::setw(25) << libName
                      << " | Good: " << std::setw(4) << good.size()
                      << " | Confidence: " << std::fixed << std::setprecision(2)
                      << confidence << "%\n";

            if (confidence > maxConfidence) {
                maxConfidence        = confidence;
                result.bestMatchFile = libName;
                result.goodMatches   = static_cast<int>(good.size());
                result.confidence    = confidence;
            }
        }

        return result;
    }

    // ── Members ───────────────────────────────
    cv::Mat                      m_reference;
    cv::Ptr<cv::ORB>             m_orb;
    std::vector<cv::KeyPoint>    m_refKp;
    cv::Mat                      m_refDesc;
    int                          m_frameIndex;
    std::atomic<bool>            m_running;
};

#endif