#include "logger.h"
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>

static std::ofstream log_file;
static std::mutex log_mutex;

std::string currentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto itt = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&itt), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void initLogger(const std::string &filename) {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (!log_file.is_open()) {
        log_file.open(filename, std::ios::app);
        if (log_file.is_open()) {
            log_file << std::fixed << std::setprecision(15);
            log_file << "========== New Run @ " << currentTimestamp() << " ==========\n";
        }
    }
}

void logInfo(const std::string &message) {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file.is_open()) {
        log_file << "[" << currentTimestamp() << "] " << message << "\n";
    }
}

void logResult(const std::string &backend, int num_taxa, int num_sites, int num_patterns, double time_sec,
               double likelihood) {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file.is_open()) {
        log_file << "[RESULT] backend=" << backend
                 << " taxa=" << num_taxa
                 << " sites=" << num_sites
                 << " patterns=" << num_patterns
                 << " time=" << time_sec << "s"
                 << " likelihood=" << likelihood << "\n";
    }
}

void closeLogger() {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file.is_open()) {
        log_file << "========== End Run ==========\n\n";
        log_file.close();
    }
}
