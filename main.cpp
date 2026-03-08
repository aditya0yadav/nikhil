#include "MatchService.hpp"
#include <csignal>
#include <memory>

// ── Global service pointer for signal handler ──
static std::unique_ptr<MatchService> g_service;

void handleSignal(int) {
    std::cout << "\n[LOG] Interrupt received – stopping...\n";
    if (g_service) g_service->stop();
}

int main() {
    // Graceful Ctrl+C / SIGTERM shutdown
    std::signal(SIGINT,  handleSignal);
    std::signal(SIGTERM, handleSignal);

    std::cout << "=========================================\n";
    std::cout << "  Vision Match System – Standalone Mode  \n";
    std::cout << "=========================================\n";
    std::cout << "  Reference : " << REFERENCE_IMAGE  << "\n";
    std::cout << "  Library   : " << LIBRARY_DIR      << "\n";
    std::cout << "  Frames out: " << OUTPUT_IMAGE_DIR << "\n";
    std::cout << "  CSV log   : " << CSV_PATH         << "\n";
    std::cout << "  Interval  : " << CAPTURE_INTERVAL_SEC << "s\n";
    std::cout << "-----------------------------------------\n";
    std::cout << "  Press Ctrl+C to stop.\n\n";

    g_service = std::make_unique<MatchService>();
    g_service->run();   // blocking loop

    std::cout << "[LOG] Exited cleanly.\n";
    return 0;
}