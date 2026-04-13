#include <iostream>
#include "inference_core.h"

int main(int argc, char** argv) {
    try {
        const Config config = parseArgs(argc, argv);
        runDemo(config);
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << "\n";
        return 1;
    }
}
