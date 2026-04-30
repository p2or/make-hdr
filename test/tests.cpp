#define CATCH_CONFIG_MAIN
#include "../modules/armadillo/tests2/catch.hpp"
#include <vector>
#include <memory>
#include <iostream>

#include "../source/resources.h"
#include "../source/solver.h"

// Dummy Image structure to mock OpenFX without linkage dependencies!
struct MockImage {
    std::vector<float> data;
    int width, height;

    MockImage(int w, int h) : width(w), height(h) {
        data.resize(width * height * 4, 0.0f);
    }

    void setPixel(int x, int y, float r, float g, float b) {
        int idx = (y * width + x) * 4;
        data[idx]     = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
    }

    // This is the EXACT signature the solver templates are looking for now
    void* getPixelAddress(int x, int y) {
        if (x < 0 || x >= width || y < 0 || y >= height) return nullptr;
        return &data[(y * width + x) * 4];
    }
};

TEST_CASE("Debevec and Robertson Kernels are Mathematically Consistent", "[solvers]") {
    
    const int input_depth = 256;
    int width = 2;
    int height = 2;

    std::vector<std::shared_ptr<MockImage>> sources;
    std::vector<float> exp_times = { 1.0f, 2.0f, 4.0f };
    std::vector<float> exp_times_log = { 0.0f, std::log(2.0f), std::log(4.0f) };

    // Fill Mock Images with a linear ramp
    for(size_t i = 0; i < exp_times.size(); ++i) {
        auto img = std::make_shared<MockImage>(width, height);
        img->setPixel(0, 0, 0.1f * exp_times[i], 0.0f, 0.0f);
        img->setPixel(1, 0, 0.2f * exp_times[i], 0.0f, 0.0f);
        img->setPixel(0, 1, 0.4f * exp_times[i], 0.0f, 0.0f);
        img->setPixel(1, 1, 0.8f * exp_times[i], 0.0f, 0.0f);
        sources.push_back(img);
    }

    std::vector<fx::point> points = { {0,0}, {1,0}, {0,1}, {1,1} };

    // Constant weighting for test simplicity
    std::vector<float> input_weights(input_depth, 1.0f);

    SECTION("Robertson Solver Outputs Logarithmic Response") {
        std::vector<double> response(input_depth, 0.0);
        
        robertson_solver<float, MockImage>(
            0,              // channel 0 (red)
            input_depth,
            5,              // iterations
            sources,
            points,
            exp_times,
            input_weights,
            response.data()
        );

        // Very basic assertions to ensure it hasn't NaN'd out and generated a real curve
        REQUIRE(response[input_depth/2] == Approx(0.0).margin(0.01)); // mid value anchor
        REQUIRE(response[input_depth-1] > response[0]);

        bool is_monotonic = true;
        for (int i = 1; i < input_depth; ++i) {
            if (response[i] < response[i-1] && std::abs(response[i] - response[i-1]) > 1e-6) {
                is_monotonic = false;
                break;
            }
        }
        REQUIRE(is_monotonic == true);
    }

    SECTION("Debevec Solver Outputs Logarithmic Response") {
        std::vector<double> response(input_depth, 0.0);
        
        debevec_solver<float, MockImage>(
            0,
            input_depth,
            10.0f,          // smoothness lambda
            sources,
            points,
            exp_times_log,
            input_weights,
            response.data()
        );

        REQUIRE(response[input_depth/2] == Approx(0.0).margin(0.01)); // mid value anchor
        REQUIRE(response[input_depth-1] > response[0]);

        bool is_monotonic = true;
        for (int i = 1; i < input_depth; ++i) {
            if (response[i] < response[i-1] && std::abs(response[i] - response[i-1]) > 1e-6) {
                is_monotonic = false;
                break;
            }
        }
        REQUIRE(is_monotonic == true);
    }
}
