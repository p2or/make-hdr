#define CATCH_CONFIG_MAIN
#include "../modules/armadillo/tests2/catch.hpp"
#include <vector>
#include <memory>
#include <cmath>

#include "../source/resources.h"
#include "../source/solver.h"

struct MockImage {
    std::vector<float> data;
    int width, height;

    MockImage(int w, int h) : width(w), height(h) {
        data.resize(width * height * 4, 0.0f);
    }

    void setPixel(int x, int y, float r, float g, float b) {
        int idx = (y * width + x) * 4;
        data[idx] = r; data[idx + 1] = g; data[idx + 2] = b;
    }

    void* getPixelAddress(int x, int y) {
        if (x < 0 || x >= width || y < 0 || y >= height) return nullptr;
        return &data[(y * width + x) * 4];
    }
};

// Checks that a log-domain response curve is monotonic and free of NaN/Inf.
static void check_curve(const std::vector<double>& r) {
    for (int i = 0; i < (int)r.size(); ++i) {
        REQUIRE_FALSE((std::isnan(r[i]) || std::isinf(r[i])));
    }
    REQUIRE(r[r.size() - 1] > r[0]);
    for (int i = 1; i < (int)r.size(); ++i)
        REQUIRE_FALSE(r[i] < r[i - 1] - 1e-6);
}

// Checks reciprocity: exp(response[bin]) / t should be the same across all
// non-clipped sources for each sample point.
static void check_reciprocity(const std::vector<double>& response,
                               const std::vector<float>& true_E,
                               const std::vector<float>& exp_times,
                               int input_depth, double eps)
{
    for (int j = 0; j < (int)true_E.size(); ++j) {
        double first_E = -1.0;
        for (int s = 0; s < (int)exp_times.size(); ++s) {
            const float z = true_E[j] * exp_times[s];
            if (z >= 1.0f) continue;
            const int bin = (int)(z * (input_depth - 1));
            const double E_est = std::exp(response[bin]) / exp_times[s];
            if (first_E < 0.0) first_E = E_est;
            else REQUIRE(E_est == Approx(first_E).epsilon(eps));
        }
    }
}

TEST_CASE("Solvers output a valid log-domain response curve", "[solvers]") {

    const int input_depth = 256;
    const std::vector<float> exp_times     = { 1.0f, 2.0f, 4.0f };
    const std::vector<float> exp_times_log = { 0.0f, std::log(2.0f), std::log(4.0f) };
    const std::vector<float> input_weights(input_depth, 1.0f);

    std::vector<std::shared_ptr<MockImage>> sources;
    for (float t : exp_times) {
        auto img = std::make_shared<MockImage>(2, 2);
        img->setPixel(0, 0, 0.1f * t, 0.f, 0.f);
        img->setPixel(1, 0, 0.2f * t, 0.f, 0.f);
        img->setPixel(0, 1, 0.4f * t, 0.f, 0.f);
        img->setPixel(1, 1, 0.8f * t, 0.f, 0.f);
        sources.push_back(img);
    }
    const std::vector<fx::point> points = { {0,0}, {1,0}, {0,1}, {1,1} };

    SECTION("Robertson") {
        std::vector<double> response(input_depth, 0.0);
        robertson_solver<float, MockImage>(0, input_depth, 5, sources, points,
                                           exp_times, input_weights, response.data());
        REQUIRE(response[input_depth / 2] == Approx(0.0).margin(0.01));
        check_curve(response);
    }

    SECTION("Debevec") {
        std::vector<double> response(input_depth, 0.0);
        debevec_solver<float, MockImage>(0, input_depth, 10.0f, sources, points,
                                         exp_times_log, input_weights, response.data());
        REQUIRE(response[input_depth / 2] == Approx(0.0).margin(0.01));
        check_curve(response);
    }
}

TEST_CASE("Reciprocity: consistent irradiance estimate across exposures", "[solvers]") {

    const int input_depth = 256;
    const std::vector<float> true_E        = { 0.05f, 0.10f, 0.15f, 0.20f };
    const std::vector<float> exp_times     = { 1.0f, 2.0f, 4.0f };
    const std::vector<float> exp_times_log = { 0.0f, std::log(2.0f), std::log(4.0f) };
    const std::vector<float> input_weights(input_depth, 1.0f);
    const int n = (int)true_E.size();

    std::vector<std::shared_ptr<MockImage>> sources;
    for (float t : exp_times) {
        auto img = std::make_shared<MockImage>(n, 1);
        for (int j = 0; j < n; ++j)
            img->setPixel(j, 0, true_E[j] * t, 0.f, 0.f);
        sources.push_back(img);
    }
    std::vector<fx::point> points;
    for (int j = 0; j < n; ++j)
        points.push_back({j, 0});

    SECTION("Debevec") {
        std::vector<double> response(input_depth, 0.0);
        debevec_solver<float, MockImage>(0, input_depth, 10.0f, sources, points,
                                         exp_times_log, input_weights, response.data());
        check_reciprocity(response, true_E, exp_times, input_depth, 0.05);
    }

    SECTION("Robertson") {
        std::vector<double> response(input_depth, 0.0);
        robertson_solver<float, MockImage>(0, input_depth, 20, sources, points,
                                           exp_times, input_weights, response.data());
        check_reciprocity(response, true_E, exp_times, input_depth, 0.10);
    }
}
