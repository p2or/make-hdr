//
//  solver.h
//  MakeHDR
//
//  Created by Vahan Sosoyan 2024.
//

#ifndef solver_h
#define solver_h

#include "resources.h"

template<typename ptype>
void solver(const int channel,
            const int input_depth,
            const float smoothness,
            const std::vector<std::shared_ptr<OFX::Image>>& sources,
            const std::vector<fx::point>& points,
            const std::vector<float>& exp_times_log,
            const std::vector<float>& input_weights,
            double* response)
{   
    const int sources_size = (int)sources.size();
    const int samples_size = (int)points.size();

    const int m = samples_size * sources_size + input_depth + 1;
    const int n = input_depth + samples_size;

    arma::mat a = arma::mat(m, n).zeros();
    arma::vec b = arma::vec(m).zeros();
    arma::vec s = arma::vec(n).zeros();

    int k = 0;
    for (int i = 0; i < samples_size; ++i)
    {
        for (int j = 0; j < sources_size; ++j)
        {           
            ptype* sample = (ptype*)sources[j]->getPixelAddress(points[i].x, points[i].y);
                     
            float sample_flt = sample == nullptr ? 0 : sample[channel];

            // Clamp 0 to 1
            sample_flt = std::min<ptype>(sample_flt, 1.f);
            sample_flt = std::max<ptype>(sample_flt, 0.f);

            const int sample_int = (int)(sample_flt * (input_depth - 1));

            const float wij = input_weights[sample_int];

            a.at(k, sample_int) = wij;
            a.at(k, input_depth + i) = -wij;
            b.at(k, 0) = wij * exp_times_log[j];
            k++;
        }
    }
    
    // Fix the scaling
    a.at(k, input_depth / 2) = 1;
    k++;

    // Smoothness equations
    const float lambda = smoothness * (input_depth / 256.f);

    for (int i = 0; i < (input_depth - 2); ++i)
    {
        float wi = input_weights[i + 1];

        a.at(k, i) = lambda * wi;
        a.at(k, i + 1) = -2 * lambda * wi;
        a.at(k, i + 2) = lambda * wi;
        k++;
    }

    bool success = arma::solve(s, a, b);
    
    // Fallback to SVD pseudo-inverse if the system is singular or ill-conditioned
    if (!success)
    {
        arma::mat a_pinv;
        success = arma::pinv(a_pinv, a);
        if (success)
            s = a_pinv * b;
    }

    if (success)
    {
        for (int i = 0; i < input_depth; ++i)
            response[i] = s[i];
    }
    else
        spdlog::error("{}: Solver has faild for channel {}!", fx::label , channel);
}

#endif