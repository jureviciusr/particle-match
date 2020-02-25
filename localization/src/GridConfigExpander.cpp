//
// Created by rokas on 17.6.19.
//

#include <tbb/parallel_for.h>
#include "GridConfigExpander.hpp"

/**
 * Given our grid / net, create a list of matching configurations
 */
std::vector<fast_match::MatchConfig>
GridConfigExpander::createListOfConfigs(cv::Size templ_size, cv::Size image_size) {
    /* Creating the steps for all the parameters (i.e. translation, rotation, and scaling) */
    std::vector<float> tx_steps = net->getXTranslationSteps(),
            ty_steps = net->getYTranslationSteps(),
            r_steps = net->getRotationSteps(),
            s_steps = net->getScaleSteps();


    /* Getting the proper number of steps for each configuration parameters */
    int ntx_steps = static_cast<int>( tx_steps.size()),
            nty_steps = static_cast<int>( ty_steps.size()),
            ns_steps = static_cast<int>( s_steps.size()),
            nr_steps = static_cast<int>( r_steps.size()),
            nr2_steps = nr_steps;

    /* Refine the number of steps for the 2nd rotation parameter */
    if (fabs((net->boundsRotate.second - net->boundsRotate.first) - (2 * M_PI)) < 0.1) {
        nr2_steps = (int) count_if(r_steps.begin(), r_steps.end(), [&](float r) {
            return r < (-M_PI / 2 + net->stepsRotate / 2);
        });
    }

    int grid_size = ntx_steps * nty_steps * ns_steps * ns_steps * nr_steps * nr2_steps;

    std::vector<fast_match::MatchConfig> configs(grid_size);

    /* Iterate thru each possible affine configuration steps */
    tbb::parallel_for(0, ntx_steps, 1, [&](int tx_index) {
        float tx = tx_steps[tx_index];

        for (int ty_index = 0; ty_index < nty_steps; ty_index++) {
            float ty = ty_steps[ty_index];

            for (int r1_index = 0; r1_index < nr_steps; r1_index++) {
                float r1 = r_steps[r1_index];

                for (int r2_index = 0; r2_index < nr2_steps; r2_index++) {
                    float r2 = r_steps[r2_index];

                    for (int sx_index = 0; sx_index < ns_steps; sx_index++) {
                        float sx = s_steps[sx_index];

                        for (int sy_index = 0; sy_index < ns_steps; sy_index++) {
                            float sy = s_steps[sy_index];

                            /* Maybe there's a better way for indexing when multithreading ... */
                            int grid_index = (tx_index * nty_steps * nr_steps * nr2_steps * ns_steps * ns_steps)
                                             + (ty_index * nr_steps * nr2_steps * ns_steps * ns_steps)
                                             + (r1_index * nr2_steps * ns_steps * ns_steps)
                                             + (r2_index * ns_steps * ns_steps)
                                             + (sx_index * ns_steps)
                                             + sy_index;

                            configs[grid_index].init(tx, ty, r2, sx, sy, r1);
                        }
                    }
                }
            }
        }
    });


    return configs;
}

/**
 * Randomly expands the configuration
 */
std::vector<fast_match::MatchConfig>
GridConfigExpander::randomExpandConfigs(std::vector<fast_match::MatchConfig> &configs, int level,
                                                   int no_of_points, float delta_factor) {

    float factor = (float) pow(delta_factor, level);

    float half_step_tx = net->stepsTransX / factor,
            half_step_ty = net->stepsTransY / factor,
            half_step_r = net->stepsRotate / factor,
            half_step_s = net->stepsScale / factor;

    int no_of_configs = static_cast<int>( configs.size());

    /* Create random vectors that contain values which are either -1, 0, or 1 */
    cv::Mat random_vec(no_of_points * no_of_configs, 6, CV_32SC1);
    rng.fill(random_vec, cv::RNG::NORMAL, 0, 0.5);
    random_vec.convertTo(random_vec, CV_32FC1);

    /* Convert our vector of configurations into a large matrix */
    std::vector<cv::Mat> configs_mat(no_of_configs);
    for (int i = 0; i < no_of_configs; i++)
        configs_mat[i] = configs[i].asMatrix();

    cv::Mat expanded;
    vconcat(configs_mat, expanded);
    expanded = repeat(expanded, no_of_points, 1);

    std::vector<float> ranges_vec = {
            half_step_tx, half_step_ty, half_step_r, half_step_s, half_step_s, half_step_r
    };

    cv::Mat ranges = cv::repeat(cv::Mat(ranges_vec).t(), no_of_points * no_of_configs, 1);

    /* The expanded configs is the original configs plus some random changes */
    cv::Mat expanded_configs = expanded + random_vec.mul(ranges);

    return fast_match::MatchConfig::fromMatrix(expanded_configs);
}

GridConfigExpander::GridConfigExpander() : ConfigExpanderBase() {}

