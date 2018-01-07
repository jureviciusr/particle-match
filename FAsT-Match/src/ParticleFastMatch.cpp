//
// Created by rokas on 17.6.20.
//

#include <tbb/parallel_for.h>
#include <chrono>
#include <utility>
#include "ParticleFastMatch.hpp"
#include "Utilities.hpp"
#include "AffineTransformation.hpp"

#define WITHIN(val, top_left, bottom_right) (\
            val.x > top_left.x && val.y > top_left.y && \
            val.x < bottom_right.x && val.y < bottom_right.y )


ParticleFastMatch::ParticleFastMatch(
        const cv::Point2i& startLocation,
        const cv::Size& mapSize,
        double radius,
        float epsilon,
        int particleCount,
        float quantile_,
        float kld_error_,
        int bin_size_,
        bool use_gaussian
) {
    particles.init(startLocation, mapSize, radius, particleCount, use_gaussian);
    kld_error = kld_error_;
    binSize = bin_size_;
    this->epsilon = epsilon;
    this->no_of_points = (int) round(10 / (epsilon * epsilon));
    xs = cv::Mat(1, no_of_points, CV_32SC1);
    ys = cv::Mat(1, no_of_points, CV_32SC1);
    if (ztable.empty()) {
        buildZTable();
    }
    zvalue = 4.1;
    float confidence = quantile_ - 0.5f; // ztable is from right side of mean
    confidence = fmin(0.49998f, fmax(0.f, confidence));
    for (unsigned int i = 0; i < ztable.size(); i++) {
        if (ztable[i] >= confidence) {
            zvalue = i / 100.0f;
            break;
        }
    }

}

void ParticleFastMatch::visualizeParticles(cv::Mat image) {
    visualizer.visualiseParticles(std::move(image), particles);
}


/*
vector<Point2f> ParticleFastMatch::filterParticles() {
    // Randomly sample points
    Mat xs(1, no_of_points, CV_32SC1),
            ys(1, no_of_points, CV_32SC1);
    rng.fill(xs, RNG::UNIFORM, 1, templ.cols);
    rng.fill(ys, RNG::UNIFORM, 1, templ.rows);


    std::vector<fast_match::MatchConfig> pConfigs = particles.getConfigs();
    vector<AffineTransformation> affines = configsToAffine(pConfigs, insiders);
    // Filter out configurations that fall outside of the boundaries
    // the internal logic of configsToAffine has more information
    std::vector<fast_match::MatchConfig> temp_configs;
    for (int i = 0; i < insiders.size(); i++)
        if (insiders[i] == true)
            temp_configs.push_back(pConfigs[i]);
    pConfigs = temp_configs;

    // For the configs, calculate the scores / distances
    distances = evaluateConfigs(image, templ, affines, xs, ys, true);
    // Find the minimum distance
    auto min_itr = min_element(distances.begin(), distances.end());
    int min_index = static_cast<int>(min_itr - distances.begin());
    double best_distance = distances[min_index];

    //best_distances[level] = best_distance;

    auto max_itr = max_element(distances.begin(), distances.end());
    int max_index = static_cast<int>(max_itr - distances.begin());
    double worst_distance = distances[max_index];

    best_config = pConfigs[min_index];
    best_trans = best_config.getAffineMatrix();

    particles.printProbabilities();

    return Utilities::calcCorners(image.size(), templ.size(), best_trans);
}
*/


vector<Point> ParticleFastMatch::filterParticles(const cv::Point2f& movement, cv::Mat& bestTransform) {
    int i = 0;
    Particles newParticles = {};
    int     support_particles = 0,
            samplingCount = minParticles;
    std::vector < std::string > bins;
    double bestProbability = +INFINITY;
    std::sort(particles.begin(), particles.end(), std::less<>());
    initTemplatePixels();
    unsigned long particleIndex = 0;
    do {
        // Sample previous particle from previous belief
        newParticles.addParticle(particles.sample());
        // Predict next state
        newParticles[particleIndex].propagate(movement);
        // Calculate particle belief
        double probability;
        cv::Mat transform = evaluateParticle(newParticles[particleIndex], i, probability);
        i++;
        if(probability < bestProbability) {
            bestTransform = transform;
            bestProbability = probability;
        }
        cv::Mat bestView = Utilities::extractMapPart(image, templ.size(), transform);
        float ccoef = Utilities::calculateCorrCoeff(bestView, templ);
        float prob;
        float lowBound = 0.2f;
        if(ccoef > 0.f) {
            prob = lowBound + (ccoef * (1 - lowBound));
        } else {
            prob = lowBound - (std::abs(ccoef) * lowBound);
        }
        newParticles[particleIndex].setProbability(prob/*((ccoef + 1) / 2.f)*/);

        std::string bin = newParticles[particleIndex].serialize(binSize);
        particleIndex++;
        if (!(std::find(bins.begin(), bins.end(), bin) != bins.end())) {
            // Mark bin as taken
            bins.push_back(bin);
            // Update number with support
            support_particles++;
            if (support_particles >= 2) {
                // update desired number
                int k = support_particles - 1;
                k = (int) ceil(k / (2 * kld_error) * pow(1 - 2 / (9.0 * k) + sqrt(2 / (9.0 * k)) * zvalue, 3));
                if (k > samplingCount) {
                    samplingCount = k;
                }
                if (samplingCount < minParticles) {
                    samplingCount = minParticles;
                }
            }
        }
    } while (newParticles.size() < samplingCount);
    particles.assign(newParticles.begin(), newParticles.end());
    particles.normalize();
    return Utilities::calcCorners(image.size(), templ.size(), bestTransform);
}


vector<Point> ParticleFastMatch::evaluateParticlesv2() {
    return particles.evaluate(image, templ, no_of_points);
}

void ParticleFastMatch::propagateParticles(const cv::Point2f& movement) {
    particles.propagate(movement);

}

void ParticleFastMatch::setDirection(const double &_d) {
    Particle::setDirection(_d);
}

vector<double>
ParticleFastMatch::evaluateConfigs(Mat &templ, vector<AffineTransformation> &affine_matrices, Mat &xs, Mat &ys,
                                   bool photometric_invariance) {
    int r1x = (int) (0.5 * (templ.cols - 1)),
            r1y = (int) (0.5 * (templ.rows - 1)),
            r2x = (int) (0.5 * (image.cols - 1)),
            r2y = (int) (0.5 * (image.rows - 1));

    int no_of_configs = static_cast<int>(affine_matrices.size());
    int no_of_points = xs.cols;

    /* Create a lookup array for our template values based on the given random x and y points */
    int *xs_ptr = xs.ptr<int>(0),
            *ys_ptr = ys.ptr<int>(0);

    vector<float> vals_i1(static_cast<unsigned long>(no_of_points));
    for (int i = 0; i < no_of_points; i++)
        vals_i1[i] = templ.at<float>(ys_ptr[i] - 1, xs_ptr[i] - 1);

    /* Recenter our indices */
    Mat xs_centered = xs.clone() - (r1x + 1),
            ys_centered = ys.clone() - (r1y + 1);

    int *xs_ptr_cent = xs_centered.ptr<int>(0),
            *ys_ptr_cent = ys_centered.ptr<int>(0);

    vector<double> distances(static_cast<unsigned long>(no_of_configs), 0.0);

    /* Calculate the score for each configurations on each of our randomly sampled points */
    tbb::parallel_for(0, no_of_configs, 1, [&](int i) {

        float a11 = affine_matrices[i].T.at<float>(0, 0),
                a12 = affine_matrices[i].T.at<float>(0, 1),
                a13 = affine_matrices[i].T.at<float>(0, 2),
                a21 = affine_matrices[i].T.at<float>(1, 0),
                a22 = affine_matrices[i].T.at<float>(1, 1),
                a23 = affine_matrices[i].T.at<float>(1, 2);

        int particleId = affine_matrices[i].id;

        double tmp_1 = (r2x + 1) + a13 + 0.5;
        double tmp_2 = (r2y + 1) + a23 + 0.5 + 1 * image.rows;
        double score = 0.0;

        if (!photometric_invariance) {
            for (int j = 0; j < no_of_points; j++) {
                int target_x = int(a11 * xs_ptr_cent[j] + a12 * ys_ptr_cent[j] + tmp_1),
                        target_y = int(a21 * xs_ptr_cent[j] + a22 * ys_ptr_cent[j] + tmp_2);

                //score += abs(vals_i1[j] - paddedCurrentImage.at<float>(target_y - 1, target_x - 1) );
                if (target_x - 1 >= 0 && target_x - 1 < paddedCurrentImage.size().width)
                    score += abs(vals_i1[j] - paddedCurrentImage.at<float>(target_y - 1, target_x - 1));

            }
        } else {
            vector<double> xs_target(static_cast<unsigned long>(no_of_points)),
                    ys_target(static_cast<unsigned long>(no_of_points));

            double sum_x = 0.0,
                    sum_y = 0.0,
                    sum_x_squared = 0.0,
                    sum_y_squared = 0.0;

            for (int j = 0; j < no_of_points; j++) {
                int target_x = int(a11 * xs_ptr_cent[j] + a12 * ys_ptr_cent[j] + tmp_1),
                        target_y = int(a21 * xs_ptr_cent[j] + a22 * ys_ptr_cent[j] + tmp_2);

                float xi = vals_i1[j],
                        yi = paddedCurrentImage.at<float>(target_y - 1, target_x - 1);

                xs_target[j] = xi;
                ys_target[j] = yi;

                sum_x += xi;
                sum_y += yi;

                sum_x_squared += (xi * xi);
                sum_y_squared += (yi * yi);
            }

            double epsilon = 1e-7;
            double mean_x = sum_x / no_of_points,
                    mean_y = sum_y / no_of_points,
                    sigma_x = sqrt((sum_x_squared - (sum_x * sum_x) / no_of_points) / no_of_points) + epsilon,
                    sigma_y = sqrt((sum_y_squared - (sum_y * sum_y) / no_of_points) / no_of_points) + epsilon;

            double sigma_div = sigma_x / sigma_y;
            double temp = -mean_x + sigma_div * mean_y;


            for (int j = 0; j < no_of_points; j++)
                score += fabs(xs_target[j] - sigma_div * ys_target[j] + temp);
        }

        distances[i] = score / no_of_points;
        /*if(particleId >= 0) {
            if(particleId < particles.size()) {
                particles[particleId].setMinimalProbability(static_cast<float>(distances[i]));
            } else {
                std::cerr << "Warning: Particle deos not exist? .. \n";
                return false;
            }
        }*/
    });


    return distances;

}

vector<AffineTransformation> ParticleFastMatch::configsToAffine(vector<fast_match::MatchConfig> &configs, vector<bool> &insiders) {
    int no_of_configs = static_cast<int>(configs.size());
    vector<Mat> affines((unsigned long) no_of_configs);

    /* The boundary, between -10 to image size + 10 */
    Point2d top_left(-10., -10.);
    Point2d bottom_right(image.cols + 10, image.rows + 10);


    /* These are for the calculations of affine transformed corners */
    int r1x = (templ.cols - 1) / 2,
            r1y = (templ.rows - 1) / 2,
            r2x = (image.cols - 1) / 2,
            r2y = (image.rows - 1) / 2;

    Mat corners = (Mat_<float>(3, 4) << 1 - (r1x + 1), templ.cols - (r1x + 1), templ.cols - (r1x + 1), 1 - (r1x + 1),
            1 - (r1y + 1), 1 - (r1y + 1), templ.rows - (r1y + 1), templ.rows - (r1y + 1),
            1.0, 1.0, 1.0, 1.0);

    Mat transl = (Mat_<float>(4, 2) << r2x + 1, r2y + 1,
            r2x + 1, r2y + 1,
            r2x + 1, r2y + 1,
            r2x + 1, r2y + 1);

    insiders.assign((unsigned long) no_of_configs, false);

    /* Convert each configuration to corresponding affine transformation matrix */
    tbb::parallel_for(0, no_of_configs, 1, [&](int i) {
        Mat affine = configs[i].getAffineMatrix();

        /* Check if our affine transformed rectangle still fits within our boundary */
        Mat affine_corners = (affine * corners).t();
        affine_corners = affine_corners + transl;

        if (WITHIN(affine_corners.at<Point2f>(0, 0), top_left, bottom_right) &&
            WITHIN(affine_corners.at<Point2f>(1, 0), top_left, bottom_right) &&
            WITHIN(affine_corners.at<Point2f>(2, 0), top_left, bottom_right) &&
            WITHIN(affine_corners.at<Point2f>(3, 0), top_left, bottom_right)) {

            affines[i] = affine;
            insiders[i] = true;
        }
    });

    /* Filter out empty affine matrices (which initially don't fit within the preset boundary) */
    /* It's done this way, so that I could parallelize the loop */
    vector<AffineTransformation> result;
    for (int i = 0; i < no_of_configs; i++) {
        if (insiders[i]){
            AffineTransformation transformation;
            transformation.T = affines[i];
            transformation.id = configs[i].getId();
            result.push_back(transformation);
        }
    }

    return result;

}

void ParticleFastMatch::initTemplatePixels() {
    rng.fill(xs, RNG::UNIFORM, 1, templ.cols);
    rng.fill(ys, RNG::UNIFORM, 1, templ.rows);
}

void ParticleFastMatch::initPaddedImage() {
    /* Use a paddedCurrentImage image, to avoid boundary checking */
    paddedCurrentImage = cv::Mat(image.rows * 3, image.cols, image.type(), cv::Scalar(0.0));
    image.copyTo(Mat(paddedCurrentImage, cv::Rect(0, image.rows, image.cols, image.rows)));
}

Mat ParticleFastMatch::evaluateParticle(Particle& particle, int id, double &bestProbability) {
    std::vector<fast_match::MatchConfig> pConfigs = particle.getConfigs(id);
    vector<AffineTransformation> affines = configsToAffine(pConfigs, insiders);
    /* Filter out configurations that fall outside of the boundaries */
    /* the internal logic of configsToAffine has more information */
    std::vector<fast_match::MatchConfig> temp_configs;
    for (int i = 0; i < insiders.size(); i++)
        if (insiders[i] == true)
            temp_configs.push_back(pConfigs[i]);
    pConfigs = temp_configs;

    /* For the configs, calculate the scores / distances */
    vector<double> pDistances = evaluateConfigs(templ, affines, xs, ys, true);

    auto min_itr = min_element(pDistances.begin(), pDistances.end());
    int min_index = static_cast<int>(min_itr - pDistances.begin());
    bestProbability = pDistances[min_index];

    //particle.setProbability((float) bestProbability);

    return pConfigs[min_index].getAffineMatrix();
}

/**
 * Builds a z-table which is necessary for the statiscal kld-sampling.
 */
void ParticleFastMatch::buildZTable() {
    float tmp;
    std::ifstream ifile("ztable.data");

    if (ifile.is_open()) {
        while (!ifile.eof()) {
            ifile >> tmp;
            ztable.push_back(tmp);
        }
    } else {
        std::cerr << "ztable.data does not exist. Error!\n";
        std::exit(-1);
    }

}

void ParticleFastMatch::setImage(const Mat &image) {
    fast_match::FAsTMatch::setImage(image);
    initPaddedImage();
}

void ParticleFastMatch::setTemplate(const Mat &templ) {
    fast_match::FAsTMatch::setTemplate(templ);
}

uint32_t ParticleFastMatch::particleCount() const {
    return static_cast<uint32_t>(particles.size());
}

cv::Point2i ParticleFastMatch::getPredictedLocation() const {
    return particles.getWeightedSum();
}

void ParticleFastMatch::setScale(float min, float max, uint32_t searchSteps) {
    particles.setScale(min, max, searchSteps);
}
