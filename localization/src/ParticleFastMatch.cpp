//
// Created by rokas on 17.6.20.
//

#include "ParticleFastMatch.hpp"
#include "Utilities.hpp"

#include <chrono>
#include <utility>
#include <fstream>

#include <opencv2/features2d.hpp>

#include <tbb/task_scheduler_observer.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_for.h>


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
    switch (matching) {

        case PearsonCorrelation:break;
        case BriskMatch:
            detector = cv::BRISK::create(20);
            break;
        case ORBMatch:
            detector = cv::ORB::create();
            break;
    }
    /*if(!detector.empty()) {
        //matcher = std::make_shared<cv::cuda::DescriptorMatcher>();
        matcher = cv::cuda::DescriptorMatcher::createBFMatcher(detector->defaultNorm());
    }*/

}

void ParticleFastMatch::visualizeParticles(cv::Mat image, const cv::Point2i& offset) {
    visualizer.visualiseParticles(std::move(image), particles, offset);
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

#ifdef USE_CV_GPU
vector<Point> ParticleFastMatch::filterParticlesAffine(const cv::Point2f &movement, cv::Mat &bestTransform) {
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
    tbb::parallel_for(0, (int) newParticles.size(), 1, [&] (int ip) {
        double probability;
        cv::Mat transform = evaluateParticle(newParticles[ip], ip, probability);
        //i++;
        if(probability < bestProbability) {
            bestTransform = transform;
            bestProbability = probability;
        }
        cv::Mat bestView = Utilities::extractWarpedMapPart(imageGray, templ.size(), transform);
        //cv::cuda::GpuMat bestView = Utilities::extractWarpedMapPart(imageGrayGpu, templ.size(), transform);
        calculateSimilarity(bestView, newParticles[ip]);
    });

    particles.assign(newParticles.begin(), newParticles.end());
    particles.normalize();
    return Utilities::calcCorners(image.size(), templ.size(), bestTransform);
}
#endif

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
    vector<AffineTransformation> affines = configsToAffine(pConfigs, particle.insiders);
    /* Filter out configurations that fall outside of the boundaries */
    /* the internal logic of configsToAffine has more information */
    std::vector<fast_match::MatchConfig> temp_configs;
    for (int i = 0; i < particle.insiders.size(); i++)
        if (particle.insiders[i] == true)
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

void ParticleFastMatch::setTemplate(const Mat &templ_) {
    fast_match::FAsTMatch::setTemplate(templ_);
    switch (matching) {
#ifdef USE_CV_GPU
        case BriskMatch:
        case ORBMatch: {
            std::vector<cv::KeyPoint> keypointsA;
            detector->detect(templGrayGpu, keypointsA);
            detector->compute(templGrayGpu, keypointsA, templGpuDescriptors);
            break;
        }
#endif
        case PearsonCorrelation: {
            if (samplingPoints.empty()) {
                for (int y_ = 0; y_ < (float) (templ_.rows * templ_.cols) * 0.1f; y_++) {
                    samplingPoints.emplace_back(
                            (int) (Utilities::uniform_dist() * templ_.cols),
                            (int) (Utilities::uniform_dist() * templ_.rows)
                    );
                }

                // Sorting points in an order that might avoid potential cache misses
                std::sort(samplingPoints.begin(), samplingPoints.end(),[] (const cv::Point& a, const cv::Point& b) {
                    return a.y == b.y ? a.x < b.x : a.y < b.y;
                });
            }
            templateSample = ImageSample(FAsTMatch::templGray, samplingPoints, templGrayAvg);
        }
        default: break;
    }
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

std::vector<cv::Point> ParticleFastMatch::filterParticles(const cv::Point2f &movement, cv::Mat &bestTransform) {
    Particles newParticles = {};
    int     support_particles = 0,
            samplingCount = minParticles;
    std::vector <std::string> bins;
    std::sort(particles.begin(), particles.end(), std::less<>());
    unsigned long particleIndex = 0;
    do {
        // Sample previous particle from previous belief
        newParticles.addParticle(particles.sample());
        // Predict next state
        newParticles[particleIndex].propagate(movement);
        std::string bin = newParticles[particleIndex].serialize(binSize);
        particleIndex++;
        if (std::find(bins.begin(), bins.end(), bin) == bins.end()) {
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
    } while (particleIndex < samplingCount);

    tbb::parallel_for_each(newParticles.begin(), newParticles.end(), [&] (Particle& particle) {
        cv::Mat rot_mat = particle.mapTransformation();
        ImageSample mapSample(imageGray, samplingPoints, rot_mat, particle.toPoint());
        auto ccoef = (float) templateSample.calcSimilarity(mapSample);
        particle.setCorrelation(ccoef);
        particle.setProbability(convertProbability(ccoef));
    });
    std::sort(particles.begin(), particles.end(), std::less<>());
    particles.assign(newParticles.begin(), newParticles.end());
    particles.normalize();
    return particles.front().getCorners();
}

cv::Mat ParticleFastMatch::getBestParticleView(cv::Mat map) {
    return particles.front().getMapImage(imageGray, cv::Size(640, 480));
}

float ParticleFastMatch::calculateSimilarity(cv::Mat im) const {
    switch (matching) {
        case PearsonCorrelation: {
            float ccoef = Utilities::calculateCorrCoeff(std::move(im), templ);
            float prob;
            float lowBound = 0.2f;
            if(ccoef > 0.f) {
                prob = lowBound + (ccoef * (1 - lowBound));
            } else {
                prob = lowBound - (std::abs(ccoef) * lowBound);
            }
            return prob;
        }
        case ORBMatch: {
            using namespace std::chrono;

            std::vector<cv::KeyPoint> keypointsA, keypointsB;
            cv::Mat descriptorsA, descriptorsB;
            high_resolution_clock::time_point t1 = high_resolution_clock::now();

            detector->detectAndCompute(im, cv::Mat(), keypointsA, descriptorsA);
            detector->detectAndCompute(templ, cv::Mat(), keypointsB, descriptorsB);

            cv::BFMatcher matcher(cv::NORM_HAMMING, true);

            std::vector<cv::DMatch> matches;
            matcher.match(descriptorsA, descriptorsB, matches);

            // 3. Extract point coordinates
            std::vector<cv::Point2f> pts1, pts2;
            for (auto& m : matches) {
                pts1.push_back(keypointsA[m.queryIdx].pt);
                pts2.push_back(keypointsB[m.trainIdx].pt);
            }

            // 4. Estimate homography with RANSAC
            std::vector<unsigned char> inliersMask(pts1.size());
            cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, inliersMask);

            if (H.empty()) {
                std::cerr << "Homography estimation failed!\n";
                return 0.0;
            }

            // 5. Count inliers
            int inliersCount = 0;
            for (size_t i = 0; i < inliersMask.size(); i++)
                if (inliersMask[i]) inliersCount++;

            // 6. Inlier ratio is our probability
            return static_cast<double>(inliersCount) / static_cast<double>(matches.size());
        }

#ifdef USE_CV_GPU
        case ORBMatch:
        case BriskMatch: {
            using namespace std::chrono;

            std::vector<cv::KeyPoint> keypointsA;

            cv::Mat descriptorsA;
            cv::cuda::GpuMat gpuDescriptors;
            high_resolution_clock::time_point t1 = high_resolution_clock::now();
            detector->detect(im, keypointsA);
            detector->compute(im, keypointsA, descriptorsA);
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
            std::cout << "Detect / compute: " << duration << "\n";

            std::vector<cv::DMatch> matches;
            t1 = high_resolution_clock::now();
            gpuDescriptors.upload(descriptorsA);
            matcher->match(templGpuDescriptors, gpuDescriptors, matches);
            t2 = high_resolution_clock::now();
            duration = duration_cast<milliseconds>( t2 - t1 ).count();
            std::cout << "Match: " << duration << "\n";

            float maxDistance = 55.f;
            float weight = 0.0f;

            for(const auto & singleMatch : matches) {
                if(singleMatch.distance < maxDistance) {
                    float normDist = 1.f - (singleMatch.distance / (templDescriptors.cols * 8.f));
                    weight += normDist;
                }
            }

            return weight / (float) templDescriptors.rows;
        }
#endif
    }
}

float glf(float x, float v = .3f, float B = 5.f, float Q = 1.f, float C = 1.f, float A = 0.f, float K = 1.f) {
    return (A + (K - A) / std::pow(C + (Q * std::exp(-B * x)), 1.f / v));
}

float normalGLF(float x, float v) {
    //float a = static_cast<float>(1.f / (sigma * std::sqrt(2.f * M_PI)));
    //float exponent = std::exp(-(std::pow((x - mu) / sigma, 2.0f) / 2.0f));
    //return a * exponent;
    return glf(x, v) / glf(1.0f, v);
}

float hprelu(float ccoef, float lowBound) {
    float prob;
    if (lowBound < 0.0f) {
        lowBound = std::abs(lowBound);
        if (ccoef > (lowBound + 0.00001f)) {
            float slope = 1 + lowBound;
            prob = (slope * ccoef) - (slope * lowBound) + (lowBound * lowBound);
        } else {
            prob = 0.f;
        }
    } else {
        if (ccoef > 0.f) {
            prob = lowBound + (ccoef * (1.f - lowBound));
        } else {
            prob = lowBound - (std::abs(ccoef) * lowBound);
        }
    }
    return prob;
}

float ParticleFastMatch::convertProbability(float in) const {
    switch (conversionMode) {
        case HPRELU:
            return hprelu(in, lowBound);
        case GLF:
            return normalGLF(in, lowBound);
        case Softmax:
            return std::exp(in);
    }
}
#ifdef USE_CV_GPU
void ParticleFastMatch::calculateSimilarity(cv::cuda::GpuMat im, Particle& particle) const {
    switch (matching) {
        case PearsonCorrelation: {
            float ccoef = Utilities::calculateCorrCoeff(im, templGrayGpu);
            particle.setCorrelation(ccoef);
            particle.setProbability(convertProbability(ccoef));
            break;
        }
        case BriskMatch: {
            std::cerr << "BRISK GPU IMPLEMENTATION IS NOT AVAILABLE\n";
            exit(10);
        }
        case ORBMatch: {
            using namespace std::chrono;
            std::vector<cv::KeyPoint> keypointsA;

            cv::cuda::GpuMat gpuDescriptors;
            high_resolution_clock::time_point t1 = high_resolution_clock::now();
            detector->detectAndCompute(im, cv::noArray(), keypointsA, gpuDescriptors);
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
            std::cout << "Detect / compute: " << duration << "\n";
            std::vector<cv::DMatch> matches;
            t1 = high_resolution_clock::now();
            matcher->match(templGpuDescriptors, gpuDescriptors, matches);
            t2 = high_resolution_clock::now();
            duration = duration_cast<milliseconds>( t2 - t1 ).count();
            std::cout << "Match: " << duration << "\n";

            float maxDistance = 55.f;
            float weight = 0.0f;

            for(const auto & singleMatch : matches) {
                if(singleMatch.distance < maxDistance) {
                    float normDist = 1.f - (singleMatch.distance / (templGpuDescriptors.cols * 8.f));
                    weight += normDist;
                }
            }
            particle.setCorrelation(-INFINITY);
            particle.setProbability(weight / (float) templGpuDescriptors.rows);
            break;
        }
    }
}
#endif
float ParticleFastMatch::getLowBound() const {
    return lowBound;
}

void ParticleFastMatch::setLowBound(float lowBound) {
    ParticleFastMatch::lowBound = lowBound;
}

const Particles &ParticleFastMatch::getParticles() const {
    return particles;
}

std::string ParticleFastMatch::conversionModeString() const {
    switch (conversionMode) {
        case HPRELU: return "HPRELU";
        case GLF: return "GLF";
        case Softmax: return "Softmax";
    }
}
