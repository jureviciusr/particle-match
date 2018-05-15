//
// Created by rokas on 17.6.20.
//

#define WITHIN(val, top_left, bottom_right) (\
            val.x > top_left.x && val.y > top_left.y && \
            val.x < bottom_right.x && val.y < bottom_right.y )


double zeroIfNan(double x) {
    if (x * 0.0 == 0.0)
        return x;
    return 0;
}

#include <opencv2/imgproc.hpp>
#include <FAsT-Match/MatchConfig.h>
#include <tbb/parallel_for.h>
#include "Utilities.hpp"

using namespace cv;

/**
 * Preprocess image, by first converting it to grayscale
 * then normalizing the value within 0.0 - 1.0 range
 * and finally make sure that the dimensions are in odd values
 **/
Mat Utilities::preprocessImage(const Mat &image) {
    Mat temp = image.clone();
    if (temp.channels() != 1)
        cvtColor(temp, temp, CV_BGR2GRAY);

    if (temp.type() != CV_32FC1)
        temp.convertTo(temp, CV_32FC1, 1.0 / 255.0);

    return makeOdd(temp);
}


/**
 * If the image dimension is of odd value, leave it as is
 * if it's even, then minus 1 from the dimension
 */
Mat Utilities::makeOdd(Mat &image) {
    int rows = (image.rows % 2 == 0) ? image.rows - 1 : image.rows;
    int cols = (image.cols % 2 == 0) ? image.cols - 1 : image.cols;
    return Mat(image, Rect(0, 0, cols, rows)).clone();
}

/**
 * From the given affine matrix, calculate the four corners of the affine transformed
 * rectangle
 */
std::vector<cv::Point> Utilities::calcCorners(cv::Size image_size, cv::Size templ_size, cv::Mat &affine) {
    float r1x = 0.5f * (templ_size.width - 1),
            r1y = 0.5f * (templ_size.height - 1),
            r2x = 0.5f * (image_size.width - 1),
            r2y = 0.5f * (image_size.height - 1);

    float a11 = affine.at<float>(0, 0),
            a12 = affine.at<float>(0, 1),
            a13 = affine.at<float>(0, 2),
            a21 = affine.at<float>(1, 0),
            a22 = affine.at<float>(1, 1),
            a23 = affine.at<float>(1, 2);

    float templ_w = templ_size.width,
            templ_h = templ_size.height;

    // The four corners of affine transformed template
    double c1x = a11 * (1 - (r1x + 1)) + a12 * (1 - (r1y + 1)) + (r2x + 1) + a13;
    double c1y = a21 * (1 - (r1x + 1)) + a22 * (1 - (r1y + 1)) + (r2y + 1) + a23;

    double c2x = a11 * (templ_w - (r1x + 1)) + a12 * (1 - (r1y + 1)) + (r2x + 1) + a13;
    double c2y = a21 * (templ_w - (r1x + 1)) + a22 * (1 - (r1y + 1)) + (r2y + 1) + a23;

    double c3x = a11 * (templ_w - (r1x + 1)) + a12 * (templ_h - (r1y + 1)) + (r2x + 1) + a13;
    double c3y = a21 * (templ_w - (r1x + 1)) + a22 * (templ_h - (r1y + 1)) + (r2y + 1) + a23;

    double c4x = a11 * (1 - (r1x + 1)) + a12 * (templ_h - (r1y + 1)) + (r2x + 1) + a13;
    double c4y = a21 * (1 - (r1x + 1)) + a22 * (templ_h - (r1y + 1)) + (r2y + 1) + a23;

    return std::vector<cv::Point2i> {
            Point((int) c1x, (int) c1y),
            Point((int) c2x, (int) c2y),
            Point((int) c3x, (int) c3y),
            Point((int) c4x, (int) c4y)
    };
}

cv::Mat Utilities::getMapRoiMask(const cv::Size &image_size, const cv::Size &templ_size, cv::Mat &affine) {
    std::vector<cv::Point2i> corners = calcCorners(image_size, templ_size, affine);
    convexHull(corners, corners);                         //to assure correct point order
    Mat roi(image_size, CV_8U, Scalar(0));  //black image
    fillConvexPoly(roi, corners, Scalar(255));           //draw ROI in white
    return roi;
}


cv::Mat Utilities::extractWarpedMapPart(
        cv::InputArray map,
        const cv::Size &templ_size,
        const cv::Mat &affine
) {
    cv::Mat T = cv::Mat::zeros(cv::Size(3, 2), CV_32F);
    affine.copyTo(T);
    std::vector<cv::Point> corners = calcCorners(map.size(), templ_size, T);
    cv::Rect roi = cv::boundingRect(corners);
    T.at<float>(0, 2) = corners[0].x - roi.x;
    T.at<float>(1, 2) = corners[0].y - roi.y;
    cv::Mat mapPart = map.getMat()(roi);
    cv::Mat destination(templ_size, CV_8UC1);
    cv::warpAffine(mapPart, destination, T, templ_size, CV_INTER_NN | CV_WARP_INVERSE_MAP);
    return destination;
}

cv::cuda::GpuMat Utilities::extractWarpedMapPart(
        cv::cuda::GpuMat map,
        const cv::Size &templ_size,
        const cv::Mat &affine
) {
    cv::Mat T = cv::Mat::zeros(cv::Size(3, 2), CV_32F);
    affine.copyTo(T);
    std::vector<cv::Point> corners = calcCorners(map.size(), templ_size, T);
    cv::Rect roi = cv::boundingRect(corners);
    T.at<float>(0, 2) = corners[0].x - roi.x;
    T.at<float>(1, 2) = corners[0].y - roi.y;
    cv::cuda::GpuMat mapPart = map(roi);
    cv::cuda::GpuMat destination(templ_size, CV_8UC1);
    cv::cuda::warpAffine(mapPart, destination, T, templ_size, CV_INTER_NN | CV_WARP_INVERSE_MAP);
    return destination;
}

double Utilities::calculateCorrelation(cv::Mat scene, cv::Mat templ) {
    cv::Scalar avgPixels(cv::sum(scene) / (scene.cols * scene.rows));
    cv::Mat sceneRoi = (scene - avgPixels);
    auto crossCorr = cv::sum(templ.mul(sceneRoi));
    auto templSqr = templ.mul(templ);
    auto templSqrSum = cv::sum(templSqr);
    auto sceneSqr = sceneRoi.mul(sceneRoi);
    auto sceneSqrSum = cv::sum(sceneSqr);
    cv::Scalar normalizer;
    cv::sqrt(templSqrSum.mul(sceneSqrSum), normalizer);
    auto normCrossCorr = crossCorr.div(normalizer);
    int cn = CV_MAT_CN(scene.type());
    return (zeroIfNan(normCrossCorr.val[0])
                  + zeroIfNan(normCrossCorr.val[1])
                  + zeroIfNan(normCrossCorr.val[2])
                  + zeroIfNan(normCrossCorr.val[3])) / cn;
}

float Utilities::calculateCorrCoeff(cv::Mat scene, cv::Mat templ) {
    cv::Mat result(cv::Size(1, 1), CV_32FC1);
    cv::matchTemplate(scene, templ, result, CV_TM_CCOEFF_NORMED);
    return result.at<float>(0, 0);
}

float Utilities::calculateCorrCoeff(cv::cuda::GpuMat scene, cv::cuda::GpuMat templ) {
    cv::cuda::GpuMat result(cv::Size(1, 1), CV_32FC1);
    cv::Mat resultL(cv::Size(1, 1), CV_32FC1);
    cv::Ptr<cv::cuda::TemplateMatching> match = cv::cuda::createTemplateMatching(CV_8UC1, CV_TM_CCOEFF_NORMED);
    match->match(scene, templ, result);
    result.download(resultL);
    return resultL.at<float>(0, 0);
}

cv::Mat Utilities::photometricNormalization(cv::Mat scene, cv::Mat templ) {
    double sum_x = cv::sum(scene)[0];
    double sum_y = cv::sum(templ)[0];
    cv::Mat newScene, newTempl;
    scene.convertTo(newScene, CV_32FC1);
    templ.convertTo(newTempl, CV_32FC1);
    double sum_x_squared = cv::sum(newScene.mul(newScene))[0];
    double sum_y_squared = cv::sum(newTempl.mul(newTempl))[0];
    double epsilon = 1e-7;
    double no_of_points = scene.rows * scene.cols;
    double mean_x = sum_x / no_of_points,
            mean_y = sum_y / no_of_points,
            sigma_x = sqrt((sum_x_squared - (sum_x * sum_x) / no_of_points) / no_of_points) + epsilon,
            sigma_y = sqrt((sum_y_squared - (sum_y * sum_y) / no_of_points) / no_of_points) + epsilon;
    double sigma_div = sigma_x / sigma_y;
    double temp = -mean_x + sigma_div * mean_y;
    return (templ * sigma_div) + temp;
}


/**
 * Get threshold based on the given delta
 * the hardcoded values are claimed to be experimentally drawn
 */
float Utilities::getThresholdPerDelta(float delta) {
    static const float p[2] = {0.1341, 0.0278};
    static const float safety = 0.02;

    return p[0] * delta + p[1] - safety;
}

double Utilities::normal_dist() {
    static std::minstd_rand g;
    static std::normal_distribution<double> normal(0.0, 0.33333);
    return normal(g);
}

double Utilities::gausian_noise(double u) {
    // Three sigma rule
    double noise = normal_dist();

    // Clamp [-1;1]
    return (noise < -1 ? -1 : noise > 1 ? 1 : noise) * u;
}

double Utilities::uniform_dist() {
    static std::uniform_real_distribution<double> uniform(0.0, 1.0);
    static std::minstd_rand g;
    return uniform(g);
}

/**
 * From given list of configurations, convert them into affine matrices.
 * But filter out all the rectangles that are out of the given boundaries.
 **/
std::vector<cv::Mat> Utilities::configsToAffine(
        std::vector<fast_match::MatchConfig> &configs, std::vector<bool> &insiders,
        const cv::Size& imageSize, const cv::Size& templSize
) {
    auto no_of_configs = static_cast<int>(configs.size());
    std::vector<cv::Mat> affines((unsigned long) no_of_configs);

    /* The boundary, between -10 to image size + 10 */
    Point2d top_left(-10., -10.);
    Point2d bottom_right(imageSize.width + 10, imageSize.height + 10);


    /* These are for the calculations of affine transformed corners */
    int r1x = (int) (0.5 * (templSize.width - 1)),
            r1y = (int) (0.5 * (templSize.height - 1)),
            r2x = (int) (0.5 * (templSize.width - 1)),
            r2y = (int) (0.5 * (templSize.height - 1));

    Mat corners = (Mat_<float>(3, 4) << 1 - (r1x + 1), templSize.width - (r1x + 1), templSize.width - (r1x + 1), 1 - (r1x + 1),
            1 - (r1y + 1), 1 - (r1y + 1), templSize.height - (r1y + 1), templSize.height - (r1y + 1),
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
    std::vector<cv::Mat> result;
    for (int i = 0; i < no_of_configs; i++) {
        if (insiders[i])
            result.push_back(affines[i]);
    }

    return result;
}

cv::Point Utilities::calculateLocationInMap(
        const cv::Size& image_size,
        const cv::Size& templ_size,
        const cv::Mat& affine,
        const cv::Point& templPoint
) {
    auto r1x = static_cast<float>(0.5 * (templ_size.width - 1)),
            r1y = static_cast<float>(0.5 * (templ_size.height - 1)),
            r2x = static_cast<float>(0.5 * (image_size.width - 1)),
            r2y = static_cast<float>(0.5 * (image_size.height - 1));

    float a11 = affine.at<float>(0, 0),
            a12 = affine.at<float>(0, 1),
            a13 = affine.at<float>(0, 2),
            a21 = affine.at<float>(1, 0),
            a22 = affine.at<float>(1, 1),
            a23 = affine.at<float>(1, 2);

    return cv::Point(
            (int) (a11 * ((templPoint.x + 1)- (r1x + 1)) + a12 * ((templPoint.y + 1) - (r1y + 1)) + (r2x + 1) + a13),
            (int) (a21 * ((templPoint.x + 1) - (r1x + 1)) + a22 * ((templPoint.y + 1) - (r1y + 1)) + (r2y + 1) + a23)
    );
}

cv::Mat Utilities::extractMapPart(const cv::Mat &map,
                                  const cv::Size &size, const cv::Point &position, double angle, float scale) {
    // Calculate how much data to crop out
    cv::Mat image;
    float imangle = std::atan(((float) size.width) / ((float) size.height));
    int roiReserver = (int) std::ceil(size.width / std::sin(imangle));
    int centerLoc = roiReserver / 2;
    // Cut part of map
    cv::Mat region = map(cv::Rect(position.x - centerLoc, position.y - centerLoc, roiReserver, roiReserver));
    // Prepare cropping after rotation
    cv::Rect rotationRoi(centerLoc - (size.width / 2), centerLoc - (size.height / 2), size.width, size.height);
    // Actual rotation
    cv::Mat view(region.size(), region.type());
    cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point(centerLoc, centerLoc), angle, scale);
    cv::warpAffine(region, view, rot_mat, cv::Size(roiReserver, roiReserver));// cv::Size(roiReserver, roiReserver));
    view(rotationRoi).copyTo(image);
    return image;
}

cv::cuda::GpuMat Utilities::extractMapPart(const cv::cuda::GpuMat &map,
                                  const cv::Size &size, const cv::Point &position, double angle, float scale) {
    // Calculate how much data to crop out
    cv::cuda::GpuMat image;
    float imangle = std::atan(((float) size.width) / ((float) size.height));
    int roiReserver = (int) std::ceil(size.width / std::sin(imangle));
    int centerLoc = roiReserver / 2;
    // Cut part of map
    cv::cuda::GpuMat region = map(cv::Rect(position.x - centerLoc, position.y - centerLoc, roiReserver, roiReserver));
    // Prepare cropping after rotation
    cv::Rect rotationRoi(centerLoc - (size.width / 2), centerLoc - (size.height / 2), size.width + 1, size.height + 1);
    // Actual rotation
    cv::cuda::GpuMat view(region.size(), region.type());
    cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point(centerLoc, centerLoc), angle, scale);
    cv::cuda::warpAffine(region, view, rot_mat, cv::Size(roiReserver, roiReserver));// cv::Size(roiReserver, roiReserver));
    view(rotationRoi).copyTo(image);
    return image;
}

