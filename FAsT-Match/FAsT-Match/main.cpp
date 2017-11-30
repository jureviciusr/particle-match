//
//  main.cpp
//  FAsT-Match
//
//  Created by Saburo Okita on 22/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <chrono>

#include "FAsTMatch.h"
#include "MatchConfig.h"

using namespace std;
using namespace std::chrono;
using namespace cv;

int main(int argc, const char * argv[])
{

    Mat image = imread( "image.png" );
    Mat templ = imread( "template.png" );
    
    fast_match::FAsTMatch fast_match;
    fast_match.init( 0.15f, 0.85f, false, 0.5f, 2.0f );

    double distance;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    vector<Point2f> corners = fast_match.apply(image, templ, distance, -M_PI);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    cout << "Time took: " << (double) duration_cast<milliseconds>( t2 - t1 ).count() / 1000. << "\n";
    
    namedWindow("");
    moveWindow("", 0, 0);
    
    line( image, corners[0], corners[1], Scalar(0, 0, 255), 2);
    line( image, corners[1], corners[2], Scalar(0, 0, 255), 2);
    line( image, corners[2], corners[3], Scalar(0, 0, 255), 2);
    line( image, corners[3], corners[0], Scalar(0, 0, 255), 2);
    
    Mat appended( image.rows, 2 * image.cols, CV_8UC3, Scalar(0, 0, 0) );

    putText(appended, "Template: ", Point(50, 50), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);


    //QtFont font = fontQt("Times");
    //addText( appended, "Template: ", Point( 50, 50 ), font );
    templ.copyTo( Mat(appended, Rect((image.cols - templ.cols) / 2, (image.rows - templ.cols) / 2, templ.cols, templ.rows)) );
    image.copyTo( Mat(appended, Rect( image.cols, 0, image.cols, image.rows)) );
    
    imshow("", appended );
    waitKey(0);
    
    return 0;
}

