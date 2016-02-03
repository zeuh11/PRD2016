//#include "opencv2/objdetect.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//using namespace std;
//using namespace cv;
//
///* Function Headers */
//void detectAndDisplay( Mat frame );
//
///* Global variables */
//String face_cascade_name = "haarcascade_frontalface_alt.xml"; //classifieur pour la détection du visage
//String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml"; //classifieur pour la détection des yeux
//
//CascadeClassifier face_cascade; //pour instancier un objet de classe cascade classifier qui va pouvoir ouvrir le fichier xml du classifieur
//CascadeClassifier eyes_cascade;
//
//String window_name = "Capture - Face detection";
//
///* @function main */
//int main( void )
//{
//    VideoCapture capture;
//    Mat frame;
//    
//    //-- 1. Chargement du classifieur en cascade
//    
//    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
//    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
//    
//    //-- 2. Lecture du flux vidéo
//    capture.open( -1 );
//    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
//    while (  capture.read(frame) )
//    {
//        if( frame.empty() )
//        {
//            printf(" --(!) No captured frame -- Break!");
//            break;
//        }
//        
//        //-- 3. Application du classifieur pour chaque frame (capture)
//        detectAndDisplay( frame ); //appel de la fonction de détection
//        int c = waitKey(10); 
//        if( (char)c == 27 ) { break; } // escape
//    }
//    return 0;
//}
///* @function detectAndDisplay */
//void detectAndDisplay( Mat frame )
//{
//    std::vector<Rect> faces;
//    Mat frame_gray;
//    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
//    equalizeHist( frame_gray, frame_gray );
//    //-- Detect faces
//    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
//    for( size_t i = 0; i < faces.size(); i++ )
//    {
//        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
//        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
//        Mat faceROI = frame_gray( faces[i] );
//        std::vector<Rect> eyes;
//        //-- In each face, detect eyes
//        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
//        for( size_t j = 0; j < eyes.size(); j++ )
//        {
//            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
//            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
//            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
//        }
//    }
//    //-- Show what you got
//    imshow( window_name, frame );
//}

/*
 * Author: Samyak Datta (datta[dot]samyak[at]gmail.com)
 *
 * A program to detect facial feature points using
 * Haarcascade classifiers for face, eyes, nose and mouth
 *
 */

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

// Functions for facial feature detection
static void help();
static void detectFaces(Mat&, vector<Rect_<int> >&, string);
static void detectEyes(Mat&, vector<Rect_<int> >&, string);
static void detectNose(Mat&, vector<Rect_<int> >&, string);
static void detectMouth(Mat&, vector<Rect_<int> >&, string);
static void detectFacialFeaures(Mat&, const vector<Rect_<int> >, string, string, string);

string input_image_path;
string face_cascade_path, eye_cascade_path, nose_cascade_path, mouth_cascade_path;

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, "{eyes||}{nose||}{mouth||}{help h||}");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    input_image_path = parser.get<string>(0);
    face_cascade_path = parser.get<string>(1);
    eye_cascade_path = parser.has("eyes") ? parser.get<string>("eyes") : "";
    nose_cascade_path = parser.has("nose") ? parser.get<string>("nose") : "";
    mouth_cascade_path = parser.has("mouth") ? parser.get<string>("mouth") : "";
    if (input_image_path.empty() || face_cascade_path.empty())
    {
        cout << "IMAGE or FACE_CASCADE are not specified";
        return 1;
    }
    // Load image and cascade classifier files
    Mat image;
    image = imread(input_image_path);
    
    // Detect faces and facial features
    vector<Rect_<int> > faces;
    detectFaces(image, faces, face_cascade_path);
    detectFacialFeaures(image, faces, eye_cascade_path, nose_cascade_path, mouth_cascade_path);
    
    imshow("Result", image);
    
    waitKey(0);
    return 0;
}

static void help()
{
    cout << "\nThis file demonstrates facial feature points detection using Haarcascade classifiers.\n"
    "The program detects a face and eyes, nose and mouth inside the face."
    "The code has been tested on the Japanese Female Facial Expression (JAFFE) database and found"
    "to give reasonably accurate results. \n";
    
    cout << "\nUSAGE: ./cpp-example-facial_features [IMAGE] [FACE_CASCADE] [OPTIONS]\n"
    "IMAGE\n\tPath to the image of a face taken as input.\n"
    "FACE_CASCSDE\n\t Path to a haarcascade classifier for face detection.\n"
    "OPTIONS: \nThere are 3 options available which are described in detail. There must be a "
    "space between the option and it's argument (All three options accept arguments).\n"
    "\t-eyes=<eyes_cascade> : Specify the haarcascade classifier for eye detection.\n"
    "\t-nose=<nose_cascade> : Specify the haarcascade classifier for nose detection.\n"
    "\t-mouth=<mouth-cascade> : Specify the haarcascade classifier for mouth detection.\n";
    
    
    cout << "EXAMPLE:\n"
    "(1) ./cpp-example-facial_features image.jpg face.xml -eyes=eyes.xml -mouth=mouth.xml\n"
    "\tThis will detect the face, eyes and mouth in image.jpg.\n"
    "(2) ./cpp-example-facial_features image.jpg face.xml -nose=nose.xml\n"
    "\tThis will detect the face and nose in image.jpg.\n"
    "(3) ./cpp-example-facial_features image.jpg face.xml\n"
    "\tThis will detect only the face in image.jpg.\n";
    
    cout << " \n\nThe classifiers for face and eyes can be downloaded from : "
    " \nhttps://github.com/Itseez/opencv/tree/master/data/haarcascades";
    
    cout << "\n\nThe classifiers for nose and mouth can be downloaded from : "
    " \nhttps://github.com/Itseez/opencv_contrib/tree/master/modules/face/data/cascades\n";
}

static void detectFaces(Mat& img, vector<Rect_<int> >& faces, string cascade_path)
{
    CascadeClassifier face_cascade;
    face_cascade.load(cascade_path);
    
    face_cascade.detectMultiScale(img, faces, 1.15, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    return;
}

static void detectFacialFeaures(Mat& img, const vector<Rect_<int> > faces, string eye_cascade,
                                string nose_cascade, string mouth_cascade)
{
    for(unsigned int i = 0; i < faces.size(); ++i)
    {
        // Mark the bounding box enclosing the face
        Rect face = faces[i];
        rectangle(img, Point(face.x, face.y), Point(face.x+face.width, face.y+face.height),
                  Scalar(255, 0, 0), 1, 4);
        
        // Eyes, nose and mouth will be detected inside the face (region of interest)
        Mat ROI = img(Rect(face.x, face.y, face.width, face.height));
        
        // Check if all features (eyes, nose and mouth) are being detected
        bool is_full_detection = false;
        if( (!eye_cascade.empty()) && (!nose_cascade.empty()) && (!mouth_cascade.empty()) )
            is_full_detection = true;
        
        // Detect eyes if classifier provided by the user
        if(!eye_cascade.empty())
        {
            vector<Rect_<int> > eyes;
            detectEyes(ROI, eyes, eye_cascade);
            
            // Mark points corresponding to the centre of the eyes
            for(unsigned int j = 0; j < eyes.size(); ++j)
            {
                Rect e = eyes[j];
                circle(ROI, Point(e.x+e.width/2, e.y+e.height/2), 3, Scalar(0, 255, 0), -1, 8);
                /* rectangle(ROI, Point(e.x, e.y), Point(e.x+e.width, e.y+e.height),
                 Scalar(0, 255, 0), 1, 4); */
            }
        }
        
        // Detect nose if classifier provided by the user
        double nose_center_height = 0.0;
        if(!nose_cascade.empty())
        {
            vector<Rect_<int> > nose;
            detectNose(ROI, nose, nose_cascade);
            
            // Mark points corresponding to the centre (tip) of the nose
            for(unsigned int j = 0; j < nose.size(); ++j)
            {
                Rect n = nose[j];
                circle(ROI, Point(n.x+n.width/2, n.y+n.height/2), 3, Scalar(0, 255, 0), -1, 8);
                nose_center_height = (n.y + n.height/2);
            }
        }
        
        // Detect mouth if classifier provided by the user
        double mouth_center_height = 0.0;
        if(!mouth_cascade.empty())
        {
            vector<Rect_<int> > mouth;
            detectMouth(ROI, mouth, mouth_cascade);
            
            for(unsigned int j = 0; j < mouth.size(); ++j)
            {
                Rect m = mouth[j];
                mouth_center_height = (m.y + m.height/2);
                
                // The mouth should lie below the nose
                if( (is_full_detection) && (mouth_center_height > nose_center_height) )
                {
                    rectangle(ROI, Point(m.x, m.y), Point(m.x+m.width, m.y+m.height), Scalar(0, 255, 0), 1, 4);
                }
                else if( (is_full_detection) && (mouth_center_height <= nose_center_height) )
                    continue;
                else
                    rectangle(ROI, Point(m.x, m.y), Point(m.x+m.width, m.y+m.height), Scalar(0, 255, 0), 1, 4);
            }
        }
         
    }
    
    return;
}

static void detectEyes(Mat& img, vector<Rect_<int> >& eyes, string cascade_path)
{
    CascadeClassifier eyes_cascade;
    eyes_cascade.load(cascade_path);
    
    eyes_cascade.detectMultiScale(img, eyes, 1.20, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    return;
}

static void detectNose(Mat& img, vector<Rect_<int> >& nose, string cascade_path)
{
    CascadeClassifier nose_cascade;
    nose_cascade.load(cascade_path);
    
    nose_cascade.detectMultiScale(img, nose, 1.20, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    return;
}

static void detectMouth(Mat& img, vector<Rect_<int> >& mouth, string cascade_path)
{
    CascadeClassifier mouth_cascade;
    mouth_cascade.load(cascade_path);
    
    mouth_cascade.detectMultiScale(img, mouth, 1.20, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    return;
}

