//
//  main.cpp
//  FaceDetectionVJ
//
//  Created by zineb adib on 27/01/2016.
//  Copyright © 2016 zineb adib. All rights reserved.
//




#include <iostream>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>





/*-------Déclaration des fonctions------*/

char key; // va servir juste à aqq qqrrêter le programme quand on le veut avec la touche q
void detectFaces(IplImage *img);
CvHaarClassifierCascade *cascade; //contient le classifieur
CvMemStorage *storage; //définit l'espace mémoire alloué au flux vidéo


int main(void) {
    
    CvCapture *capture; //flux vidéo récupéré par la cam
    
    IplImage *img; //image à l'instant t du flux vidéo
    const char *filename = "haarcascade_frontalface_alt.xml";
    
    /* Chargement du classifieur */
    cascade = (CvHaarClassifierCascade*)cvLoadHaarClassifierCascade( filename, cvSize(24, 24) );
    
    
    /*-------FLUX VIDEO------*/
    
    
    /* Ouverture du flux video de la camera */
    
    //capture = cvCreateCameraCapture(-1);
    
    capture = cvCreateCameraCapture(CV_CAP_ANY);
    //capture = cvCaptureFromCAM(CV_CAP_ANY);
    //capture = cvCreateFileCapture("/Users/zinebadib/Desktop/S1L11357V1.avi" );// si jamais on a un problème avec la webcam
    
    
    /*-------FLUX VIDEO------*/
    
    
    /* Initialisation de l’espace memoire */
    storage = cvCreateMemStorage(0);
    
    /* Creation d’une fenetre */
    cvNamedWindow("Window-FT", 1);
    
    /* Boucle de traitement: tant qu'on a pas appuyé sur la touche q pour arrêter le programme, on stocke l'image en cours du flux vidéo */
    while(key != 'q')
    {
        img = cvQueryFrame(capture);
        detectFaces(img);
        key = cvWaitKey(10); //le système attend 10mins avant d'arrêter le programme
    }
    
    /* Liberation de l’espace memoire*/
    cvReleaseCapture(&capture);
    cvDestroyWindow("Window-FT");
    cvReleaseHaarClassifierCascade(&cascade);
    cvReleaseMemStorage(&storage);
    return 0;
}

//Face detection function
void detectFaces(IplImage *img)
{
    /* Declaration des variables */
    int i;
    CvSeq *faces = cvHaarDetectObjects(img, cascade, storage, 1.1, 3, 0, cvSize(40,40)); //si jamais trop lent augmenter la taille à 100,100
    
    //dessiner un rectangle autour des visages détectés
    for(i=0; i<(faces?faces->total:0); i++)
    {
        CvRect *r = (CvRect*)cvGetSeqElem(faces, i);
        cvRectangle(img, cvPoint(r->x, r->y), cvPoint(r->x + r->width, r->y + r->height), CV_RGB(255, 0, 0), 1, 8, 0);
    }
    
    cvShowImage("Window-FT", img); //affichage
}

