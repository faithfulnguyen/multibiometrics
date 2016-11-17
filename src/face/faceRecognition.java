/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package face;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.BORDER_DEFAULT;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_imgcodecs;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.GaussianBlur;
import static org.bytedeco.javacpp.opencv_imgproc.equalizeHist;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import org.bytedeco.javacpp.opencv_objdetect;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_FIND_BIGGEST_OBJECT;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_SCALE_IMAGE;

/**
 *
 * @author Nguyen Trung Tin
 */
public class faceRecognition {
    private opencv_objdetect.CascadeClassifier face_cascade;
    private ArrayList<ArrayList<Mat>> trainData;
    private ArrayList<ArrayList<Mat>> testData;
    private ArrayList<ArrayList<Mat>> faceDataTrain;
    private ArrayList<ArrayList<Mat>> faceDataTest;
   
    public faceRecognition(){
        this.trainData = new ArrayList<ArrayList<Mat>>();
        this.testData = new ArrayList<ArrayList<Mat>>();
        this.faceDataTest = new ArrayList<ArrayList<Mat>>();
        this.faceDataTrain = new ArrayList<ArrayList<Mat>>();
        File folder = new File("");
        String fileName = folder.getAbsolutePath() + "/src/haarcascades/haarcascade_frontalface_default.xml";
        this.face_cascade = new opencv_objdetect.CascadeClassifier();
        this.face_cascade.load(fileName);
    }
    
    public void readImage() throws IOException{
        File folder = new File("");
        String fileName = folder.getAbsolutePath() + "/src/Database/face";
        System.out.println("Read face images!");
        File[] listOfFiles = new File(fileName).listFiles();
        Arrays.sort(listOfFiles);
        for(int idx = 0; idx < listOfFiles.length / 8 ; idx++){
            ArrayList<Mat> trt = new ArrayList<>();
            ArrayList<Mat> tst = new ArrayList<>();
            ArrayList<Mat> fctrt = new ArrayList<>();
            ArrayList<Mat> fctst = new ArrayList<>();
            for (int i = 0; i < 8; i++) {
                if (listOfFiles[i + idx * 8].getName().contains(".jpg")){
                    String name =  listOfFiles[i + idx * 8].getName();
                    Mat image = imread(fileName + "/" + name, opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
                    Mat face = faceDetector(image);
                    resize(face, face, new Size(60, 60));
                    equalizeHist(face, face);  
                    resize(image, image, new Size(60, 60));
                    GaussianBlur(image, image, new Size(3, 3), 0, 0, BORDER_DEFAULT);
                    if(i < 6){
                        trt.add(image);
                        fctrt.add(face);
                    }else{
                        tst.add(image);
                        fctst.add(face);
                    }
                }
            }
            this.faceDataTrain.add(fctrt);
            this.faceDataTest.add(fctst);
            this.trainData.add(trt);
            this.testData.add(tst);
        }
    }
    
    public Mat faceDetector(Mat image){
        opencv_core.RectVector objectList = new opencv_core.RectVector();
         //Only support biggest face
        face_cascade.detectMultiScale( image, objectList, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, 
                                        new Size(120, 90), new Size(image.cols(), image.cols()) );
        Mat face_resized = new Mat();
        if (objectList.size() <= 0 ){
                resize(image, image, new Size(60, 60));
                return image;
        }
        
        opencv_core.Rect maxRect = new opencv_core.Rect();
        maxRect = objectList.get(0);
//        maxRect.x((int)(maxRect.x() + maxRect.x() * 0.1));
//        maxRect.width((int)(maxRect.width() + maxRect.width() * 0.1));
        face_resized = image.apply(maxRect);
        return face_resized;
    }
   
    public ArrayList<ArrayList<opencv_core.Mat>> getDataTrain(){
        return this.trainData;
    }
   
    public ArrayList<ArrayList<opencv_core.Mat>> getDataTest(){
        return this.testData;
    }
    public ArrayList<ArrayList<opencv_core.Mat>> getFaceDataTest(){
        return this.faceDataTest;
    }
    
    public ArrayList<ArrayList<opencv_core.Mat>> getFaceDataTrain(){
        return this.faceDataTrain;
    }
}
