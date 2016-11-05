/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multibiometric;
import fingerprint.fingerprintRecognition;
import face.faceRecognition;
import java.io.IOException;
import java.util.ArrayList;
import org.bytedeco.javacpp.indexer.DoubleRawIndexer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_core.normalize;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.calcHist;
import static org.bytedeco.javacpp.opencv_imgproc.resize;


/**
 *
 * @author Nguyen Trung Tin
 */
public class MultiBiometric {
    private final ArrayList<ArrayList<Mat>> trainData;
    private final ArrayList<ArrayList<Mat>> testData;
    public MultiBiometric(){
        this.trainData = new ArrayList<>();
        this.testData = new ArrayList<>();
    }
    /**
     * @param args the command line arguments
     * @throws java.io.IOException
     */
    
    public static void main(String[] args) throws IOException {
        // TODO code application logic here
        faceRecognition face = new faceRecognition();
        fingerprintRecognition fingerprtn = new fingerprintRecognition();
        MultiBiometric mlt = new MultiBiometric();
        face.readImage();
        fingerprtn.readImage();
        System.out.println("fusion faces and fingerprints");
        ArrayList<ArrayList<Mat>> tmpTstfc = face.getDataTest();
        ArrayList<ArrayList<Mat>> tmpTstfg = fingerprtn.getDataTest();
        mlt.fusionImagesTest(tmpTstfc, tmpTstfg);
        mlt.fusionImagesTrain(face.getDataTrain(), fingerprtn.getDataTrain());
        mlt.match();
    }
  
    public Mat fusionImage(Mat face, Mat fgprint){
        Mat fusion = new Mat(face.rows() + fgprint.rows(), fgprint.cols(), face.type());
        Mat fc = fusion.apply(new Rect(0, 0, face.rows(), face.cols()));
        face.copyTo(fc);
        Mat fg = fusion.apply(new Rect(0, face.rows(), fgprint.rows(), fgprint.cols()));
        fgprint.copyTo(fg);
        return fusion;
    }
    
    public void fusionImagesTest(ArrayList<ArrayList<Mat>> face, ArrayList<ArrayList<Mat>> fgprint){
        for(int i = 0; i < face.size(); i++){
            ArrayList<Mat> tmp = new ArrayList<>();
            for(int j = 0; j < face.get(0).size(); j++){
                Mat fImage = fusionImage(face.get(i).get(j), fgprint.get(i).get(j));
                imwrite("tst" + i + j + ".jpg", fImage); 
                Mat hist = baseProcess.Process.gaborSubWindow(fImage);
                tmp.add(hist);
            }
            this.testData.add(tmp);
        }
    }
    
    public void fusionImagesTrain(ArrayList<ArrayList<Mat>> face, ArrayList<ArrayList<Mat>> fgprint){
        for(int i = 0; i < face.size(); i++){
            ArrayList<Mat> tmp = new ArrayList<>();
            for(int j = 0; j < face.get(0).size(); j++){
                Mat fImage = fusionImage(face.get(i).get(j), fgprint.get(i).get(j));
                imwrite("trt" + i + j + ".jpg", fImage);
                Mat hist = baseProcess.Process.gaborSubWindow(fImage);
                tmp.add(hist);
            }
            this.trainData.add(tmp);
        }
    }
    
    public double[] findClass(Mat hist){
        double[] score = new double[this.trainData.size()];
        for(int i = 0; i < this.trainData.size(); i++){
            double tmp = 0;
            for(int j = 0; j < this.trainData.get(i).size(); j++){
                double dis = baseProcess.Process.chiSquare(hist, this.trainData.get(i).get(j));
                tmp += dis;
            }
            tmp = tmp / (this.trainData.get(0).size() * 1.0);
            score[i] = tmp; 
        }
        double[] min = new double[2];
        min[0] = 10000000;
        for (int i = 0; i < score.length; i++){
            if(min[0] > score[i]){
                min[0] = score[i];
                min[1] = i;
            }
        }
        return min;
    }
    
    public void match(){
        int err = 0;
        int[] label = new int[this.testData.size() * this.testData.get(0).size()];
        for(int i = 0; i < this.testData.size(); i++){
            for(int j = 0; j < this.testData.get(0).size(); j++){
                label[i* this.testData.get(0).size() + j] = i;
            }
        }
        for(int i = 0; i < this.testData.size(); i++){
            double[] a;
            for(int ele = 0; ele < this.testData.get(0).size(); ele++){
                a = this.findClass(this.testData.get(i).get(ele));
                if(a[1] != label[i * this.testData.get(0).size() + ele ])
                    err++;
                System.out.println( ele + ": " + "Predict: " + a[1] + " : " +  label[i * this.testData.get(0).size() + ele ] + " Distance: " + a[0]);
            }
            System.out.println(".......................");
        }
        System.out.println("Error : " + err + " Total: " + this.testData.size() *  this.testData.get(0).size());
        System.out.println( "Accuracy rate: " + (1 - ( err * 1.0) / (this.testData.size() * this.testData.get(0).size())));
    }
     
}

