/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fingerprint;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.BORDER_DEFAULT;
import org.bytedeco.javacpp.opencv_imgcodecs;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import org.bytedeco.javacpp.opencv_imgproc;
import static org.bytedeco.javacpp.opencv_imgproc.GaussianBlur;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 *
 * @author Nguyen Trung Tin
 */
public class fingerprintRecognition {
    
    // instance
    private final ArrayList<ArrayList<opencv_core.Mat>> trainData;
    private final ArrayList<ArrayList<opencv_core.Mat>> testData;
   
    // method
    public fingerprintRecognition(){
        this.trainData = new ArrayList<>();
        this.testData = new ArrayList<>();
    }
    
    public void readImage(){
        File folder = new File("");
        String fileName = folder.getAbsolutePath() + "/src/Database/crop";
        System.out.println("Read fingerprint images!");
        File[] listOfFiles = new File(fileName).listFiles();
        Arrays.sort(listOfFiles);
        for(int idx = 0; idx < listOfFiles.length / 8 ; idx++){
            ArrayList<opencv_core.Mat> trt = new ArrayList<>();
            ArrayList<opencv_core.Mat> tst = new ArrayList<>();
            for (int i = 0; i < 8; i++) {
                if (listOfFiles[i + idx * 8].getName().contains(".tif")){
                    String name =  listOfFiles[i + idx * 8].getName();
                    opencv_core.Mat image = imread(fileName + "/" + name, opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
                    resize(image, image, new opencv_core.Size(60, 60));
                    opencv_imgproc.equalizeHist(image, image);
                    opencv_core.Mat norlImg = normalizeSubWindow(image);
                    if(i < 6){
                        trt.add(norlImg);
                    }else
                        tst.add(norlImg);
                }
            }
            this.trainData.add(trt);
            this.testData.add(tst);
        }
    }

    public static opencv_core.Mat normalizeSubWindow(opencv_core.Mat image){
        int ut = 139;
        int vt = 100;//1607;  
        double u = meanMatrix(image);
        double v = variance(image, u);
        UByteRawIndexer idx = image.createIndexer();
        for(int i = 0; i < image.rows(); i++){
            for(int j = 0; j < image.cols() ; j++){
                double beta = Math.sqrt((vt * 1.0 / v ) * (Math.pow(idx.get(i, j) - u, 2)));
                if(idx.get(i, j) > ut){
                    idx.put(i, j, 0, (int)ut + (int)beta);
                }
                else idx.put(i, j, 0, Math.abs((int)ut - (int)beta));      
            }
        }
        return image;
    }
    
    public static double variance(opencv_core.Mat image, double mean){
        double var = 0; 
        UByteRawIndexer idx = image.createIndexer();
        for(int i = 0; i < image.rows(); i++){
            for(int j = 0; j < image.cols(); j++){
                var += Math.pow((idx.get(i, j) - mean), 2);
            }
        }
        var /= (image.cols() * image.rows());
        return var;
    }
    
    public static double meanMatrix(opencv_core.Mat img){
        double sum = 0;
        UByteRawIndexer idx = img.createIndexer();
        for(int i = 0; i < img.rows(); i++){
            for(int j = 0; j < img.cols(); j++){
                sum += idx.get(i, j, 0);
            }
        }
        sum /= (img.cols() * img.rows());
        return sum;
        
    }
    
    public ArrayList<ArrayList<opencv_core.Mat>> getDataTrain(){
        return this.trainData;
    }
   
    public ArrayList<ArrayList<opencv_core.Mat>> getDataTest(){
        return this.testData;
    }
}
