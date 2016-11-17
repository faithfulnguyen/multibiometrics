/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package baseProcess;

import org.bytedeco.javacpp.indexer.DoubleRawIndexer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_core.CV_64F;
import static org.bytedeco.javacpp.opencv_core.CV_PI;
import org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.normalize;
import static org.bytedeco.javacpp.opencv_imgproc.calcHist;
import static org.bytedeco.javacpp.opencv_imgproc.filter2D;
import static org.bytedeco.javacpp.opencv_imgproc.getGaborKernel;

/**
 *
 * @author Nguyen Trung Tin
 */
public class Process {
    
    public static double euclideanDistance(Mat img, Mat img1){
        double score = 0;
        DoubleRawIndexer idx = img.createIndexer();
        DoubleRawIndexer idx1 = img1.createIndexer();
        for(int i = 0; i < img.cols(); i++){
            score += Math.pow((idx.get(0, i) - idx1.get(0, i)), 2);
        }
        score = Math.sqrt(score);
        return score;
    }
    
    public static double chiSquare(Mat img, Mat img1){
        float dis = 0;
        DoubleRawIndexer idx = img.createIndexer();
        DoubleRawIndexer idx1 = img1.createIndexer();
        for(int i = 0; i < img.cols(); i++){
            if((idx.get(0, i) + idx1.get(0, i)) == 0){
               dis += 0;
            }
            else{
               dis += Math.pow((idx.get(0, i) - idx1.get(0, i)), 2)/(idx.get(0, i) + idx1.get(0, i));
            }
        }
        return dis;
   }
   
    public static Mat calcLBP(Mat image){
        Mat lbp = new Mat(image.rows() - 2, image.cols() - 2, image.type());
        UByteRawIndexer dst1Idx = image.createIndexer();
        UByteRawIndexer dst = lbp.createIndexer();
        int rows = image.rows(), cols = image.cols();
        for( int r = 1; r < rows - 1; r++){
            for( int c = 1; c < cols - 1; c++){
                float center = dst1Idx.get(r, c, 0);
                int code = 0;
                if(dst1Idx.get(r - 1, c - 1, 0) >= center){
                        code += 128;
                }
                if(dst1Idx.get(r - 1, c, 0) >= center){
                        code += 64;
                }
                if(dst1Idx.get(r - 1, c + 1, 0) >= center){
                    code += 32;
                }
                if(dst1Idx.get(r, c + 1, 0) >= center){
                    code += 16;
                }
                if(dst1Idx.get(r + 1, c + 1, 0) >= center){
                    code += 8;
                }
                if(dst1Idx.get(r + 1, c, 0) >= center){
                    code += 4;
                }
                if(dst1Idx.get(r + 1, c - 1, 0) >= center){
                    code += 2;
                }
                if(dst1Idx.get(r, c - 1, 0) >= center){
                    code += 1;
                }
                dst.put(r - 1, c - 1, code);
            }
        }
        return lbp;	
    }
    
    public static double sumMatrix(Mat rect){
        double sum = 0;
        UByteIndexer idx = rect.createIndexer();
        for(int i = 0; i < rect.rows(); i++){
            for(int j = 0; j < rect.cols(); j++){
                sum += idx.get(i, j , 0);
            }
        }
        return sum;
    }
    
    public static Mat calHistogramImage(Mat image, int s){
        float[] range = { 0, 256 };
        int[] chanel = { 0 };
        int[] sz = { s };
        Boolean uniform = true; 
        Boolean accumulate = false;
        Mat hist = new Mat(256, 1, opencv_core.CV_64F);
        calcHist(image, 1, chanel, new Mat(), hist, 1, sz, range, uniform, accumulate );
        hist = hist.reshape(1, 1);
        normalize(hist, hist);
        return hist;
    }

    public static opencv_core.Mat gaborSubWindow(Mat image){
        int index = 0;
        double lm = 1, gm = 0.01, ps = CV_PI/8;
        double theta = 0;
        //double[] sig = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 };
        double sig = 3;//3
        int block = 10;
        int siz = image.cols() / block;
        int ori = 15;
        int shst = 256;
        Mat hist = new Mat(1, ori * siz * shst, CV_64F);
        DoubleRawIndexer dstIdx = hist.createIndexer();
        for(int r = 0; r < image.cols(); r += block){
            for(int g = 0; g < ori; g++){
                opencv_core.Mat kernel = getGaborKernel(new opencv_core.Size(4, 4), sig, theta, lm, gm, ps, CV_32F);
                opencv_core.Mat gabor = new opencv_core.Mat(image.rows(), image.cols(), image.type());
                Mat tmp = image.apply(new opencv_core.Rect(r, 0, block, image.rows())); 
                filter2D(tmp, gabor, image.type(), kernel);  
                Mat lbp = calcLBP(gabor);
                Mat hs =  calHistogramImage(lbp, shst);
                FloatRawIndexer hdx = hs.createIndexer();
                for(int h = 0; h < hs.cols(); h++){
                    dstIdx.put(0, h + index, hdx.get(0, h));
                }
                index += shst;
                theta += 11.25;
            }
            theta = 0;     
        }      
        return hist;
    }

}
