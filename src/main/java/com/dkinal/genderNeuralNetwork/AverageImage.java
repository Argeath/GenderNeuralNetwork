package com.dkinal.genderNeuralNetwork;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.*;

/**
 * @author Dominik Kinal <kinaldominik@gmail.com>
 */
public class AverageImage implements Serializable {
    private static final long serialVersionUID = 1L;

    int width;
    int height;

    int[][] image;
    int weight;

    public AverageImage(Size size) {
        width = (int)size.width;
        height = (int)size.height;
        weight = 0;

        image = new int[(int)size.height][(int)size.width];
        for(int i = 0; i < size.height; i++)
            for(int j = 0; j < size.width; j++)
                image[i][j] = 0;
    }

    public void addImage(Mat mat) {
        double sumValue;
        int value;

        // Srednia ze wszystkich plikow dla kazdego pixela
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                sumValue = getValue(row, col) * weight + mat.get(row, col)[0];

                value = (int) (sumValue / (weight + 1));

                setValue(row, col, value);
            }
        }

        weight++;
    }

    /**
     *  Przerabia zdjecie, aby mialo srednia wartosc pikseli rowna 0.5.
     *  Aby jedno zdjecie podczas uczenia nie mialo wiekszego wplywu niz inne.
     * @param mat Zdjecie
     * @return Usrednione Zdjecie
     */
    public static Mat toMedial(Mat mat) {
        Mat mat2 = new Mat(mat.size(), mat.type());

        double mediumPixel = 0;
        double sumOfPixelByRow = 0;
        double sumOfPixelByColInRow = 0;

        int rows = mat.rows();
        int cols = mat.cols();

        for(int x=0; x<rows; x++){
            sumOfPixelByRow=0;
            for(int y=0; y<cols; y++){
                sumOfPixelByRow = sumOfPixelByRow + mat.get(x, y)[0];
            }

            sumOfPixelByRow /= cols; // srednia w rzedzie
            sumOfPixelByColInRow += sumOfPixelByRow;
        }

        mediumPixel = sumOfPixelByColInRow/rows; // srednia wartosc na zdjeciu

        int perfectMediumPixel = 255/2;


        int mediumValue = 0;
        int pixelValue = 0;
        for(int x=0; x<rows; x++){
            for(int y=0; y<cols; y++){
                pixelValue = (int) mat.get(x, y)[0];

                mediumValue = (int) (pixelValue*perfectMediumPixel/mediumPixel);

                if(mediumValue>255){
                    mediumValue=255;
                }

                mat2.put(x, y, mediumValue);
            }
        }

        return mat2;
    }

    /**
     * Oblicza srednia roznic dla kazdego pixela
     * @param mat Zdjecie porownywane
     * @return Podobienstwo (0 - 1)
     */
    public double compareMat(Mat mat) {
        double pixel1, pixel2, pixelError;
        double sumOfError = 0;

        for(int row=0; row < height; row++) {
            for(int col=0; col < width; col++) {
                pixel1 = (double)getValue(row, col);
                pixel2 = mat.get(row, col)[0];

                pixelError = Math.abs(pixel1 - pixel2);
                sumOfError += pixelError;
            }
        }

        double error = sumOfError / (width * height * 255);

        return 1 - error;
    }

    private int getValue(int row, int col) {
        return image[row][col];
    }

    private void setValue(int row, int col, int value) {
        image[row][col] = value;
    }

    public void saveAsJPG(String path) {
        Mat img = new Mat(height, width, CvType.CV_8UC1);

        for(int row=0; row < height; row++) {
            for(int col=0; col < width; col++) {
                img.put(row, col, getValue(row, col));
            }
        }

        Imgcodecs.imwrite(path, img);
    }

    public void saveImage(String path) {

        try(ObjectOutputStream stream = new ObjectOutputStream(new FileOutputStream(path))) {
            stream.writeObject(this);
            System.out.println("Saved image to " + path);

        } catch(Exception ex) {
            ex.printStackTrace();
        }
    }

    public static AverageImage loadImage(String path) {
        AverageImage img = null;
        try(ObjectInputStream stream = new ObjectInputStream(new FileInputStream(path))) {

            img = (AverageImage) stream.readObject();

        } catch(Exception ex) {
            ex.printStackTrace();
        }

        return img;
    }

    public int getWeight() {
        return weight;
    }
}
