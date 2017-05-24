package com.dkinal.genderNeuralNetwork;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;

/**
 * @author Dominik Kinal <kinaldominik@gmail.com>
 */
public class Trainer {
    public static final Size imageDataSize60 = new Size(60, 60);
    public static final Size imageDataSize90 = new Size(90, 90);
    public static final Size imageDataSize120 = new Size(120, 120);

    public static final String KnowledgePath = "src/res/knowledge/";
    public static final String TrainingPath = "src/res/training/";
    public static final String TestPath = "src/res/test/";

    public static final int FEMALE = 0;
    public static final int MALE = 1;

    private Size size;

    private AverageImage[] images; // 0 - female, 1 - male

    public Trainer() {
        size = imageDataSize90;

        images = new AverageImage[2];
        for(int i = 0; i < 2; i++)
            images[i] = new AverageImage(size);
    }

    /**
     * Tworzy średnią ze wszystkich zdjęć dla mezczyzny/kobiety.
     * @param gender plec
     * @param files zdjecia
     */
    private void trainType(int gender, String[] files) {
        for(String path : files) {
            Mat mat = Imgcodecs.imread(path, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Imgproc.resize(mat, mat, size);
            mat = AverageImage.toMedial(mat);

            images[gender].addImage(mat);

            System.out.println("Training (" + gender + "): " + images[gender].getWeight());
        }
    }

    public void saveKnowledge() {
        for(int i = 0; i < 2; i++) {
            images[i].saveImage(KnowledgePath + i + ".dat");
            images[i].saveAsJPG(KnowledgePath + i + ".jpg");
        }
    }

    public boolean loadKnowledge() {
        boolean found = true;

        for(int i = 0; i < 2; i++) {
            File f = new File(KnowledgePath + i + ".dat");

            if(f.exists() && !f.isDirectory())
                images[i] = AverageImage.loadImage(KnowledgePath + i + ".dat");
            else {
                System.out.println("Knowledge not found: " + KnowledgePath + i + ".dat");
                found = false;
            }
        }

        return found;
    }

    public void train() throws IOException {
        String[] femaleFiles = getFilesOfType(TrainingPath,"female");
        String[] maleFiles = getFilesOfType(TrainingPath,"male");

        trainType(FEMALE, femaleFiles);
        trainType(MALE, maleFiles);
    }

    static String[] getFilesOfType(String path, String type) throws IOException {

        File[] files = new File(path + type).listFiles();
        if(files == null)
            throw new IOException("0 files");

        String[] list = new String[files.length];

        for(int i = 0; i < files.length; i++) {
            list[i] = files[i].getAbsolutePath();
        }

        return list;
    }

    public int[] predict(Mat mat) {
        //mat = AverageImage.toMedial(mat);

        double femaleSimilarity = images[FEMALE].compareMat(mat);
        double maleSimilarity = images[MALE].compareMat(mat);

//        System.out.println("FEMALE: " + femaleSimilarity);
//        System.out.println("MALE: " + maleSimilarity);

        return new int[] {(int)(femaleSimilarity*1000), (int)(maleSimilarity*1000)+10};
    }

    int[] test(String path) {
        Mat mat = Imgcodecs.imread(path, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
        Imgproc.resize(mat, mat, size);
        return predict(mat);
    }

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) throws IOException {
        Trainer trainer = new Trainer();
        if(!trainer.loadKnowledge()) {
            trainer.train();
            trainer.saveKnowledge();
        }

//        trainer.test(TestPath + "0.jpg");
//        trainer.test(TestPath + "1.jpg");

        int femaleRight = 0;
        int maleRight = 0;
        String[] females = getFilesOfType(TestPath, "female");
        String[] males = getFilesOfType(TestPath, "male");
        int[] results;
        for(String s : females) {
            results = trainer.test(s);
            if(results[0] > results[1])
                femaleRight++;
        }
        System.out.println("Female right: " + femaleRight + " / " + females.length);

        for(String s : males) {
            results = trainer.test(s);
            if(results[1] > results[0])
                maleRight++;
        }
        System.out.println("Male right: " + maleRight + " / " + males.length);
    }
}
