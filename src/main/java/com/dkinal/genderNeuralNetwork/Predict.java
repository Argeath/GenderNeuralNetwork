package com.dkinal.genderNeuralNetwork;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;


public class Predict {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public static void main(String[] args) {
		
		Trainer trainer = new Trainer();
        trainer.loadKnowledge();

        File[] files = new File("src/res/test/school").listFiles();
        int index = 0;
		for(File s : files) {
            String imageFilePath = s.getAbsolutePath();
            try {
                Mat[] faces = new FaceDetector().snipFace(imageFilePath, new Size(90, 90));


                int faceNo = 1;
                Mat grayed = new Mat(90, 90, CvType.CV_8UC1);

                for (Mat face : faces) {
                    Imgproc.cvtColor(face, grayed, Imgproc.COLOR_RGB2GRAY);
                    int prediction = trainer.predict(grayed);

                    if (prediction > 0) {
                        Imgcodecs.imwrite("src/res/test/school/" + index++ + "_" + faceNo + "_female_" + prediction + ".jpg", face);
                    } else {
                        Imgcodecs.imwrite("src/res/test/school/" + index++ + "_" + faceNo + "_male_" + prediction + ".jpg", face);
                    }

                    faceNo++;
                }
            } catch(Exception ex) {
                System.out.println("Photo: " + s.getName() + " exception");
            }
        }
	}
}
