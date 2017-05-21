package com.dkinal.genderNeuralNetwork;

import com.dkinal.genderNeuralNetwork.weightedPixel.WeightedStandardImage;
import com.dkinal.genderNeuralNetwork.weightedPixel.WeightedStandardPixelTrainer;
import org.opencv.core.Core;

import java.io.File;
import java.io.IOException;


public class Train {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	private static String trainingFolderPath = "src/res/trainingData";

	public static void main(String[] args) throws IOException {
		System.out.println("Welcome to OpenCV " + Core.VERSION);
		//list of image files//////////////////////////////////////////////////////////////////////////////////////////
		
		File trainingFolder = new File(trainingFolderPath);
		String[] trainingSubfolderPaths = trainingFolder.list((current, name) -> new File(current, name).isDirectory());

		String[] femaleFiles = getFilesOfType("female");
		String[] maleFiles = getFilesOfType("male");

		///test
		/*/for(int i=filePaths.length-1; i>=0; i--){
			System.out.println("filePaths: " + filePaths[i]);
			System.out.println("ids: " + ids[i]);
		}/**/
		
		
		//train////////////////////////////////////////////////////////////////////////////////////////////////////////
		WeightedStandardPixelTrainer weightedStandardPixelTrainer = new WeightedStandardPixelTrainer();
		weightedStandardPixelTrainer.train(femaleFiles, maleFiles);
		WeightedStandardImage weightedStandardImage = weightedStandardPixelTrainer.getWeightedStandardImage();
		
		weightedStandardImage.saveKnowledge("src/res/knowledge/Knowledge.log");
		
		System.out.println("Operation successful!!!");
	}

	static String[] getFilesOfType(String type) throws IOException {

		File[] files = new File(trainingFolderPath + "\\" + type).listFiles();
		if(files == null)
			throw new IOException("0 files");

		String[] list = new String[files.length];

		for(int i = 0; i < files.length; i++) {
			list[i] = files[i].getAbsolutePath();
		}

		return list;
	}
}
