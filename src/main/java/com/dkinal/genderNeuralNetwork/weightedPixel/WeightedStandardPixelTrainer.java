/************************************************************************************************************************
* Developer: Minhas Kamal(BSSE-0509, IIT, DU)																			*
* Date: 08-Sep-2014																										*
*************************************************************************************************************************/

package com.dkinal.genderNeuralNetwork.weightedPixel;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.FileReader;

public class WeightedStandardPixelTrainer {
	public static final Size imageDataSize60 = new Size(60, 60);
	public static final Size imageDataSize90 = new Size(90, 90);
	public static final Size imageDataSize120 = new Size(120, 120);
	
	private Size imageSize;
	private int imageRow;
	private int imageCol;
	
	private WeightedStandardImage weightedStandardImage;
	
	
	public WeightedStandardPixelTrainer(){
		this.imageSize = imageDataSize90;
		imageRow = (int)this.imageSize.width;
		imageCol = (int)this.imageSize.height;
		this.weightedStandardImage = new WeightedStandardImage(2, imageSize);
	}
	
	public WeightedStandardPixelTrainer(Size imageSize){
		this.imageSize = imageSize;
        imageRow = (int)this.imageSize.width;
        imageCol = (int)this.imageSize.height;
		this.weightedStandardImage = new WeightedStandardImage(2, imageSize);
	}
	
	
	//TRAIN/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	public void train(String[] femaleFiles, String[] maleFiles){
        trainType(0, femaleFiles);
        trainType(1, maleFiles);
	}

	private void trainType(int typeNo, String[] files) {
	    int index = 0;
	    for(String path : files) {
            Mat mat = Imgcodecs.imread(path, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Imgproc.resize(mat, mat, imageSize);
            mat = toMedial(mat);

            double sumValue;
            int value;

            for (int row = 0; row < imageRow; row++) {
                for (int col = 0; col < imageCol; col++) {
                    sumValue = (weightedStandardImage.getStandardImages(typeNo, row, col) *
                            weightedStandardImage.getWeight(typeNo)) +
                            mat.get(row, col)[0];

                    value = (int) (sumValue / (weightedStandardImage.getWeight(typeNo) + 1));

                    weightedStandardImage.setStandardImages(typeNo, row, col, (short) value);
                }
            }

            weightedStandardImage.incrementWeight(typeNo);
            System.out.println(index + ": image No: " + weightedStandardImage.getWeight(typeNo));    //show progress

            index++;
        }
    }
	
	/**
	 * 
	 * @param mat
	 * @return
	 */
	private Mat toMedial(Mat mat){
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
			
			sumOfPixelByRow = sumOfPixelByRow/cols;
			sumOfPixelByColInRow = sumOfPixelByColInRow + sumOfPixelByRow;
		}
		
		mediumPixel = sumOfPixelByColInRow/rows;
		
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
	
	
	//LOAD/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * 
	 * @param filePath
	 */
	public void load(String filePath){
		
		String mainString = readFile(filePath);
		
		if(!mainString.substring(0, 24).equals("<WeightedStandardImage/>")){
			//file is corrupted
			return ;
		}
		
		int startIndex=0;
		int stopIndex=0;
		
		//total image types
		startIndex = mainString.indexOf("<types>", 24);
		stopIndex = mainString.indexOf("</types>", 24);
		int types = Integer.parseInt(mainString.substring(startIndex+7, stopIndex));
		
		//all image size
		startIndex = mainString.indexOf("<size>", stopIndex);
		stopIndex = mainString.indexOf(",", stopIndex);
		int width = Integer.parseInt(mainString.substring(startIndex+6, stopIndex));
		
		startIndex = stopIndex+1;
		stopIndex = mainString.indexOf("</size>", stopIndex);	
		int height = Integer.parseInt(mainString.substring(startIndex, stopIndex));

		
		imageSize = new Size(width, height);
		
		
		WeightedStandardImage weightedStandardImage = new WeightedStandardImage(types, imageSize);
		
		stopIndex = mainString.indexOf("<data>", stopIndex);
		for(int i=0; i<types; i++){
			//image id
			startIndex = mainString.indexOf("<id>", stopIndex);
			stopIndex = mainString.indexOf("</id>", stopIndex);
			int id = Integer.parseInt(mainString.substring(startIndex+4, stopIndex));
			
			//image number or weight
			startIndex = mainString.indexOf("<weight>", stopIndex);
			stopIndex = mainString.indexOf("</weight>", stopIndex);
			int weight = Integer.parseInt(mainString.substring(startIndex+8, stopIndex));
			weightedStandardImage.setWeight(i, weight);
			
			startIndex = mainString.indexOf("<stdImage>", stopIndex);
			stopIndex = startIndex+10;
//			stopIndex = mainString.indexOf("</stdImage>", stopIndex);
//			System.out.println(mainString.substring(startIndex+11, stopIndex-1));
			
			short pixel;
			for(int row=0, col; row<width; row++){
				for(col=0; col<height; col++){
					//standard image data
					startIndex = stopIndex + 1;
					stopIndex = mainString.indexOf(',', startIndex);
//					System.out.println(mainString.substring(startIndex, stopIndex));
					pixel = Short.parseShort(mainString.substring(startIndex, stopIndex));
					weightedStandardImage.setStandardImages(i, row, col, pixel);
				}
				
				stopIndex++;
			}
		}
		
		
		/*///test only
		System.out.println("type: " + types + "\nsize: " + width + "," + height + "\n ");
		for(int i=0; i<types; i++){
			System.out.println(i + "id: " + weightedStandardImage.getId(i));
			System.out.println(i + "weight: " + weightedStandardImage.getWeight(i));
			System.out.println(i + "data: " + weightedStandardImage.getStandardImages(i).dump());
		}
		/**/
		
		this.weightedStandardImage = weightedStandardImage;
		
		return ;
	}
	
	/**
	 * 
	 * @param filePath
	 * @return
	 */
	private String readFile(String filePath) {
		String tempString = new String(); // for temporary data store
		String Information = new String(); // #contains the full file

		try {
			BufferedReader mainBR = new BufferedReader(new FileReader(filePath)); 

			tempString = mainBR.readLine();
			while (tempString!=null) { // reading step by step
				Information = Information + tempString + "\n";
				tempString = mainBR.readLine();
			}

			mainBR.close(); // closing the file

		}catch (Exception e) {
			e.printStackTrace();
		}
		
		return Information;
	}
	
	
	//PREDICT//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * 
	 * @param matSample
	 * @return
	 */
	public int predict(Mat matSample){
		int id = 0;
		float similarity = 0;
		
		Imgproc.resize(matSample, matSample, imageSize);
//		matSample = toMedial(matSample);
		
		int types = weightedStandardImage.getTypes();
		for(int i=0; i<types; i++){
			float currentSimilarity = compareMatDiv(weightedStandardImage.getStandardImages(i), matSample) + 
					compareMatDif(weightedStandardImage.getStandardImages(i), matSample);
			
			//System.out.println(currentSimilarity);	///test
			
			if(currentSimilarity > similarity){
				similarity = currentSimilarity;
				id = i;
			}
		}
		
		if(similarity<20){	//if image is not recognized
			return -1;
		}
		
		return id;
	}
	
	/**
	 * 
	 * @param mat1
	 * @param mat2
	 * @return
	 */
	private float compareMatDiv(Mat mat1, Mat mat2){
		if(!mat1.size().equals(mat2.size())){
			System.out.println("Incompatible Data!");
			return -1;
		}
		
		float similarity = 0;
		
		int rows = mat1.rows();
		int cols = mat1.cols();
		
		float pixel1 = 0;
		float pixel2 = 0;
		float pixelSimilarity = 0;
		float sumOfSimilarityByRow = 0;
		float sumOfSimilarityByColInRow = 0;
		
		for(int row=0; row<rows; row++){
			sumOfSimilarityByRow = 0;
			
			for(int col=0; col<cols; col++){
				pixel1 = (float) mat1.get(row, col)[0];
				pixel2 = (float) mat2.get(row, col)[0];
				
				if(pixel1==pixel2){
					pixelSimilarity = 100;
				}else if(pixel1>pixel2){
					pixelSimilarity = ( (255+pixel2) / (255+pixel1) ) * 100;
				}else{
					pixelSimilarity = (pixel1/pixel2) * 100;
				}
				
				sumOfSimilarityByRow = sumOfSimilarityByRow + pixelSimilarity;
			}
			
			sumOfSimilarityByRow = sumOfSimilarityByRow/cols;
			sumOfSimilarityByColInRow = sumOfSimilarityByColInRow + sumOfSimilarityByRow;
		}
		
		similarity = sumOfSimilarityByColInRow/rows;
		
		
		if(similarity>50){
			similarity = similarity-50;
		}else{
			similarity = 50-similarity;
		}
		
		
		return similarity;
	}
	
	/**
	 * 
	 * @param mat1
	 * @param mat2
	 * @return
	 */
	private float compareMatDif(Mat mat1, Mat mat2){
		if(!mat1.size().equals(mat2.size())){
			System.out.println("Incompatible Data!");
			return -1;
		}
		
		float similarity = 0;
		
		int rows = mat1.rows();
		int cols = mat1.cols();
		
		float pixel1 = 0;
		float pixel2 = 0;
		float pixelSimilarity = 0;
		float sumOfSimilarityByRow = 0;
		float sumOfSimilarityByColInRow = 0;
		
		for(int row=0; row<rows; row++){
			sumOfSimilarityByRow = 0;
			
			for(int col=0; col<cols; col++){
				pixel1 = (float) mat1.get(row, col)[0];
				pixel2 = (float) mat2.get(row, col)[0];
				
				if(pixel1==pixel2){
					pixelSimilarity = 100;
				}else if(pixel1<pixel2){
					pixelSimilarity = (255-pixel2+pixel1)/255 * 100;
				}else{
					pixelSimilarity = (255-pixel1+pixel2)/255 * 100;
				}
				
				sumOfSimilarityByRow = sumOfSimilarityByRow + pixelSimilarity;
			}
			
			sumOfSimilarityByRow = sumOfSimilarityByRow/cols;
			sumOfSimilarityByColInRow = sumOfSimilarityByColInRow + sumOfSimilarityByRow;
		}
		
		similarity = sumOfSimilarityByColInRow/rows;
		
		
		if(similarity>50){
			similarity = similarity-50;
		}else{
			similarity = 50-similarity;
		}
		
		
		return similarity;
	}
	
	
	//GETTER_SETTER////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	public WeightedStandardImage getWeightedStandardImage(){
		return weightedStandardImage;
	}
	
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**///test only
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		//list of image files//////////////////////////////////////////////////////////////////////////////////////////
		/*String[] stringFilePaths = {"resource/Face_Male_Female/images1", "resource/Face_Male_Female/images2"};
		
		ArrayList<String> filePathList = new ArrayList<String>();
		ArrayList<Integer> idList = new ArrayList<Integer>();
		
		int id=0;
		for(String stringFilePath: stringFilePaths){
			
			File[] files = new File(stringFilePath).listFiles();
			
			for(File file: files){
				filePathList.add(file.getAbsolutePath());
				idList.add(id);
			}
			
			id++;
		}
		
		String[] filePaths = new String[filePathList.size()];
		filePathList.toArray(filePaths);
		Integer[] ids = new Integer[idList.size()];
		idList.toArray(ids);
		*/
		
		///test
		/*for(int i=filePaths.length-1; i>=0; i--){
			System.out.println("filePaths: " + filePaths[i]);
			System.out.println("ids: " + ids[i]);
		}*/
		
		
		//train and predict////////////////////////////////////////////////////////////////////////////////////////////
		WeightedStandardPixelTrainer weightedStandardPixelTrainer = new WeightedStandardPixelTrainer();
		/*weightedStandardPixelTrainer.train(filePaths, ids);
		WeightedStandardImage weightedStandardImage = weightedStandardPixelTrainer.getWeightedStandardImage();
		weightedStandardImage.saveKnowledge("resource/Face_Male_Female/Knowledge.log");*/
		//experience file
		weightedStandardPixelTrainer.load("src/res/knowledge/KnowledgeAlphabet.log");
		/*WeightedStandardImage weightedStandardImage = weightedStandardPixelTrainer.getWeightedStandardImage();
		System.out.println(weightedStandardImage.dump());*/
		
		//sample file
		String imageFilePath = "C:\\Users\\admin\\Desktop\\1.jpg";
		Mat mat = Imgcodecs.imread(imageFilePath, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
		
		int prediction = weightedStandardPixelTrainer.predict(mat);
		System.out.println("Prediction is: " + prediction);
		
		
		/*Mat mat = weightedStandardImage.getStandardImages(0);
		Imgcodecs.imwrite("resource/Face_Male_Female/stdImage4.png" , mat);*/
		
		System.out.println("Operation Successful!!!");
	}
	/**/
}
