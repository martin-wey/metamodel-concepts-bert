package ecore2text;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import ecore2text.EcoreToTextTransformation;

import java.io.IOException;

public class CodeGeneratorRunner {

	private static int nbModels = 0;
	
	private static int minClsPerModel = 2;
	
	private static int maxClsPerModel = 15;
	
	public static void main(String[] args) {
		String repoBasePath = args[0];
		String outputBasePath = args[1];
		String repoCsvAnalysisFilePath = Paths.get(repoBasePath, "/analysis/stats.txt").toString();
		String line = "";
        String cvsSplitBy = ",";
		
		try (BufferedReader br = new BufferedReader(new FileReader(repoCsvAnalysisFilePath))) {
			while ((line = br.readLine()) != null) {
				String[] ecoreModelData = line.split(cvsSplitBy);
				Path ecoreModelPath = Paths.get(repoBasePath, ecoreModelData[0]);
				int ecoreModelNbCls = Integer.parseInt(ecoreModelData[3]);
				
				// Check whether the file exists and does not exceed the maximum number of classes allowed
				if (Files.exists(ecoreModelPath) && ecoreModelNbCls >= minClsPerModel && ecoreModelNbCls <= maxClsPerModel) {
					System.out.println("Ecore [file= " + ecoreModelData[0] + " , cls= " + ecoreModelNbCls + "]");
					try {
						String jsonResult = new EcoreToTextTransformation().generate(ecoreModelPath.toString());
						Path outputPath = Paths.get(outputBasePath, ecoreModelData[0].replace("ecore", "json"));
						Files.createDirectories(outputPath.getParent());
						Files.createFile(outputPath);
						Files.write(outputPath, jsonResult.getBytes());
						nbModels++;
					} catch (IOException e) {
						e.printStackTrace();
						System.out.println("An error occured.");
					} 
					catch (Exception e) {
						e.printStackTrace();
						System.out.println("An error occured.");
					}
				}	
			}
			System.out.println("Nb ecore models : " + nbModels);
		} catch(FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
