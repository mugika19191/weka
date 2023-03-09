package labo5;

import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.filters.unsupervised.attribute.ReplaceWithMissingValue;

public class Ariketa1 {
	public static void main(String[] args) throws Exception{
		
		DataSource source = new DataSource("/home/imanol/Escritorio/uni/2022-2023/Weka/Praktika/weka-3-8-6/data/breast-cancer.arff");
		Instances data = source.getDataSet();
		if(data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes() -1);
			
		}
		
		Instances train = null;
		Instances test = null;
		StratifiedRemoveFolds filter = new StratifiedRemoveFolds();
		filter.setNumFolds(10);
		for(int f= 1; f <= filter.getNumFolds(); f++) {
			
			filter.setInputFormat(data);
			filter.setFold(f);
			filter.setInvertSelection(false);
			Instances fold = Filter.useFilter(data, filter);
			
			if(f>=1 && f<=7) {
				if(train == null) {
					train = fold;
				}else {
					for(int i = 0; i< fold.numInstances(); i++) {
						train.add(fold.get(i));
					}
				}
			}else {
				if(test == null) {
					test = fold;
				}else {
					for (int i = 0; i < fold.numInstances(); i++) {
						test.add(fold.get(i));
					}
				}
			}
			
		}
		System.out.println(data.numInstances());
		System.out.println(train.numInstances());
		System.out.println(test.numInstances());
		
		for (int i = 0; i < test.numInstances(); i++) {
			test.instance(i).setClassMissing();
			
		}
		
		ArffSaver saveTrain = new ArffSaver();
		saveTrain.setInstances(train);
		saveTrain.setFile(new File("/home/imanol/Escritorio/uni/2022-2023/Weka/Praktika/respuestas_Labos/Labo5/ariketa1_train.arff"));
		saveTrain.writeBatch();
		
		ArffSaver testBlindSaver = new ArffSaver();
		testBlindSaver.setInstances(test);
		testBlindSaver.setFile(new File("/home/imanol/Escritorio/uni/2022-2023/Weka/Praktika/respuestas_Labos/Labo5/ariketa1_testBlind.arff"));
		testBlindSaver.writeBatch();
		
		
	}
	
	
	

}
