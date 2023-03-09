package labo5;
import java.io.File;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class Ariketa2 {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("/home/imanol/Escritorio/uni/2022-2023/Weka/Praktika/weka-3-8-6/data/breast-cancer.arff");
		Instances data = source.getDataSet();
		if(data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes()-1);
		}
		
		System.out.println("dataren atributu kopurua hurrengoa da: " + data.numAttributes());
		AttributeSelection attFilter = new AttributeSelection();
		attFilter.setEvaluator(new CfsSubsetEval());
		attFilter.setSearch(new BestFirst());
		attFilter.setInputFormat(data);
		Instances trainFSS = Filter.useFilter(data, attFilter);
		
		System.out.println("Atributu kopurua (Datuak aurreprozesatu ostean): " + trainFSS.numAttributes());
		
		NaiveBayes clasifier = new NaiveBayes();
		clasifier.buildClassifier(trainFSS);
		
		SerializationHelper.write("/home/imanol/Escritorio/uni/2022-2023/Weka/Praktika/respuestas_Labos/Labo5/NB.model",clasifier);
		ArffSaver save = new ArffSaver();
		save.setInstances(trainFSS);
		save.setFile(new File("/home/imanol/Escritorio/uni/2022-2023/Weka/Praktika/respuestas_Labos/Labo5/ariketa2_train.arff"));
		save.writeBatch();
	}
}
