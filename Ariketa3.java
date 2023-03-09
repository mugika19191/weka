package labo5;

import java.io.File;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.beans.DataSetEvent;

public class Ariketa3 {
	public static void main(String[] args) throws Exception{
		
		DataSource sourceTrain = new DataSource("/home/imanol/Escritorio/uni/2022-2023/Weka/Praktika/respuestas_Labos/Labo5/ariketa2_train.arff");
		Instances train = sourceTrain.getDataSet();
		if(train.classIndex() == -1) {
			train.setClassIndex(train.numAttributes()-1);
		}
		
		DataSource sourceTest = new DataSource("/home/imanol/Escritorio/uni/2022-2023/Weka/Praktika/respuestas_Labos/Labo5/ariketa1_testBlind.arff");
		Instances test = sourceTest.getDataSet();
		if(test.classIndex() == -1) {
			test.setClassIndex(test.numAttributes() -1 );
		}
		Classifier clasifier = (Classifier) SerializationHelper.read("/home/imanol/Escritorio/uni/2022-2023/Weka/Praktika/respuestas_Labos/Labo5/NB.model");
		
		System.out.println("Atributu kopurua (Datuak arreprozesatu baino lehen): " + test.numAttributes());
		Remove filter = new Remove();
		filter.setInvertSelection(false);
		filter.setInputFormat(train);
		
		Instances testEgokitua = Filter.useFilter(test, filter);
		System.out.println("Atributu kopurua(Datuak prozesatu eta gero): " + testEgokitua.numAttributes());
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(testEgokitua);
		saver.setFile(new File("/home/imanol/Escritorio/uni/2022-2023/Weka/Praktika/respuestas_Labos/Labo5/Ariketa3_testEgokituta"));
		saver.writeBatch();
		
	}
}
