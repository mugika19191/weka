// 
// Decompiled by Procyon v0.5.36
// 

package crossValidation;

import java.util.Calendar;
import java.text.SimpleDateFormat;
import java.io.Writer;
import java.io.PrintWriter;
import java.io.FileWriter;
import weka.core.Instances;
import weka.classifiers.Classifier;
import java.util.Random;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.converters.ConverterUtils;

public class Nagusia
{
    public static void main(final String[] args) throws Exception {
        if (args.length == 0) {
            System.out.println(String.valueOf(args.length) + " parametro sartu dituzu");
            System.out.println("java -jar estimateNaiveBayes.jar trainPath.arff kalitatea.txt");
            System.out.println("1. Datu sortaren kokapena (path) .arff  formatuan (input). Aurre-baldintza: klasea azken atributuan egongo da.");
            System.out.println("2. Emaitzak idazteko irteerako fitxategiaren path-a (output).");
        }
        else {
            final ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
            final Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            final NaiveBayes model = new NaiveBayes();
            final Evaluation eval = new Evaluation(data);
            eval.crossValidateModel((Classifier)model, data, 5, new Random(1L));
            System.out.println(eval.toSummaryString());
            System.out.println("Instantzia kopurua: " + data.numInstances());
            System.out.println("Atributu kopurua: " + data.numAttributes());
            System.out.println("Datu sorta honetan, lehenengo atributuak  " + data.numDistinctValues(0) + " balio desberdin hartu ditzake.");
            System.out.println("Datu sorta honetan, azken-aurreko atributuak  " + data.attributeStats(data.numAttributes() - 2).missingCount + " missing value ditu.");
            int min = Integer.MAX_VALUE;
            int minClassValue = 0;
            for (int i = 0; i < data.numClasses(); ++i) {
                final int x = data.attributeStats(data.classIndex()).nominalCounts[i];
                if (x < min) {
                    min = x;
                    minClassValue = i;
                }
            }
            System.out.println("Klase minoritarioa: " + data.attribute(data.classIndex()).value(minClassValue));
            System.out.println("Klase minoritarioa instantzia kop : " + min);
            System.out.println("Klase minoritarioa f_measure: " + eval.fMeasure(minClassValue));
            fitxategiaSortu(eval, args[1]);
        }
    }
    
    private static void fitxategiaSortu(final Evaluation eval, final String directory) {
        try {
            final FileWriter file = new FileWriter(directory);
            final PrintWriter pw = new PrintWriter(file);
            pw.println("Fitxategia sortu:" + directory);
            final String data = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
            pw.println("Exekuzioa data--> " + data);
            pw.println("Nahasmen-Matrizea: " + eval.toMatrixString());
            pw.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
}
