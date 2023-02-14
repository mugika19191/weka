// 
// Decompiled by Procyon v0.5.36
// 

package holdout;

import java.util.Calendar;
import java.text.SimpleDateFormat;
import java.io.Writer;
import java.io.PrintWriter;
import java.io.FileWriter;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.core.converters.ConverterUtils;

public class Nagusia
{
    public static void main(final String[] args) throws Exception {
        final ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        final Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        final Randomize filterRandom = new Randomize();
        filterRandom.setRandomSeed(1);
        filterRandom.setInputFormat(data);
        final Instances RandomData = Filter.useFilter(data, (Filter)filterRandom);
        final RemovePercentage filterRemove = new RemovePercentage();
        filterRemove.setInputFormat(RandomData);
        filterRemove.setPercentage(30.0);
        final Instances train = Filter.useFilter(RandomData, (Filter)filterRemove);
        System.out.println("Train tiene estas instancias " + train.numInstances());
        filterRemove.setInputFormat(RandomData);
        filterRemove.setPercentage(30.0);
        filterRemove.setInvertSelection(true);
        final Instances test = Filter.useFilter(RandomData, (Filter)filterRemove);
        System.out.println("Test tiene estas instancias " + test.numInstances());
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);
        final NaiveBayes model = new NaiveBayes();
        model.buildClassifier(train);
        final Evaluation eval = new Evaluation(train);
        eval.evaluateModel((Classifier)model, test, new Object[0]);
        System.out.println(eval.toMatrixString());
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
