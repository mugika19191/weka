// 
// Decompiled by Procyon v0.5.36
// 

package weka2partziala;

import java.util.Iterator;
import weka.core.Instances;
import weka.classifiers.evaluation.Prediction;
import java.io.Writer;
import java.io.BufferedWriter;
import java.io.FileWriter;
import weka.core.SerializationHelper;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Debug;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.SMO;
import weka.core.converters.ConverterUtils;

public class SMOmodel
{
    public static void main(final String[] args) throws Exception {
        final ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        final Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        final ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(args[1]);
        final Instances test = source.getDataSet();
        test.setClassIndex(test.numAttributes() - 1);
        for (int i = 0; i < data.numAttributes(); ++i) {
            System.out.println("ID: " + data.attribute(i).name());
            System.out.println("Desberdin kopurua: " + data.attributeStats(i).distinctCount);
            System.out.println();
        }
        for (int i = 0; i < data.attributeStats(data.numAttributes() - 1).distinctCount; ++i) {
            System.out.println("ID: " + data.attribute(data.numAttributes() - 1).value(i));
            System.out.println("ID: " + data.attributeStats(data.numAttributes() - 1).nominalCounts[i]);
            System.out.println();
        }
        final SMO smo = new SMO();
        smo.buildClassifier(data);
        final PolyKernel pk = new PolyKernel();
        double max = 0.0;
        double ber = 0.0;
        for (int j = 0; j < 6; ++j) {
            pk.setExponent((double)j);
            smo.setKernel((Kernel)pk);
            final Evaluation eval = new Evaluation(data);
            eval.crossValidateModel((Classifier)smo, data, 3, (Random)new Debug.Random(1L));
            if (max < eval.weightedFMeasure()) {
                max = eval.weightedFMeasure();
                ber = j;
            }
            System.out.println(String.valueOf(j) + "-garren FMeasure: " + eval.weightedFMeasure());
        }
        pk.setExponent(ber);
        smo.setKernel((Kernel)pk);
        SerializationHelper.write(args[2], (Object)smo);
        final Evaluation ezZintz = new Evaluation(data);
        ezZintz.evaluateModel((Classifier)smo, data, new Object[0]);
        System.out.println(ezZintz.precision(1));
        final BufferedWriter bw = new BufferedWriter(new FileWriter(args[3]));
        bw.write("Precision: " + (int)ezZintz.precision(1));
        bw.newLine();
        bw.write("Recall: " + (int)ezZintz.recall(1));
        bw.newLine();
        bw.write("F-Measure: " + (int)ezZintz.fMeasure(1));
        bw.newLine();
        bw.write(ezZintz.toMatrixString());
        bw.flush();
        bw.close();
        final Classifier rSMO = (Classifier)SerializationHelper.read(args[4]);
        final Evaluation iragarpen = new Evaluation(test);
        iragarpen.evaluateModel(rSMO, test, new Object[0]);
        int k = 0;
        final BufferedWriter bw2 = new BufferedWriter(new FileWriter(args[4]));
        for (final Prediction p : iragarpen.predictions()) {
            String iragarpena = data.attribute(data.numAttributes() - 1).value((int)p.predicted());
            final String erreala = data.attribute(data.numAttributes() - 1).value((int)p.actual());
            String errorea = "";
            if (Double.isNaN(p.actual())) {
                iragarpena = "?";
            }
            if (iragarpena != erreala) {
                errorea = "$";
            }
            else {
                errorea = "-";
            }
            bw2.write("\t" + k + "\t\t" + iragarpena + "\t\t" + erreala + "\t\t   " + errorea);
            bw2.newLine();
            ++k;
        }
        bw2.flush();
        bw2.close();
    }
}