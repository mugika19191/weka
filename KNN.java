package weka2partziala;

import java.io.Writer;
import java.io.PrintWriter;
import java.io.FileWriter;
import weka.core.Instances;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Debug;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.classifiers.Evaluation;
import weka.core.SelectedTag;
import weka.core.MinkowskiDistance;
import weka.core.FilteredDistance;
import weka.core.ManhattanDistance;
import weka.core.EuclideanDistance;
import weka.core.DistanceFunction;
import weka.core.ChebyshevDistance;
import weka.core.neighboursearch.LinearNNSearch;
import weka.classifiers.lazy.IBk;
import weka.core.converters.ConverterUtils;

public class KNN
{
    public static void main(final String[] args) throws Exception {
        final ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[1]);
        final Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        final IBk knn = new IBk();
        final LinearNNSearch chebyshevDistance = new LinearNNSearch();
        chebyshevDistance.setDistanceFunction((DistanceFunction)new ChebyshevDistance());
        final LinearNNSearch euclideanDistance = new LinearNNSearch();
        euclideanDistance.setDistanceFunction((DistanceFunction)new EuclideanDistance());
        final LinearNNSearch manhattanDistance = new LinearNNSearch();
        manhattanDistance.setDistanceFunction((DistanceFunction)new ManhattanDistance());
        final LinearNNSearch filteredDistance = new LinearNNSearch();
        filteredDistance.setDistanceFunction((DistanceFunction)new FilteredDistance());
        final LinearNNSearch minkowskiDistance = new LinearNNSearch();
        minkowskiDistance.setDistanceFunction((DistanceFunction)new MinkowskiDistance());
        final LinearNNSearch[] distantziak = { chebyshevDistance, euclideanDistance, manhattanDistance, filteredDistance, minkowskiDistance };
        final SelectedTag[] tags = { new SelectedTag(1, IBk.TAGS_WEIGHTING), new SelectedTag(2, IBk.TAGS_WEIGHTING), new SelectedTag(4, IBk.TAGS_WEIGHTING) };
        int kaux = 0;
        LinearNNSearch daux = null;
        SelectedTag waux = null;
        double fmeasureaux = 0.0;
        double fmeasuremax = 0.0;
        final Evaluation eval = new Evaluation(data);
        for (int k = 1; k < data.numInstances() / 4; ++k) {
            knn.setKNN(k);
            LinearNNSearch[] array;
            for (int length = (array = distantziak).length, i = 0; i < length; ++i) {
                final LinearNNSearch d = array[i];
                knn.setNearestNeighbourSearchAlgorithm((NearestNeighbourSearch)d);
                SelectedTag[] array2;
                for (int length2 = (array2 = tags).length, j = 0; j < length2; ++j) {
                    final SelectedTag w = array2[j];
                    knn.setDistanceWeighting(w);
                    eval.crossValidateModel((Classifier)knn, data, k, (Random)new Debug.Random(1L));
                    fmeasureaux = eval.weightedFMeasure();
                    if (fmeasureaux > fmeasuremax) {
                        fmeasuremax = fmeasureaux;
                        kaux = k;
                        daux = d;
                        waux = w;
                    }
                }
            }
        }
        System.out.println(" Fmeasure optimoa: " + fmeasuremax + ", hurrengo parametroekin lortu dena: ");
        System.out.println(" k = " + kaux);
        System.out.println(" d = " + daux.distanceFunctionTipText());
        System.out.println(" w = " + waux);
        fitxategia_idatzi(eval, args[1]);
    }
    
    private static void fitxategia_idatzi(final Evaluation ev, final String direktorio) {
        try {
            final FileWriter file = new FileWriter(direktorio);
            final PrintWriter pw = new PrintWriter(file);
            pw.println("Direktorioa -->" + direktorio);
            pw.println("Nahasmen matrizea -->" + ev.toMatrixString());
            pw.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
}