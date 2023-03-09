
package weka2partziala;

import weka.classifiers.Classifier;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.core.SelectedTag;
import weka.core.MinkowskiDistance;
import weka.core.FilteredDistance;
import weka.core.ManhattanDistance;
import weka.core.EuclideanDistance;
import weka.core.DistanceFunction;
import weka.core.ChebyshevDistance;
import weka.core.neighboursearch.LinearNNSearch;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class KNN_MARTIN
{
    public static void main(final String[] args) {
        if (args.length != 1) {
            System.out.println("El n\u00c3ºmero de par\u00c3¡metros que est\u00c3¡s utilizando no es el correcto. ");
            System.out.println("java -jar kNN </path/data.arff>\n");
        }
        else {
            try {
                final ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
                final Instances data = source.getDataSet();
                data.setClassIndex(data.numAttributes() - 1);
                System.out.println("Fichero utilizado: " + args[0]);
                System.out.println(" ");
                System.out.println("N\u00c3ºmero de instancias en el fichero seleccionado: " + data.numInstances());
                System.out.println("");
                optimizarFMeasureConkdw(data);
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
    
    public static void optimizarFMeasureConkdw(final Instances data) throws Exception {
        final IBk ibk = new IBk();
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
        final LinearNNSearch[] distancias = { chebyshevDistance, euclideanDistance, manhattanDistance, filteredDistance, minkowskiDistance };
        final SelectedTag[] etiquetas = { new SelectedTag(1, IBk.TAGS_WEIGHTING), new SelectedTag(2, IBk.TAGS_WEIGHTING), new SelectedTag(4, IBk.TAGS_WEIGHTING) };
        int kaux = 0;
        LinearNNSearch daux = null;
        SelectedTag waux = null;
        double fmeasureaux = 0.0;
        double fmeasuremax = 0.0;
        for (int k = 1; k < data.numInstances() / 4; ++k) {
            ibk.setKNN(k);
            LinearNNSearch[] array;
            for (int length = (array = distancias).length, i = 0; i < length; ++i) {
                final LinearNNSearch d = array[i];
                ibk.setNearestNeighbourSearchAlgorithm((NearestNeighbourSearch)d);
                SelectedTag[] array2;
                for (int length2 = (array2 = etiquetas).length, j = 0; j < length2; ++j) {
                    final SelectedTag w = array2[j];
                    ibk.setDistanceWeighting(w);
                    final Evaluation eval = new Evaluation(data);
                    eval.crossValidateModel((Classifier)ibk, data, 10, new Random(1L));
                    fmeasureaux = eval.weightedFMeasure();
                    System.out.println(fmeasureaux);
                    if (fmeasureaux > fmeasuremax) {
                        fmeasuremax = fmeasureaux;
                        kaux = k;
                        daux = d;
                        waux = w;
                    }
                }
            }
        }
        System.out.println("   ");
        System.out.println("  ____________________________________________________________________________________________________ ");
        System.out.println("   ");
        System.out.println(" La fmeasure \u00c3³ptima es: " + fmeasuremax + " y se ha conseguido utilizando los siguientes par\u00c3¡metros: ");
        System.out.println(" k = " + kaux);
        System.out.println(" d = " + daux.distanceFunctionTipText());
        System.out.println(" w = " + waux);
    }
}
