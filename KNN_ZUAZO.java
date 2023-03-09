
package weka2partziala;

import java.io.Writer;
import java.io.PrintWriter;
import java.io.FileWriter;
import weka.core.SerializationHelper;
import weka.core.MinkowskiDistance;
import weka.core.ManhattanDistance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.Instances;
import weka.classifiers.Classifier;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.classifiers.lazy.IBk;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils;
import weka.core.Tag;

public class KNN_ZUAZO
{
    static int k;
    static int w;
    static int d;
    static Tag tag;
    static double maxPrecision;
    
    static {
        KNN_ZUAZO.k = 1;
        KNN_ZUAZO.w = 0;
        KNN_ZUAZO.d = 0;
        KNN_ZUAZO.maxPrecision = Double.MIN_VALUE;
    }
    
    public static void main(final String[] args) throws Exception {
        final ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[1]);
        final Instances data = source.getDataSet();
        data.setClassIndex(0);
        System.out.println("Instantzia totalak " + data.numInstances());
        System.out.println("Atributu totalak :" + data.numAttributes());
        final int kkopurua = data.numInstances();
        int min = Integer.MAX_VALUE;
        int minClassIndex = 0;
        for (int i = 0; i < data.numClasses(); ++i) {
            final int x = data.attributeStats(data.classIndex()).nominalCounts[i];
            System.out.println(String.valueOf(data.attribute(data.classIndex()).value(i)) + "-->" + x + " instantzia kopurua");
            if (x < min) {
                min = x;
                minClassIndex = i;
            }
        }
        System.out.println("Klase minoritarioa: " + data.attribute(data.classIndex()).value(minClassIndex));
        System.out.println();
        System.out.println("---------------kNN Parametro Ekorketa-----------");
        final LinearNNSearch[] distantziak = distantziak();
        System.out.println("Distantziak kargatuta");
        System.out.println(1);
        System.out.println(2);
        System.out.println(4);
        final SelectedTag[] tags = { new SelectedTag(1, IBk.TAGS_WEIGHTING), new SelectedTag(2, IBk.TAGS_WEIGHTING), new SelectedTag(4, IBk.TAGS_WEIGHTING) };
        System.out.println("Tags kargatuta");
        System.out.println("K, kopurua :" + kkopurua);
        final IBk model = new IBk();
        for (int j = 1; j < kkopurua; ++j) {
            model.setKNN(j);
            for (int k = 0; k < distantziak.length; ++k) {
                model.setNearestNeighbourSearchAlgorithm((NearestNeighbourSearch)distantziak[k]);
                for (int l = 0; l < tags.length; ++l) {
                    model.setDistanceWeighting(tags[l]);
                    final Evaluation eval = new Evaluation(data);
                    eval.crossValidateModel((Classifier)model, data, 3, new Random(3L));
                    if (eval.precision(minClassIndex) > KNN_ZUAZO.maxPrecision) {
                        KNN_ZUAZO.maxPrecision = eval.precision(minClassIndex);
                        KNN_ZUAZO.k = j;
                        KNN_ZUAZO.d = k;
                        final SelectedTag tagOna = model.getDistanceWeighting();
                        KNN_ZUAZO.tag = tagOna.getSelectedTag();
                        KNN_ZUAZO.w = l;
                    }
                }
            }
        }
        System.out.println("PRECISION maximoa -> " + KNN_ZUAZO.maxPrecision);
        System.out.println("K hoberena -> " + KNN_ZUAZO.k);
        System.out.println("Distantzia mota hoberena -> " + motaLortu(KNN_ZUAZO.d));
        System.out.println("Tag -> " + KNN_ZUAZO.tag);
        model.setKNN(KNN_ZUAZO.k);
        model.setNearestNeighbourSearchAlgorithm((NearestNeighbourSearch)distantziak[KNN_ZUAZO.d]);
        model.setDistanceWeighting(tags[KNN_ZUAZO.w]);
        model.buildClassifier(data);
        modeloa_idatzi(args[2], (Classifier)model);
    }
    
    private static String motaLortu(final int d) {
        String mota = "";
        if (d == 0) {
            mota = "EuclideanDistance";
        }
        else if (d == 1) {
            mota = "ManhattanDistance";
        }
        else {
            mota = "MinkowskiDistance";
        }
        return mota;
    }
    
    private static LinearNNSearch[] distantziak() throws Exception {
        final LinearNNSearch euclideanDistance = new LinearNNSearch();
        euclideanDistance.setDistanceFunction((DistanceFunction)new EuclideanDistance());
        final LinearNNSearch manhattanDistance = new LinearNNSearch();
        euclideanDistance.setDistanceFunction((DistanceFunction)new ManhattanDistance());
        final LinearNNSearch minkowskiDistance = new LinearNNSearch();
        euclideanDistance.setDistanceFunction((DistanceFunction)new MinkowskiDistance());
        return new LinearNNSearch[] { euclideanDistance, manhattanDistance, minkowskiDistance };
    }
    
    private static void modeloa_idatzi(final String direktorio, final Classifier modeloa) throws Exception {
        SerializationHelper.write(direktorio, (Object)modeloa);
    }
    
    private static void modeloa_irakurri(final String direktorioa) throws Exception {
        SerializationHelper.read(direktorioa);
        System.out.println("> Modeloa kargatu da: ");
        System.out.println("..................................................");
        final Classifier sailkatzaile = (Classifier)SerializationHelper.read(direktorioa);
        System.out.println(sailkatzaile.toString());
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