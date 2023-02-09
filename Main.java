import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class Main {
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:/Users/mugik/Downloads/1. Praktika Datuak-20230126/heart-c.arff");
        Instances data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        System.out.println("Lehenengo laborategiko ariketak:");
        System.out.println("1- Hemendik atera dira datuak: C:/Users/mugik/Downloads/1. Praktika Datuak-20230126/heart-c.arff");
        System.out.println("2- Instantzia kopurua: "+ data.numInstances());
        System.out.println("3- Atributu kopurua: "+ data.numAttributes());
        System.out.println("4- Lehenengo atributuak har ditzakeen balio ezberdinak: "+ data.attributeStats(0).distinctCount);
        System.out.println("5- azken atributuak hartzen dituen balioak eta beraien maiztasuna:");
        //data.attributeStats(data.numAttributes()-1);
        for (int i=0;i<data.attributeStats(data.numAttributes()-1).nominalCounts.length;i++){
            System.out.println("Balioa: "+data.attribute(data.numAttributes()-1).value(i) +" Maiztasuna "+data.attributeStats(data.numAttributes()-1).nominalCounts[i]);
        }
        System.out.println("6- Azken aurreko atributuak dituen missing value kopurua: "+ data.attributeStats(data.numAttributes()-2).missingCount );
    }
}