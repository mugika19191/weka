import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
public class Main {
    public static void main(String[] args) throws Exception {
        /////////////////////EZ BADAUDE ARGUMENTURIK///////////////////////////////////////
        if (args.length==0){
            System.out.println("\nJava proiektu hau erabiltzeko bi argumentu jarri behar dira: \n");
            System.out.println("1- Datuak dauzkan .arff dokumentuaren path-a.");
            System.out.println("2- Lortutako emaitzak non gordeko diren zehazten duen path-a. \n");
            System.out.println("Sartu berriro argumentuak!");
            return; // amaitu programa
        }
        /////////////////////ARGUMENTUAK ONDO SARTZEN BADIRA///////////////////////////////////////
        System.out.println("Bigarren laborategia lehenengo zatia: \n");
        System.out.println("Erabilitako path-ak:");
        for (int i=0; i< args.length;i++){
            System.out.println(args[i]);
        }
        //////////////////////////////////////////////////////////////////////////////////
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);
        ////////////////////////////////////////////////////////////////////////////////////
        AttributeSelection filter= new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search=new BestFirst();
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);
        Instances newData = Filter.useFilter(data, filter);

        NaiveBayes estimador= new NaiveBayes();


        Evaluation evaluator = new Evaluation(newData);
        evaluator.crossValidateModel(estimador, newData, 5, new Random(1)); // Random(1): the seed=1 means "no shuffle" :-!
        //
        double acc=evaluator.pctCorrect();
        double inc=evaluator.pctIncorrect();
        double kappa=evaluator.kappa();
        double mae=evaluator.meanAbsoluteError();
        double rmse=evaluator.rootMeanSquaredError();
        double rae=evaluator.relativeAbsoluteError();
        double rrse=evaluator.rootRelativeSquaredError();
        double confMatrix[][]= evaluator.confusionMatrix();


//////////////////////IDATZI LORTUTAKO DATUAK///////////////////////
        BufferedWriter buffer=new BufferedWriter(new FileWriter(args[1]));
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("uuuu/MM/dd HH:mm:ss");
        LocalDateTime now = LocalDateTime.now();
        buffer.write(dtf.format(now)+"\n");
        buffer.write("\n"+args[0]+"\n");
        buffer.write(args[1]+"\n");
        buffer.write("\n"+"Correctly Classified Instances  " + acc +"\n");
        buffer.write("Incorrectly Classified Instances  " + inc+"\n");
        buffer.write("Kappa statistic  " + kappa+"\n");
        buffer.write("Mean absolute error  " + mae+"\n");
        buffer.write("Root mean squared error  " + rmse+"\n");
        buffer.write("Relative absolute error  " + rae+"\n");
        buffer.write("Root relative squared error  " + rrse+"\n");
        buffer.write("\n"+evaluator.toMatrixString());

        buffer.flush();
        buffer.close();
    }
}