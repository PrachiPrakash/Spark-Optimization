package org.iiitb.driver;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.util.MLUtils;
import org.iiitb.mllib.LinearRegressionWithCG;

import org.apache.spark.api.java.function.Function;


import scala.Tuple2;

public class Main 
{
	public static void main(String args[])
	{
		SparkConf sconf = new SparkConf();
		sconf.setAppName("MLDemo");
		
		SparkContext jsc = new SparkContext(sconf);
		
		JavaRDD<LabeledPoint> dataset = MLUtils.loadLibSVMFile(jsc, args[0]).toJavaRDD();
		
		
		
		
		
		
		LinearRegressionWithCG lr = new LinearRegressionWithCG();
		//lr.setIntercept(true);
		
		final LinearRegressionModel model = lr.train(JavaRDD.toRDD(dataset));
		System.out.println("hello");
		System.out.println(model.weights().toString());
		//System.out.println(dataset.take(1).toString());
		
		/*JavaRDD<Tuple2<Double, Double>> valuesAndPreds = dataset.map(
				  new Function<LabeledPoint, Tuple2<Double, Double>>() {
				    public Tuple2<Double, Double> call(LabeledPoint point) {
				      double prediction = model.predict(point.features());
				      return new Tuple2<Double,Double>(prediction, point.label());
				    }
				  }
				);
	
		double MSE = new JavaDoubleRDD(valuesAndPreds.map(
				  new Function<Tuple2<Double, Double>, Object>() {
				    public Object call(Tuple2<Double, Double> pair) {
				      return Math.pow(pair._1() - pair._2(), 2.0);
				    }
				  }
				).rdd()).mean();
		
				System.out.println("training Mean Squared Error = " + MSE);*/
	}
	
	
}
