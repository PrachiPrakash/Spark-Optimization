package org.iiitb.driver;

import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.iiitb.mllib.LogisticRegressionWithCG;

import scala.Tuple2;

public class LogisticRegressionWithCGDemo 
{
	public static void main(String args[])
	{
		SparkConf sconf = new SparkConf();
		sconf.setAppName("MLDemo");
		
		SparkContext jsc = new SparkContext(sconf);
		
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc, args[0]).toJavaRDD();
		//JavaRDD<LabeledPoint> data2 = MLUtils.loadLibSVMFile(jsc, args[0]+"/test.data.t").toJavaRDD();
		
		JavaRDD<LabeledPoint> d2 = data.map(new Function<LabeledPoint, LabeledPoint>() {

			@Override
			public LabeledPoint call(LabeledPoint arg0) throws Exception {
				if(arg0.label() == 2)
					return new LabeledPoint(0.0, MLUtils.appendBias( arg0.features()));
				else
					return new LabeledPoint(1.0, MLUtils.appendBias( arg0.features()));
				
			}
			
		});
		
		

		
		JavaRDD<LabeledPoint>[] splits = d2.randomSplit(new double[] {0.8, 0.2}, 11L);
		JavaRDD<LabeledPoint> train = splits[0].cache();
		JavaRDD<LabeledPoint> test = splits[1];
		
		

		LogisticRegressionWithCG reg = new LogisticRegressionWithCG();
		
		final LogisticRegressionModel mod = reg.run(JavaRDD.toRDD(train));
		//mod.clearThreshold();
		
		// Compute raw scores on the test set.
		JavaRDD<Tuple2<Double, Double>> predictionAndLabels = test.map(
		  new Function<LabeledPoint, Tuple2<Double, Double>>() {
		    @Override
		    public Tuple2<Double, Double> call(LabeledPoint p) {
		      Double prediction = mod.predict(p.features());
		      double mp = 0.0;
		      if(prediction >= 0.5)
		    	  mp = 1.0;
		      else
		    	  mp = 0.0;
		      
		      if(mp == p.label())
		    	  return new Tuple2<Double, Double>(1.0, 1.0);
		      else
		    	  return new Tuple2<Double, Double>(0.0, 1.0);
		    }
		  }
		);
		JavaPairRDD<Double, Double> precision =predictionAndLabels.mapToPair(new PairFunction<Tuple2<Double,Double>, Double,Double>() {

			@Override
			public Tuple2<Double, Double> call(Tuple2<Double, Double> arg0) throws Exception {
				// TODO Auto-generated method stub
				return arg0;
			}
		});
		precision = precision.reduceByKey(new Function2<Double, Double, Double>() {
			
			@Override
			public Double call(Double arg0, Double arg1) throws Exception {
				// TODO Auto-generated method stub
				return arg0+arg1;
			}
		});
		List<Tuple2<Double,Double>> ac = precision.collect();
		double res = 0.0;
		
		for(Tuple2<Double,Double> i:ac){
			if(i._1() == 1.0)
				res = i._2();
				
		}
		double tot = test.count();
		
		
		System.out.println("final W is:"+mod.weights());
		System.out.println("Accuracy is:"+res/tot);
		
		
		
		
	}
}
