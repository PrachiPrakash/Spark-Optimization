package org.iiitb.driver;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.optimization.GradientDescent;
import org.apache.spark.mllib.optimization.LeastSquaresGradient;
import org.apache.spark.mllib.optimization.Optimizer;
import org.apache.spark.mllib.optimization.SimpleUpdater;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;


import scala.Tuple2;

public class Main3 
{
	public static void main(String args[])
	{
		SparkConf sc = new SparkConf();
		sc.setAppName("Gradient Demo");
		
		SparkContext jsc = new SparkContext(sc);
		
		JavaRDD<LabeledPoint> la = MLUtils.loadLibSVMFile(jsc, args[0]).toJavaRDD();
		
		JavaRDD<Tuple2<Object, Vector>> dataset = la.map(new Function<LabeledPoint, Tuple2<Object,Vector>>() {

			public Tuple2<Object, Vector> call(LabeledPoint arg0) throws Exception {
				
				return new Tuple2<Object, Vector>(arg0.label(),arg0.features());
			}
		
		});
		int maxIter = Integer.parseInt(args[1]);
		
		GradientDescent op = new GradientDescent(new LeastSquaresGradient(), new SimpleUpdater());
		op.setConvergenceTol(0.01);
		op.setMiniBatchFraction(1.0);
		op.setNumIterations(maxIter);
		op.setStepSize(0.00001);
		Vector w = Vectors.dense(new double[]{0.0,0.0,0.0}); 
		
		long s  = System.currentTimeMillis();
		w = op.optimize(JavaRDD.toRDD(dataset), w);
		long e = System.currentTimeMillis();
		
		System.out.println(w);
		System.out.println("Optimization time = "+(e-s));
		//System.out.println("Total Number of calls "+op.getNoCalls());
		//System.out.println("Total Loss calls in bracket phase "+op.numLossCallBracket);
		//System.out.println("Total Loss calls in zoom phase "+op.numLossCallsZoom);
		//System.out.println("Total grad calls in bracket phase "+op.numGradCallsBracket);
		//System.out.println("Total grad calls in zoom phase "+op.numGradCallsZoom);
		
		
		
	}
}
