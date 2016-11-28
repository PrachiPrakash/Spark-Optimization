package org.iiitb.mllib;

import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.optimization.HingeGradient;
import org.apache.spark.mllib.optimization.LogisticGradient;
import org.apache.spark.mllib.optimization.Optimizer;
import org.apache.spark.mllib.optimization.SimpleUpdater;
import org.apache.spark.mllib.optimization.SquaredL2Updater;
import org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.iiitb.optimization.ConjugateGradientOptimizer;

public class LogisticRegressionWithCG extends GeneralizedLinearAlgorithm<LogisticRegressionModel>
{

	@Override
	public LogisticRegressionModel createModel(Vector arg0, double arg1) 
	{
		// TODO Auto-generated method stub
		return new LogisticRegressionModel(arg0, arg1);
	}

	@Override
	public Optimizer optimizer() 
	{
		ConjugateGradientOptimizer op = new ConjugateGradientOptimizer(0.001, 100,12);
		op.setGd(new LogisticGradient());
		op.setUp(new SquaredL2Updater());
		return op;

	}
	public static void train(RDD<LabeledPoint> data)
	{
		
	}

}
