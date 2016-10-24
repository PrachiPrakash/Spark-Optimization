package org.iiitb.mllib;


import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.optimization.GradientDescent;
import org.apache.spark.mllib.optimization.LeastSquaresGradient;
import org.apache.spark.mllib.optimization.Optimizer;
import org.apache.spark.mllib.optimization.SimpleUpdater;
import org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.rdd.RDD;

public class LinearRegressionWithCG extends GeneralizedLinearAlgorithm<LinearRegressionModel>
{

	@Override
	public LinearRegressionModel createModel(Vector arg0, double arg1) 
	{
		// TODO Auto-generated method stub
		return new LinearRegressionModel(arg0, arg1);
	}

	@Override
	public Optimizer optimizer() 
	{
		GradientDescent gd = new GradientDescent(new LeastSquaresGradient(), new SimpleUpdater());
		gd.setMiniBatchFraction(1.0);
		gd.setStepSize(0.001);
		gd.setNumIterations(300);
		
		return gd;

	}
	
	public LinearRegressionModel train(RDD<LabeledPoint> data)
	{
		return this.run(data);
	}

}
