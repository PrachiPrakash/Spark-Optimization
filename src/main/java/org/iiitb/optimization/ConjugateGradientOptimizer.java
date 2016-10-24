package org.iiitb.optimization;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.optimization.Gradient;
import org.apache.spark.mllib.optimization.LeastSquaresGradient;
import org.apache.spark.mllib.optimization.Optimizer;
import org.apache.spark.mllib.optimization.SimpleUpdater;
import org.apache.spark.mllib.optimization.Updater;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;

public class ConjugateGradientOptimizer implements Optimizer 
{

	private static final long serialVersionUID = 1L;
	private double stepSize;
	private int numIteration;
	private Gradient gd;
	private Updater update;
	
	public ConjugateGradientOptimizer(double stepSize, int numIteration)
	{
		this.stepSize = stepSize;
		this.numIteration = numIteration;
		gd = new LeastSquaresGradient();
		update = new SimpleUpdater();
	}

	@Override
	public Vector optimize(RDD<Tuple2<Object, Vector>> data, Vector w) 
	{
		int i = 1;
		JavaRDD<Tuple2<Object, Vector>> da = data.toJavaRDD();
		JavaSparkContext jsc = new JavaSparkContext(da.context());
		final Broadcast<Double> sz = jsc.broadcast((double)da.count());
		double gotgo = 0.0;
		Vector d= null;
		Vector gold = null;
		double beta = 0.0;
		
		boolean converged = false;
		
		while(!converged && i <= numIteration){
			
			final Broadcast<Vector> bw = jsc.broadcast(w);
			
			//compute the gradient vector of all the rows
			JavaRDD<Vector> gradRDD = da.map(new Function<Tuple2<Object,Vector>,Vector>() {
				public Vector call(Tuple2<Object, Vector> in)
				{
					Vector features = in._2;
					Tuple2<Vector,Object> grad = gd.compute(features, (Double)in._1, bw.getValue());
					BLAS.scal(1.0/sz.getValue(), grad._1);
					
					return grad._1;
				}
			});
			
			JavaPairRDD<Integer,Double> expandedGradient = gradRDD.flatMapToPair(new PairFlatMapFunction<Vector,Integer,Double>() {

				@Override
				public Iterable<Tuple2<Integer, Double>> call(Vector arg0) throws Exception 
				{
					double vec[] = arg0.toArray();
					List<Tuple2<Integer,Double>> lis = new ArrayList<Tuple2<Integer,Double>>();
					for(int i=0; i<vec.length; i++)
						lis.add(new Tuple2<Integer, Double>(i,vec[i]));
					return lis;
				}
			
				
			});
			
			 JavaPairRDD<Integer,Double> r = expandedGradient.reduceByKey(new Function2<Double, Double, Double>() {
				
				@Override
				public Double call(Double arg0, Double arg1) throws Exception {
					// TODO Auto-generated method stub
					return arg0 + arg1;
				}
			});
			 
			 List<Tuple2<Integer, Double>>  t= r.collect();
			 double temp[] = new double[t.size()];
			 
			 int k=0;
			 for(Tuple2<Integer, Double> tu:t)
				 temp[k++] = tu._2;
			 
			 Vector newGrad = Vectors.dense(temp);
			
			 
			 if(i == 1){
				 double temp_d[] = new double[newGrad.size()]; 
				 d = Vectors.dense(temp_d);
				 BLAS.copy(newGrad, d);
				 BLAS.scal(-1, d);
			 }
			 else{
				gotgo = BLAS.dot(gold, gold); 
				double temp_gold[] = new double[newGrad.size()];
				Vector gold_temp = Vectors.dense(temp_gold);
				
				BLAS.copy(newGrad, gold_temp);
				BLAS.axpy(-1.0, gold, gold_temp);
				
				beta = BLAS.dot(newGrad, gold_temp)/gotgo;
				BLAS.scal(beta,d);
				BLAS.axpy(-1, newGrad, d);
				
				
			 }
			gold = newGrad;
			double temp_d[] = new double[d.size()];
			Vector d_temp = Vectors.dense(temp_d);
			BLAS.copy(d, d_temp);
			BLAS.scal(0.00001,d_temp);
			BLAS.axpy(1,d_temp , w);
			 
			 
		
			 System.out.println("The value after "+ i +" iteration is"+w.toString());
	
			i +=1;
		}
		
		return w;
		
	}
	
	private boolean isConverged(Vector oldW, Vector newW,double con)
	{
		BLAS.axpy(-1.0, newW, oldW);
		double diff = norm(oldW);
		
		return (diff <con * Math.max(norm(newW), 1.0));
	}
	
	private double norm(Vector v)
	{
		double w[] = v.toArray();
		double sum = 0.0;
		for(int i=0; i<w.length; i++)
			sum += w[i]*w[i];
		return Math.sqrt(sum);
	}
	

}
