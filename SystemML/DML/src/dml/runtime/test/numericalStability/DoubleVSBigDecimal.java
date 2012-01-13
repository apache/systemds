package dml.runtime.test.numericalStability;

import java.math.BigDecimal;

public class DoubleVSBigDecimal {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		if(args.length<1)
		{
			System.out.println("DoubleVSBigDecimal <# tries>");
			System.exit(-1);
		}
		int n=10000;//Integer.parseInt(args[0]);
		long start=System.currentTimeMillis();
		double sum=0;
		int factor=1000;
		int m=n*factor;
		for(int i=1; i<m; i++)
		{
			sum=(1.0/(double)i);
			//sum+=(double)i*(double)i;
			//sum+=(double)i;
			//sum-=(double)i;
		}
		double time=((double)System.currentTimeMillis()-(double)start)/(double) factor;
		System.out.println("time: "+time);
		
		start=System.currentTimeMillis();
		BigDecimal sumBig=new BigDecimal(0);
		for(int i=1; i<n; i++)
		{
			BigDecimal.ONE.divide(new BigDecimal(i), SetUp.mc);
			//new BigDecimal(i).multiply(new BigDecimal(i));
			//sumBig=sumBig.add(new BigDecimal(i));
			//BigDecimal.ONE.subtract(new BigDecimal(i));
		}
		long bigTime=System.currentTimeMillis()-start;
		
		System.out.println("big time: "+bigTime);
		
		System.out.println((double)bigTime/(double)time);
	}

}
