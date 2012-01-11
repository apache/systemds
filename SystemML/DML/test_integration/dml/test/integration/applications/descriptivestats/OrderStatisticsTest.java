package dml.test.integration.applications.descriptivestats;

import java.util.Arrays;

import org.junit.Test;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;

public class OrderStatisticsTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "applications/descriptivestats/";
		availableTestConfigurations.put("SimpleQuantileTest",
				new TestConfiguration("SimpleQuantileTest", new String[] { "median", "weighted_median" }));
		availableTestConfigurations.put("QuantileTest",
				new TestConfiguration("QuantileTest", new String[] { "quantile", "weighted_quantile" }));
		availableTestConfigurations.put("IQMTest",
				new TestConfiguration("IQMTest", new String[] { "iqm", "weighted_iqm" }));
		
	}
	
	public static double quantile(double[] vector, double p)
	{
		int i=(int)Math.ceil(p*vector.length);
		//System.out.println("P = "+p+", i="+i+", value: "+vector[i-1]);
		return vector[i-1];
	}
	
	public static double IQM(double[] vector) {
		int start=(int)Math.ceil(0.25*vector.length);
		int end=(int)Math.ceil(0.75*vector.length);
		double sum=0;
		for(int i=start; i<end; i++)
			sum+=vector[i];
		return sum/(end-start);
	}
	
	public static double[] sort(double[][] vector)
	{
		double[] ret=new double[vector.length];
		for(int i=0; i<ret.length; i++)
			ret[i]=vector[i][0];
		Arrays.sort(ret);
		return ret;
	}
	
	public static double[][] blowUp(double[][] vector, double[][] weight) {
		int totalNum=0;
		for(int i=0; i<weight.length; i++)		
			totalNum+=weight[i][0];
		double[][] ret=new double[totalNum][1];
		int index=0;
		for(int i=0; i<weight.length; i++)
		{	for(int j=0; j<weight[i][0]; j++)
				ret[index++][0]=vector[i][0];
		}
		return ret;
	}	
	
	public static void round(double[][] weight) {
		for(int i=0; i<weight.length; i++)
			weight[i][0]=Math.floor(weight[i][0]);
	}
	
	@Test
	public void testSimpleQuantile()
	{
		int rows = 10;
        int cols = 1;

        TestConfiguration config = getTestConfiguration("SimpleQuantileTest");
        config.addVariable("rows", rows);

        loadTestConfiguration("SimpleQuantileTest");

        createHelperMatrix();
        double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, System.currentTimeMillis());
        double[][] weight = getRandomMatrix(rows, 1, 1, 10, 1, System.currentTimeMillis());
        round(weight);
        
        double[] sorted=sort(vector);
        double median = quantile(sorted, 0.5);
        
        writeInputMatrix("vector", vector);
        writeExpectedHelperMatrix("median", median);
        
        
        double[] fullsorted=sort(blowUp(vector, weight));
        double weightedmedian=quantile(fullsorted, 0.5);
        writeInputMatrix("weight", weight);
        writeExpectedHelperMatrix("weighted_median", weightedmedian);
        
        runTest();

        compareResults(5e-14);
	}
	
	@Test
	public void testQuantile()
	{
		int rows1 = 10;
		int rows2 = 5;
        int cols = 1;

        TestConfiguration config = getTestConfiguration("QuantileTest");
        config.addVariable("rows1", rows1);
        config.addVariable("rows2", rows2);

        loadTestConfiguration("QuantileTest");

        createHelperMatrix();
        double[][] vector = getRandomMatrix(rows1, 1, 0, 1, 1, System.currentTimeMillis());
        double[][] weight = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
        round(weight);
        double[][] prob = getRandomMatrix(rows2, 1, 0, 1, 1, System.currentTimeMillis());
        
        double[][] q=new double[prob.length][1];
        double[] sorted=sort(vector);
        for(int i=0; i<prob.length; i++)
        	q[i][0]= quantile(sorted, prob[i][0]);
        
        writeInputMatrix("vector", vector);
        writeInputMatrix("prob", prob);
        writeExpectedMatrix("quantile", q);
        
        double[][] wq=new double[prob.length][1];
        double[] fullsorted=sort(blowUp(vector, weight));
        for(int i=0; i<prob.length; i++)
        	wq[i][0]=quantile(fullsorted, prob[i][0]);
        writeInputMatrix("weight", weight);
        writeExpectedMatrix("weighted_quantile", wq);
        
        runTest();

        compareResults(5e-14);
	}
	
	@Test
	public void testIQM()
	{
		int rows = 10;
        int cols = 1;

        TestConfiguration config = getTestConfiguration("IQMTest");
        config.addVariable("rows", rows);

        loadTestConfiguration("IQMTest");

        createHelperMatrix();
        double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, System.currentTimeMillis());
        double[][] weight = getRandomMatrix(rows, 1, 1, 10, 1, System.currentTimeMillis());
        round(weight);
        
        double[] sorted=sort(vector);
        double iqm = IQM(sorted);
        
        writeInputMatrix("vector", vector);
        writeExpectedHelperMatrix("iqm", iqm);
        
        
        double[] fullsorted=sort(blowUp(vector, weight));
        double weightediqm=IQM(fullsorted);
        writeInputMatrix("weight", weight);
        writeExpectedHelperMatrix("weighted_iqm", weightediqm);
        
        runTest();

        compareResults(5e-14);
	}
}
