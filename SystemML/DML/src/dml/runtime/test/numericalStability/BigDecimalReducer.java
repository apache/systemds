package dml.runtime.test.numericalStability;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.Iterator;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;


public class BigDecimalReducer extends MapReduceBase
implements Reducer<NullWritable, Text, NullWritable, Text>{

	MathContext mc=SetUp.mc;
	BigDecimal three=new BigDecimal(3);
	BigDecimal four=new BigDecimal(4);
	BigDecimal six=new BigDecimal(6);
	
	@Override
	public void reduce(NullWritable dummyKey, Iterator<Text> values,
			OutputCollector<NullWritable, Text> out, Reporter report)
			throws IOException {
		
		BigDecimal sum=new BigDecimal(0.0, mc);
		BigDecimal sum2=new BigDecimal(0.0, mc);
		BigDecimal sum3=new BigDecimal(0.0, mc);
		BigDecimal sum4=new BigDecimal(0.0, mc);
		long count=0;
		
		while(values.hasNext())
		{
			String str=values.next().toString();
			//System.out.println(str);
			String[] strs=str.split("#");
			if(strs.length!=5)
				throw new IOException("expect 5 numbers instead of "+strs.length);
			count+=Long.parseLong(strs[0]);
			sum=sum.add(new BigDecimal(strs[1]));
			sum2=sum2.add(new BigDecimal(strs[2]));
			sum3=sum3.add(new BigDecimal(strs[3]));
			sum4=sum4.add(new BigDecimal(strs[4]));
		}
		BigDecimal n=new BigDecimal(count);
		String outStr="Summation: "+sum;
		BigDecimal mu=sum.divide(n, mc);
		outStr+="\nMean: "+mu+"\nVariance: ";
		BigDecimal nminus1=new BigDecimal(count-1);
		//var=(s2/n - mu*mu)*(n/(n-1))=(s2-s1*mu)/(n-1)
		BigDecimal var=sum2.subtract(sum.multiply(mu)).divide(nminus1, mc);
		BigDecimal std=BigFunctions.sqrt(var, mc.getPrecision());
		outStr+=var+"\nStd: "+std;
		
		//g1 = (s3 - 3*mu*s2 + 3*mu^2*s1 -n*mu^3)/(n*std_dev^3)
		// t=s3 - 3*mu*s2
		
		BigDecimal t=sum3.subtract(sum2.multiply(mu).multiply(three));
		//t=t+3*mu^2*s1
		t=t.add(sum.multiply(three).multiply(mu.pow(2)));
		//t=t-n*mu^3
		t=t.subtract(mu.pow(3).multiply(n));		
		BigDecimal g1=t.divide(n.multiply(var).multiply(std), mc);
		outStr+="\nSkewness: "+g1;
		
		//g2 = (sum(V^4) - 4*s3*mu + 6*s2*mu^2 - 3*n*mu^4)/(n*std_dev^4) - 3
		//t=sum(V^4) - 4*s3*mu
		
		t=sum4.subtract(sum3.multiply(mu).multiply(four));
		//t=t+ 6*s2*mu^2
		t=t.add(sum2.multiply(mu.pow(2)).multiply(six));
		//t=t- 3*n*mu^4
		t=t.subtract(mu.pow(4).multiply(n).multiply(three));
		BigDecimal g2=t.divide(var.pow(2).multiply(n), mc).subtract(three);
		outStr+="\nKurtosis: "+g2;		
	
		out.collect(NullWritable.get(), new Text(outStr));
		//System.out.println(outStr);
	}
}
