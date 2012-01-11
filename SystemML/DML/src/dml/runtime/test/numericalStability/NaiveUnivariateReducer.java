package dml.runtime.test.numericalStability;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

public class NaiveUnivariateReducer extends MapReduceBase
implements Reducer<DoubleWritable, Text, NullWritable, Text>{
	
	double sum=0;
	double sum2=0;
	double sum3=0;
	double sum4=0;
	long count=0;
	private OutputCollector<NullWritable, Text> cachedCollector=null;
	private boolean firsttime=true;
	@Override
	public void reduce(DoubleWritable dummyKey, Iterator<Text> values,
			OutputCollector<NullWritable, Text> out, Reporter report)
			throws IOException {
		
		if(firsttime)
		{
			cachedCollector=out;
			firsttime=false;
		}
		
		while(values.hasNext())
		{
			String str=values.next().toString();
			//System.out.println(str);
			String[] strs=str.split("#");
			if(strs.length!=5)
				throw new IOException("expect 5 numbers instead of "+strs.length);
			count+=Long.parseLong(strs[0]);
			sum+=Double.parseDouble(strs[1]);
			sum2+=Double.parseDouble(strs[2]);
			sum3+=Double.parseDouble(strs[3]);
			sum4+=Double.parseDouble(strs[4]);
		}
		
		//System.out.println(outStr);
	}

	public void close() throws IOException
	{
		double n=count;
		String outStr="Summation: "+sum;
		double mu=sum/n;
		outStr+="\nMean: "+mu+"\nVariance: ";
		double nminus1=count-1;
		//var=(s2/n - mu*mu)*(n/(n-1))=(s2-s1*mu)/(n-1)
		double var=(sum2-sum*mu)/nminus1;
		double std=Math.sqrt(var);
		outStr+=var+"\nStd: "+std;
		
		//g1 = (s3 - 3*mu*s2 + 3*mu^2*s1 -n*mu^3)/(n*std_dev^3)
		// t=s3 - 3*mu*s2
		
		double t=sum3-(sum2*mu*3.0);
		//t=t+3*mu^2*s1
		t=t+sum*3.0*Math.pow(mu, 2);
		//t=t-n*mu^3
		t=t-n*Math.pow(mu, 3);
		
		double g1=t/n/var/std;
		outStr+="\nSkewness: "+g1;
		
		//g2 = (sum(V^4) - 4*s3*mu + 6*s2*mu^2 - 3*n*mu^4)/(n*std_dev^4) - 3
		//t=sum(V^4) - 4*s3*mu
		
		t=sum4-sum3*mu*4;
		//t=t+ 6*s2*mu^2
		t=t+sum2*Math.pow(mu, 2)*6;
		//t=t- 3*n*mu^4
		t=t-Math.pow(mu, 4)*n*3;
		double g2=t/Math.pow(var,2)/n - 3;
		outStr+="\nKurtosis: "+g2;		
	
		if(cachedCollector!=null)
		{
			String str=n+"#"+sum+"#"+sum2+"#"+sum3+"#"+sum4;
			cachedCollector.collect(NullWritable.get(), new Text(outStr));
		}
	}
}
