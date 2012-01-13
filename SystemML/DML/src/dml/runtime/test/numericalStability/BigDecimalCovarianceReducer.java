package dml.runtime.test.numericalStability;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.Iterator;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

public class BigDecimalCovarianceReducer extends MapReduceBase
implements Reducer<NullWritable, Text, NullWritable, Text>{

	MathContext mc=SetUp.mc;
	
	@Override
	public void reduce(NullWritable dummyKey, Iterator<Text> values,
			OutputCollector<NullWritable, Text> out, Reporter report)
			throws IOException {
		
		BigDecimal sumx=new BigDecimal(0.0, mc);
		BigDecimal sumy=new BigDecimal(0.0, mc);
		BigDecimal sumx2=new BigDecimal(0.0, mc);
		BigDecimal sumy2=new BigDecimal(0.0, mc);
		BigDecimal sumxy=new BigDecimal(0.0, mc);
		long count=0;
		
		while(values.hasNext())
		{
			String str=values.next().toString();
			//System.out.println(str);
			String[] strs=str.split("#");
			if(strs.length!=6)
				throw new IOException("expect 4 numbers instead of "+strs.length);
			count+=Long.parseLong(strs[0]);
			sumx=sumx.add(new BigDecimal(strs[1]));
			sumy=sumy.add(new BigDecimal(strs[2]));
			sumx2=sumx2.add(new BigDecimal(strs[3]));
			sumy2=sumy2.add(new BigDecimal(strs[4]));
			sumxy=sumxy.add(new BigDecimal(strs[5]));
		}
		BigDecimal n=new BigDecimal(count);
		
		BigDecimal nminus1=new BigDecimal(count-1);
		//var=(s2/n - mu*mu)*(n/(n-1))=(s2-s1*mu)/(n-1)
		BigDecimal stdx=BigFunctions.sqrt(sumx2.subtract(sumx.multiply(sumx).divide(n, mc)).divide(nminus1, mc), mc.getPrecision());
		BigDecimal stdy=BigFunctions.sqrt(sumy2.subtract(sumy.multiply(sumy).divide(n, mc)).divide(nminus1, mc), mc.getPrecision());
		
		//covariance=(sxy-(sumx.sumy/n)/(n-1)
		BigDecimal covariance=(sumxy.subtract(sumx.multiply(sumy).divide(n, mc))).divide(nminus1, mc);
		BigDecimal pearsonR=covariance.divide(stdx.multiply(stdy), mc);
		String outStr="Covariance: "+covariance+"\nPearsonR: "+pearsonR;
		out.collect(NullWritable.get(), new Text(outStr));
		//System.out.println(outStr);
	}
}
