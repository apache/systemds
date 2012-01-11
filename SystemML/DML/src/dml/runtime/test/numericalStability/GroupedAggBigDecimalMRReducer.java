package dml.runtime.test.numericalStability;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.Iterator;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import dml.runtime.matrix.io.WeightedCell;


public class GroupedAggBigDecimalMRReducer extends MapReduceBase
implements Reducer<IntWritable, DoubleWritable, NullWritable, Text >{

	Text buff=new Text();	
	MathContext mc=SetUp.mc;
	
	@Override
	public void reduce(IntWritable key, Iterator<DoubleWritable> values,
			OutputCollector<NullWritable, Text> out, Reporter report)
			throws IOException {
		BigDecimal sum=new BigDecimal(0.0, mc);
		BigDecimal sum2=new BigDecimal(0.0, mc);
		long n=0;
		while(values.hasNext())
		{
			BigDecimal v=new BigDecimal(values.next().get());
			sum=sum.add(v);
			sum2=sum2.add(v.pow(2));
			n++;
		}
		buff.set(n+"\t"+sum+"\t"+sum2+"\t"+key);
		out.collect(NullWritable.get(), buff);
	}	
}
