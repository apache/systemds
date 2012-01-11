package dml.runtime.test.numericalStability;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.WeightedPair;

public class BigDecimalCovarianceMapper extends MapReduceBase
implements Mapper<MatrixIndexes, WeightedPair, NullWritable, Text>{
	
	private boolean firsttime=true;
	private OutputCollector<NullWritable, Text> cachedCollector=null;
	
	private MathContext mc=SetUp.mc;
	private BigDecimal sumx=new BigDecimal(0.0, mc);
	private BigDecimal sumy=new BigDecimal(0.0, mc);
	private BigDecimal sumx2=new BigDecimal(0.0, mc);
	private BigDecimal sumy2=new BigDecimal(0.0, mc);
	private BigDecimal sumxy=new BigDecimal(0.0, mc);
	private long n=0;
	
	@Override
	public void map(MatrixIndexes index, WeightedPair wpair,
			OutputCollector<NullWritable, Text> out,
			Reporter reporter) throws IOException {
		if(firsttime)
		{
			cachedCollector=out;
			firsttime=false;
		}
		
		BigDecimal x=new BigDecimal(wpair.getValue());
		BigDecimal y=new BigDecimal(wpair.getOtherValue());
		sumx=sumx.add(x);
		sumy=sumy.add(y);
		sumx2=sumx2.add(x.multiply(x));
		sumy2=sumy2.add(y.multiply(y));
		sumxy=sumxy.add(x.multiply(y));
		n++;
	}

	public void close() throws IOException
	{
		if(cachedCollector!=null)
		{
			String str=n+"#"+sumx+"#"+sumy+"#"+sumx2+"#"+sumy2+"#"+sumxy;
			cachedCollector.collect(NullWritable.get(), new Text(str));
		}
	}
}
