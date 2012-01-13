package dml.runtime.test.numericalStability;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.matrix.io.Converter;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.mapred.MRJobConfiguration;

public class NaiveUnivariateMapper extends MapReduceBase
implements Mapper<Writable, Writable, DoubleWritable, Text>{

	private boolean firsttime=true;
	private OutputCollector<DoubleWritable, Text> cachedCollector=null;
	//converter to convert the input record into indexes and matrix value (can be a cell or a block)
	private Converter inputConverter=null;
	
	//the block sizes for the representative matrices
	protected int brlen=0;
	protected int bclen=0;

	private double sum=0;
	private double sum2=0;
	private double sum3=0;
	private double sum4=0;
	private long n=0;
	
	private void update(double v)
	{
		sum+=v;
		sum2+=Math.pow(v,2);
		sum3+=Math.pow(v,3);
		sum4+=Math.pow(v,4);
		n++;
	}
	
	@Override
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<DoubleWritable, Text> out, Reporter report)
			throws IOException {
		if(firsttime)
		{
			cachedCollector=out;
			firsttime=false;
		}
		
		inputConverter.setBlockSize(brlen, bclen);
		inputConverter.convert(rawKey, rawValue);
		
		//apply unary instructions on the converted indexes and values
		while(inputConverter.hasNext())
		{
			Pair<MatrixIndexes, MatrixCell> pair=inputConverter.next();
		//	System.out.println("convert to: "+pair);
			MatrixIndexes indexes=pair.getKey();
			MatrixCell value=pair.getValue();
			update(value.getValue());
		}
		
	}

	public void close() throws IOException
	{
		if(cachedCollector!=null)
		{
			String str=n+"#"+sum+"#"+sum2+"#"+sum3+"#"+sum4;
			cachedCollector.collect(new DoubleWritable(sum), new Text(str));
		}
	}

	public void configure(JobConf job)
	{
		super.configure(job);
		//get input converter information
		inputConverter=MRJobConfiguration.getInputConverter(job, (byte)0);
		//get the block sizes of the representative matrices
		brlen=MRJobConfiguration.getNumRowsPerBlock(job, (byte)0);
		bclen=MRJobConfiguration.getNumColumnsPerBlock(job, (byte)0);
	}

}
