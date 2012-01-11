package dml.runtime.test.numericalStability;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;

import dml.api.DMLScript;
import dml.runtime.matrix.io.Converter;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.runtime.util.MapReduceTool;

public class SetUp {

	public static MathContext mc=new MathContext(1000, RoundingMode.HALF_UP);
	public static MathContext outmc=new MathContext(100, RoundingMode.HALF_UP);
	
	//computed value (q) and the real value (c)
	public static BigDecimal calculateLRE(BigDecimal q, BigDecimal c)
	{
		//LRE=-log10(|q-c|/|c|)
		/*
		Rules:
	    q should be close to c (less than factor of 2). If they are not, set LRE to zero
	    If LRE is greater than number of the digits in c, set LRE to number of the digits in c.
	    If LRE is less than unity, set it to zero.
		 */
		BigDecimal LRE;
		BigDecimal numDigitsInc=new BigDecimal(c.precision());
		
		if(c.compareTo(BigDecimal.ZERO)!=0)
		{
			BigDecimal factor=q.divide(c, mc);
	//		System.out.println("-- factor: "+factor);
		
			//q should be close to c (less than factor of 2). If they are not, set LRE to zero
			if(factor.compareTo(new BigDecimal(2))>=0 || factor.compareTo(new BigDecimal(0.5))<=0)
				return BigDecimal.ZERO;
		}
		
		
		BigDecimal temp=q.subtract(c).abs();
	//	System.out.println("-- temp: "+temp);
		if(temp.compareTo(new BigDecimal("1E-999"))<=0)//if q==c, return # digits in c
			return numDigitsInc;
		
		//if c=0, return LRE=-log10(|q|)
		//otherwise return LRE=-log10(|q-c|/|c|)
		if(c.compareTo(BigDecimal.ZERO)!=0)
			temp=temp.divide(c.abs(), outmc);
		
//		System.out.println("-- temp: "+temp);
		
		if(temp.compareTo(new BigDecimal("1E-999"))==0)//if q==c, return # digits in c
			return numDigitsInc;
		
		LRE=BigFunctions.ln(temp, outmc.getPrecision());
	//	System.out.println("-- LRE: "+LRE);
		LRE=LRE.divide(BigFunctions.ln(BigDecimal.TEN, outmc.getPrecision()), outmc).negate();
	//	System.out.println("-- LRE: "+LRE);
		if(LRE.compareTo(numDigitsInc )>0)
			LRE=numDigitsInc;
		else if(LRE.compareTo(BigDecimal.ONE)<0)
			LRE=BigDecimal.ZERO;
	//	System.out.println("-- LRE: "+LRE);
		return LRE;
	}
	
	public static HashMap<String, BigDecimal> compareResult(String bigDecimalResult, String systemMLResultFolder) throws IOException
	{	
		HashMap<String, BigDecimal> map=getBigDecimalResults(bigDecimalResult);
	//	System.out.println(map);
		return compareResult(map, systemMLResultFolder);
	}
	
	public static HashMap<String, BigDecimal> compareResult(HashMap<String, BigDecimal> bigDecimalResult, String systemMLResultFolder) throws IOException
	{
		//Univariate:Summation, Mean, Variance, Std, Skewness, Kurtosis
		//Grouped: Eta, AnovaF
		//Covariance: Covariance, PearsonR
		
		for(Entry<String, BigDecimal> e: bigDecimalResult.entrySet())
		{
			BigDecimal q=getSystemMLResult(systemMLResultFolder, e.getKey());
		//	System.out.println(e.getKey()+"\n true: "+e.getValue()+"\n get: "+q);
			if(q==null)
			{
				e.setValue(null);
				continue;
			}
			if(e.getKey().equals("Eta")&&q.compareTo(BigDecimal.ZERO)<0)
				e.setValue(BigDecimal.ZERO);
			else
				e.setValue(calculateLRE(q, e.getValue()));
		}
		return bigDecimalResult;
	}

	public static BigDecimal getSystemMLResult(String folder,	String name) {
		double[][] array;
		try {
			array = MapReduceTool.readMatrixFromHDFS(folder+"/"+name, InputInfo.BinaryCellInputInfo, 1, 1, 1, 1);
		} catch (IOException e) {
			return null;
		}
		if(array[0][0]==Double.NaN)
			return null;
		return new BigDecimal(array[0][0]);
	}

	private static HashMap<String, BigDecimal> getBigDecimalResults(String bigDecimalResult) throws IOException
	{
		JobConf job = new JobConf();
		FileInputFormat.addInputPath(job, new Path(bigDecimalResult));
		HashMap<String, BigDecimal> map=new HashMap<String, BigDecimal>();
		try {

			TextInputFormat informat=new TextInputFormat();
			informat.configure(job);
			InputSplit[] splits= informat.getSplits(job, 1);	
			LongWritable key=LongWritable.class.newInstance();
			Text value=Text.class.newInstance();
			
			for(InputSplit split: splits)
			{
				RecordReader reader=informat.getRecordReader(split, job, Reporter.NULL);
				while(reader.next(key, value))
				{
					String[] strs=value.toString().split(": ");
					map.put(strs[0], new BigDecimal(strs[1]));
				}
			}
			
			return map;
			
		} catch (Exception e) {
			throw new IOException(e);
		}
	}
	
	private static HashMap<String, BigDecimal> getSystemMLResults(String bigDecimalResult) throws IOException
	{
		JobConf job = new JobConf();
		FileInputFormat.addInputPath(job, new Path(bigDecimalResult));
		HashMap<String, BigDecimal> map=new HashMap<String, BigDecimal>();
		try {

			TextInputFormat informat=new TextInputFormat();
			informat.configure(job);
			InputSplit[] splits= informat.getSplits(job, 1);	
			LongWritable key=LongWritable.class.newInstance();
			Text value=Text.class.newInstance();
			
			for(InputSplit split: splits)
			{
				RecordReader reader=informat.getRecordReader(split, job, Reporter.NULL);
				while(reader.next(key, value))
				{
					String[] strs=value.toString().split(": ");
					BigDecimal parsed=null;
					try{
						parsed=new BigDecimal(strs[1]);
					}catch(NumberFormatException e)
					{
						parsed=null;
					}
					map.put(strs[0], parsed);
				}
			}
			
			return map;
			
		} catch (Exception e) {
			throw new IOException(e);
		}
	}
	
	public static void main(String[] args) throws Exception
	{
		if(args.length<4)
		{
			System.out.println("SetUp <BigDecimalResult> <ComputedResult> <forEta&AnovaF> <forUnstableUnivariate?>");
			System.exit(-1);
		}
		
		boolean forUnstableUnivariate=Boolean.parseBoolean(args[3]);
		HashMap<String, BigDecimal> map=null;
		if(forUnstableUnivariate)
		{
			map=getBigDecimalResults(args[0]);
			HashMap<String, BigDecimal> map2=getSystemMLResults(args[1]);
			for(Entry<String, BigDecimal> e: map.entrySet())
			{
				BigDecimal q=map2.get(e.getKey());
			//	System.out.println(e.getKey()+"\n true: "+e.getValue()+"\n get: "+q);
				if(q==null)
				{
					e.setValue(null);
					continue;
				}
				if(e.getKey().equals("Eta")&&q.compareTo(BigDecimal.ZERO)<0)
					e.setValue(BigDecimal.ZERO);
				else
					e.setValue(calculateLRE(q, e.getValue()));
			}
		}else
		{
			
			if(Boolean.parseBoolean(args[2]))
			{
				map=GroupedAggBigDecimalMR.computeBivariateStats(args[0]);
				map=compareResult(map, args[1]);
			}else
			{
				map=compareResult(args[0], args[1]);
			}
		}
		
		for(Entry<String, BigDecimal> e: map.entrySet())
		{
			if(e.getValue()!=null)
				System.out.println(e.getKey()+"\t"+e.getValue());//.round(outmc));
			else
				System.out.println(e.getKey()+"\tNULL");
		}
		
	}
}
