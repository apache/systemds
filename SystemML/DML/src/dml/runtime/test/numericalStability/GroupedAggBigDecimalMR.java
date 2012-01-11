package dml.runtime.test.numericalStability;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.HashMap;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.Counters.Group;

import dml.runtime.matrix.CombineMR;
import dml.runtime.matrix.io.Converter;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.mapred.MRJobConfiguration;

import dml.runtime.util.MapReduceTool;

public class GroupedAggBigDecimalMR {

	public static void runJob(String input, int numReducers, int replication, String output) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(GroupedAggBigDecimalMR.class);
		job.setJobName("GroupedAggBigDecimalMR");
	
		FileInputFormat.addInputPath(job, new Path(input));
		job.setInputFormat(SequenceFileInputFormat.class);
		//set up the number of reducers
		job.setNumReduceTasks(numReducers);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
				
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(GroupedAggBigDecimalMRMapper.class);
		
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(DoubleWritable.class);
		
		//configure reducer
		job.setReducerClass(GroupedAggBigDecimalMRReducer.class);
		MapReduceTool.deleteFileIfExistOnHDFS(output);
		FileOutputFormat.setOutputPath(job, new Path(output));
		
		RunningJob runjob=JobClient.runJob(job);		
	}
	
	public static HashMap<String, BigDecimal> computeBivariateStats(String dir) throws IOException
	{
		JobConf job = new JobConf();
		FileInputFormat.addInputPath(job, new Path(dir));
		TextInputFormat informat=new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits= informat.getSplits(job, 1);
		LongWritable lineNumber=new LongWritable();
		Text line=new Text();
		MathContext mc=SetUp.mc;
		
		MathContext outmc=new MathContext(10, RoundingMode.HALF_UP);
		long c=0;
		BigDecimal s=new BigDecimal(0);
		BigDecimal s2=new BigDecimal(0);
		BigDecimal accumGroupVar=new BigDecimal(0);
		BigDecimal fnominator=new BigDecimal(0);
		long r=0;
		for(InputSplit split: splits)
		{
			RecordReader reader=informat.getRecordReader(split, job, Reporter.NULL);
			while(reader.next(lineNumber, line))
			{
				String[] strs=line.toString().split("\t");
				long ci=Long.parseLong(strs[0]);
				BigDecimal ni=new BigDecimal(ci);
				BigDecimal si=new BigDecimal(strs[1]);
				BigDecimal s2i=new BigDecimal(strs[2]);
				int group=Integer.parseInt(strs[3]);
				if(group>r)
					r=group;
				
				c+=ci;
				s=s.add(si);
				s2=s2.add(s2i);
					
				BigDecimal mui=si.divide(ni, mc);
				//(n-1)*var=(s2-s1*mu)
				BigDecimal accumvari=s2i.subtract(si.multiply(mui));
				accumGroupVar=accumGroupVar.add(accumvari);
				fnominator=fnominator.add(si.multiply(mui));
			//	System.out.println("group: "+group+" s: "+si.round(outmc)+", s2: "+s2i.round(outmc)+", mu: "+mui.round(outmc)+", accumvari: "+accumvari.round(outmc));
			}
		}
		BigDecimal W=new BigDecimal(c);
		BigDecimal R=new BigDecimal(r);
		BigDecimal mu=s.divide(W, mc);
		BigDecimal accumVar=s2.subtract(s.multiply(mu));
		//System.out.println("accumGroupVar: "+accumGroupVar);
		BigDecimal eta=BigFunctions.sqrt(BigDecimal.ONE.subtract(accumGroupVar.divide(accumVar, mc)), mc.getPrecision());
		fnominator=fnominator.subtract(W.multiply(mu.multiply(mu)));
		//System.out.println("fnominator: "+fnominator);
		fnominator=fnominator.divide(R.subtract(BigDecimal.ONE), mc);
		//System.out.println("fnominator after: "+fnominator);
		BigDecimal denominator=accumGroupVar.divide(W.subtract(R), mc);
		//System.out.println("denominator after: "+denominator);
		BigDecimal f=fnominator.divide(denominator, mc);
		HashMap<String, BigDecimal> map=new HashMap<String, BigDecimal>(2);
		map.put("Eta", eta);
		map.put("AnovaF", f);
		return map;
		//System.out.println("total -- s: "+s+", s2: "+s2+", mu: "+mu+", accumvari: "+accumVar+", fnominator: "+fnominator);
		//System.out.println("Eta: "+eta+"\nF: "+f);
	}
	
	public static void main(String[] args) throws Exception {
		
		if(args.length<6)
		{
			System.out.println("GroupedAggBigDecimalMR <numerial input> <categorical input> <num Rows> <# reducers> <output> <blockSize>");
			System.exit(-1);
		}
		
		boolean inBlockRepresentation=false;
		String V=args[0];
		String U=args[1];
		String[] inputs=new String[]{V, U};
		int npb=Integer.parseInt(args[5]);
		InputInfo[] inputInfos=new InputInfo[]{InputInfo.TextCellInputInfo, InputInfo.TextCellInputInfo};
		if(npb>1)
		{
			inputInfos=new InputInfo[]{InputInfo.BinaryBlockInputInfo, InputInfo.BinaryBlockInputInfo};
			inBlockRepresentation=true;
		}
		long r=Long.parseLong(args[2]);
		long[] rlens=new long[]{r, r};
		long[] clens=new long[]{1, 1};
		int[] brlens=new int[]{npb, npb};
		int[] bclens=new int[]{1, 1};
		String combineInstructions="combinebinary\u00b0false\u00b7BOOLEAN\u00b00\u00b7DOUBLE\u00b01\u00b7DOUBLE\u00b02\u00b7DOUBLE";
		int numReducers=Integer.parseInt(args[3]);
		int replication=1;
		byte[] resultIndexes=new byte[]{2};
		String UV="UV";
		String[] outputs=new String[]{UV};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.WeightedPairOutputInfo};
		CombineMR.runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, 
				combineInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);
		
		GroupedAggBigDecimalMR.runJob(UV, numReducers, replication, args[4]);
	}
}
