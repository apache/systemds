package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Vector;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixPackedCell;
import dml.runtime.matrix.io.TaggedMatrixPackedCell;
import dml.runtime.matrix.io.TaggedMatrixValue;

public class GMRMapper extends MapperBase 
implements Mapper<Writable, Writable, Writable, Writable>{

	//whether this is a map only job
	private boolean mapOnlyJob=false;
	
	//the final result indexes that needed to be outputted
	protected byte[] resultIndexes=null;
	
	//the counters to record how many nonZero cells have been produced for each output
	protected long[] resultsNonZeros=null;
	
	//the output converter if this is a map only job
	private CollectMultipleConvertedOutputs collectFinalMultipleOutputs=null;
	
	//tempory variables
	private MatrixIndexes indexBuffer=new MatrixIndexes();
	//private MatrixValue valueBuffer=null;
	private TaggedMatrixValue taggedValueBuffer=null;
	private HashMap<Byte, Vector<Integer>> tagMapping;
		
	public void map(Writable rawKey, Writable rawValue, OutputCollector<Writable, Writable> out, 
			Reporter reporter) throws IOException 
	{
	//	System.out.println("in mapper: \n"+rawValue);
		commonMap(rawKey, rawValue, out, reporter);
	}
	
	protected void specialOperationsForActualMap(int index, OutputCollector<Writable, Writable> out, 
			Reporter reporter) throws IOException {
		
		//apply all instructions
		processMapperInstructionsForMatrix(index);
		
		//output the results needed by the reducer
		if(mapOnlyJob)
			processMapFinalOutput(index, indexBuffer, taggedValueBuffer, 
					collectFinalMultipleOutputs, reporter, tagMapping);
		else
			processMapOutputToReducerForGMR(index, indexBuffer, taggedValueBuffer, out);
	}
	
	
	protected void processMapOutputToReducerForGMR(int index, MatrixIndexes indexBuffer, 
			TaggedMatrixValue taggedValueBuffer, OutputCollector<Writable, Writable> out) throws IOException
	{
			
		for(byte output: outputIndexes.get(index))
		{
			IndexedMatrixValue result=cachedValues.get(output);
			if(result==null)
				continue;
			indexBuffer.setIndexes(result.getIndexes());
			////////////////////////////////////////
		//	taggedValueBuffer.getBaseObject().copy(result.getValue());
			if(valueClass.equals(MatrixCell.class))
				taggedValueBuffer.getBaseObject().copy(result.getValue());
			else
				taggedValueBuffer.setBaseObject(result.getValue());
			////////////////////////////////////////
			taggedValueBuffer.setTag(output);
			out.collect(indexBuffer, taggedValueBuffer);
		//	System.out.println("map output: "+indexBuffer+"\n"+taggedValueBuffer);
		}	
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
		//assign the temporay vairables
		try {
		//	System.out.println(valueClass.getName());
		//	System.out.println(MatrixCell.class.getName());
			if(job.getMapOutputValueClass().equals(TaggedMatrixPackedCell.class))
				taggedValueBuffer=TaggedMatrixValue.createObject(MatrixPackedCell.class);
			else
				taggedValueBuffer=TaggedMatrixValue.createObject(valueClass);		
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
		//decide whether it is a maponly job
		mapOnlyJob=(job.getNumReduceTasks()<=0);
		if(!mapOnlyJob)
			return;
		
		//get the indexes of the final output matrices
		resultIndexes=MRJobConfiguration.getResultIndexes(job);
		
		//initialize the counters for the nonZero cells
		resultsNonZeros=new long[resultIndexes.length];
		Arrays.fill(resultsNonZeros, 0);
		
		tagMapping=new HashMap<Byte, Vector<Integer>>();
		for(int i=0; i<resultIndexes.length; i++)
		{
			byte output=resultIndexes[i];
			Vector<Integer> vec=tagMapping.get(output);
			if(vec==null)
			{
				vec=new Vector<Integer>();
				tagMapping.put(output, vec);
			}
			vec.add(i);
		}
		//for map only job, get the map output converters 
		collectFinalMultipleOutputs=MRJobConfiguration.getMultipleConvertedOutputs(job);
	}
	
	public void close() throws IOException
	{
		if(collectFinalMultipleOutputs!=null)
			collectFinalMultipleOutputs.close();
	}
}
