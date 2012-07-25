package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixPackedCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixPackedCell;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixValue;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class GMRMapper extends MapperBase 
implements Mapper<Writable, Writable, Writable, Writable>{

	//whether this is a map only job
	private boolean mapOnlyJob=false;
	
	//the final result indexes that needed to be outputted for maponly job
	protected byte[] resultIndexes=null;
	protected byte[] resultDimsUnknown=null;

	//output converters for maponly job
	protected CollectMultipleConvertedOutputs collectFinalMultipleOutputs;

	//the counters to record how many nonZero cells have been produced for each output
	// for maponly job
	protected long[] resultsNonZeros=null;
	protected long[] resultsMaxRowDims=null;
	protected long[] resultsMaxColDims=null;
	protected String dimsUnknownFilePrefix;
	
	//cached reporter to report the number of nonZeros for each reduce task
	protected Reporter cachedReporter=null;
	protected boolean firsttime=true;
	
	protected String mapperID;
	
	//tempory variables
	private MatrixIndexes indexBuffer=new MatrixIndexes();
	//private MatrixValue valueBuffer=null;
	private TaggedMatrixValue taggedValueBuffer=null;
	private HashMap<Byte, Vector<Integer>> tagMapping;
		
	public void map(Writable rawKey, Writable rawValue, OutputCollector<Writable, Writable> out, 
			Reporter reporter) throws IOException 
	{
		if(firsttime)
		{
			cachedReporter=reporter;
			firsttime=false;
		}
		
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
			ArrayList<IndexedMatrixValue> results= cachedValues.get(output);
			if(results==null)
				continue;
			for(IndexedMatrixValue result: results)
			{
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
	}
	
	protected void processMapFinalOutput(int index, MatrixIndexes indexBuffer,
			TaggedMatrixValue taggedValueBuffer, CollectMultipleConvertedOutputs collectFinalMultipleOutputs,
			Reporter reporter, HashMap<Byte, Vector<Integer>> tagMapping) throws IOException
	{
		for(byte output: outputIndexes.get(index))
		{
			IndexedMatrixValue result=cachedValues.getFirst(output);
			if(result==null)
				continue;
			indexBuffer.setIndexes(result.getIndexes());
			////////////////////////////////////////
			//	taggedValueBuffer.getBaseObject().copy(result.getValue());
			taggedValueBuffer.setBaseObject(result.getValue());
			////////////////////////////////////////
			taggedValueBuffer.setTag(output);
			for(int outputIndex: tagMapping.get(output))
			{
				collectOutput_N_Increase_Counter(indexBuffer, taggedValueBuffer.getBaseObject(), outputIndex, 
					reporter);
			}
		}	
	}

	protected void collectOutput_N_Increase_Counter(MatrixIndexes indexes, MatrixValue value, 
			int i, Reporter reporter) throws IOException
	{
		collectOutput_N_Increase_Counter(indexes, value, i, reporter, collectFinalMultipleOutputs, 
				resultDimsUnknown, resultsNonZeros, resultsMaxRowDims, resultsMaxColDims);
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
		
		mapperID = job.get("mapred.task.id");
		dimsUnknownFilePrefix = job.get("dims.unknown.file.prefix");
		
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
		resultDimsUnknown = MRJobConfiguration.getResultDimsUnknown(job);
		
		//initialize SystemML Counters (defined in MRJobConfiguration)
		resultsNonZeros=new long[resultIndexes.length];
		resultsMaxRowDims=new long[resultIndexes.length];
		resultsMaxColDims=new long[resultIndexes.length];
		
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
		if(cachedReporter!=null)
		{
			String[] parts = mapperID.split("_");
			String jobID = "job_" + parts[1] + "_" + parts[2];
			int taskid;
			if ( parts[0].equalsIgnoreCase("task")) {
				taskid = Integer.parseInt(parts[parts.length-1]);
			}
			else if ( parts[0].equalsIgnoreCase("attempt")) {
				taskid = Integer.parseInt(parts[parts.length-2]);
			}
			else {
				throw new RuntimeException("Unrecognized format for reducerID: " + mapperID);
			}
			//System.out.println("Inside GMRMapper.close(): jobID = " + jobID + ", taskID = " + taskid);
			
			if(mapOnlyJob)
			{
				boolean dimsUnknown = false;
				for(int i=0; i<resultIndexes.length; i++) {
					cachedReporter.incrCounter(MRJobConfiguration.NUM_NONZERO_CELLS, Integer.toString(i), resultsNonZeros[i]);
					
					if ( resultDimsUnknown!=null && resultDimsUnknown[i] != (byte) 0 ) {
						dimsUnknown = true;
						// Each counter is of the form: (group, name)
						// where group = max_rowdim_resultindex; name = taskid
						//System.out.println("--> before i="+i+", row = " + cachedReporter.getCounter("max_rowdim_"+i, ""+taskid).getCounter() + ", col = " + cachedReporter.getCounter("max_coldim_"+i, ""+taskid).getCounter());
						//cachedReporter.getCounter(MRJobConfiguration.MAX_ROW_DIMENSION, Integer.toString(i)).increment(resultsMaxRowDims[i]);
						//cachedReporter.getCounter(MRJobConfiguration.MAX_COL_DIMENSION, Integer.toString(i)).increment(resultsMaxColDims[i]);
						//System.out.println("--> after i="+i+", row = " + cachedReporter.getCounter("max_rowdim_"+i, ""+taskid).getCounter() + ", col = " + cachedReporter.getCounter("max_coldim_"+i, ""+taskid).getCounter());
					}
				}
				//System.out.println("DimsUnknown = " + dimsUnknown);
				if ( dimsUnknown ) {
					// every task creates a file with max_row and max_col dimensions found in that task
					MapReduceTool.writeDimsFile(dimsUnknownFilePrefix + "/" + jobID + "_dimsFile/" + "m_" + taskid , resultDimsUnknown, resultsMaxRowDims, resultsMaxColDims);
				}
			}
		}
		
		if(collectFinalMultipleOutputs!=null)
			collectFinalMultipleOutputs.close();
	}
}
