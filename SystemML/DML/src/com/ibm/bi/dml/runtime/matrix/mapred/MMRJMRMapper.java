package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.TripleIndexes;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class MMRJMRMapper extends MapperBase 
implements Mapper<Writable, Writable, Writable, Writable>{
	
	//the aggregate binary instruction for this mmcj job
	private TripleIndexes triplebuffer=new TripleIndexes();
	private TaggedMatrixValue taggedValue=null;
	private HashMap<Byte, Long> numRepeats=new HashMap<Byte, Long>();
	private HashSet<Byte> aggBinInput1s=new HashSet<Byte>();
	private HashSet<Byte> aggBinInput2s=new HashSet<Byte>();
	
	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
		//apply all instructions
		processMapperInstructionsForMatrix(index);
		
		for(byte output: outputIndexes.get(index))
		{
			IndexedMatrixValue result=cachedValues.getFirst(output);
			if(result==null)
				continue;
			
			//output the left matrix
			if(aggBinInput1s.contains(output))
			{
				for(long j=0; j<numRepeats.get(output); j++)
				{
					triplebuffer.setIndexes(result.getIndexes().getRowIndex(), j+1, result.getIndexes().getColumnIndex());
					taggedValue.setBaseObject(result.getValue());
					taggedValue.setTag(output);
					out.collect(triplebuffer, taggedValue);
				//	System.out.println("output to reducer: "+triplebuffer+"\n"+taggedValue);
				}
			}else if(aggBinInput2s.contains(output))//output the right matrix
			{
				for(long i=0; i<numRepeats.get(output); i++)
				{
					triplebuffer.setIndexes(i+1, result.getIndexes().getColumnIndex(), result.getIndexes().getRowIndex());
					taggedValue.setBaseObject(result.getValue());
					taggedValue.setTag(output);
					out.collect(triplebuffer, taggedValue);
				//	System.out.println("output to reducer: "+triplebuffer+"\n"+taggedValue);
				}
			}else //output other matrix that are not involved in aggregate binary
			{
				triplebuffer.setIndexes(result.getIndexes().getRowIndex(), result.getIndexes().getColumnIndex(), -1);
				////////////////////////////////////////
			//	taggedValueBuffer.getBaseObject().copy(result.getValue());
				taggedValue.setBaseObject(result.getValue());
				////////////////////////////////////////
				taggedValue.setTag(output);
				out.collect(triplebuffer, taggedValue);
			//	System.out.println("output to reducer: "+triplebuffer+"\n"+taggedValue);
			}
		}	
	}

	@Override
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
		commonMap(rawKey, rawValue, out, reporter);
		
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
		taggedValue=TaggedMatrixValue.createObject(valueClass);
		AggregateBinaryInstruction[] aggBinInstructions;
		try {
			aggBinInstructions = MRJobConfiguration.getAggregateBinaryInstructions(job);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		
		for(AggregateBinaryInstruction aggBinInstruction: aggBinInstructions)
		{
			MatrixCharacteristics mc=MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, aggBinInstruction.input2);
			long matrixNumColumn=mc.numColumns;
			int blockNumColumn=mc.numColumnsPerBlock;
			numRepeats.put(aggBinInstruction.input1, (long)Math.ceil((double)matrixNumColumn/(double)blockNumColumn));
			
			mc=MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, aggBinInstruction.input1);
			long matrixNumRow=mc.numRows;
			int blockNumRow=mc.numRowsPerBlock;
			numRepeats.put(aggBinInstruction.input2, (long)Math.ceil((double)matrixNumRow/(double)blockNumRow));
			aggBinInput1s.add(aggBinInstruction.input1);
			aggBinInput2s.add(aggBinInstruction.input2);
		}
	}
}
