package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.io.TaggedMatrixValue;
import dml.runtime.matrix.io.TripleIndexes;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class MMRJMRMapper extends MapperBase 
implements Mapper<Writable, Writable, Writable, Writable>{
	
	//the aggregate binary instruction for this mmcj job
	private AggregateBinaryInstruction aggBinInstruction;
	private TripleIndexes triplebuffer=new TripleIndexes();
	private TaggedMatrixValue taggedValue=null;
	private HashMap<Byte, Long> numRepeats=new HashMap<Byte, Long>();
	
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
			if(output==aggBinInstruction.input1)
			{
				for(long j=0; j<numRepeats.get(aggBinInstruction.input1); j++)
				{
					triplebuffer.setIndexes(result.getIndexes().getRowIndex(), j+1, result.getIndexes().getColumnIndex());
					taggedValue.setBaseObject(result.getValue());
					taggedValue.setTag(aggBinInstruction.input1);
					out.collect(triplebuffer, taggedValue);
				//	System.out.println("output to reducer: "+triplebuffer+"\n"+taggedValue);
				}
			}else if(output==aggBinInstruction.input2)//output the right matrix
			{
				for(long i=0; i<numRepeats.get(aggBinInstruction.input2); i++)
				{
					triplebuffer.setIndexes(i+1, result.getIndexes().getColumnIndex(), result.getIndexes().getRowIndex());
					taggedValue.setBaseObject(result.getValue());
					taggedValue.setTag(aggBinInstruction.input2);
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
		AggregateBinaryInstruction[] ins;
		try {
			ins = MRJobConfiguration.getAggregateBinaryInstructions(job);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		if(ins.length!=1)
			throw new RuntimeException("MMRJ only perform one aggregate binary instruction");
		aggBinInstruction=ins[0];
		
	
		MatrixCharacteristics mc=MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, aggBinInstruction.input2);
		long matrixNumColumn=mc.numColumns;
		int blockNumColumn=mc.numColumnsPerBlock;
		numRepeats.put(aggBinInstruction.input1, (long)Math.ceil((double)matrixNumColumn/(double)blockNumColumn));
		
		mc=MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, aggBinInstruction.input1);
		long matrixNumRow=mc.numRows;
		int blockNumRow=mc.numRowsPerBlock;
		numRepeats.put(aggBinInstruction.input2, (long)Math.ceil((double)matrixNumRow/(double)blockNumRow));
		
		////////////////////////////////////////////////////////
	/*	for(int i=0; i<representativeMatrixes.size(); i++)
		{
			if(representativeMatrixes.get(i)==aggBinInstruction.input1)
			{
				MatrixCharacteristics mc=MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, aggBinInstruction.input2);
				long matrixNumColumn=mc.numColumns;
				int blockNumColumn=mc.numColumnsPerBlock;
				numRepeats.put(aggBinInstruction.input1, (long)Math.ceil((double)matrixNumColumn/(double)blockNumColumn));
			}
			
			if(representativeMatrixes.get(i)==aggBinInstruction.input2)
			{
				MatrixCharacteristics mc=MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, aggBinInstruction.input1);
				long matrixNumRow=mc.numRows;
				int blockNumRow=mc.numRowsPerBlock;
				numRepeats.put(aggBinInstruction.input2, (long)Math.ceil((double)matrixNumRow/(double)blockNumRow));
			}
		}*/
	}
}
