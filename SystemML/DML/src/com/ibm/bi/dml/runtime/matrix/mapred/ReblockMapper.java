package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.TaggedPartialBlock;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class ReblockMapper extends MapperBase 
implements Mapper<Writable, Writable, Writable, Writable>
{
	private MatrixIndexes indexBuffer=new MatrixIndexes();
	private TaggedPartialBlock partialBuffer=new TaggedPartialBlock();
	private boolean firsttime=true;
	private OutputCollector<Writable, Writable> cachedCollector;
	private boolean outputDummyRecord=false;
	private HashMap<Byte, MatrixCharacteristics> dimensions=new HashMap<Byte, MatrixCharacteristics>();
	@Override
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
		if(firsttime)
		{
			cachedCollector=out;
			firsttime=false;
		}
		commonMap(rawKey, rawValue, out, reporter);
	}
	
	@Override
	public void configure(JobConf job) {
		MRJobConfiguration.setMatrixValueClass(job, false);
		super.configure(job);
		outputDummyRecord=MapReduceTool.getUniqueKeyPerTask(job, true).equals("0");
		if(outputDummyRecord)
		{
			//System.out.println("outputDummyRecord is true! in "+MapReduceTool.getUniqueKeyPerTask(job, true));
			//parse the reblock instructions 
			ReblockInstruction[] reblockInstructions;
			try {
				reblockInstructions = MRJobConfiguration.getReblockInstructions(job);
			} catch (DMLUnsupportedOperationException e) {
				throw new RuntimeException(e);
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
			for(ReblockInstruction ins: reblockInstructions)
				dimensions.put(ins.output, MRJobConfiguration.getMatrixCharactristicsForReblock(job, ins.output));
		}
	}

	public void close() throws IOException
	{
		super.close();
		if(!outputDummyRecord || cachedCollector==null)
			return;
		partialBuffer.getBaseObject().set(-1, -1, 0);
		long r, c;
		long rlen, clen;
		int brlen, bclen;
		for(Entry<Byte, MatrixCharacteristics> e: dimensions.entrySet())
		{
			partialBuffer.setTag(e.getKey());
			rlen=e.getValue().numRows;
			clen=e.getValue().numColumns;
			brlen=e.getValue().numRowsPerBlock;
			bclen=e.getValue().numColumnsPerBlock;
			r=1;
			for(long i=0; i<rlen; i+=brlen)
			{
				c=1;
				for(long j=0; j<clen; j+=bclen)
				{
					indexBuffer.setIndexes(r, c);
					cachedCollector.collect(indexBuffer, partialBuffer);
				//	System.out.println("in mapper: "+indexBuffer+": "+partialBuffer);
					c++;
				}
				r++;
			}
		}
	}
	
	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
		//apply all instructions
		processMapperInstructionsForMatrix(index);
		
		//apply reblock instructions and output
		processReblockInMapperAndOutput(index, indexBuffer, partialBuffer, out);
	}
}
