/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.instructions.mr.CSVWriteInstruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.Pair;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;

public class CSVWriteMapper extends MapperBase implements Mapper<WritableComparable, Writable, TaggedFirstSecondIndexes, MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	HashMap<Byte, ArrayList<Byte>> inputOutputMap=new HashMap<Byte, ArrayList<Byte>>();
	TaggedFirstSecondIndexes outIndexes=new TaggedFirstSecondIndexes();
	
	@Override
	public void map(WritableComparable rawKey, Writable rawValue,
			OutputCollector<TaggedFirstSecondIndexes, MatrixBlock> out,
			Reporter reporter) throws IOException {
		long start=System.currentTimeMillis();
		
	//	System.out.println("read in Mapper: "+rawKey+": "+rawValue);
		
		//for each represenattive matrix, read the record and apply instructions
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			//convert the record into the right format for the representatice matrix
			inputConverter.setBlockSize(brlens[i], bclens[i]);
			inputConverter.convert(rawKey, rawValue);
			
			byte thisMatrix=representativeMatrixes.get(i);
			
			//apply unary instructions on the converted indexes and values
			while(inputConverter.hasNext())
			{
				Pair<MatrixIndexes, MatrixBlock> pair=inputConverter.next();
				MatrixIndexes indexes=pair.getKey();
				
				MatrixBlock value=pair.getValue();
				
				outIndexes.setIndexes(indexes.getRowIndex(), indexes.getColumnIndex());
				ArrayList<Byte> outputs=inputOutputMap.get(thisMatrix);
				for(byte output: outputs)
				{
					outIndexes.setTag(output);
					out.collect(outIndexes, value);
					//LOG.info("Mapper output: "+outIndexes+", "+value+", tag: "+output);
				}
			}
		}
		reporter.incrCounter(Counters.MAP_TIME, System.currentTimeMillis()-start);
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
		try {
			CSVWriteInstruction[] ins = MRJobConfiguration.getCSVWriteInstructions(job);
			for(CSVWriteInstruction in: ins)
			{
				ArrayList<Byte> outputs=inputOutputMap.get(in.input);
				if(outputs==null)
				{
					outputs=new ArrayList<Byte>();
					inputOutputMap.put(in.input, outputs);
				}
				outputs.add(in.output);
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
		// do nothing
	}
	
}
