/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Vector;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.TaggedInt;
import com.ibm.bi.dml.runtime.matrix.data.WeightedCell;


public class GroupedAggMRMapper extends MapperBase
	implements Mapper<MatrixIndexes, MatrixValue, TaggedInt, WeightedCell>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	//block instructions that need to be performed in part by mapper
	protected Vector<Vector<GroupedAggregateInstruction>> groupAgg_instructions=new Vector<Vector<GroupedAggregateInstruction>>();
	private IntWritable outKeyValue=new IntWritable();
	private TaggedInt outKey=new TaggedInt(outKeyValue, (byte)0);
	private WeightedCell outValue=new WeightedCell();

	@Override
	public void map(MatrixIndexes key, MatrixValue value,
			        OutputCollector<TaggedInt, WeightedCell> out, Reporter reporter) 
	    throws IOException 
	{
		for(int i=0; i<representativeMatrixes.size(); i++)
			for(GroupedAggregateInstruction ins : groupAgg_instructions.get(i))
			{
				//set the tag once for the block
				outKey.setTag(ins.output);
				
				//get block and unroll into weighted cells
				//(it will be in dense format)
				MatrixBlock block = (MatrixBlock) value;
				
				int rlen = block.getNumRows();
				int clen = block.getNumColumns();
				if( clen == 2 ) //w/o weights
				{
					for( int r=0; r<rlen; r++ )
					{
						outKeyValue.set((int)block.quickGetValue(r, 1));
						outValue.setValue(block.quickGetValue(r, 0));
						outValue.setWeight(1);
						out.collect(outKey, outValue);		
					}
				}
				else //w/ weights
				{
					for( int r=0; r<rlen; r++ )
					{
						outKeyValue.set((int)block.quickGetValue(r, 1));
						outValue.setValue(block.quickGetValue(r, 0));
						outValue.setWeight(block.quickGetValue(r, 2));
						out.collect(outKey, outValue);		
					}
				}
			}
	}

	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException 
	{
		
	}
	
	@Override
	public void configure(JobConf job)
	{
		super.configure(job);
		
		try 
		{
			GroupedAggregateInstruction[] grpaggIns = MRJobConfiguration.getGroupedAggregateInstructions(job);
			if( grpaggIns == null )
				throw new RuntimeException("no GroupAggregate Instructions found!");
			
			Vector<GroupedAggregateInstruction> vec = new Vector<GroupedAggregateInstruction>();
			for(int i=0; i<representativeMatrixes.size(); i++)
			{
				byte input=representativeMatrixes.get(i);
				for(GroupedAggregateInstruction ins : grpaggIns)
					if(ins.input == input)
						vec.add(ins);
				groupAgg_instructions.add(vec);
			}
		} 
		catch (Exception e) 
		{
			throw new RuntimeException(e);
		} 
	}
}
