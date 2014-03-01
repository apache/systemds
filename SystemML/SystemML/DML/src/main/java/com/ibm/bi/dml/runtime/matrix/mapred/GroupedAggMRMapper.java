/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
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
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.TaggedInt;
import com.ibm.bi.dml.runtime.matrix.io.WeightedCell;
import com.ibm.bi.dml.runtime.matrix.io.WeightedPair;


public class GroupedAggMRMapper extends MapperBase
implements Mapper<MatrixIndexes, WeightedPair, TaggedInt, WeightedCell>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	//block instructions that need to be performed in part by mapper
	protected Vector<Vector<GroupedAggregateInstruction>> groupAgg_instructions=new Vector<Vector<GroupedAggregateInstruction>>();
	private IntWritable outKeyValue=new IntWritable();
	private TaggedInt outKey=new TaggedInt(outKeyValue, (byte)0);
	private WeightedCell outValue=new WeightedCell();
	
	public void configure(JobConf job)
	{
		super.configure(job);
		try {
			GroupedAggregateInstruction[] grpaggIns=MRJobConfiguration.getGroupedAggregateInstructions(job);
			if(grpaggIns==null)
				throw new RuntimeException("no GroupAggregate Instructions found!");
			Vector<GroupedAggregateInstruction> vec=new Vector<GroupedAggregateInstruction>();
			for(int i=0; i<representativeMatrixes.size(); i++)
			{
				byte input=representativeMatrixes.get(i);
				for(GroupedAggregateInstruction ins: grpaggIns)
				{
					if(ins.input==input)
						vec.add(ins);
				}
				groupAgg_instructions.add(vec);
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
	}

	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
		
	}

	@Override
	public void map(MatrixIndexes index, WeightedPair wpair,
			OutputCollector<TaggedInt, WeightedCell> out,
			Reporter reporter) throws IOException {
		for(int i=0; i<representativeMatrixes.size(); i++)
			for(GroupedAggregateInstruction ins: groupAgg_instructions.get(i))
			{
				outKeyValue.set((int)wpair.getOtherValue());
				outKey.setTag(ins.output);
				outValue.setValue(wpair.getValue());
				outValue.setWeight(wpair.getWeight());
				out.collect(outKey, outValue);
			//	System.out.println("map output: "+outKey+" -- "+outValue);
			}
	}
}
