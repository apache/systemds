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
import java.util.HashSet;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.COV;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.instructions.cp.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.mr.CM_N_COVInstruction;
import com.ibm.bi.dml.runtime.matrix.data.CM_N_COVCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.COVOperator;


public class CMCOVMRReducer extends ReduceBase
implements Reducer<TaggedFirstSecondIndexes, MatrixValue, MatrixIndexes, MatrixValue>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private CM_N_COVInstruction[] cmNcovInstructions=null;
	private CM_N_COVCell cmNcovCell=new CM_N_COVCell(); 
	private COV covFn=COV.getCOMFnObject();
	private HashMap<Byte, CM> cmFn = new HashMap<Byte, CM>();
	private MatrixIndexes outIndex=new MatrixIndexes(1, 1);
	private MatrixCell outCell=new MatrixCell();
	private HashMap<Byte, ArrayList<Integer>> outputIndexesMapping=new HashMap<Byte, ArrayList<Integer>>();
	protected HashSet<Byte> covTags=new HashSet<Byte>();
	private CM_COV_Object zeroObj=null;
	
	//the dimension for all the representative matrices 
	//(they are all the same, since coming from the same files)
	protected HashMap<Byte, Long> rlens=null;
	protected HashMap<Byte, Long> clens=null;
	
	@Override
	public void reduce(TaggedFirstSecondIndexes index,
			Iterator<MatrixValue> values,
			OutputCollector<MatrixIndexes, MatrixValue> out, Reporter report)
			throws IOException {
		commonSetup(report);
		cmNcovCell.setCM_N_COVObject(0, 0, 0);
		ValueFunction fn=cmFn.get(index.getTag());
		if(covTags.contains(index.getTag()))
			fn=covFn;
		while(values.hasNext())
		{
			CM_N_COVCell cell=(CM_N_COVCell) values.next();
			try {
				fn.execute(cmNcovCell.getCM_N_COVObject(), cell.getCM_N_COVObject());
			} catch (DMLRuntimeException e) {
				throw new IOException(e);
			}
		}
		
		//add 0 values back in
	/*	long totaln=rlens.get(index.getTag())*clens.get(index.getTag());
		long zerosToAdd=totaln-(long)(cmNcovCell.getCM_N_COVObject().w);
		for(long i=0; i<zerosToAdd; i++)
		{
			try {
				fn.execute(cmNcovCell.getCM_N_COVObject(), zeroObj);
			} catch (DMLRuntimeException e) {
				throw new IOException(e);
			}
		}*/
		
		long totaln=rlens.get(index.getTag())*clens.get(index.getTag());
		long zerosToAdd=totaln-(long)(cmNcovCell.getCM_N_COVObject().w);
		if(zerosToAdd>0)
		{
			zeroObj.w=zerosToAdd;
			try {
				fn.execute(cmNcovCell.getCM_N_COVObject(), zeroObj);
			} catch (DMLRuntimeException e) {
				throw new IOException(e);
			}
		}
		
		for(CM_N_COVInstruction in: cmNcovInstructions)
		{
			if(in.input==index.getTag())
			{
				try {
					outCell.setValue(cmNcovCell.getCM_N_COVObject().getRequiredResult(in.getOperator()));
				} catch (DMLRuntimeException e) {
					throw new IOException(e);
				}
				
				ArrayList<Integer> outputIndexes = outputIndexesMapping.get(in.output);
				for(int i: outputIndexes)
				{
					collectOutput_N_Increase_Counter(outIndex, outCell, i, report);
				//	System.out.println("final output: "+outIndex+" -- "+outCell);
				}
			}
		}
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
		try {
			cmNcovInstructions=MRJobConfiguration.getCM_N_COVInstructions(job);
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
		rlens=new HashMap<Byte, Long>();
		clens=new HashMap<Byte, Long>();
		for(CM_N_COVInstruction ins: cmNcovInstructions)
		{
			if(ins.getOperator() instanceof COVOperator)
				covTags.add(ins.input);
			else //CMOperator
				cmFn.put(ins.input, CM.getCMFnObject(((CMOperator)ins.getOperator()).getAggOpType()));
			outputIndexesMapping.put(ins.output, getOutputIndexes(ins.output));
			rlens.put(ins.input, MRJobConfiguration.getNumRows(job, ins.input));
			clens.put(ins.input, MRJobConfiguration.getNumColumns(job, ins.input));
		}
		zeroObj=new CM_COV_Object();
		zeroObj.w=1;
	}
}
