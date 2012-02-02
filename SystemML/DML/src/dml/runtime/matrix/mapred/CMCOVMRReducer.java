package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Vector;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.functionobjects.CM;
import dml.runtime.functionobjects.COV;
import dml.runtime.functionobjects.ValueFunction;
import dml.runtime.instructions.CPInstructions.CM_COV_Object;
import dml.runtime.instructions.MRInstructions.CM_N_COVInstruction;
import dml.runtime.matrix.io.CM_N_COVCell;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import dml.runtime.matrix.operators.COVOperator;
import dml.utils.DMLRuntimeException;

public class CMCOVMRReducer extends ReduceBase
implements Reducer<TaggedFirstSecondIndexes, MatrixValue, MatrixIndexes, MatrixValue>{

	private CM_N_COVInstruction[] cmNcovInstructions=null;
	private CM_N_COVCell cmNcovCell=new CM_N_COVCell(); 
	private COV covFn=COV.getCOMFnObject();
	private CM cmFn=CM.getCMFnObject();
	private MatrixIndexes outIndex=new MatrixIndexes(1, 1);
	private MatrixCell outCell=new MatrixCell();
	private HashMap<Byte, Vector<Integer>> outputIndexesMapping=new HashMap<Byte, Vector<Integer>>();
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
		ValueFunction fn=cmFn;
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
				
				Vector<Integer> outputIndexes = outputIndexesMapping.get(in.output);
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
			outputIndexesMapping.put(ins.output, getOutputIndexes(ins.output));
			rlens.put(ins.input, MRJobConfiguration.getNumRows(job, ins.input));
			clens.put(ins.input, MRJobConfiguration.getNumColumns(job, ins.input));
		}
		zeroObj=new CM_COV_Object();
		zeroObj.w=1;
	}
}
