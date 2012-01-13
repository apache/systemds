package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.functionobjects.CM;
import dml.runtime.instructions.CPInstructions.CM_COV_Object;
import dml.runtime.instructions.CPInstructions.KahanObject;
import dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.TaggedInt;
import dml.runtime.matrix.io.WeightedCell;
import dml.runtime.matrix.operators.AggregateOperator;
import dml.runtime.matrix.operators.CMOperator;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;

public class GroupedAggMRReducer extends ReduceBase
implements Reducer<TaggedInt, WeightedCell, MatrixIndexes, MatrixCell >{

	private MatrixIndexes outIndex=new MatrixIndexes(1, 1);
	private MatrixCell outCell=new MatrixCell();
	private HashMap<Byte, GroupedAggregateInstruction> grpaggInsructions=new HashMap<Byte, GroupedAggregateInstruction>();
	private CM_COV_Object cmObj=new CM_COV_Object(); 
	private CM cmFn=CM.getCMFnObject();
	private HashMap<Byte, Vector<Integer>> outputIndexesMapping=new HashMap<Byte, Vector<Integer>>();
	@Override
	public void reduce(TaggedInt key,
			Iterator<WeightedCell> values,
			OutputCollector<MatrixIndexes, MatrixCell> out, Reporter report)
			throws IOException {
		commonSetup(report);
		GroupedAggregateInstruction ins=grpaggInsructions.get(key.getTag());
		Operator op=ins.getOperator();
		if(op instanceof CMOperator)
		{
			cmObj.reset();
			while(values.hasNext())
			{
				WeightedCell value=values.next();
				try {
					cmFn.execute(cmObj, value.getValue(), value.getWeight());
				} catch (DMLRuntimeException e) {
					throw new IOException(e);
				}
			}
			try {
				outCell.setValue(cmObj.getRequiredResult(op));
			} catch (DMLRuntimeException e) {
				throw new IOException(e);
			}
			
		}else if(op instanceof AggregateOperator)
		{
			AggregateOperator aggop=(AggregateOperator) op;
				
			if(aggop.correctionExists)
			{
				KahanObject buffer=new KahanObject(aggop.initialValue, 0);
				while(values.hasNext())
				{
					WeightedCell value=values.next();
					try {
						aggop.increOp.fn.execute(buffer, value.getValue()*value.getWeight());
					} catch (DMLRuntimeException e) {
						throw new IOException(e);
					}
				}
				outCell.setValue(buffer._sum);
			}
			else
			{
				double v=aggop.initialValue;
				while(values.hasNext())
				{
					WeightedCell value=values.next();
					try {
						v=aggop.increOp.fn.execute(v, value.getValue()*value.getWeight());
					} catch (DMLRuntimeException e) {
						throw new IOException(e);
					}
				}
				outCell.setValue(v);
			}
			
		}else
			throw new IOException("cannot execute instruciton "+ins);
		
		outIndex.setIndexes((long)key.getBaseObject().get(), 1);
		cachedValues.reset();
		cachedValues.set(key.getTag(), outIndex, outCell);
		//System.out.println("after cm: "+outIndex+" -- "+outCell);
		processReducerInstructions();
		//output the final result matrices
		outputResultsFromCachedValues(report);
		
	/*	Vector<Integer> outputIndexes = outputIndexesMapping.get(key.getTag());
		for(int i: outputIndexes)
		{
			collectOutput_N_Increase_Counter(outIndex, outCell, i, report);
			//System.out.println("final output: "+outIndex+" -- "+outCell);
		}*/
		
	}

	public void configure(JobConf job)
	{
		super.configure(job);
		//valueClass=MatrixCell.class;
		try {
			GroupedAggregateInstruction[] grpaggIns=MRJobConfiguration.getGroupedAggregateInstructions(job);
			if(grpaggIns==null)
				throw new RuntimeException("no GroupAggregate Instructions found!");
			for(GroupedAggregateInstruction ins: grpaggIns)
			{
				grpaggInsructions.put(ins.output, ins);
				outputIndexesMapping.put(ins.output, getOutputIndexes(ins.output));
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
	}
}
