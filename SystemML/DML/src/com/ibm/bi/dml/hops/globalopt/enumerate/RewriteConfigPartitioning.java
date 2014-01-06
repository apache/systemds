/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.InterestingPropertyType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;


/**
 * 
 */
public class RewriteConfigPartitioning extends RewriteConfig
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//valid instance configurations
	private static int[] _defValues = new int[]{ PDataPartitionFormat.NONE.ordinal(),
		                                         PDataPartitionFormat.ROW_WISE.ordinal(), 
		                                         PDataPartitionFormat.COLUMN_WISE.ordinal() };  
	
	public RewriteConfigPartitioning()
	{
		super( RewriteConfigType.DATA_PARTITIONING, -1 );
	}
	
	@Override
	public int[] getDefinedValues()
	{
		return _defValues;
	}

	@Override
	public InterestingProperty getInterestingProperty()
	{
		//direct mapping from rewrite config to interesting property
		return new InterestingProperty(InterestingPropertyType.PARTITION_FORMAT, getValue());
	}
	
/*


	@Override
	public void applyToHop(Hop hop) {
		ExecType eType = ExecType.MR;
		if(this.getValue().equals(CP))
		{
		 eType = ExecType.CP;
		}
		if(this.getValue().equals(UNKNOWN))
		{	
			eType = null;
		}
		
		hop.setForcedExecType(eType);
	}
	
	@Override
	public String toString() {
		StringBuilder buffer = new StringBuilder();
		buffer.append("[");
		buffer.append(this.getName());
		buffer.append(", ");
		String valString = getValueString();
		buffer.append(valString);
		buffer.append(", def: [");
		for(Integer dv : this.getDefinedValues()) 
		{
			buffer.append(dv);
			buffer.append(", ");
		}
		buffer.append("]]");
		
		return buffer.toString();
	}

	@Override
	public String getValueString() {
		if(this.getValue() == null) { 
			return null;
		}
		String valString = "MR";
		if(this.getValue().equals(CP)) {
			valString = "CP";
		}
		return valString;
	}

	@Override
	public boolean isValidForOperator(Hop operator) {
		
		if(getValue().equals(CP)) {
			if(operator instanceof ReblockOp ) {
				return false;
			}
		
			if(operator instanceof DataOp) { 
					DataOp data = (DataOp)operator;
					if((data.get_dataop() == DataOpTypes.PERSISTENTREAD || data.get_dataop() == DataOpTypes.PERSISTENTWRITE)
							&& this.getValue().equals(LocationParam.CP) 
							&& !data.get_dataType().equals(DataType.SCALAR)) {
						System.out.println("data op in CP: " + data + ", " + data.get_dataop());
						return false;
					}
					
					//read matrices always from HDFS
					if((data.get_dataop() == DataOpTypes.TRANSIENTREAD || data.get_dataop() == DataOpTypes.TRANSIENTWRITE) 
							&& data.get_dataType() == DataType.MATRIX 
							&& this.getValue().equals(LocationParam.CP)) {
						return false;
					}
					
			}
			
			if(operator instanceof LiteralOp && this.getValue() == LocationParam.MR) {
				return false;
			}
			
			operator.refreshMemEstimates(new MemoTable());
			double memEstimate = operator.getMemEstimate();
			double budget = OptimizerUtils.getMemBudget(true);
			if(memEstimate > budget) 
				return false;
		}
		
		return true;
	}

	@Override
	public RewriteConfig extractParamFromHop(Hop hop) {
		ExecType forcedExecType = hop.getForcedExecType();
		Integer extractedExecType = CP;
		if(forcedExecType == null) {
			extractedExecType = UNKNOWN;
		}else if(forcedExecType.equals(ExecType.MR)) {
			extractedExecType = MR;
		}
		RewriteConfig extracted = this.createInstance(extractedExecType);
		return extracted;
	}
*/

}
