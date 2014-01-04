/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.HashSet;
import java.util.Set;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.MemoTable;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.ReblockOp;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;


public class LocationParam extends ConfigParam 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String NAME = "dataLocation";
	public static final int MR = 0;
	public static final int CP = 1;
	private static final int UNKNOWN = -1;
	
	@Override
	public Set<InterestingProperty> createsInterestingProperties() {
		Set<InterestingProperty> retVal = new HashSet<InterestingProperty>();
		InterestingProperty locationProp = new DataLocationProperty();
		locationProp.setValue(new Integer(this.getValue()));
		retVal.add(locationProp);
		return retVal;
	}

	@Override
	public Rewrite requiresRewrite(InterestingProperty toCreate) {
		LocationRewrite requiredRewrite = new LocationRewrite();
		requiredRewrite.setExecLocation(this.value);
		return requiredRewrite;
	}

	@Override
	public String getName() {
		return NAME;
	}
	
	public ConfigParam createInstance(Integer value) {
		ConfigParam param = new LocationParam();
		param.setName(this.getName());
		param.setDefinedValues(this.getDefinedValues().toArray(new Integer[this.getDefinedValues().size()]));
		param.setValue(value);
		return param;
	}


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

	/**
	 * @return
	 */
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
	public ConfigParam extractParamFromHop(Hop hop) {
		ExecType forcedExecType = hop.getForcedExecType();
		Integer extractedExecType = CP;
		if(forcedExecType == null) {
			extractedExecType = UNKNOWN;
		}else if(forcedExecType.equals(ExecType.MR)) {
			extractedExecType = MR;
		}
		ConfigParam extracted = this.createInstance(extractedExecType);
		return extracted;
	}

}
