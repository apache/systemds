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
import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.FileFormatTypes;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.globalopt.CrossBlockOp;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.OutputParameters.Format;

/**
 *
 */
public class FormatParam extends ConfigParam 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String NAME = "format";
	
	public static final Integer TEXT = 0;
	public static final Integer BINARY_BLOCK = 1;
	public static final Integer BINARY_CELL = 2;
	
	@Override
	public String getName() {
		return NAME;
	}
	
	public FormatParam(Integer... definedValues) {
		this.setDefinedValues(definedValues);
	}
	
	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.optimizer.enumeration.ConfigParam#applyToHop(com.ibm.bi.dml.hops.Hops)
	 */
	@Override
	public void applyToHop(Hop hop) {
		// TODO Auto-generated method stub
		//should only apply to FuncOps and here the statement block is needed
		//generate Reblocks for the rest???
	}

	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.optimizer.enumeration.ConfigParam#createInstance(java.lang.Long)
	 */
	@Override
	public ConfigParam createInstance(Integer value) {
		ConfigParam param = new FormatParam();
		param.setDefinedValues(this.getDefinedValues().toArray(new Integer[this.getDefinedValues().size()]));
		param.setValue(value);
		return param;
	}

	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.optimizer.enumeration.ConfigParam#createsInterestingProperties()
	 */
	@Override
	public Set<InterestingProperty> createsInterestingProperties() {
		Set<InterestingProperty> properties = new HashSet<InterestingProperty>();
		FormatProperty formatProperty = new FormatProperty();
		Integer value = new Integer(this.getValue()); 
		formatProperty.setValue(value);
		properties.add(formatProperty);
		return properties;
	}


	/**
	 * Binary block is valid for every operator. Binary Cell and Text can atm only 
	 * be configured for external Functions.
	 */
	@Override
	public boolean isValidForOperator(Hop operator) {
		return true;
	}
	
	public boolean isFormatValid(Hop operator) {
		if(this.value.equals(BINARY_BLOCK)) {
			return true;
		}
		
		if(operator instanceof CrossBlockOp) {
			return true;
		}
		
		if(operator instanceof FunctionOp) {
			if(!this.getValue().equals(BINARY_CELL)) {
				return true;
			}
		}
		
		if(operator instanceof DataOp) {
			return true;
		}
		return false;
	}

	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.optimizer.enumeration.ConfigParam#requiresRewrite(com.ibm.bi.dml.optimizer.enumeration.InterestingProperty)
	 */
	@Override
	public Rewrite requiresRewrite(InterestingProperty toCreate) {
		Rewrite rewriteToReturn = null;
		int requiredFormat = (int)toCreate.getValue();
		if(requiredFormat != this.getValue()) { 
			ReblockRewrite requiredRewrite = new ReblockRewrite();
			FormatParam requiredFormatParam = new FormatParam();
			requiredFormatParam.setValue(requiredFormat);
			requiredRewrite.setFormat(requiredFormatParam );
			//BINARY_BLOCK needs the block sizes here...
			rewriteToReturn = requiredRewrite;
		}
		return rewriteToReturn;
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
		
		String valString = "TEXT";
		if(this.getValue().equals(BINARY_BLOCK)) {
			valString = "BINARY_BLOCK";
		}
		if(this.getValue().equals(BINARY_CELL)) {
			valString = "BINARY_CELL";
		}
		return valString;
	}
	
	@Override
	public boolean equals(Object o) {
		if(o == this)
			return true;
		if(o instanceof FormatParam) {
			FormatParam oParam = (FormatParam)o;
			return oParam.getValue().equals(this.getValue());
		}
		
		if(o instanceof FileFormatTypes) {
			FileFormatTypes ffType = (FileFormatTypes)o;
			
			if(ffType.equals(FileFormatTypes.TEXT) && this.getValue().equals(FormatParam.TEXT)) {
				return true;
			}
			
			if(ffType.equals(FileFormatTypes.BINARY) && !this.getValue().equals(FormatParam.TEXT)) {
				return true;
			}
			
		}
		
		return false;
	}

	@Override
	public ConfigParam extractParamFromHop(Hop hop) {
		
		try {
			Lop lop = hop.constructLops();
			Format format = lop.getOutputParameters().getFormat();
			Integer extractedFormat = BINARY_BLOCK;
			if(format.equals(Format.TEXT)) {
				extractedFormat = TEXT;
			} else if(hop.get_cols_in_block() == -1L) {
				extractedFormat = BINARY_CELL;
			}
			
			ConfigParam extracted = this.createInstance(extractedFormat);
			return extracted;
		} catch (HopsException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (LopsException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		return null;
	}
}
