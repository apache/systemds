/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.HashSet;
import java.util.Set;

import com.ibm.bi.dml.hops.Hop;


public class BlockSizeParam extends ConfigParam 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String NAME = "blockSize";

	
	@Override
	public Set<InterestingProperty> createsInterestingProperties() {
		InterestingProperty property = new BlockSizeProperty();
		property.setValue(this.getValue());
		Set<InterestingProperty> retVal = new HashSet<InterestingProperty>();
		retVal.add(property);
		return retVal;
	}

	@Override
	public Rewrite requiresRewrite(InterestingProperty toCreate) {
		//...
		return null;
	}

	@Override
	public String getName() {
		return NAME;
	}
	
	public ConfigParam createInstance(Integer value) {
		ConfigParam param = new BlockSizeParam();
		param.setName(this.getName());
		param.setDefinedValues(this.getDefinedValues().toArray(new Integer[this.getDefinedValues().size()]));
		param.setValue(value);
		return param;
	}

	@Override
	public void applyToHop(Hop hop) {
		hop.set_cols_in_block(this.value);
		hop.set_rows_in_block(this.value);
	}

	@Override
	public boolean isValidForOperator(Hop operator) {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public String getValueString() {
		return String.valueOf(this.getValue());
	}

	@Override
	public ConfigParam extractParamFromHop(Hop hop) {
		//TODO: rectangular blocksize
		Integer extractedBlockSize = (int)hop.get_rows_in_block();
		ConfigParam extracted = this.createInstance(extractedBlockSize);
		return extracted;
	}
}
