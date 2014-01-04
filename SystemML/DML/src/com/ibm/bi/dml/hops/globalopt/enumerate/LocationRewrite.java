/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.lops.LopProperties.ExecType;

public class LocationRewrite extends Rewrite 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Integer execLocation = -1;
	
	@Override
	public void apply(OptimizedPlan plan) {
		Hop operator = plan.getOperator();
		ExecType eType = ExecType.MR;
		if(execLocation.equals(LocationParam.CP))
			eType = ExecType.CP;
		operator.setForcedExecType(eType);

	}

	public Integer getExecLocation() {
		return execLocation;
	}

	public void setExecLocation(Integer execLocation) {
		this.execLocation = execLocation;
	}

}
