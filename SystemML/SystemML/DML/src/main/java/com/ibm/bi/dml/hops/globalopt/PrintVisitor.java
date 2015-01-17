/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.LiteralOp;

public class PrintVisitor implements HopsVisitor 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private String varName;
	
	public PrintVisitor(String string) {
		this.varName = string;
	}

	//TODO: replace string concatenation by StringBuilder oder StringBuffer
	@Override
	public Flag preVisit(Hop hops) {
		
		if(!(hops instanceof LiteralOp)) 
		{	
			if(hops.getMetaData() != null)
			{
				System.out.println(hops.getMetaData().toString());
			}else {
				System.out.println(hops.getName() + ", " + hops.getClass().getSimpleName());
			}
			if(hops instanceof DataOp)
			{
				DataOp dataOp = (DataOp)hops;
				System.out.println("; format: " + dataOp.getFormatType() + "; type: " + dataOp.get_dataop());
				
			}else 
			{ 
				System.out.println();
			}
		}
		return Flag.GO_ON;
	}
	
	@Override
	public boolean matchesPattern(Hop hops) {
		if(this.varName != null) {
			
		}
		return true;
	}

	@Override
	public Flag postVisit(Hop hops) {
		return Flag.GO_ON;
	}

	public Flag visit(Hop hops) {
		return Flag.GO_ON;
	}

	@Override
	public boolean traverseBackwards() {
		return false;
	}
}
