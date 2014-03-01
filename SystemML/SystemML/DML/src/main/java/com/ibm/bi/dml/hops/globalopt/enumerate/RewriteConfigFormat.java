/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.FormatType;
import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.InterestingPropertyType;

/**
 * 
 */
public class RewriteConfigFormat extends RewriteConfig
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//valid instance configurations 
	private static int[] _defValues = new int[]{ FormatType.BINARY_BLOCK.ordinal(), 
		                                         FormatType.BINARY_CELL.ordinal(),
		                                         FormatType.TEXT_CELL.ordinal()};  
	
    public RewriteConfigFormat()
 	{
 		super( RewriteConfigType.FORMAT_CHANGE, -1 );
 	}
    
    public RewriteConfigFormat(int value)
 	{
 		super( RewriteConfigType.FORMAT_CHANGE, value );
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
		return new InterestingProperty(InterestingPropertyType.FORMAT, getValue());
	}
	
/*



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

*/

}
