/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import com.ibm.bi.dml.runtime.util.UtilFunctions;

public abstract class ConstIdentifier extends Identifier 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ConstIdentifier(){
		super();
	}
	
	public long getLongValue() throws LanguageException {
		if ( this instanceof IntIdentifier )
			return ((IntIdentifier)this).getValue();
		else if ( this instanceof DoubleIdentifier ) 
			return UtilFunctions.toLong(((DoubleIdentifier) this).getValue());
		else
			throw new LanguageException("Invalid variable type");
	}
 
}
