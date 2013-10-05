package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.utils.DMLException;

@SuppressWarnings("serial")
public class LopsException extends DMLException 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public LopsException(String message)
	{
		super(message);
	}

	public LopsException(Exception e) {
		super(e);
	}
	
	public LopsException(String message, Throwable cause) {
	    super(message, cause);
	}

}
