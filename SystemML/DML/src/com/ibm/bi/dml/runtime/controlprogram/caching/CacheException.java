/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.caching;

import com.ibm.bi.dml.runtime.DMLRuntimeException;

public class CacheException extends DMLRuntimeException
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = 1L;

	public CacheException ()
	{
		super ("Cache Exception");
	}

	public CacheException (String message)
	{
		super (message);
	}

	public CacheException (Exception cause)
	{
		super (cause);
	}

	public CacheException (String message, Exception cause)
	{
		super (message, cause);
	}

}
