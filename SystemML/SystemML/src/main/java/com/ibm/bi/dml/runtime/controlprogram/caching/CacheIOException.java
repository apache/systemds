/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.caching;


public class CacheIOException extends CacheException
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;

	public CacheIOException ()
	{
		super ();
	}

	public CacheIOException (String message)
	{
		super (message);
	}

	public CacheIOException (Exception cause)
	{
		super (cause);
	}

	public CacheIOException (String message, Exception cause)
	{
		super (message, cause);
	}

}
