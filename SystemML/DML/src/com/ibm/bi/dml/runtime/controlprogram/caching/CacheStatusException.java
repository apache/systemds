package com.ibm.bi.dml.runtime.controlprogram.caching;


public class CacheStatusException extends CacheException
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;

	public CacheStatusException (String message)
	{
		super (message);
	}

	public CacheStatusException (Exception cause)
	{
		super (cause);
	}

	public CacheStatusException (String message, Exception cause)
	{
		super (message, cause);
	}

}
