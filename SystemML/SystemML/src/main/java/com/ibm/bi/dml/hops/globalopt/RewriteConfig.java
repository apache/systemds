/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import com.ibm.bi.dml.hops.Hop.FileFormatTypes;
import com.ibm.bi.dml.hops.globalopt.InterestingProperties.Location;
import com.ibm.bi.dml.lops.LopProperties.ExecType;



/**
 * This RewriteConfig represents an instance configuration of a particular rewrite.
 * 
 */
public class RewriteConfig 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/*
	public enum RewriteConfigType {
		BLOCK_SIZE,
		FORMAT_CHANGE,
		EXEC_TYPE,
		DATA_PARTITIONING,
		VECTORIZATION,
		REPLICATION_FACTOR
	}
	*/
	
	private ExecType _rewriteSetExecType  = null;
	private int      _rewriteSetBlockSize = -1;
	private FileFormatTypes _rewriteFormat = null;
	
	public RewriteConfig( ExecType et, int bs, FileFormatTypes format )
	{
		_rewriteSetExecType = et;
		_rewriteSetBlockSize = bs;
		_rewriteFormat = format;
	}
	
	public RewriteConfig( RewriteConfig rc )
	{
		_rewriteSetExecType = rc._rewriteSetExecType;
		_rewriteSetBlockSize = rc._rewriteSetBlockSize;
		_rewriteFormat = rc._rewriteFormat;
	}
	
	public ExecType getExecType()
	{
		return _rewriteSetExecType;
	}
	
	public int getBlockSize()
	{
		return _rewriteSetBlockSize;
	}
	
	public FileFormatTypes getFormat()
	{
		return _rewriteFormat;
	}
	
	/**
	 * 
	 * @return
	 */
	public InterestingProperties deriveInterestingProperties()
	{
		int bs = _rewriteSetBlockSize; 
		Location loc = (_rewriteSetExecType==ExecType.CP) ? Location.MEM : Location.HDFS;
		
		return new InterestingProperties(bs, null, loc, null, -1, true);
	}

	
	@Override
	public String toString() 
	{
		return "RC["+_rewriteSetExecType+","+_rewriteSetBlockSize+"]";//+_type+"="+_value + "]";
	}

}
