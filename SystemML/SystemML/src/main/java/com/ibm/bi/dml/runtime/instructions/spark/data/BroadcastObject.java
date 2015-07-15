/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.data;

import org.apache.spark.broadcast.Broadcast;

public class BroadcastObject extends LineageObject
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private Broadcast<PartitionedMatrixBlock> _bcHandle = null;
	
	public BroadcastObject( Broadcast<PartitionedMatrixBlock> bvar, String varName )
	{
		_bcHandle = bvar;
		_varName = varName;
	}
	
	/**
	 * 
	 * @return
	 */
	public Broadcast<PartitionedMatrixBlock> getBroadcast()
	{
		return _bcHandle;
	}
}
