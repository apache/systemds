/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.data;

import org.apache.spark.api.java.JavaPairRDD;

public class RDDObject extends LineageObject
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private JavaPairRDD<?,?> _rddHandle = null;
	
	//meta data on origin of given rdd handle
	private boolean _checkpointed = false; //created via checkpoint instruction
	private boolean _hdfsfile = false;     //created from hdfs file
	
	public RDDObject( JavaPairRDD<?,?> rddvar, String varName)
	{
		_rddHandle = rddvar;
		_varName = varName;
	}
	
	/**
	 * 
	 * @return
	 */
	public JavaPairRDD<?,?> getRDD()
	{
		return _rddHandle;
	}
	
	public void setCheckpointRDD( boolean flag )
	{
		_checkpointed = flag;
	}
	
	public boolean isCheckpointRDD() 
	{
		return _checkpointed;
	}
	
	public void setHDFSFile( boolean flag )
	{
		_hdfsfile = flag;
	}
	
	public boolean isHDFSFile()
	{
		return _hdfsfile;
	}
	

	/**
	 * 
	 * @return
	 */
	public boolean allowsShortCircuitRead()
	{
		boolean ret = false;
		
		if( isCheckpointRDD() && getLineageChilds().size() == 1 ) {
			LineageObject lo = getLineageChilds().get(0);
			ret = ( lo instanceof RDDObject && ((RDDObject)lo).isHDFSFile() );
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean allowsShortCircuitCollect()
	{
		return ( isCheckpointRDD() && getLineageChilds().size() == 1
			     && getLineageChilds().get(0) instanceof RDDObject );
	}
}
