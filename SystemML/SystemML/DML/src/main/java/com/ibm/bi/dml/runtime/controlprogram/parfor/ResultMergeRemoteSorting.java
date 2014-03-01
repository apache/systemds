/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

public class ResultMergeRemoteSorting extends WritableComparator
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	protected ResultMergeRemoteSorting()
	{
		super(ResultMergeTaggedMatrixIndexes.class, true);
	}
	
	@SuppressWarnings("rawtypes")
	@Override
    public int compare(WritableComparable k1, WritableComparable k2) 
	{
		ResultMergeTaggedMatrixIndexes key1 = (ResultMergeTaggedMatrixIndexes)k1;
		ResultMergeTaggedMatrixIndexes key2 = (ResultMergeTaggedMatrixIndexes)k2;

		int ret = key1.getIndexes().compareTo(key2.getIndexes());
		if( ret == 0 ) //same indexes, secondary sort
		{
			ret = ((key1.getTag() == key2.getTag()) ? 0 : 
				   (key1.getTag() < key2.getTag())? -1 : 1);
		}	
			
		return ret; 
		
    }
}
