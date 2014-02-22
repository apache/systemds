/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;

import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;

public class ResultMergeRemotePartitioning implements Partitioner<ResultMergeTaggedMatrixIndexes, TaggedMatrixBlock> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	private long _numColBlocks = -1;
	
	
    @Override
    public int getPartition(ResultMergeTaggedMatrixIndexes key, TaggedMatrixBlock val, int numPartitions) 
    {
    	//MB: Result merge might deal with lots of data but only few
    	//different indexes (many worker result blocks for one final
    	//result block). Hence, balanced partitioning it even more important
    	//and unfortunately, our default hash function results in significant
    	//load imbalance for those cases. Based on the known result dimensions
    	//we can create a better partitioning scheme. However, it still makes
    	//the assumption that there is no sparsity skew between blocks.
    	
    	MatrixIndexes ix = key.getIndexes();
    	int blockid = (int) (ix.getRowIndex() * _numColBlocks + ix.getColumnIndex());
    	int partition = blockid % numPartitions;
    	
        //int hash = key.getIndexes().hashCode();
        //int partition = hash % numPartitions;
        
    	return partition;
    }

	@Override
	public void configure(JobConf job) 
	{
		long[] tmp = MRJobConfiguration.getResultMergeMatrixCharacteristics( job );
		long clen = tmp[1]; 
		int bclen = (int) tmp[3];
		_numColBlocks = clen/bclen + ((clen%bclen!=0)? 1 : 0);
	}
}
