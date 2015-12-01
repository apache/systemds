/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.controlprogram.parfor;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;

import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.TaggedMatrixBlock;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;

public class ResultMergeRemotePartitioning implements Partitioner<ResultMergeTaggedMatrixIndexes, TaggedMatrixBlock> 
{
	
	
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
