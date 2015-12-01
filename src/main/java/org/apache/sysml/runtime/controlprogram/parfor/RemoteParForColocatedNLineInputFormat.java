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

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;

import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.lib.NLineInputFormat;

import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;

/**
 * Specific extension of NLineInputFormat in order to ensure data colocation
 * for partitioned matrices although those matrices are not directly passed to the
 * MR job as an input.
 * 
 */
public class RemoteParForColocatedNLineInputFormat extends NLineInputFormat
{
	
	@Override
	public InputSplit[] getSplits(JobConf job, int numSplits) throws IOException 
	{	
		InputSplit[] tmp = super.getSplits(job, numSplits);
		
		//get partitioning information
		PDataPartitionFormat dpf = MRJobConfiguration.getPartitioningFormat(job);
		int blen = -1;
		switch( dpf ) {
			case ROW_WISE:          blen = 1; break;
			case ROW_BLOCK_WISE:    blen = MRJobConfiguration.getPartitioningBlockNumRows(job); break;
			case COLUMN_WISE:       blen = 1; break;
			case COLUMN_BLOCK_WISE: blen = MRJobConfiguration.getPartitioningBlockNumCols(job); break;
			default: 
				//do nothing
		}		
		String fname = MRJobConfiguration.getPartitioningFilename(job);

		//create wrapper splits 
		InputSplit[] ret = new InputSplit[ tmp.length ];
		for( int i=0; i<tmp.length; i++ )
		{
			//check for robustness of subsequent cast
			if( tmp[i] instanceof FileSplit ) 
				ret[i] = new RemoteParForColocatedFileSplit( (FileSplit) tmp[i], fname, blen );
			else
				ret[i] = tmp[i];
		}
		return ret;
	}
}
