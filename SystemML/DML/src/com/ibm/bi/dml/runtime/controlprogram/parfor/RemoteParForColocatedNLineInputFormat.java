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
