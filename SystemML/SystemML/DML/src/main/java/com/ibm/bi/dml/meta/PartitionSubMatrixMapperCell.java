/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;

import java.io.IOException;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.data.Converter;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.Pair;
import com.ibm.bi.dml.runtime.matrix.data.PartialBlock;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class PartitionSubMatrixMapperCell extends MapReduceBase 
implements Mapper<Writable, Writable, TaggedFirstSecondIndexes, PartialBlock> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	

	private Converter inputConverter=null;
	private PartialBlock partialBuffer=new PartialBlock();
	PartitionParams pp = new PartitionParams() ;
	int rSize, cSize ;
	int brlen, bclen ;
	long numRows, numColumns ;
	MatrixIndexes[] blockIndexes;
	
	// Object creation happens only once
	Pair<Long, Long> pair = new Pair<Long, Long>() ;
	TaggedFirstSecondIndexes tfsi = new TaggedFirstSecondIndexes() ;
	
	int getOutputMatrixId(MatrixIndexes indexes, MatrixIndexes blockIndexA) {
		long i = indexes.getRowIndex();
		long j = indexes.getColumnIndex() ;
		long blockI = i/rSize, blockJ = j/cSize ;
		
		if(blockI == (blockIndexA.getRowIndex()-1) && blockJ == (blockIndexA.getColumnIndex()-1))
			return 0 ;
		else if(blockI == (blockIndexA.getRowIndex()-1))
			return 1 ;
		else if(blockJ == (blockIndexA.getColumnIndex()-1))
			return 2 ;
		else
			return 3 ;
	}
	
	Pair<Long, Long> getMatrixIndexes(MatrixIndexes indexes, MatrixIndexes blockIndexA) {
		// Return the new matrix index corresponding to indexes
		long i = indexes.getRowIndex();
		long j = indexes.getColumnIndex() ;
		int opMatrixId = getOutputMatrixId(indexes, blockIndexA) ;
		
		long blockI = i/rSize, blockJ = j/cSize ;
		long remI = i%rSize, remJ=j%cSize ;
		
		if(opMatrixId == 0) {
			pair.set(remI, remJ) ;
		}
		else if(opMatrixId == 1) {
			// How many B blocks are to my left ?
			long yIndex = (blockJ < (blockIndexA.getColumnIndex()-1)) ? j : j-cSize ;
			pair.set(remI, yIndex) ;
		}
		else if(opMatrixId == 2) {
			long xIndex = (blockI < (blockIndexA.getRowIndex()-1)) ? i : i-rSize ;
			pair.set(xIndex, remJ) ;
		}
		else {
			long yIndex = (blockJ < (blockIndexA.getColumnIndex()-1)) ? j : j-cSize ;
			long xIndex = (blockI < (blockIndexA.getRowIndex()-1)) ? i : i-rSize ;
			pair.set(xIndex, yIndex) ;
		}
		return pair ;
	}
	
	@Override
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<TaggedFirstSecondIndexes, PartialBlock> out, Reporter reporter)
			throws IOException {
		inputConverter.setBlockSize(0, 0); // 2 x 2 matrix blocks..
		inputConverter.convert(rawKey, rawValue);
		while(inputConverter.hasNext()) {
			Pair<MatrixIndexes, MatrixValue> pair=inputConverter.next();
			
			MatrixIndexes indexes=pair.getKey();
			indexes.setIndexes(indexes.getRowIndex()-1, indexes.getColumnIndex()-1);
			MatrixCell value=(MatrixCell) pair.getValue();
		
			
			int maxIters = (pp.toReplicate) ? blockIndexes.length : 1; 
						
			for(int blockIndexCounter = 0 ; blockIndexCounter < maxIters; blockIndexCounter++)
			{
				int opMatrixId = getOutputMatrixId(indexes, blockIndexes[blockIndexCounter]) + 4*blockIndexCounter;
					
				Pair<Long,Long> opIndexes = getMatrixIndexes(indexes, blockIndexes[blockIndexCounter]) ;	
				long bi=(opIndexes.getKey()/brlen)+1;
				long bj=(opIndexes.getValue()/bclen)+1;
				
				int i=(int) (opIndexes.getKey()%brlen);
				int j=(int) (opIndexes.getValue()%bclen);
				partialBuffer.set(i, j, value.getValue());

				System.out.println("PartitionSubMatrixMapperCell -- partialBuffer setting: " + i + "," + j + value.getValue());
				
				tfsi.setTag((byte) opMatrixId) ;
				tfsi.setIndexes(bi, bj) ;
		
				out.collect(tfsi, partialBuffer) ;
			} 	
		}
	}
	
	public void initializeBlockIndexes() {
		int numRowGroups = (int) Math.ceil((double)numRows/(double)rSize) ;
		int numColGroups = (int) Math.ceil((double)numColumns/(double)cSize) ;
		
		blockIndexes = new MatrixIndexes[numRowGroups * numColGroups] ;
		for(int i = 0 ; i < numRowGroups; i++)
			for(int j = 0; j < numColGroups; j++)
				blockIndexes[i * numColGroups + j] = new MatrixIndexes(i+1,j+1) ;
	}
	
	@Override
	public void configure(JobConf job) {
		//get input converter information
		inputConverter=MRJobConfiguration.getInputConverter(job, (byte)0);
		brlen=MRJobConfiguration.getNumRowsPerBlock(job, (byte)0);
		bclen=MRJobConfiguration.getNumColumnsPerBlock(job, (byte)0);		
		numRows = MRJobConfiguration.getNumRows(job, (byte) 0) ;
		numColumns = MRJobConfiguration.getNumColumns(job, (byte) 0) ;
		pp = MRJobConfiguration.getPartitionParams(job) ;
		rSize = (int) Math.ceil((double)numRows/(double)pp.numRowGroups) ;
		cSize = (int) Math.ceil((double)numColumns/(double)pp.numColGroups) ;
		//System.out.println("rsize = " + rSize + " and cSize = " + cSize) ;
		
		initializeBlockIndexes() ;
	}
}