/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;

import java.io.IOException;

import org.apache.commons.math3.random.Well1024a;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.Pair;


public class SubMatrixBlockMapperMethod extends BlockMapperMethod 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	long rSizeBlock, cSizeBlock ;
	MatrixIndexes longPair = new MatrixIndexes() ;
	MatrixIndexes[] blockIndexes ;
	long numRows, numColumns ;

	public SubMatrixBlockMapperMethod(PartitionParams pp, MultipleOutputs multipleOutputs,
			long numRows, long numColumns, int brlen, int bclen) {
		super(pp, multipleOutputs);
		this.numRows = numRows ;
		this.numColumns = numColumns ;
		
		long rSize = (long) Math.ceil((double)numRows/(double)pp.numRowGroups) ;
		long cSize = (long) Math.ceil((double)numColumns/(double)pp.numColGroups) ;
		long x = rSize % brlen ;
		if(x != 0) rSize = rSize + x ;
		long y = cSize % bclen ;
		if(y != 0) cSize = cSize + y ;

		//System.out.println("rsize = " + rSize + " and cSize = " + cSize) ;
		// we need to ensure rSize is chosen such that it divides brlen.
		// choose rSize and cSize appropriately here.
		rSizeBlock = rSize / brlen ;
		cSizeBlock = cSize / bclen ;

		//initializeBlockIndexes() ;
		int numRowGroups = (int) Math.ceil((double)numRows/(double)rSize) ;
		int numColGroups = (int) Math.ceil((double)numColumns/(double)cSize) ;
		blockIndexes = new MatrixIndexes[numRowGroups * numColGroups] ;
		for(int i = 0 ; i < numRowGroups; i++)
			for(int j = 0; j < numColGroups; j++)
				blockIndexes[i * numColGroups + j] = new MatrixIndexes(i+1,j+1) ;
	}
	
	byte getOutputMatrixId(MatrixIndexes indexes, MatrixIndexes blockIndexA) {
		long i = indexes.getRowIndex();
		long j = indexes.getColumnIndex() ;
		long blockI = i/rSizeBlock, blockJ = j/cSizeBlock ;

		if(blockI == (blockIndexA.getRowIndex()-1) && blockJ == (blockIndexA.getColumnIndex()-1))
			return 0 ;
		else if(blockI == (blockIndexA.getRowIndex()-1))
			return 1 ;
		else if(blockJ == (blockIndexA.getColumnIndex()-1))
			return 2 ;
		else
			return 3 ;
	}

	MatrixIndexes getMatrixIndexes(MatrixIndexes indexes, MatrixIndexes blockIndexA) {
		// Return the new matrix index corresponding to indexes
		long i = indexes.getRowIndex();
		long j = indexes.getColumnIndex();
		int opMatrixId = getOutputMatrixId(indexes, blockIndexA) ;

		long blockI = i/rSizeBlock, blockJ = j/cSizeBlock ;
		long remI = i%rSizeBlock, remJ=j%cSizeBlock ;

		if(opMatrixId == 0) {
			longPair.setIndexes(remI, remJ) ;
		}
		else if(opMatrixId == 1) {
			// How many B blocks are to my left ?
			long yIndex = (blockJ < (blockIndexA.getColumnIndex()-1)) ? j : j-cSizeBlock ;
			longPair.setIndexes(remI, yIndex) ;
		}
		else if(opMatrixId == 2) {
			long xIndex = (blockI < (blockIndexA.getRowIndex()-1)) ? i : i-rSizeBlock ;
			longPair.setIndexes(xIndex, remJ) ;
		}
		else {
			long yIndex = (blockJ < (blockIndexA.getColumnIndex()-1)) ? j : j-cSizeBlock ;
			long xIndex = (blockI < (blockIndexA.getRowIndex()-1)) ? i : i-rSizeBlock ;
			longPair.setIndexes(xIndex, yIndex) ;
		}
		return longPair ;
	}
	

	@Override
	void execute(Well1024a currRandom, Pair<MatrixIndexes, MatrixBlock> pair, Reporter reporter, OutputCollector out) throws IOException {
		MatrixIndexes indexes=pair.getKey();
		MatrixBlock block =(MatrixBlock) pair.getValue();

		for(int blockIndexCounter = 0 ; blockIndexCounter < blockIndexes.length; blockIndexCounter++) {
			int opMatrixId = (pp.toReplicate) ? getOutputMatrixId(indexes, blockIndexes[blockIndexCounter]) + 4*blockIndexCounter :
				getOutputMatrixId(indexes, blockIndexes[blockIndexCounter]);
			
			MatrixIndexes opIndexes = getMatrixIndexes(indexes, blockIndexes[blockIndexCounter]) ;
			opIndexes.setIndexes(opIndexes.getRowIndex() + 1, opIndexes.getColumnIndex() + 1);
			multipleOutputs.getCollector(""+opMatrixId, reporter).collect(opIndexes, block) ;
			
			
			System.out.println("opMatrixId = " + opMatrixId + " opIndexes = " + opIndexes + " block = " + block) ;
		}
	}

}
