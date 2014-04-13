/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.io;

import org.apache.commons.math.random.Well1024a;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.RandCPInstruction;

public class MatrixBlock extends MatrixBlockDSM
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixBlock() {
		super();
	}
	
	public MatrixBlock(int rlen, int clen, boolean sparse) {
		super(rlen, clen, sparse);
	}

	public MatrixBlock(int rlen, int clen, boolean sparse, int estnnzs){
		super(rlen, clen, sparse, estnnzs);
	}
	
	
	/**
	 * Function to generate the random matrix with specified dimensions (block sizes are not specified).
	 *  
	 * @param rows
	 * @param cols
	 * @param sparsity
	 * @param min
	 * @param max
	 * @param pdf
	 * @param seed
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock randOperations(int rows, int cols, double sparsity, double min, double max, String pdf, long seed) throws DMLRuntimeException
	{
		int blocksize = ConfigurationManager.getConfig().getIntValue(DMLConfig.DEFAULT_BLOCK_SIZE);
		return randOperations(
				rows, cols, blocksize, blocksize, 
				sparsity, min, max, pdf, seed);
	}
	
	/**
	 * Function to generate the random matrix with specified dimensions and block dimensions.
	 * @param rows
	 * @param cols
	 * @param rowsInBlock
	 * @param colsInBlock
	 * @param sparsity
	 * @param min
	 * @param max
	 * @param pdf
	 * @param seed
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock randOperations(int rows, int cols, int rowsInBlock, int colsInBlock, double sparsity, double min, double max, String pdf, long seed) throws DMLRuntimeException {
		Well1024a bigrand = RandCPInstruction.setupSeedsForRand(seed);
		MatrixBlock m = new MatrixBlock();
		if ( pdf.equalsIgnoreCase("normal") ) {
			// for normally distributed values, min and max are specified as an invalid value NaN.
			m.getRandomMatrix(pdf, rows, cols, rowsInBlock, colsInBlock, sparsity, Double.NaN, Double.NaN, bigrand, -1);
		}
		else {
			m.getRandomMatrix(pdf, rows, cols, rowsInBlock, colsInBlock, sparsity, min, max, bigrand, -1);
		}
		return m;
	}
	
	public static MatrixBlock seqOperations(double from, double to, double incr) throws DMLRuntimeException {
		MatrixBlock m = new MatrixBlock();
		m.getSequence(from, to, incr);
		return m;
	}
}
