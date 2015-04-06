/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.Function;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.Converter;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.Pair;
import com.ibm.bi.dml.runtime.matrix.data.TextToBinaryCellConverter;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


public class ConvertTextLineToBinaryCellFunction implements Function<String, Tuple2<MatrixIndexes, MatrixCell>> {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = -3672377410407066396L;
	private int brlen; 
	private int bclen;
	
	public ConvertTextLineToBinaryCellFunction(int brlen, int bclen) {
		this.brlen = brlen;
		this.bclen = bclen;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public Tuple2<MatrixIndexes, MatrixCell> call(String line) throws Exception {
		@SuppressWarnings("rawtypes")
		Converter converter = new TextToBinaryCellConverter();
		converter.setBlockSize(brlen, bclen);
		converter.convert(null, new Text(line));
		
		Pair<MatrixIndexes, MatrixValue> retVal = null;
		if(converter.hasNext()) {
			retVal = converter.next();
			
			// ------------------------------------------------------------------------------------------
			// Get appropriate indexes for blockIndexes and cell
			// For line: 1020 704 2.362153706180234 (assuming default block size: 1000 X 1000),
			// blockRowIndex = 2, blockColIndex = 1, rowIndexInBlock = 19, colIndexInBlock = 703 ... TODO: double check this !!!
			long blockRowIndex = UtilFunctions.blockIndexCalculation(retVal.getKey().getRowIndex(), (int) brlen);
			long blockColIndex = UtilFunctions.blockIndexCalculation(retVal.getKey().getColumnIndex(), (int) bclen);
			long rowIndexInBlock = UtilFunctions.cellInBlockCalculation(retVal.getKey().getRowIndex(), brlen);
			long colIndexInBlock = UtilFunctions.cellInBlockCalculation(retVal.getKey().getColumnIndex(), bclen);
			// Perform sanity check
			if(blockRowIndex <= 0 || blockColIndex <= 0 || rowIndexInBlock < 0 || colIndexInBlock < 0) {
				throw new Exception("Error computing indexes for the line:" + line);
			}
			// ------------------------------------------------------------------------------------------
			
			MatrixIndexes blockIndexes = new MatrixIndexes(blockRowIndex, blockColIndex);
			MatrixCell cell = new MatrixCell(rowIndexInBlock, colIndexInBlock, ((MatrixCell)retVal.getValue()).getValue());
			
			return new Tuple2<MatrixIndexes, MatrixCell>(blockIndexes, cell);
		}
		
		// In case of header for matrix format
		return new Tuple2<MatrixIndexes, MatrixCell>(new MatrixIndexes(-1, -1), null);
	}
	
}

