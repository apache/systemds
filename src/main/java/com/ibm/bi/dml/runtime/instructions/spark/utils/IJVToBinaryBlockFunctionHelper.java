package com.ibm.bi.dml.runtime.instructions.spark.utils;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.hadoop.io.Text;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.ReblockBuffer;
import com.ibm.bi.dml.runtime.util.FastStringTokenizer;

public class IJVToBinaryBlockFunctionHelper implements Serializable {
	private static final long serialVersionUID = -7952801318564745821L;
	//internal buffer size (aligned w/ default matrix block size)
	private static final int BUFFER_SIZE = 4 * 1000 * 1000; //4M elements (32MB)
	private int _bufflen = -1;
	
	private long _rlen = -1;
	private long _clen = -1;
	private int _brlen = -1;
	private int _bclen = -1;
	
	public IJVToBinaryBlockFunctionHelper(MatrixCharacteristics mc) throws DMLRuntimeException
	{
		if(!mc.dimsKnown()) {
			throw new DMLRuntimeException("The dimensions need to be known in given MatrixCharacteristics for given input RDD");
		}
		_rlen = mc.getRows();
		_clen = mc.getCols();
		_brlen = mc.getRowsPerBlock();
		_bclen = mc.getColsPerBlock();
		
		//determine upper bounded buffer len
		_bufflen = (int) Math.min(_rlen*_clen, BUFFER_SIZE);
		
	}
	
	// ----------------------------------------------------
	// Can extend this by having type hierarchy
	public MatrixCell textToMatrixCell(Text txt) {
		FastStringTokenizer st = new FastStringTokenizer(' ');
		//get input string (ignore matrix market comments)
		String strVal = txt.toString();
		if( strVal.startsWith("%") ) 
			return null;
		
		//parse input ijv triple
		st.reset( strVal );
		long row = st.nextLong();
		long col = st.nextLong();
		double val = st.nextDouble();
		return new MatrixCell(row, col, val);
	}
	
	public MatrixCell matrixEntryToMatrixCell(MatrixEntry entry) {
		long row = entry.i();
		long col = entry.j();
		double val = entry.value();
		return new MatrixCell(row, col, val);
	}
	
	// ----------------------------------------------------
	
	Iterable<Tuple2<MatrixIndexes, MatrixBlock>> convertToBinaryBlock(Object arg0, RDDConverterTypes converter)  throws Exception {
		ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		ReblockBuffer rbuff = new ReblockBuffer(_bufflen, _rlen, _clen, _brlen, _bclen);
	
		Iterator<?> iter = (Iterator<?>) arg0;
		while( iter.hasNext() ) {
			MatrixCell cell = null;
			switch(converter) {
				case MATRIXENTRY_TO_MATRIXCELL:
					cell = matrixEntryToMatrixCell((MatrixEntry) iter.next());
					break;
					
				case TEXT_TO_MATRIX_CELL:
					cell = textToMatrixCell((Text) iter.next());
					break;
				
				default:
					throw new Exception("Invalid converter for IJV data:" + converter.toString());
			}
			
			if(cell == null) {
				continue;
			}
			
			//flush buffer if necessary
			if( rbuff.getSize() >= rbuff.getCapacity() )
				flushBufferToList(rbuff, ret);
			
			//add value to reblock buffer
			rbuff.appendCell(cell.getRowIndex(), cell.getColIndex(), cell.getValue());
		}
		
		//final flush buffer
		flushBufferToList(rbuff, ret);
	
		return ret;
	}
	
	/**
	 * 
	 * @param rbuff
	 * @param ret
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 */
	private void flushBufferToList( ReblockBuffer rbuff,  ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret ) 
		throws IOException, DMLRuntimeException
	{
		//temporary list of indexed matrix values to prevent library dependencies
		ArrayList<IndexedMatrixValue> rettmp = new ArrayList<IndexedMatrixValue>();					
		rbuff.flushBufferToBinaryBlocks(rettmp);
		ret.addAll(SparkUtils.fromIndexedMatrixBlock(rettmp));
	}
}