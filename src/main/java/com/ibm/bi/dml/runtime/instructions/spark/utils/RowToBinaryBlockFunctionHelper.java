package com.ibm.bi.dml.runtime.instructions.spark.utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.hadoop.io.Text;
import org.apache.spark.sql.Row;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.io.IOUtilFunctions;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

import org.apache.spark.mllib.linalg.Vector;


/**
 * This functions allows to map rdd partitions of csv rows into a set of partial binary blocks.
 * 
 * NOTE: For this csv to binary block function, we need to hold all output blocks per partition 
 * in-memory. Hence, we keep state of all column blocks and aggregate row segments into these blocks. 
 * In terms of memory consumption this is better than creating partial blocks of row segments.
 * 
 */
public class RowToBinaryBlockFunctionHelper implements Serializable 
{
	private static final long serialVersionUID = -4948430402942717043L;
	
	private long _rlen = -1;
	private long _clen = -1;
	private int _brlen = -1;
	private int _bclen = -1;
	private String _delim = null;
	private boolean _fill = false;
	private double _fillValue = 0;
	
	public RowToBinaryBlockFunctionHelper(MatrixCharacteristics mc)
	{
		_rlen = mc.getRows();
		_clen = mc.getCols();
		_brlen = mc.getRowsPerBlock();
		_bclen = mc.getColsPerBlock();
	}
	
	public RowToBinaryBlockFunctionHelper(MatrixCharacteristics mc, String delim, boolean fill, double fillValue)
	{
		_rlen = mc.getRows();
		_clen = mc.getCols();
		_brlen = mc.getRowsPerBlock();
		_bclen = mc.getColsPerBlock();
		_delim = delim;
		_fill = fill;
		_fillValue = fillValue;
	}
	
	boolean emptyFound = false;
	
	// ----------------------------------------------------
	public double[] textToDoubleArray(Text row) {
		String[] parts = IOUtilFunctions.split(row.toString(), _delim);
		double[] ret = new double[parts.length];
		int ix = 0;
		for(String part : parts) {
			emptyFound |= part.isEmpty() && !_fill;
			double val = (part.isEmpty() && _fill) ?
					_fillValue : Double.parseDouble(part);
			ret[ix++] = val;
		}
		return ret;
	}
	public double[] rowToDoubleArray(Row row) throws Exception {
		double[] ret = new double[row.length()];
		for(int i = 0; i < row.length(); i++) {
			ret[i] = getDoubleValue(row, i);
		}
		return ret;
	}
	
	public double[] vectorToDoubleArray(Vector arg) throws Exception {
		return arg.toDense().values();
	}
	// ----------------------------------------------------

	public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> convertToBinaryBlock(Object arg0, RDDConverterTypes converter) 
		throws Exception 
	{
		ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();

		int ncblks = (int)Math.ceil((double)_clen/_bclen);
		MatrixIndexes[] ix = new MatrixIndexes[ncblks];
		MatrixBlock[] mb = new MatrixBlock[ncblks];
		
		@SuppressWarnings("unchecked")
		Iterator<Tuple2<?,Long>> iter = (Iterator<Tuple2<?, Long>>) arg0;
		while( iter.hasNext() )
		{
			Tuple2<?,Long> tmp = iter.next();
			// String row = tmp._1();
			long rowix = tmp._2() + 1;
			
			long rix = UtilFunctions.computeBlockIndex(rowix, _brlen);
			int pos = UtilFunctions.computeCellInBlock(rowix, _brlen);
		
			//create new blocks for entire row
			if( ix[0] == null || ix[0].getRowIndex() != rix ) {
				if( ix[0] !=null )
					flushBlocksToList(ix, mb, ret);
				long len = UtilFunctions.computeBlockSize(_rlen, rix, _brlen);
				createBlocks(rowix, (int)len, ix, mb);
			}
			
			//process row data
			emptyFound = false;
			double[] parts = null;
			switch(converter) {
				case TEXT_TO_DOUBLEARR:
					parts = textToDoubleArray((Text) tmp._1());
					break;
				case ROW_TO_DOUBLEARR:
					parts = rowToDoubleArray((Row) tmp._1());
					break;
				case VECTOR_TO_DOUBLEARR:
					parts = vectorToDoubleArray((Vector) ((Row) tmp._1()).get(0));
					break;
				default:
					throw new Exception("Invalid converter for row-based data:" + converter.toString());
			}
			
			for( int cix=1, pix=0; cix<=ncblks; cix++ ) 
			{
				int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _bclen);				
				for( int j=0; j<lclen; j++ ) {
					double val = parts[pix++];
					mb[cix-1].appendValue(pos, j, val);
				}	
			}
	
			//sanity check empty cells filled w/ values
			if(converter == RDDConverterTypes.TEXT_TO_DOUBLEARR)
				IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(((Text) tmp._1()).toString(), _fill, emptyFound);
		}
	
		//flush last blocks
		flushBlocksToList(ix, mb, ret);
	
		return ret;
	}
		
	// Creates new state of empty column blocks for current global row index.
	private void createBlocks(long rowix, int lrlen, MatrixIndexes[] ix, MatrixBlock[] mb)
	{
		//compute row block index and number of column blocks
		long rix = UtilFunctions.computeBlockIndex(rowix, _brlen);
		int ncblks = (int)Math.ceil((double)_clen/_bclen);
		
		//create all column blocks (assume dense since csv is dense text format)
		for( int cix=1; cix<=ncblks; cix++ ) {
			int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _bclen);				
			ix[cix-1] = new MatrixIndexes(rix, cix);
			mb[cix-1] = new MatrixBlock(lrlen, lclen, false);		
		}
	}
	
	// Flushes current state of filled column blocks to output list.
	private void flushBlocksToList( MatrixIndexes[] ix, MatrixBlock[] mb, ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret ) 
		throws DMLRuntimeException
	{
		int len = ix.length;			
		for( int i=0; i<len; i++ )
			if( mb[i] != null ) {
				ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(ix[i],mb[i]));
				mb[i].examSparsity(); //ensure right representation
			}	
	}
	
	public static double getDoubleValue(Row row, int index) throws Exception {
		try {
			return row.getDouble(index);
		} catch(Exception e) {
			try {
				// Causes lock-contention for Java 7
				return Double.parseDouble(row.get(index).toString());
			}
			catch(Exception e1) {
				throw new Exception("Only double types are supported as input to SystemML. The input argument is \'" + row.get(index) + "\'");
			}
		}
	}
}

