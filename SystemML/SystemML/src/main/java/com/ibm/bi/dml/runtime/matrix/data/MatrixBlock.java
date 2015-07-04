/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;

import org.apache.commons.math3.random.Well1024a;
import org.apache.hadoop.io.DataInputBuffer;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.Hop.OpOp2;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.lops.MapMultChain.ChainType;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.lops.WeightedSquaredLoss.WeightsType;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.CTable;
import com.ibm.bi.dml.runtime.functionobjects.DiagIndex;
import com.ibm.bi.dml.runtime.functionobjects.Divide;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.ReduceAll;
import com.ibm.bi.dml.runtime.functionobjects.SortIndex;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.instructions.cp.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.cp.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.cp.KahanObject;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.mr.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.matrix.data.LibMatrixBincell.BinaryAccessType;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.COVOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.QuaternaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;
import com.ibm.bi.dml.runtime.util.FastBufferedDataInputStream;
import com.ibm.bi.dml.runtime.util.FastBufferedDataOutputStream;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


public class MatrixBlock extends MatrixValue implements Externalizable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = 7319972089143154056L;
	
	//sparsity nnz threshold, based on practical experiments on space consumption and performance
	public static final double SPARSITY_TURN_POINT = 0.4;
	//sparsity threshold for ultra-sparse matrix operations (40nnz in a 1kx1k block)
	public static final double ULTRA_SPARSITY_TURN_POINT = 0.00004; 
	//basic header (int rlen, int clen, byte type)
	public static final int HEADER_SIZE = 9;
	
	public enum BlockType{
		EMPTY_BLOCK,  
		ULTRA_SPARSE_BLOCK, //ultra sparse representation, in-mem same as sparse
		SPARSE_BLOCK, //sparse representation, see sparseRows 
		DENSE_BLOCK, //dense representation, see denseBlock			
	}
	
	//matrix meta data
	protected int rlen       = -1;
	protected int clen       = -1;
	protected boolean sparse = true;
	protected long nonZeros   = 0;
	
	//matrix data (sparse or dense)
	protected double[] denseBlock    = null;
	protected SparseRow[] sparseRows = null;
		
	//sparse-block-specific attributes (allocation only)
	protected int estimatedNNzsPerRow = -1; 
		
	//ctable-specific attributes
	protected int maxrow = -1;
	protected int maxcolumn = -1;
	
	//grpaggregate-specific attributes (optional)
	protected int numGroups = -1;
	
	//diag-specific attributes (optional)
	protected boolean diag = false;
	
	
	////////
	// Matrix Constructors
	//
	
	public MatrixBlock()
	{
		rlen = 0;
		clen = 0;
		sparse = true;
		nonZeros = 0;
		maxrow = 0;
		maxcolumn = 0;
	}
	public MatrixBlock(int rl, int cl, boolean sp)
	{
		rlen = rl;
		clen = cl;
		sparse = sp;
		nonZeros = 0;
		maxrow = 0;
		maxcolumn = 0;
	}
	
	public MatrixBlock(int rl, int cl, boolean sp, long estnnzs)
	{
		this(rl, cl, sp);
		estimatedNNzsPerRow=(int)Math.ceil((double)estnnzs/(double)rl);	
	}
	
	public MatrixBlock(MatrixBlock that)
	{
		this.copy(that);
	}
	
	////////
	// Initialization methods
	// (reset, init, allocate, etc)
	
	public void reset()
	{
		reset(-rlen);
	}
	
	public void reset(long estnnzs)
	{
		estimatedNNzsPerRow=(int)Math.ceil((double)estnnzs/(double)rlen);
		if(sparse)
		{
			resetSparse();
		}
		else
		{
			if(denseBlock!=null)
			{
				if(denseBlock.length<rlen*clen)
					denseBlock=null;
				else
					Arrays.fill(denseBlock, 0, rlen*clen, 0);
			}
		}
		nonZeros=0;
		
		//operation-specific attributes
		maxrow = rlen;
		maxcolumn = clen;
		numGroups = -1;
	}
	
	public void reset(int rl, int cl) {
		rlen=rl;
		clen=cl;
		nonZeros=0;
		reset();
	}
	
	public void reset(int rl, int cl, long estnnzs) {
		rlen=rl;
		clen=cl;
		nonZeros=0;
		reset(estnnzs);
	}
		
	public void reset(int rl, int cl, boolean sp)
	{
		sparse=sp;
		reset(rl, cl);
	}
	
	public void reset(int rl, int cl, boolean sp, long estnnzs)
	{
		sparse=sp;
		reset(rl, cl, estnnzs);
	}
	
	public void resetSparse()
	{
		if(sparseRows!=null)
		{
			for(int i=0; i<Math.min(rlen, sparseRows.length); i++)
				if(sparseRows[i]!=null)
					sparseRows[i].reset(estimatedNNzsPerRow, clen);
		}
	}
	
	public void resetDenseWithValue(int rl, int cl, double v) 
		throws DMLRuntimeException 
	{	
		estimatedNNzsPerRow=-1;
		rlen=rl;
		clen=cl;
		sparse=false;
		
		if(v==0)
		{
			reset();
			return;
		}
		
		//allocate dense block
		allocateDenseBlock();
		
		//init with constant value (non-zero, see above)
		int limit = rlen * clen;
		Arrays.fill(denseBlock, 0, limit, v);
		nonZeros=limit;
	}
	
	/**
	 * NOTE: This method is designed only for dense representation.
	 * 
	 * @param arr
	 * @param r
	 * @param c
	 * @throws DMLRuntimeException
	 */
	public void init(double[][] arr, int r, int c) 
		throws DMLRuntimeException 
	{	
		//input checks 
		if ( sparse )
			throw new DMLRuntimeException("MatrixBlockDSM.init() can be invoked only on matrices with dense representation.");
		if( r*c > rlen*clen )
			throw new DMLRuntimeException("MatrixBlockDSM.init() invoked with too large dimensions ("+r+","+c+") vs ("+rlen+","+clen+")");
		
		//allocate or resize dense block
		allocateDenseBlock();
		
		//copy and compute nnz
		for(int i=0, ix=0; i < r; i++, ix+=clen) 
			System.arraycopy(arr[i], 0, denseBlock, ix, arr[i].length);
		recomputeNonZeros();
		
		maxrow = r;
		maxcolumn = c;
	}
	
	/**
	 * NOTE: This method is designed only for dense representation.
	 * 
	 * @param arr
	 * @param r
	 * @param c
	 * @throws DMLRuntimeException
	 */
	public void init(double[] arr, int r, int c) 
		throws DMLRuntimeException 
	{	
		//input checks 
		if ( sparse )
			throw new DMLRuntimeException("MatrixBlockDSM.init() can be invoked only on matrices with dense representation.");
		if( r*c > rlen*clen )
			throw new DMLRuntimeException("MatrixBlockDSM.init() invoked with too large dimensions ("+r+","+c+") vs ("+rlen+","+clen+")");
		
		//allocate or resize dense block
		allocateDenseBlock();
		
		//copy and compute nnz 
		System.arraycopy(arr, 0, denseBlock, 0, arr.length);
		recomputeNonZeros();
		
		maxrow = r;
		maxcolumn = c;
	}
	
	/**
	 * 
	 * @param val
	 * @param r
	 * @param c
	 * @throws DMLRuntimeException
	 */
	public void init(double val, int r, int c)
		throws DMLRuntimeException 
	{	
		//input checks 
		if ( sparse )
			throw new DMLRuntimeException("MatrixBlockDSM.init() can be invoked only on matrices with dense representation.");
		if( r*c > rlen*clen )
			throw new DMLRuntimeException("MatrixBlockDSM.init() invoked with too large dimensions ("+r+","+c+") vs ("+rlen+","+clen+")");
		
		if( val != 0 ) {
			//allocate or resize dense block
			allocateDenseBlock();
			
			if( r*c == rlen*clen ) { //FULL MATRIX INIT
				//memset value  
				Arrays.fill(denseBlock, val);
			}
			else { //PARTIAL MATRIX INIT
				//rowwise memset value 
				for(int i=0, ix=0; i < r; i++, ix+=clen) 
					Arrays.fill(denseBlock, ix, ix+c, val);
			}
			
			//set non zeros to input dims
			nonZeros = r*c;
		}
		
		maxrow = r;
		maxcolumn = c;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isAllocated()
	{
		if( sparse )
			return (sparseRows!=null);
		else
			return (denseBlock!=null);
	}
	
	/**
	 * @throws DMLRuntimeException 
	 * 
	 */
	public void allocateDenseBlock() 
		throws DMLRuntimeException 
	{
		allocateDenseBlock( true );
	}
	
	/**
	 * 
	 * @param clearNNZ
	 * @throws DMLRuntimeException
	 */
	public void allocateDenseBlock(boolean clearNNZ) 
		throws DMLRuntimeException 
	{
		long limit = (long)rlen * clen;
		
		//check max size constraint (16GB dense), since java arrays are limited to 2^(32-1) elements)
		if( limit > Integer.MAX_VALUE ) {
			throw new DMLRuntimeException("Dense in-memory matrix block ("+rlen+"x"+clen+") exceeds supported size of "+Integer.MAX_VALUE+" elements (16GB). " +
					                      "Please, reduce the JVM heapsize to execute this in MR.");
		}
		
		//allocate block if non-existing or too small (guaranteed to be 0-initialized),
		if(denseBlock == null || denseBlock.length < limit ) {
			denseBlock = new double[(int)limit];
		}
		
		//clear nnz if necessary
		if( clearNNZ ) {
			nonZeros = 0;
		}
	}
	
	/**
	 * 
	 */
	public void allocateSparseRowsBlock()
	{
		allocateSparseRowsBlock(true);
	}
	
	/**
	 * 
	 * @param clearNNZ
	 */
	public void allocateSparseRowsBlock(boolean clearNNZ)
	{	
		//allocate block if non-existing or too small (guaranteed to be 0-initialized),
		if( sparseRows == null ) {
			sparseRows=new SparseRow[rlen];
		}
		else if( sparseRows.length < rlen ) {
			SparseRow[] oldSparseRows=sparseRows;
			sparseRows = new SparseRow[rlen];
			for(int i=0; i<Math.min(oldSparseRows.length, rlen); i++) {
				sparseRows[i]=oldSparseRows[i];
			}
		}
		
		//clear nnz if necessary
		if( clearNNZ ) {
			nonZeros = 0;
		}
	}
	
	
	/**
	 * This should be called only in the read and write functions for CP
	 * This function should be called before calling any setValueDenseUnsafe()
	 * 
	 * @param rl
	 * @param cl
	 * @throws DMLRuntimeException 
	 */
	public void allocateDenseBlockUnsafe(int rl, int cl) 
		throws DMLRuntimeException
	{
		sparse=false;
		rlen=rl;
		clen=cl;
		
		//allocate dense block
		allocateDenseBlock();
	}
	
	
	/**
	 * Allows to cleanup all previously allocated sparserows or denseblocks.
	 * This is for example required in reading a matrix with many empty blocks 
	 * via distributed cache into in-memory list of blocks - not cleaning blocks 
	 * from non-empty blocks would significantly increase the total memory consumption.
	 * 
	 */
	public void cleanupBlock( boolean dense, boolean sparse )
	{
		if(dense)
			denseBlock = null;
		if(sparse)
			sparseRows = null;
	}
	
	////////
	// Metadata information 
	
	public int getNumRows()
	{
		return rlen;
	}
	
	/**
	 * NOTE: setNumRows() and setNumColumns() are used only in tertiaryInstruction (for contingency tables)
	 * and pmm for meta corrections.
	 * 
	 * @param _r
	 */
	public void setNumRows(int r) 
	{
		rlen = r;
	}
	
	public int getNumColumns()
	{
		return clen;
	}
	
	public void setNumColumns(int c) 
	{
		clen = c;
	}
	
	public long getNonZeros()
	{
		return nonZeros;
	}
	
	public boolean isVector() 
	{
		return (rlen == 1 || clen == 1);
	}
	
	
	/**
	 * Return the maximum row encountered WITHIN the current block
	 *  
	 */
	public int getMaxRow() 
	{
		if (!sparse) 
			return getNumRows();
		else 
			return maxrow;
	}
	
	public void setMaxRow(int r)
	{
		maxrow = r;
	}
	
	
	/**
	 * Return the maximum column encountered WITHIN the current block
	 * 
	 */
	public int getMaxColumn() 
	{
		if (!sparse) 
			return getNumColumns();
		else 
			return maxcolumn;
	}
	
	public void setMaxColumn(int c) 
	{
		maxcolumn = c;
	}
	
	@Override
	public boolean isEmpty()
	{
		return isEmptyBlock(false);
	}
	
	public boolean isEmptyBlock()
	{
		return isEmptyBlock(true);
	}
	
	
	public boolean isEmptyBlock(boolean safe)
	{
		boolean ret = false;
		if( sparse && sparseRows==null )
			ret = true;
		else if( !sparse && denseBlock==null ) 	
			ret = true;
		if( nonZeros==0 )
		{
			//prevent under-estimation
			if(safe)
				recomputeNonZeros();
			ret = (nonZeros==0);
		}
		return ret;
	}
	
	public void setDiag()
	{
		diag = true;
	}
	
	public boolean isDiag()
	{
		return diag;
	}
	
	////////
	// Data handling
	
	public double[] getDenseArray()
	{
		if(sparse)
			return null;
		return denseBlock;
	}
	
	public SparseRow[] getSparseRows()
	{
		if(!sparse)
			return null;
		return sparseRows;
	}
	
	public SparseRowsIterator getSparseRowsIterator()
	{
		//check for valid format, should have been checked from outside
		if( !sparse )
			throw new RuntimeException("getSparseCellInterator should not be called for dense format");
		
		return new SparseRowsIterator(rlen, sparseRows);
	}
	
	public SparseRowsIterator getSparseRowsIterator(int rowStart, int rowNum)
	{
		//check for valid format, should have been checked from outside
		if( !sparse )
			throw new RuntimeException("getSparseCellInterator should not be called for dense format");
		
		return new SparseRowsIterator(rowStart, rowStart+rowNum, sparseRows);
	}
	
	@Override
	public void getCellValues(Collection<Double> ret) 
	{
		int limit=rlen*clen;
		if(sparse)
		{
			if(sparseRows==null)
			{
				for(int i=0; i<limit; i++)
					ret.add(0.0);
			}else
			{
				for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
				{
					if(sparseRows[r]==null) 
						continue;
					double[] container=sparseRows[r].getValueContainer();
					for(int j=0; j<sparseRows[r].size(); j++)
						ret.add(container[j]);
				}
				int zeros=limit-ret.size();
				for(int i=0; i<zeros; i++)
					ret.add(0.0);
			}
		}else
		{
			if(denseBlock==null)
			{
				for(int i=0; i<limit; i++)
					ret.add(0.0);
			}else
			{
				for(int i=0; i<limit; i++)
					ret.add(denseBlock[i]);
			}
		}
	}

	@Override
	public void getCellValues(Map<Double, Integer> ret) 
	{
		int limit=rlen*clen;
		if(sparse)
		{
			if(sparseRows==null)
			{
				ret.put(0.0, limit);
			}else
			{
				for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
				{
					if(sparseRows[r]==null) 
						continue;
					double[] container=sparseRows[r].getValueContainer();
					for(int j=0; j<sparseRows[r].size(); j++)
					{
						Double v=container[j];
						Integer old=ret.get(v);
						if(old!=null)
							ret.put(v, old+1);
						else
							ret.put(v, 1);
					}
				}
				int zeros=limit-ret.size();
				Integer old=ret.get(0.0);
				if(old!=null)
					ret.put(0.0, old+zeros);
				else
					ret.put(0.0, zeros);
			}
			
		}else
		{
			if(denseBlock==null)
			{
				ret.put(0.0, limit);
			}else
			{
				for(int i=0; i<limit; i++)
				{
					double v=denseBlock[i];
					Integer old=ret.get(v);
					if(old!=null)
						ret.put(v, old+1);
					else
						ret.put(v, 1);
				}	
			}
		}
	}
	
	@Override
	public double getValue(int r, int c) 
	{
		if(r>rlen || c > clen)
			throw new RuntimeException("indexes ("+r+","+c+") out of range ("+rlen+","+clen+")");
		
		if(sparse)
		{
			if(sparseRows==null || sparseRows.length<=r || sparseRows[r]==null)
				return 0;
			return sparseRows[r].get(c);
		}else
		{
			if(denseBlock==null)
				return 0;
			return denseBlock[r*clen+c]; 
		}
	}
	
	@Override
	public void setValue(int r, int c, double v) 
	{
		if(r>rlen || c > clen)
			throw new RuntimeException("indexes ("+r+","+c+") out of range ("+rlen+","+clen+")");
		if(sparse)
		{
			if( (sparseRows==null || sparseRows.length<=r || sparseRows[r]==null) && v==0.0)
				return;
			//allocation on demand
			allocateSparseRowsBlock(false);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(estimatedNNzsPerRow, clen);
			
			if(sparseRows[r].set(c, v))
				nonZeros++;
			
		}else
		{
			if(denseBlock==null && v==0.0)
				return;

			//allocate and init dense block (w/o overwriting nnz)
			try {
				allocateDenseBlock(false);
			}
			catch(DMLRuntimeException e){
				throw new RuntimeException(e);
			}
				
			int index=r*clen+c;
			if(denseBlock[index]==0)
				nonZeros++;
			denseBlock[index]=v;
			if(v==0)
				nonZeros--;
		}
		
	}
	
	@Override
	public void setValue(CellIndex index, double v) 
	{
		setValue(index.row, index.column, v);
	}
	
	@Override
	/**
	 * If (r,c) \in Block, add v to existing cell
	 * If not, add a new cell with index (r,c).
	 * 
	 * This function intentionally avoids the maintenance of NNZ for efficiency. 
	 * 
	 */
	public void addValue(int r, int c, double v) {
		if(sparse)
		{
			//allocation on demand
			allocateSparseRowsBlock(false);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(estimatedNNzsPerRow, clen);
			double curV=sparseRows[r].get(c);
			curV+=v;
			sparseRows[r].set(c, curV);
			
		}
		else
		{
			//allocate and init dense block (w/o overwriting nnz)
			try {
				allocateDenseBlock(false);
			}
			catch(DMLRuntimeException e){
				throw new RuntimeException(e);
			}
			
			int index=r*clen+c;
			denseBlock[index]+=v;
		}
	}
	
	public double quickGetValue(int r, int c) 
	{
		if(sparse)
		{
			if(sparseRows==null || sparseRows.length<=r || sparseRows[r]==null)
				return 0;
			return sparseRows[r].get(c);
		}
		else
		{
			if(denseBlock==null)
				return 0;
			return denseBlock[r*clen+c]; 
		}
	}
	
	public void quickSetValue(int r, int c, double v) 
	{
		if(sparse)
		{
			if( (sparseRows==null || sparseRows.length<=r || sparseRows[r]==null) && v==0.0)
				return;
			//allocation on demand
			allocateSparseRowsBlock(false);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(estimatedNNzsPerRow, clen);
			
			if(sparseRows[r].set(c, v))
				nonZeros++;
			
		}
		else
		{
			if(denseBlock==null && v==0.0)
				return;		
			
			//allocate and init dense block (w/o overwriting nnz)
			try {
				allocateDenseBlock(false);
			}
			catch(DMLRuntimeException e){
				throw new RuntimeException(e);
			}
			
			int index=r*clen+c;
			if(denseBlock[index]==0)
				nonZeros++;
			denseBlock[index]=v;
			if(v==0)
				nonZeros--;
		}
	}
	
	public double getValueDenseUnsafe(int r, int c) 
	{
		if(denseBlock==null)
			return 0;
		return denseBlock[r*clen+c]; 
	}
	

	/**
	 * This can be only called when you know you have properly allocated spaces for a dense representation
	 * and r and c are in the the range of the dimension
	 * Note: this function won't keep track of the nozeros
	 */	
	public void setValueDenseUnsafe(int r, int c, double v) 
	{
		denseBlock[r*clen+c]=v;		
	}
	
	public double getValueSparseUnsafe(int r, int c) 
	{
		if(sparseRows==null || sparseRows.length<=r || sparseRows[r]==null)
			return 0;
		return sparseRows[r].get(c);	
	}
	
	/**
	 * Append value is only used when values are appended at the end of each row for the sparse representation
	 * This can only be called, when the caller knows the access pattern of the block
	 
	 * @param r
	 * @param c
	 * @param v
	 */
	public void appendValue(int r, int c, double v)
	{
		if(v==0) return;
		if(!sparse) 
			quickSetValue(r, c, v);
		else
		{
			//allocation on demand
			allocateSparseRowsBlock(false);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(estimatedNNzsPerRow, clen);
			sparseRows[r].append(c, v);
			nonZeros++;
		}
	}
	
	public void appendRow(int r, SparseRow values)
	{
		if(values==null)
			return;
		if(sparse)
		{
			//allocation on demand
			allocateSparseRowsBlock(false);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(values);
			else
				sparseRows[r].copy(values);
			nonZeros+=values.size();
			
		}
		else
		{
			int[] cols=values.getIndexContainer();
			double[] vals=values.getValueContainer();
			for(int i=0; i<values.size(); i++)
				quickSetValue(r, cols[i], vals[i]);
		}
	}
	
	/**
	 * 
	 * @param that
	 * @param rowoffset
	 * @param coloffset
	 */
	public void appendToSparse( MatrixBlock that, int rowoffset, int coloffset ) 
	{
		if( that==null || that.isEmptyBlock(false) )
			return; //nothing to append
		
		//init sparse rows if necessary
		allocateSparseRowsBlock(false);
		
		if( that.sparse ) //SPARSE <- SPARSE
		{
			for( int i=0; i<that.rlen; i++ )
			{
				SparseRow brow = that.sparseRows[i];
				if( brow!=null && !brow.isEmpty() )
				{
					int aix = rowoffset+i;
					int len = brow.size();
					int[] ix = brow.getIndexContainer();
					double[] val = brow.getValueContainer();
					
					if( sparseRows[aix]==null )
						sparseRows[aix] = new SparseRow(estimatedNNzsPerRow,clen);
					
					for( int j=0; j<len; j++ )
						sparseRows[aix].append(coloffset+ix[j], val[j]);		
				}
			}
		}
		else //SPARSE <- DENSE
		{
			for( int i=0; i<that.rlen; i++ )
			{
				int aix = rowoffset+i;
				for( int j=0, bix=i*that.clen; j<that.clen; j++ )
				{
					double val = that.denseBlock[bix+j];
					if( val != 0 )
					{
						if( sparseRows[aix]==null )//create sparserow only if required
							sparseRows[aix] = new SparseRow(estimatedNNzsPerRow,clen);
						sparseRows[aix].append(coloffset+j, val);
					}
				}
			}
		}
	}
	
	/**
	 * 
	 */
	public void sortSparseRows()
	{
		if( !sparse || sparseRows==null )
			return;
		
		for( SparseRow arow : sparseRows )
			if( arow!=null && arow.size()>1 )
				arow.sort();
	}
	
	/**
	 * Utility function for computing the min non-zero value. 
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 */
	public double minNonZero() 
		throws DMLRuntimeException
	{
		//check for empty block and return immediately
		if( isEmptyBlock() )
			return -1;
		
		//NOTE: usually this method is only applied on dense vectors and hence not really tuned yet.
		double min = Double.MAX_VALUE;
		for( int i=0; i<rlen; i++ )
			for( int j=0; j<clen; j++ ){
				double val = quickGetValue(i, j);
				if( val != 0 )
					min = Math.min(min, val);
			}
		
		return min;
	}
	
	/**
	 * Wrapper method for reduceall-min of a matrix.
	 * 
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public double min() 
		throws DMLRuntimeException
	{
		//construct operator
		AggregateOperator aop = new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject("min"));
		AggregateUnaryOperator auop = new AggregateUnaryOperator( aop, ReduceAll.getReduceAllFnObject());
		
		//execute operation
		MatrixBlock out = new MatrixBlock(1, 1, false);
		LibMatrixAgg.aggregateUnaryMatrix(this, out, auop);
		
		return out.quickGetValue(0, 0);
	}
	
	/**
	 * Wrapper method for reduceall-max of a matrix.
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 */
	public double max() 
		throws DMLRuntimeException
	{
		//construct operator
		AggregateOperator aop = new AggregateOperator(-Double.MAX_VALUE, Builtin.getBuiltinFnObject("max"));
		AggregateUnaryOperator auop = new AggregateUnaryOperator( aop, ReduceAll.getReduceAllFnObject());
		
		//execute operation
		MatrixBlock out = new MatrixBlock(1, 1, false);
		LibMatrixAgg.aggregateUnaryMatrix(this, out, auop);
		
		return out.quickGetValue(0, 0);
	}

	////////
	// sparsity handling functions
	
	/**
	 * Returns the current representation (true for sparse).
	 * 
	 */
	public boolean isInSparseFormat()
	{
		return sparse;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isUltraSparse()
	{
		double sp = ((double)nonZeros/rlen)/clen;
		//check for sparse representation in order to account for vectors in dense
		return sparse && sp<ULTRA_SPARSITY_TURN_POINT && nonZeros<40;
	}

	/**
	 * Evaluates if this matrix block should be in sparse format in
	 * memory. Note that this call does not change the representation - 
	 * for this please call examSparsity.
	 * 
	 * @return
	 */
	public boolean evalSparseFormatInMemory()
	{
		long lrlen = (long) rlen;
		long lclen = (long) clen;
		long lnonZeros = (long) nonZeros;
			
		//ensure exact size estimates for write
		if( lnonZeros<=0 ) {
			recomputeNonZeros();
			lnonZeros = (long) nonZeros;
		}	
		
		//decide on in-memory representation
		return evalSparseFormatInMemory(lrlen, lclen, lnonZeros);
	}
	

	private boolean evalSparseFormatInMemory(boolean transpose)
	{
		int lrlen = (transpose) ? clen : rlen;
		int lclen = (transpose) ? rlen : clen;
		long lnonZeros = (long) nonZeros;
		
		//ensure exact size estimates for write
		if( lnonZeros<=0 ) {
			recomputeNonZeros();
			lnonZeros = (long) nonZeros;
		}	
		
		//decide on in-memory representation
		return evalSparseFormatInMemory(lrlen, lclen, lnonZeros);
	}
	
	/**
	 * Evaluates if this matrix block should be in sparse format on
	 * disk. This applies to any serialized matrix representation, i.e.,
	 * when writing to in-memory buffer pool pages or writing to local fs
	 * or hdfs. 
	 * 
	 * @return
	 */
	public boolean evalSparseFormatOnDisk()
	{
		long lrlen = (long) rlen;
		long lclen = (long) clen;

		//ensure exact size estimates for write
		if( nonZeros <= 0 ) {
			recomputeNonZeros();
		}	
		
		//decide on in-memory representation
		return evalSparseFormatOnDisk(lrlen, lclen, nonZeros);
	}
	
	/**
	 * Evaluates if this matrix block should be in sparse format in
	 * memory. Depending on the current representation, the state of the
	 * matrix block is changed to the right representation if necessary. 
	 * Note that this consumes for the time of execution memory for both 
	 * representations.  
	 * 
	 * @throws DMLRuntimeException
	 */
	public void examSparsity() 
		throws DMLRuntimeException
	{
		//determine target representation
		boolean sparseDst = evalSparseFormatInMemory(); 
				
		//check for empty blocks (e.g., sparse-sparse)
		if( isEmptyBlock(false) )
			cleanupBlock(true, true);
		
		//change representation if required (also done for 
		//empty blocks in order to set representation flags)
		if( sparse && !sparseDst)
			sparseToDense();
		else if( !sparse && sparseDst )
			denseToSparse();
	}
	
	/**
	 * Evaluates if a matrix block with the given characteristics should be in sparse format 
	 * in memory.
	 * 
	 * @param brlen
	 * @param bclen
	 * @param nnz
	 * @return
	 */
	public static boolean evalSparseFormatInMemory( final long nrows, final long ncols, final long nnz )
	{		
		//evaluate sparsity threshold
		double lsparsity = (double)nnz/nrows/ncols;
		boolean lsparse = (lsparsity < SPARSITY_TURN_POINT);
		
		//compare size of sparse and dense representation in order to prevent
		//that the sparse size exceed the dense size since we use the dense size
		//as worst-case estimate if unknown (and it requires less io from 
		//main memory).
		double sizeSparse = estimateSizeSparseInMemory(nrows, ncols, lsparsity);
		double sizeDense = estimateSizeDenseInMemory(nrows, ncols);
		
		return lsparse && (sizeSparse<sizeDense);
	}
	
	/**
	 * Evaluates if a matrix block with the given characteristics should be in sparse format 
	 * on disk (or in any other serialized representation).
	 * 
	 * @param brlen
	 * @param bclen
	 * @param nnz
	 * @return
	 */
	public static boolean evalSparseFormatOnDisk( final long nrows, final long ncols, final long nnz )
	{
		//evaluate sparsity threshold
		double lsparsity = ((double)nnz/nrows)/ncols;
		boolean lsparse = (lsparsity < SPARSITY_TURN_POINT);
		
		double sizeUltraSparse = estimateSizeUltraSparseOnDisk( nrows, ncols, nnz );
		double sizeSparse = estimateSizeSparseOnDisk(nrows, ncols, nnz);
		double sizeDense = estimateSizeDenseOnDisk(nrows, ncols);
		
		return lsparse && (sizeSparse<sizeDense || sizeUltraSparse<sizeDense);		
	}
	
	
	////////
	// basic block handling functions	
	
	/**
	 * 
	 */
	private void denseToSparse() 
	{	
		//set target representation
		sparse = true;
		
		//early abort on empty blocks
		if(denseBlock==null)
			return;
		
		//allocate sparse target block (reset required to maintain nnz again)
		allocateSparseRowsBlock();
		reset();
		
		//copy dense to sparse
		double[] a = denseBlock;
		SparseRow[] c = sparseRows;
		
		for( int i=0, aix=0; i<rlen; i++ )
			for(int j=0; j<clen; j++, aix++)
				if( a[aix] != 0 ) {
					if( c[i]==null ) //create sparse row only if required
						c[i]=new SparseRow(estimatedNNzsPerRow, clen);
					c[i].append(j, a[aix]);
					nonZeros++;
				}
				
		//cleanup dense block
		denseBlock = null;
	}
	
	/**
	 * 
	 * @throws DMLRuntimeException
	 */
	private void sparseToDense() 
		throws DMLRuntimeException 
	{	
		//set target representation
		sparse = false;
		
		//early abort on empty blocks
		if(sparseRows==null)
			return;
		
		int limit=rlen*clen;
		if ( limit < 0 ) {
			throw new DMLRuntimeException("Unexpected error in sparseToDense().. limit < 0: " + rlen + ", " + clen + ", " + limit);
		}
		
		//allocate dense target block, but keep nnz (no need to maintain)
		allocateDenseBlock(false);
		Arrays.fill(denseBlock, 0, limit, 0);
		
		//copy sparse to dense
		SparseRow[] a = sparseRows;
		double[] c = denseBlock;
		
		for( int i=0, cix=0; i<rlen; i++, cix+=clen)
			if( a[i] != null && !a[i].isEmpty() ) {
				int alen = a[i].size();
				int[] aix = a[i].getIndexContainer();
				double[] avals = a[i].getValueContainer();
				for(int j=0; j<alen; j++)
					if( avals[j] != 0 )
						c[ cix+aix[j] ] = avals[j];
			}
		
		//cleanup sparse rows
		sparseRows = null;
	}

	public void recomputeNonZeros()
	{
		nonZeros=0;
		if( sparse && sparseRows!=null )
		{
			int limit = Math.min(rlen, sparseRows.length);
			for(int i=0; i<limit; i++)
				if(sparseRows[i]!=null)
					nonZeros += sparseRows[i].size();
		}
		else if( !sparse && denseBlock!=null )
		{
			int limit=rlen*clen;
			for(int i=0; i<limit; i++)
			{
				//HotSpot JVM bug causes crash in presence of NaNs 
				//nonZeros += (denseBlock[i]!=0) ? 1 : 0;
				if( denseBlock[i]!=0 )
					nonZeros++;
			}
		}
	}
	
	private long recomputeNonZeros(int rl, int ru, int cl, int cu)
	{
		long nnz = 0;
		if(sparse)
		{
			if(sparseRows!=null)
			{
				int rlimit = Math.min( ru+1, Math.min(rlen, sparseRows.length) );
				if( cl==0 && cu==clen-1 ) //specific case: all cols
				{
					for(int i=rl; i<rlimit; i++)
						if(sparseRows[i]!=null && !sparseRows[i].isEmpty())
							nnz+=sparseRows[i].size();	
				}
				else if( cl==cu ) //specific case: one column
				{
					for(int i=rl; i<rlimit; i++)
						if(sparseRows[i]!=null && !sparseRows[i].isEmpty())
							nnz += (sparseRows[i].get(cl)!=0) ? 1 : 0;
				}
				else //general case
				{
					int astart,aend;
					for(int i=rl; i<rlimit; i++)
						if(sparseRows[i]!=null && !sparseRows[i].isEmpty())
						{
							SparseRow arow = sparseRows[i];
							astart = arow.searchIndexesFirstGTE(cl);
							aend = arow.searchIndexesFirstGTE(cu);
							nnz += (astart!=-1) ? (aend-astart+1) : 0;
						}
				}
			}
		}
		else
		{
			if(denseBlock!=null)
			{
				for( int i=rl, ix=rl*clen; i<=ru; i++, ix+=clen )
					for( int j=cl; j<=cu; j++ )
					{
						//HotSpot JVM bug causes crash in presence of NaNs 
						//nnz += (denseBlock[ix+j]!=0) ? 1 : 0;
						if( denseBlock[ix+j]!=0 )
							nnz++;
					}
			}
		}

		return nnz;
	}
	
	/**
	 * Basic debugging primitive to check correctness of nnz, this method is not intended for
	 * production use.
	 * 
	 * @throws DMLRuntimeException
	 */
	public void checkNonZeros() 
		throws DMLRuntimeException
	{
		//take non-zeros before and after recompute nnz
		long nnzBefore = getNonZeros();
		recomputeNonZeros();
		long nnzAfter = getNonZeros();
		
		//raise exception if non-zeros don't match up
		if( nnzBefore != nnzAfter )
			throw new DMLRuntimeException("Number of non zeros incorrect: "+nnzBefore+" vs "+nnzAfter);
	}

	@Override
	public void copy(MatrixValue thatValue) 
	{
		MatrixBlock that = checkType(thatValue);
		
		//copy into automatically determined representation
		copy( that, that.evalSparseFormatInMemory() );
	}
	
	@Override
	public void copy(MatrixValue thatValue, boolean sp) 
	{
		MatrixBlock that = checkType(thatValue);
		
		if( this == that ) //prevent data loss (e.g., on sparse-dense conversion)
			throw new RuntimeException( "Copy must not overwrite itself!" );
		
		this.rlen=that.rlen;
		this.clen=that.clen;
		this.sparse=sp;
		estimatedNNzsPerRow=(int)Math.ceil((double)thatValue.getNonZeros()/(double)rlen);
		if(this.sparse && that.sparse)
			copySparseToSparse(that);
		else if(this.sparse && !that.sparse)
			copyDenseToSparse(that);
		else if(!this.sparse && that.sparse)
			copySparseToDense(that);
		else
			copyDenseToDense(that);
	}
	
	private void copySparseToSparse(MatrixBlock that)
	{
		this.nonZeros=that.nonZeros;
		if( that.isEmptyBlock(false) )
		{
			resetSparse();
			return;
		}
	
		allocateSparseRowsBlock(false);
		for(int i=0; i<Math.min(that.sparseRows.length, rlen); i++)
		{
			if(that.sparseRows[i]!=null)
			{
				if(sparseRows[i]==null)
					sparseRows[i]=new SparseRow(that.sparseRows[i]);
				else
					sparseRows[i].copy(that.sparseRows[i]);
			}else if(this.sparseRows[i]!=null)
				this.sparseRows[i].reset(estimatedNNzsPerRow, clen);
		}
	}
	
	private void copyDenseToDense(MatrixBlock that)
	{
		nonZeros = that.nonZeros;
		int limit = rlen*clen;
		
		//plain reset to 0 for empty input
		if( that.isEmptyBlock(false) )
		{
			if(denseBlock!=null)
				Arrays.fill(denseBlock, 0, limit, 0);
			return;
		}
		
		//allocate and init dense block (w/o overwriting nnz)
		try {
			allocateDenseBlock(false);
		}
		catch(DMLRuntimeException e){
			throw new RuntimeException(e);
		}
		
		//actual copy 
		System.arraycopy(that.denseBlock, 0, denseBlock, 0, limit);
	}
	
	private void copySparseToDense(MatrixBlock that)
	{
		this.nonZeros=that.nonZeros;
		if( that.isEmptyBlock(false) )
		{
			if(denseBlock!=null)
				Arrays.fill(denseBlock, 0);
			return;
		}
		
		//allocate and init dense block (w/o overwriting nnz)
		try {
			allocateDenseBlock(false);
		}
		catch(DMLRuntimeException e){
			throw new RuntimeException(e);
		}
		
		int start=0;
		for(int r=0; r<Math.min(that.sparseRows.length, rlen); r++, start+=clen)
		{
			if(that.sparseRows[r]==null) 
				continue;
			double[] values=that.sparseRows[r].getValueContainer();
			int[] cols=that.sparseRows[r].getIndexContainer();
			for(int i=0; i<that.sparseRows[r].size(); i++)
			{
				denseBlock[start+cols[i]]=values[i];
			}
		}
	}
	
	private void copyDenseToSparse(MatrixBlock that)
	{
		nonZeros = that.nonZeros;
		if( that.isEmptyBlock(false) )
		{
			resetSparse();
			return;
		}
		
		allocateSparseRowsBlock(false);
	
		for(int i=0, ix=0; i<rlen; i++)
		{
			if( sparseRows[i]!=null ) 
				sparseRows[i].reset(estimatedNNzsPerRow, clen);
			
			for(int j=0; j<clen; j++)
			{
				double val = that.denseBlock[ix++];
				if( val != 0 )
				{
					if(sparseRows[i]==null) //create sparse row only if required
						sparseRows[i]=new SparseRow(estimatedNNzsPerRow, clen);
					sparseRows[i].append(j, val);
				}
			}
		}
	}
	
	
	/**
	 * In-place copy of matrix src into the index range of the existing current matrix.
	 * Note that removal of existing nnz in the index range and nnz maintenance is 
	 * only done if 'awareDestNZ=true', 
	 * 
	 * @param rl
	 * @param ru
	 * @param cl
	 * @param cu
	 * @param src
	 * @param awareDestNZ
	 *           true, forces (1) to remove existing non-zeros in the index range of the 
	 *                 destination if not present in src and (2) to internally maintain nnz
	 *           false, assume empty index range in destination and do not maintain nnz
	 *                  (the invoker is responsible to recompute nnz after all copies are done) 
	 * @throws DMLRuntimeException 
	 */
	public void copy(int rl, int ru, int cl, int cu, MatrixBlock src, boolean awareDestNZ ) 
		throws DMLRuntimeException 
	{	
		if(sparse && src.sparse)
			copySparseToSparse(rl, ru, cl, cu, src, awareDestNZ);
		else if(sparse && !src.sparse)
			copyDenseToSparse(rl, ru, cl, cu, src, awareDestNZ);
		else if(!sparse && src.sparse)
			copySparseToDense(rl, ru, cl, cu, src, awareDestNZ);
		else
			copyDenseToDense(rl, ru, cl, cu, src, awareDestNZ);
	}

	private void copySparseToSparse(int rl, int ru, int cl, int cu, MatrixBlock src, boolean awareDestNZ)
	{	
		//handle empty src and dest
		if( src.isEmptyBlock(false) )
		{
			if( awareDestNZ && sparseRows != null )
				copyEmptyToSparse(rl, ru, cl, cu, true);
			return;		
		}
		if(sparseRows==null)
			sparseRows=new SparseRow[rlen];
		else if( awareDestNZ )
		{
			copyEmptyToSparse(rl, ru, cl, cu, true);
			//explicit clear if awareDestNZ because more efficient since
			//src will have multiple columns and only few overwriting values
		}
		
		//copy values
		int alen;
		int[] aix;
		double[] avals;
		
		for( int i=0; i<src.rlen; i++ )
		{
			SparseRow arow = src.sparseRows[i];
			if( arow != null && !arow.isEmpty() )
			{
				alen = arow.size();
				aix = arow.getIndexContainer();
				avals = arow.getValueContainer();		
				
				if( sparseRows[rl+i] == null || sparseRows[rl+i].isEmpty()  )
				{
					sparseRows[rl+i] = new SparseRow(estimatedNNzsPerRow, clen); 
					SparseRow brow = sparseRows[rl+i];
					for( int j=0; j<alen; j++ )
						brow.append(cl+aix[j], avals[j]);
					
					if( awareDestNZ )
						nonZeros += brow.size();
				}
				else if( awareDestNZ ) //general case (w/ awareness NNZ)
				{
					SparseRow brow = sparseRows[rl+i];
					int lnnz = brow.size();
					if( cl==cu && cl==aix[0] ) 
					{
						if (avals[0]==0)
							brow.delete(cl);
						else
							brow.set(cl, avals[0] );
					}
					else
					{
						brow.deleteIndexRange(cl, cu);
						for( int j=0; j<alen; j++ )
							brow.set(cl+aix[j], avals[j]);
					}
					nonZeros += (brow.size() - lnnz);
				}	
				else //general case (w/o awareness NNZ)
				{		
					SparseRow brow = sparseRows[rl+i];

					//brow.set(cl, arow);	
					for( int j=0; j<alen; j++ )
						brow.set(cl+aix[j], avals[j]);
				}				
			}
		}
	}
	
	private void copySparseToDense(int rl, int ru, int cl, int cu, MatrixBlock src, boolean awareDestNZ) 
		throws DMLRuntimeException
	{	
		//handle empty src and dest
		if( src.isEmptyBlock(false) )
		{
			if( awareDestNZ && denseBlock != null ) {
				nonZeros -= recomputeNonZeros(rl, ru, cl, cu);
				copyEmptyToDense(rl, ru, cl, cu);
			}
			return;		
		}
		if(denseBlock==null)
			allocateDenseBlock();
		else if( awareDestNZ )
		{
			nonZeros -= recomputeNonZeros(rl, ru, cl, cu);
			copyEmptyToDense(rl, ru, cl, cu);
		}

		//copy values
		int alen;
		int[] aix;
		double[] avals;
		
		for( int i=0, ix=rl*clen; i<src.rlen; i++, ix+=clen )
		{	
			SparseRow arow = src.sparseRows[i];
			if( arow != null && !arow.isEmpty() )
			{
				alen = arow.size();
				aix = arow.getIndexContainer();
				avals = arow.getValueContainer();
				
				for( int j=0; j<alen; j++ )
					denseBlock[ix+cl+aix[j]] = avals[j];
				
				if(awareDestNZ)
					nonZeros += alen;
			}
		}
	}

	private void copyDenseToSparse(int rl, int ru, int cl, int cu, MatrixBlock src, boolean awareDestNZ)
	{
		//handle empty src and dest
		if( src.isEmptyBlock(false) )
		{
			if( awareDestNZ && sparseRows != null )
				copyEmptyToSparse(rl, ru, cl, cu, true);
			return;		
		}
		if(sparseRows==null)
			sparseRows=new SparseRow[rlen];
		//no need to clear for awareDestNZ since overwritten  
		
		//copy values
		double val;		
		for( int i=0, ix=0; i<src.rlen; i++, ix+=src.clen )
		{
			int rix = rl + i;
			if( sparseRows[rix]==null || sparseRows[rix].isEmpty() )
			{
				for( int j=0; j<src.clen; j++ )
					if( (val = src.denseBlock[ix+j]) != 0 )
					{
						if( sparseRows[rix]==null )
							sparseRows[rix] = new SparseRow(estimatedNNzsPerRow, clen); 
						sparseRows[rix].append(cl+j, val); 
					}
				
				if( awareDestNZ && sparseRows[rix]!=null )
					nonZeros += sparseRows[rix].size();
			}
			else if( awareDestNZ ) //general case (w/ awareness NNZ)
			{
				SparseRow brow = sparseRows[rix];
				int lnnz = brow.size();
				if( cl==cu ) 
				{
					if ((val = src.denseBlock[ix])==0)
						brow.delete(cl);
					else
						brow.set(cl, val);
				}
				else
				{
					brow.deleteIndexRange(cl, cu);
					for( int j=0; j<src.clen; j++ )
						if( (val = src.denseBlock[ix+j]) != 0 ) 
							brow.set(cl+j, val);	
				}
				nonZeros += (brow.size() - lnnz);
			}	
			else //general case (w/o awareness NNZ)
			{
				SparseRow brow = sparseRows[rix];
				for( int j=0; j<src.clen; j++ )
					if( (val = src.denseBlock[ix+j]) != 0 ) 
						brow.set(cl+j, val);
			}
		}
	}
	
	private void copyDenseToDense(int rl, int ru, int cl, int cu, MatrixBlock src, boolean awareDestNZ) 
		throws DMLRuntimeException
	{
		//handle empty src and dest
		if( src.isEmptyBlock(false) )
		{
			if( awareDestNZ && denseBlock != null ) {
				nonZeros -= recomputeNonZeros(rl, ru, cl, cu);
				copyEmptyToDense(rl, ru, cl, cu);
			}
			return;		
		}
		if(denseBlock==null)
			allocateDenseBlock();
		//no need to clear for awareDestNZ since overwritten 
	
		if( awareDestNZ )
			nonZeros = nonZeros - recomputeNonZeros(rl, ru, cl, cu) + src.nonZeros;
		
		//copy values
		int rowLen = cu-cl+1;				
		if(clen == src.clen) //optimization for equal width
			System.arraycopy(src.denseBlock, 0, denseBlock, rl*clen+cl, src.rlen*src.clen);
		else
			for( int i=0, ix1=0, ix2=rl*clen+cl; i<src.rlen; i++, ix1+=src.clen, ix2+=clen )
				System.arraycopy(src.denseBlock, ix1, denseBlock, ix2, rowLen);
	}
	
	private void copyEmptyToSparse(int rl, int ru, int cl, int cu, boolean updateNNZ ) 
	{
		if( cl==cu ) //specific case: column vector
		{
			if( updateNNZ )
			{
				for( int i=rl; i<=ru; i++ )
					if( sparseRows[i] != null && !sparseRows[i].isEmpty() )
					{
						int lnnz = sparseRows[i].size();
						sparseRows[i].delete(cl);
						nonZeros += (sparseRows[i].size()-lnnz);
					}
			}
			else
			{
				for( int i=rl; i<=ru; i++ )
					if( sparseRows[i] != null && !sparseRows[i].isEmpty() )
						sparseRows[i].delete(cl);
			}
		}
		else
		{
			if( updateNNZ )
			{
				for( int i=rl; i<=ru; i++ )
					if( sparseRows[i] != null && !sparseRows[i].isEmpty() )
					{
						int lnnz = sparseRows[i].size();
						sparseRows[i].deleteIndexRange(cl, cu);
						nonZeros += (sparseRows[i].size()-lnnz);
					}						
			}
			else
			{
				for( int i=rl; i<=ru; i++ )
					if( sparseRows[i] != null && !sparseRows[i].isEmpty() )
						sparseRows[i].deleteIndexRange(cl, cu);
			}
		}
	}
	
	private void copyEmptyToDense(int rl, int ru, int cl, int cu)
	{
		int rowLen = cu-cl+1;				
		if(clen == rowLen) //optimization for equal width
			Arrays.fill(denseBlock, rl*clen+cl, ru*clen+cu+1, 0);
		else
			for( int i=rl, ix2=rl*clen+cl; i<=ru; i++, ix2+=clen )
				Arrays.fill(denseBlock, ix2, ix2+rowLen, 0);
	}

	
	/**
	 * Merge disjoint: merges all non-zero values of the given input into the current
	 * matrix block. Note that this method does NOT check for overlapping entries;
	 * it's the callers reponsibility of ensuring disjoint matrix blocks.  
	 * 
	 * The appendOnly parameter is only relevant for sparse target blocks; if true,
	 * we only append values and do not sort sparse rows for each call; this is useful
	 * whenever we merge iterators of matrix blocks into one target block.
	 * 
	 * @param that
	 * @param appendOnly
	 * @throws DMLRuntimeException 
	 */
	public void merge(MatrixBlock that, boolean appendOnly) 
		throws DMLRuntimeException
	{
		//check for empty input source (nothing to merge)
		if( that == null || that.isEmptyBlock(false) )
			return;
		
		//check dimensions (before potentially copy to prevent implicit dimension change) 
		//this also does a best effort check for disjoint input blocks via the number of non-zeros
		if( rlen != that.rlen || clen != that.clen )
			throw new DMLRuntimeException("Dimension mismatch on merge disjoint (target="+rlen+"x"+clen+", source="+that.rlen+"x"+that.clen+")");
		if( (long)this.nonZeros+ that.nonZeros > (long)rlen*clen )
			throw new DMLRuntimeException("Number of non-zeros mismatch on merge disjoint (target="+rlen+"x"+clen+", nnz target="+nonZeros+", nnz source="+that.nonZeros+")");
		
		//check for empty target (copy in full)
		if( this.isEmptyBlock(false) ) {
			this.copy(that);
			return;
		}
		
		//core matrix block merge (guaranteed non-empty source/target, nnz maintenance not required)
		long nnz = this.nonZeros + that.nonZeros;
		if( sparse )
			this.mergeIntoSparse(that, appendOnly);
		else
			this.mergeIntoDense(that);	
		
		//maintain number of nonzeros
		this.nonZeros = nnz;
	}
	
	/**
	 * 
	 * @param that
	 */
	private void mergeIntoDense(MatrixBlock that)
	{
		if( that.sparse ) //DENSE <- SPARSE
		{
			SparseRow[] b = that.sparseRows;
			for( int i=0; i<rlen; i++ )
				if( b[i] != null && !b[i].isEmpty() )
				{
					SparseRow brow = b[i];
					int blen = brow.size();
					int[] bix = brow.getIndexContainer();
					double[] bval = brow.getValueContainer();
					for( int j=0; j<blen; j++ )
						if( bval[j] != 0 )
							this.quickSetValue(i, bix[j], bval[j]);
				}
		}
		else //DENSE <- DENSE
		{
			double[] a = this.denseBlock;
			double[] b = that.denseBlock;
			int len = rlen * clen;
			for( int i=0; i<len; i++ )
				a[i] = ( b[i] != 0 ) ? b[i] : a[i];
		}
	}
	
	/**
	 * 
	 * @param that
	 * @param appendOnly
	 */
	private void mergeIntoSparse(MatrixBlock that, boolean appendOnly)
	{
		if( that.sparse ) //SPARSE <- SPARSE
		{
			SparseRow[] a = this.sparseRows;
			SparseRow[] b = that.sparseRows;
			for( int i=0; i<rlen; i++ ) 
			{
				if( b[i] != null && !b[i].isEmpty() )
				{
					if( a[i] == null || a[i].isEmpty() ) {
						//copy entire sparse row (no sort required)
						a[i] = new SparseRow(b[i]); 
					}
					else
					{
						boolean appended = false;
						SparseRow brow = b[i];
						int blen = brow.size();
						int[] bix = brow.getIndexContainer();
						double[] bval = brow.getValueContainer();
						for( int j=0; j<blen; j++ ) {
							if( bval[j] != 0 ) {
								this.appendValue(i, bix[j], bval[j]);
								appended = true;
							}
						}
						//only sort if value appended
						if( !appendOnly && appended )
							this.sparseRows[i].sort();		
					}
				}
			}
		}
		else //SPARSE <- DENSE
		{
			double[] b = that.denseBlock;
			for( int i=0, bix=0; i<rlen; i++, bix+=clen )
			{
				boolean appended = false;
				for( int j=0; j<clen; j++ ) {
					if( b[bix+j] != 0 ) {
						this.appendValue(i, j, b[bix+j]);
						appended = true;
					}
				}
				//only sort if value appended
				if( !appendOnly && appended )
					this.sparseRows[i].sort();
			}
		}
	}
	
	////////
	// Input/Output functions
	
	@Override
	public void readFields(DataInput in) 
		throws IOException 
	{
		//read basic header (int rlen, int clen, byte type)
		rlen = in.readInt();
		clen = in.readInt();
		byte bformat = in.readByte();
		
		//check type information
		if( bformat<0 || bformat>=BlockType.values().length )
			throw new IOException("invalid format: '"+bformat+"' (need to be 0-"+BlockType.values().length+").");
			
		BlockType format=BlockType.values()[bformat];
		try 
		{
			switch(format)
			{
				case ULTRA_SPARSE_BLOCK:
					nonZeros = readNnzInfo( in, true );
					sparse = evalSparseFormatInMemory(rlen, clen, nonZeros);
					cleanupBlock(true, true); //clean all
					if( sparse )
						readUltraSparseBlock(in);
					else
						readUltraSparseToDense(in);
					break;
				case SPARSE_BLOCK:
					nonZeros = readNnzInfo( in, false );
					sparse = evalSparseFormatInMemory(rlen, clen, nonZeros);
					cleanupBlock(sparse, !sparse); 
					if( sparse )
						readSparseBlock(in);
					else
						readSparseToDense(in);
					break;
				case DENSE_BLOCK:
					sparse = false;
					cleanupBlock(false, true); //reuse dense
					readDenseBlock(in); //always dense in-mem if dense on disk
					break;
				case EMPTY_BLOCK:
					sparse = true;
					cleanupBlock(true, true); //clean all
					nonZeros = 0;
					break;
			}
		}
		catch(DMLRuntimeException ex)
		{
			throw new IOException("Error reading block of type '"+format.toString()+"'.", ex);
		}
	}

	/**
	 * 
	 * @param in
	 * @throws IOException
	 * @throws DMLRuntimeException 
	 */
	private void readDenseBlock(DataInput in) 
		throws IOException, DMLRuntimeException 
	{
		allocateDenseBlock(true); //allocate block, clear nnz
		
		int limit = rlen*clen;
		
		if( in instanceof MatrixBlockDataInput ) //fast deserialize
		{
			MatrixBlockDataInput mbin = (MatrixBlockDataInput)in;
			nonZeros = mbin.readDoubleArray(limit, denseBlock);
		}
		else if( in instanceof DataInputBuffer && MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION ) 
		{
			//workaround because sequencefile.reader.next(key, value) does not yet support serialization framework
			DataInputBuffer din = (DataInputBuffer)in;
			MatrixBlockDataInput mbin = new FastBufferedDataInputStream(din);
			nonZeros = mbin.readDoubleArray(limit, denseBlock);			
			((FastBufferedDataInputStream)mbin).close();
		}
		else //default deserialize
		{
			for( int i=0; i<limit; i++ )
			{
				denseBlock[i]=in.readDouble();
				if(denseBlock[i]!=0)
					nonZeros++;
			}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @throws IOException
	 */
	private void readSparseBlock(DataInput in) 
		throws IOException 
	{			
		allocateSparseRowsBlock(false); 
		resetSparse(); //reset all sparse rows
		
		if( in instanceof MatrixBlockDataInput ) //fast deserialize
		{
			MatrixBlockDataInput mbin = (MatrixBlockDataInput)in;
			nonZeros = mbin.readSparseRows(rlen, sparseRows);
		}
		else if( in instanceof DataInputBuffer  && MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION ) 
		{
			//workaround because sequencefile.reader.next(key, value) does not yet support serialization framework
			DataInputBuffer din = (DataInputBuffer)in;
			MatrixBlockDataInput mbin = new FastBufferedDataInputStream(din);
			nonZeros = mbin.readSparseRows(rlen, sparseRows);		
			((FastBufferedDataInputStream)mbin).close();
		}
		else //default deserialize
		{
			for(int r=0; r<rlen; r++)
			{
				int nr=in.readInt();
				if(nr==0)
				{
					if(sparseRows[r]!=null)
						sparseRows[r].reset(estimatedNNzsPerRow, clen);
					continue;
				}
				if(sparseRows[r]==null)
					sparseRows[r]=new SparseRow(nr);
				else
					sparseRows[r].reset(nr, clen);
				for(int j=0; j<nr; j++)
					sparseRows[r].append(in.readInt(), in.readDouble());
			}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @throws IOException
	 * @throws DMLRuntimeException 
	 */
	private void readSparseToDense(DataInput in) 
		throws IOException, DMLRuntimeException 
	{
		allocateDenseBlock(false); //allocate block
		Arrays.fill(denseBlock, 0);
		
		for(int r=0; r<rlen; r++)
		{
			int nr = in.readInt();
			for( int j=0; j<nr; j++ )
			{
				int c = in.readInt();
				double val = in.readDouble(); 
				denseBlock[r*clen+c] = val;
			}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @throws IOException
	 */
	private void readUltraSparseBlock(DataInput in) 
		throws IOException 
	{	
		allocateSparseRowsBlock(false); //adjust to size
		resetSparse(); //reset all sparse rows
		
		for(long i=0; i<nonZeros; i++)
		{
			int r = in.readInt();
			int c = in.readInt();
			double val = in.readDouble();			
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(1,clen);
			sparseRows[r].append(c, val);
		}
	}
	
	/**
	 * 
	 * @param in
	 * @throws IOException
	 * @throws DMLRuntimeException 
	 */
	private void readUltraSparseToDense(DataInput in) 
		throws IOException, DMLRuntimeException 
	{	
		allocateDenseBlock(false); //allocate block
		Arrays.fill(denseBlock, 0);
		
		for(long i=0; i<nonZeros; i++)
		{
			int r = in.readInt();
			int c = in.readInt();
			double val = in.readDouble();			
			denseBlock[r*clen+c] = val;
		}
	}
	
	@Override
	public void write(DataOutput out) 
		throws IOException 
	{
		//determine format
		boolean sparseSrc = sparse;
		boolean sparseDst = evalSparseFormatOnDisk();
		
		//write first part of header
		out.writeInt(rlen);
		out.writeInt(clen);
		
		if( sparseSrc )
		{
			//write sparse to *
			if( sparseRows==null || nonZeros==0 ) 
				writeEmptyBlock(out);
			else if( nonZeros<rlen && sparseDst ) 
				writeSparseToUltraSparse(out); 
			else if( sparseDst ) 
				writeSparseBlock(out);
			else
				writeSparseToDense(out);
		}
		else
		{
			//write dense to *
			if( denseBlock==null || nonZeros==0 ) 
				writeEmptyBlock(out);
			else if( nonZeros<rlen && sparseDst )
				writeDenseToUltraSparse(out);
			else if( sparseDst )
				writeDenseToSparse(out);
			else
				writeDenseBlock(out);
		}
	}
	
	/**
	 * 
	 * @param out
	 * @throws IOException
	 */
	private void writeEmptyBlock(DataOutput out) 
		throws IOException
	{
		//empty blocks do not need to materialize row information
		out.writeByte( BlockType.EMPTY_BLOCK.ordinal() );
	}
	
	/**
	 * 
	 * @param out
	 * @throws IOException
	 */
	private void writeDenseBlock(DataOutput out) 
		throws IOException 
	{
		out.writeByte( BlockType.DENSE_BLOCK.ordinal() );
		
		int limit=rlen*clen;
		if( out instanceof MatrixBlockDataOutput ) //fast serialize
			((MatrixBlockDataOutput)out).writeDoubleArray(limit, denseBlock);
		else //general case (if fast serialize not supported)
			for(int i=0; i<limit; i++)
				out.writeDouble(denseBlock[i]);
	}
	
	/**
	 * 
	 * @param out
	 * @throws IOException
	 */
	private void writeSparseBlock(DataOutput out) 
		throws IOException 
	{
		out.writeByte( BlockType.SPARSE_BLOCK.ordinal() );
		writeNnzInfo( out, false );
		
		if( out instanceof MatrixBlockDataOutput ) //fast serialize
			((MatrixBlockDataOutput)out).writeSparseRows(rlen, sparseRows);
		else //general case (if fast serialize not supported)
		{
			int r=0;
			for(;r<Math.min(rlen, sparseRows.length); r++)
			{
				if(sparseRows[r]==null)
					out.writeInt(0);
				else
				{
					int nr=sparseRows[r].size();
					out.writeInt(nr);
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					for(int j=0; j<nr; j++)
					{
						out.writeInt(cols[j]);
						out.writeDouble(values[j]);
					}
					
				}	
			}
			for(;r<rlen; r++)
				out.writeInt(0);
		}
	}
	
	/**
	 * 
	 * @param out
	 * @throws IOException
	 */
	private void writeSparseToUltraSparse(DataOutput out) 
		throws IOException 
	{
		out.writeByte( BlockType.ULTRA_SPARSE_BLOCK.ordinal() );
		writeNnzInfo( out, true );
		
		long wnnz = 0;
		for(int r=0;r<Math.min(rlen, sparseRows.length); r++)
			if(sparseRows[r]!=null && !sparseRows[r].isEmpty() )
			{
				int alen = sparseRows[r].size();
				int[] aix = sparseRows[r].getIndexContainer();
				double[] avals = sparseRows[r].getValueContainer();
				for(int j=0; j<alen; j++) {
					out.writeInt(r);
					out.writeInt(aix[j]);
					out.writeDouble(avals[j]);
					wnnz++;
				}
			}	
		
		//validity check (nnz must exactly match written nnz)
		if( nonZeros != wnnz ) {
			throw new IOException("Invalid number of serialized non-zeros: "+wnnz+" (expected: "+nonZeros+")");
		}
	}
	
	/**
	 * 
	 * @param out
	 * @throws IOException
	 */
	private void writeSparseToDense(DataOutput out) 
		throws IOException 
	{
		//write block type 'dense'
		out.writeByte( BlockType.DENSE_BLOCK.ordinal() );
		
		//write data (from sparse to dense)
		if( sparseRows==null ) //empty block
			for( int i=0; i<rlen*clen; i++ )
				out.writeDouble(0);
		else //existing sparse block
		{
			for( int i=0; i<rlen; i++ )
			{
				if( i<sparseRows.length && sparseRows[i]!=null && !sparseRows[i].isEmpty() )
				{
					SparseRow arow = sparseRows[i];
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();
					//foreach non-zero value, fill with 0s if required
					for( int j=0, j2=0; j2<alen; j++, j2++ ) {
						for( ; j<aix[j2]; j++ )
							out.writeDouble( 0 );
						out.writeDouble( avals[j2] );
					}					
					//remaining 0 values in row
					for( int j=aix[alen-1]+1; j<clen; j++)
						out.writeDouble( 0 );
				}
				else //empty row
					for( int j=0; j<clen; j++ )
						out.writeDouble( 0 );	
			}
		}
	}
	
	/**
	 * 
	 * @param out
	 * @throws IOException
	 */
	private void writeDenseToUltraSparse(DataOutput out) throws IOException 
	{
		out.writeByte( BlockType.ULTRA_SPARSE_BLOCK.ordinal() );
		writeNnzInfo( out, true );

		long wnnz = 0;
		for(int r=0, ix=0; r<rlen; r++)
			for(int c=0; c<clen; c++, ix++)
				if( denseBlock[ix]!=0 )
				{
					out.writeInt(r);
					out.writeInt(c);
					out.writeDouble(denseBlock[ix]);
					wnnz++;
				}
		
		//validity check (nnz must exactly match written nnz)
		if( nonZeros != wnnz ) {
			throw new IOException("Invalid number of serialized non-zeros: "+wnnz+" (expected: "+nonZeros+")");
		}
	}
	
	/**
	 * 
	 * @param out
	 * @throws IOException
	 */
	private void writeDenseToSparse(DataOutput out) 
		throws IOException 
	{	
		out.writeByte( BlockType.SPARSE_BLOCK.ordinal() ); //block type
		writeNnzInfo( out, false );
		
		int start=0;
		for(int r=0; r<rlen; r++)
		{
			//count nonzeros
			int nr=0;
			for(int i=start; i<start+clen; i++)
				if(denseBlock[i]!=0.0)
					nr++;
			out.writeInt(nr);
			for(int c=0; c<clen; c++)
			{
				if(denseBlock[start]!=0.0)
				{
					out.writeInt(c);
					out.writeDouble(denseBlock[start]);
				}
				start++;
			}
		}
	}

	/**
	 * 
	 * @param in
	 * @throws IOException
	 */
	private long readNnzInfo( DataInput in, boolean ultrasparse ) 
		throws IOException
	{
		//note: if ultrasparse, int always sufficient because nnz<rlen
		// where rlen is limited to integer
				
		long lrlen = (long)rlen;
		long lclen = (long)clen;
		
		//read long if required, otherwise int (see writeNnzInfo, consistency required)
		if( lrlen*lclen > Integer.MAX_VALUE  && !ultrasparse) {
			nonZeros = in.readLong();
		}
		else {
			nonZeros = in.readInt(); 
		}
		
		return nonZeros;
	}
	
	/**
	 * 
	 * @param out
	 * @throws IOException
	 */
	private void writeNnzInfo( DataOutput out, boolean ultrasparse ) 
		throws IOException
	{
		//note: if ultrasparse, int always sufficient because nnz<rlen
		// where rlen is limited to integer
		
		long lrlen = (long)rlen;
		long lclen = (long)clen;
		
		//write long if required, otherwise int
		if( lrlen*lclen > Integer.MAX_VALUE && !ultrasparse) {
			out.writeLong( nonZeros ); 
		}
		else {
			out.writeInt( (int)nonZeros );
		}
	}
	
	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for efficient broadcast/rdd deserialization. 
	 * 
	 * @param is
	 * @throws IOException
	 */
	public void readExternal(ObjectInput is) 
		throws IOException
	{
		if( is instanceof ObjectInputStream )
		{
			//fast deserialize of dense/sparse blocks
			ObjectInputStream ois = (ObjectInputStream)is;
			FastBufferedDataInputStream fis = new FastBufferedDataInputStream(ois);
			readFields(fis);
		}
		else {
			//default deserialize (general case)
			readFields(is);
		}
	}
	
	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for efficient broadcast/rdd serialization. 
	 * 
	 * @param is
	 * @throws IOException
	 */
	public void writeExternal(ObjectOutput os) 
		throws IOException
	{
		if( os instanceof ObjectOutputStream ) {
			//fast serialize of dense/sparse blocks
			ObjectOutputStream oos = (ObjectOutputStream)os;
			FastBufferedDataOutputStream fos = new FastBufferedDataOutputStream(oos);
			write(fos);
			fos.flush();
		}
		else {
			//default serialize (general case)
			write(os);	
		}
	}
	
	/**
	 * NOTE: The used estimates must be kept consistent with the respective write functions. 
	 * 
	 * @return
	 */
	public long getExactSizeOnDisk()
	{
		//determine format
		boolean sparseSrc = sparse;
		boolean sparseDst = evalSparseFormatOnDisk();
		
		long lrlen = (long) rlen;
		long lclen = (long) clen;
		long lnonZeros = (long) nonZeros;
		
		//ensure exact size estimates for write
		if( lnonZeros <= 0 )
		{
			recomputeNonZeros();
			lnonZeros = (long) nonZeros;
		}
				
		//get exact size estimate (see write for the corresponding meaning)
		if( sparseSrc )
		{
			//write sparse to *
			if(sparseRows==null || lnonZeros==0)
				return HEADER_SIZE; //empty block
			else if( lnonZeros<lrlen && sparseDst )
				return estimateSizeUltraSparseOnDisk(lrlen, lclen, lnonZeros); //ultra sparse block
			else if( sparseDst )
				return estimateSizeSparseOnDisk(lrlen, lclen, lnonZeros); //sparse block
			else 
				return estimateSizeDenseOnDisk(lrlen, lclen); //dense block
		}
		else
		{
			//write dense to *
			if(denseBlock==null || lnonZeros==0)
				return HEADER_SIZE; //empty block
			else if( lnonZeros<lrlen && sparseDst )
				return estimateSizeUltraSparseOnDisk(lrlen, lclen, lnonZeros); //ultra sparse block
			else if( sparseDst )
				return estimateSizeSparseOnDisk(lrlen, lclen, lnonZeros); //sparse block
			else
				return estimateSizeDenseOnDisk(lrlen, lclen); //dense block
		}
	}
	
	////////
	// Estimates size and sparsity
	
	/**
	 * 
	 * @param nrows
	 * @param ncols
	 * @param sparsity
	 * @return
	 */
	public static long estimateSizeInMemory(long nrows, long ncols, double sparsity)
	{
		//determine sparse/dense representation
		boolean sparse = evalSparseFormatInMemory(nrows, ncols, (long)(sparsity*nrows*ncols));
		
		//estimate memory consumption for sparse/dense
		if( sparse )
			return estimateSizeSparseInMemory(nrows, ncols, sparsity);
		else
			return estimateSizeDenseInMemory(nrows, ncols);
	}
	
	/**
	 * 
	 * @param nrows
	 * @param ncols
	 * @return
	 */
	private static long estimateSizeDenseInMemory(long nrows, long ncols)
	{
		// basic variables and references sizes
		long size = 44;
		
		// core dense matrix block (double array)
		size += nrows * ncols * 8;
		
		return size;
	}
	
	/**
	 * 
	 * @param nrows
	 * @param ncols
	 * @param sparsity
	 * @return
	 */
	private static long estimateSizeSparseInMemory(long nrows, long ncols, double sparsity)
	{
		// basic variables and references sizes
		long size = 44;
		
		//NOTES:
		// * Each sparse row has a fixed overhead of 8B (reference) + 32B (object) +
		//   12B (3 int members), 32B (overhead int array), 32B (overhead double array),
		// * Each non-zero value requires 12B for the column-index/value pair.
		// * Overheads for arrays, objects, and references refer to 64bit JVMs
		// * If nnz < than rows we have only also empty rows.
		
		//account for sparsity and initial capacity
		long cnnz = Math.max(SparseRow.initialCapacity, (long)Math.ceil(sparsity*ncols));
		long rlen = Math.min(nrows, (long) Math.ceil(sparsity*nrows*ncols));
		size += rlen * ( 116 + 12 * cnnz ); //sparse row
		size += nrows * 8; //empty rows
		
		//OLD ESTIMATE: 
		//int len = Math.max(SparseRow.initialCapacity, (int)Math.ceil(sparsity*ncols));
		//size += nrows * (28 + 12 * len );
		
		return size;
	}
	
	/**
	 * 
	 * @param nrows
	 * @param ncols
	 * @param sparsity
	 * @return
	 */
	public static long estimateSizeOnDisk( long nrows, long ncols, long nnz )
	{
		//determine sparse/dense representation
		boolean sparse = evalSparseFormatOnDisk(nrows, ncols, nnz);
		
		//estimate memory consumption for sparse/dense 
		if( sparse && nnz<nrows )
			return estimateSizeUltraSparseOnDisk(nrows, ncols, nnz);
		else if( sparse )
			return estimateSizeSparseOnDisk(nrows, ncols, nnz);
		else
			return estimateSizeDenseOnDisk(nrows, ncols);
	}
	
	/**
	 * 
	 * @param nrows
	 * @param ncols
	 * @param sparsity
	 * @return
	 */
	private static long estimateSizeDenseOnDisk( long nrows, long ncols)
	{
		//basic header (int rlen, int clen, byte type) 
		long size = HEADER_SIZE;
		//data (all cells double)
		size += nrows * ncols * 8;

		return size;
	}
	
	/**
	 * 
	 * @param nrows
	 * @param ncols
	 * @param sparsity
	 * @return
	 */
	private static long estimateSizeSparseOnDisk( long nrows, long ncols, long nnz )
	{
		//basic header: (int rlen, int clen, byte type) 
		long size = HEADER_SIZE;
		//extended header (long nnz)
		size += (nrows*ncols > Integer.MAX_VALUE) ? 8 : 4;
		//data: (int num per row, int-double pair per non-zero value)
		size += nrows * 4 + nnz * 12;	

		return size;
	}
	
	/**
	 * 
	 * @param nrows
	 * @param ncols
	 * @param sparsity
	 * @return
	 */
	private static long estimateSizeUltraSparseOnDisk( long nrows, long ncols, long nnz )
	{
		//basic header (int rlen, int clen, byte type) 
		long size = HEADER_SIZE;
		//extended header (int nnz, guaranteed by rlen<nnz)
		size += 4;
		//data (int-int-double triples per non-zero value)
		size += nnz * 16;	
		
		return size;
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param op
	 * @return
	 */
	public static SparsityEstimate estimateSparsityOnAggBinary(MatrixBlock m1, MatrixBlock m2, AggregateBinaryOperator op)
	{
		//Since MatrixMultLib always uses a dense output (except for ultra-sparse mm)
		//with subsequent check for sparsity, we should always return a dense estimate.
		//Once, we support more aggregate binary operations, we need to change this.
		
		//WARNING: KEEP CONSISTENT WITH LIBMATRIXMULT
		//Note that it is crucial to report the right output representation because
		//in case of block reuse (e.g., mmcj) the output 'reset' refers to either
		//dense or sparse representation and hence would produce incorrect results
		//if we report the wrong representation (i.e., missing reset on ultrasparse mm). 
		
		boolean ultrasparse = (m1.isUltraSparse() || m2.isUltraSparse());
		return new SparsityEstimate(ultrasparse, m1.getNumRows()*m2.getNumRows());
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param op
	 * @return
	 */
	private static SparsityEstimate estimateSparsityOnBinary(MatrixBlock m1, MatrixBlock m2, BinaryOperator op)
	{
		SparsityEstimate est = new SparsityEstimate();
		
		//estimate dense output for all sparse-unsafe operations, except DIV (because it commonly behaves like
		//sparse-safe but is not due to 0/0->NaN, this is consistent with the current hop sparsity estimate)
		//see also, special sparse-safe case for DIV in LibMatrixBincell 
		if( !op.sparseSafe && !(op.fn instanceof Divide) ) {
			est.sparse = false;
			return est;
		}
		
		BinaryAccessType atype = LibMatrixBincell.getBinaryAccessType(m1, m2);
		boolean outer = (atype == BinaryAccessType.OUTER_VECTOR_VECTOR);
		long m = m1.getNumRows();
		long n = outer ? m2.getNumColumns() : m1.getNumColumns();
		long nz1 = m1.getNonZeros();
		long nz2 = m2.getNonZeros();
		
		//account for matrix vector and vector/vector
		long estnnz = 0;
		if( atype == BinaryAccessType.OUTER_VECTOR_VECTOR )
		{
			//for outer vector operations the sparsity estimate is exactly known
			estnnz = nz1 * nz2; 
		}
		else //DEFAULT CASE
		{		
			if( atype == BinaryAccessType.MATRIX_COL_VECTOR )
				nz2 = nz2 * n;
			else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )
				nz2 = nz2 * m;
		
			//compute output sparsity consistent w/ the hop compiler
			OpOp2 bop = op.getBinaryOperatorOpOp2();
			double sp1 = OptimizerUtils.getSparsity(m, n, nz1);
			double sp2 = OptimizerUtils.getSparsity(m, n, nz2);
			double spout = OptimizerUtils.getBinaryOpSparsity(sp1, sp2, bop, true);
			estnnz = UtilFunctions.toLong(spout * m * n);
		}
		
		est.sparse = evalSparseFormatInMemory(m, n, estnnz);
		est.estimatedNonZeros = estnnz;
		
		return est;
	}
	
	private boolean estimateSparsityOnSlice(int selectRlen, int selectClen, int finalRlen, int finalClen)
	{
		long ennz = (long)((double)nonZeros/rlen/clen*selectRlen*selectClen);
		return evalSparseFormatInMemory(finalRlen, finalClen, ennz); 
	}
	
	private boolean estimateSparsityOnLeftIndexing(long rlenm1, long clenm1, long nnzm1, long rlenm2, long clenm2, long nnzm2)
	{
		//min bound: nnzm1 - rlenm2*clenm2 + nnzm2
		//max bound: min(rlenm1*rlenm2, nnzm1+nnzm2)
		
		long ennz = Math.min(rlenm1*clenm1, nnzm1+nnzm2);
		return evalSparseFormatInMemory(rlenm1, clenm1, ennz);
	}
	
	private boolean estimateSparsityOnGroupedAgg( long rlen, long groups )
	{
		long ennz = Math.min(groups, rlen);
		return evalSparseFormatInMemory(groups, 1, ennz);
	}
	
	////////
	// Core block operations (called from instructions)
	
	public MatrixValue scalarOperations(ScalarOperator op, MatrixValue result) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlock ret = checkType(result);
		
		// estimate the sparsity structure of result matrix
		boolean sp = this.sparse; // by default, we guess result.sparsity=input.sparsity
		if (!op.sparseSafe)
			sp = false; // if the operation is not sparse safe, then result will be in dense format
		
		//allocate the output matrix block
		if( ret==null )
			ret = new MatrixBlock(rlen, clen, sp, this.nonZeros);
		else
			ret.reset(rlen, clen, sp, this.nonZeros);
		
		//core scalar operations
		LibMatrixBincell.bincellOp(this, ret, op);
		
		return ret;
	}
		
	public MatrixValue unaryOperations(UnaryOperator op, MatrixValue result) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlock ret = checkType(result);
		
		// estimate the sparsity structure of result matrix
		boolean sp = this.sparse; // by default, we guess result.sparsity=input.sparsity
		if (!op.sparseSafe)
			sp = false; // if the operation is not sparse safe, then result will be in dense format
		
		//allocate output
		if( ret == null )
			ret = new MatrixBlock(rlen, clen, sp, this.nonZeros);
		else 
			ret.reset(rlen, clen, sp);
		
		//core execute
		if( LibMatrixAgg.isSupportedUnaryOperator(op) ) 
		{
			//e.g., cumsum/cumprod/cummin/cumax
			LibMatrixAgg.aggregateUnaryMatrix(this, ret, op);
		}
		else
		{
			//copy input into target dense/sparse representation
			ret.copy(this, sp);
			
			//compute unary operations in-place of target representation
			ret.unaryOperationsInPlace(op);
		}
		
		return ret;
	}
	
	public void unaryOperationsInPlace(UnaryOperator op) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(op.sparseSafe)
			sparseUnaryOperationsInPlace(op);
		else
			denseUnaryOperationsInPlace(op);
	}
	
	/**
	 * only apply to non zero cells
	 * 
	 * @param op
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private void sparseUnaryOperationsInPlace(UnaryOperator op) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//early abort possible since sparse-safe
		if( isEmptyBlock(false) )
			return;
		
		if(sparse)
		{
			nonZeros=0;
			for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
			{
				if(sparseRows[r]==null) 
					continue;
				double[] values=sparseRows[r].getValueContainer();
				int[] cols=sparseRows[r].getIndexContainer();
				int pos=0;
				for(int i=0; i<sparseRows[r].size(); i++)
				{
					double v=op.fn.execute(values[i]);
					if(v!=0)
					{
						values[pos]=v;
						cols[pos]=cols[i];
						pos++;
						nonZeros++;
					}
				}
				sparseRows[r].truncate(pos);
			}
			
		}
		else
		{
			int limit=rlen*clen;
			nonZeros=0;
			for(int i=0; i<limit; i++)
			{
				denseBlock[i]=op.fn.execute(denseBlock[i]);
				if(denseBlock[i]!=0)
					nonZeros++;
			}
		}
	}
	
	private void denseUnaryOperationsInPlace(UnaryOperator op) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if( sparse ) //SPARSE MATRIX
		{
			double v;
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					v=op.fn.execute(quickGetValue(r, c));
					quickSetValue(r, c, v);
				}
		}
		else//DENSE MATRIX
		{
			//early abort not possible because not sparsesafe
			if(denseBlock==null)
				allocateDenseBlock();
			
			//compute values in-place and update nnz
			final int limit = rlen*clen;
			int lnnz = 0;
			for( int i=0; i<limit; i++ ) {
				denseBlock[i] = op.fn.execute(denseBlock[i]);	
				if( denseBlock[i]!=0 )
					lnnz++;
			}		
			nonZeros = lnnz;
			
			//IBM JVM bug (JDK6) causes crash for certain inputs (w/ infinities) 
			//nonZeros = 0;
			//for(int i=0; i<limit; i++)
			//{
			//	denseBlock[i]=op.fn.execute(denseBlock[i]);
			//	if(denseBlock[i]!=0)
			//		nonZeros++;
			//}
		}
	}
	
	/**
	 * 
	 */
	public MatrixValue binaryOperations(BinaryOperator op, MatrixValue thatValue, MatrixValue result) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlock that = checkType(thatValue);
		MatrixBlock ret = checkType(result);
		if( !LibMatrixBincell.isValidDimensionsBinary(this, that) ) {
			throw new RuntimeException("block sizes are not matched for binary " +
					"cell operations: "+this.rlen+"x"+this.clen+" vs "+ that.rlen+"x"+that.clen);
		}
		
		//compute output dimensions
		boolean outer = (LibMatrixBincell.getBinaryAccessType(this, that)
				         == BinaryAccessType.OUTER_VECTOR_VECTOR); 
		int rows = rlen;
		int cols = outer ? that.clen : clen;
		
		//estimate output sparsity
		SparsityEstimate resultSparse = estimateSparsityOnBinary(this, that, op);
		if( ret == null )
			ret = new MatrixBlock(rows, cols, resultSparse.sparse, resultSparse.estimatedNonZeros);
		else
			ret.reset(rows, cols, resultSparse.sparse, resultSparse.estimatedNonZeros);
		
		//core binary cell operation
		LibMatrixBincell.bincellOp( this, that, ret, op );
		
		return ret;
	}
	
	/**
	 * 
	 */
	public void binaryOperationsInPlace(BinaryOperator op, MatrixValue thatValue) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlock that=checkType(thatValue);
		if( !LibMatrixBincell.isValidDimensionsBinary(this, that) ) {
			throw new RuntimeException("block sizes are not matched for binary " +
					"cell operations: "+this.rlen+"*"+this.clen+" vs "+ that.rlen+"*"+that.clen);
		}
	
		//estimate output sparsity
		SparsityEstimate resultSparse = estimateSparsityOnBinary(this, that, op);
		if(resultSparse.sparse && !this.sparse)
			denseToSparse();
		else if(!resultSparse.sparse && this.sparse)
			sparseToDense();
				
		//core binary cell operation
		LibMatrixBincell.bincellOpInPlace(this, that, op);
	}


	
	public void incrementalAggregate(AggregateOperator aggOp, MatrixValue correction, 
			MatrixValue newWithCorrection)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//assert(aggOp.correctionExists); 
		MatrixBlock cor=checkType(correction);
		MatrixBlock newWithCor=checkType(newWithCorrection);
		KahanObject buffer=new KahanObject(0, 0);
		
		if(aggOp.correctionLocation==CorrectionLocationType.LASTROW)
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					buffer._sum=this.quickGetValue(r, c);
					buffer._correction=cor.quickGetValue(0, c);
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.quickGetValue(r, c), 
							newWithCor.quickGetValue(r+1, c));
					quickSetValue(r, c, buffer._sum);
					cor.quickSetValue(0, c, buffer._correction);
				}
			
		}else if(aggOp.correctionLocation==CorrectionLocationType.LASTCOLUMN)
		{
			if(aggOp.increOp.fn instanceof Builtin 
			   && ( ((Builtin)(aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MAXINDEX
			         || ((Builtin)(aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MININDEX )
			         ){
					// *** HACK ALERT *** HACK ALERT *** HACK ALERT ***
					// rowIndexMax() and its siblings don't fit very well into the standard
					// aggregate framework. We (ab)use the "correction factor" argument to
					// hold the maximum value in each row/column.
				
					// The execute() method for this aggregate takes as its argument
					// two candidates for the highest value. Bookkeeping about
					// indexes (return column/row index with highest value, breaking
					// ties in favor of higher indexes) is handled in this function.
					// Note that both versions of incrementalAggregate() contain
					// very similar blocks of special-case code. If one block is
					// modified, the other needs to be changed to match.
					for(int r=0; r<rlen; r++){
						double currMaxValue = cor.quickGetValue(r, 0);
						long newMaxIndex = (long)newWithCor.quickGetValue(r, 0);
						double newMaxValue = newWithCor.quickGetValue(r, 1);
						double update = aggOp.increOp.fn.execute(newMaxValue, currMaxValue);
						
						if (2.0 == update) {
							// Return value of 2 ==> both values the same, break ties
							// in favor of higher index.
							long curMaxIndex = (long) quickGetValue(r,0);
							quickSetValue(r, 0, Math.max(curMaxIndex, newMaxIndex));
						} else if(1.0 == update){
							// Return value of 1 ==> new value is better; use its index
							quickSetValue(r, 0, newMaxIndex);
							cor.quickSetValue(r, 0, newMaxValue);
						} else {
							// Other return value ==> current answer is best
						}
					}
					// *** END HACK ***
				}else{
					for(int r=0; r<rlen; r++)
						for(int c=0; c<clen; c++)
						{
							buffer._sum=this.quickGetValue(r, c);
							buffer._correction=cor.quickGetValue(r, 0);;
							buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.quickGetValue(r, c), newWithCor.quickGetValue(r, c+1));
							quickSetValue(r, c, buffer._sum);
							cor.quickSetValue(r, 0, buffer._correction);
						}
				}
		}
		else if(aggOp.correctionLocation==CorrectionLocationType.NONE)
		{
			
			//e.g., ak+ kahan plus as used in sum, mapmult, mmcj and tsmm
			if(aggOp.increOp.fn instanceof KahanPlus)
			{
				LibMatrixAgg.aggregateBinaryMatrix(newWithCor, this, cor);
			}
			else
			{
				if( newWithCor.isInSparseFormat() && aggOp.sparseSafe ) //SPARSE
				{
					SparseRow[] bRows = newWithCor.getSparseRows();
					if( bRows==null ) //early abort on empty block
						return;
					for( int r=0; r<Math.min(rlen, bRows.length); r++ )
					{
						SparseRow brow = bRows[r];
						if( brow != null && !brow.isEmpty() ) 
						{
							int blen = brow.size();
							int[] bix = brow.getIndexContainer();
							double[] bvals = brow.getValueContainer();
							for( int j=0; j<blen; j++)
							{
								int c = bix[j];
								buffer._sum = this.quickGetValue(r, c);
								buffer._correction = cor.quickGetValue(r, c);
								buffer = (KahanObject) aggOp.increOp.fn.execute(buffer, bvals[j]);
								quickSetValue(r, c, buffer._sum);
								cor.quickSetValue(r, c, buffer._correction);			
							}
						}
					}
					
				}
				else //DENSE or SPARSE (!sparsesafe)
				{
					for(int r=0; r<rlen; r++)
						for(int c=0; c<clen; c++)
						{
							buffer._sum=this.quickGetValue(r, c);
							buffer._correction=cor.quickGetValue(r, c);
							buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.quickGetValue(r, c));
							quickSetValue(r, c, buffer._sum);
							cor.quickSetValue(r, c, buffer._correction);
						}
				}
			
				//change representation if required
				//(note since ak+ on blocks is currently only applied in MR, hence no need to account for this in mem estimates)
				examSparsity(); 
			}
		}
		else if(aggOp.correctionLocation==CorrectionLocationType.LASTTWOROWS)
		{
			double n, n2, mu2;
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					buffer._sum=this.quickGetValue(r, c);
					n=cor.quickGetValue(0, c);
					buffer._correction=cor.quickGetValue(1, c);
					mu2=newWithCor.quickGetValue(r, c);
					n2=newWithCor.quickGetValue(r+1, c);
					n=n+n2;
					double toadd=(mu2-buffer._sum)*n2/n;
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
					quickSetValue(r, c, buffer._sum);
					cor.quickSetValue(0, c, n);
					cor.quickSetValue(1, c, buffer._correction);
				}
			
		}else if(aggOp.correctionLocation==CorrectionLocationType.LASTTWOCOLUMNS)
		{
			double n, n2, mu2;
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					buffer._sum=this.quickGetValue(r, c);
					n=cor.quickGetValue(r, 0);
					buffer._correction=cor.quickGetValue(r, 1);
					mu2=newWithCor.quickGetValue(r, c);
					n2=newWithCor.quickGetValue(r, c+1);
					n=n+n2;
					double toadd=(mu2-buffer._sum)*n2/n;
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
					quickSetValue(r, c, buffer._sum);
					cor.quickSetValue(r, 0, n);
					cor.quickSetValue(r, 1, buffer._correction);
				}
		}
		else
			throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
	}
	
	public void incrementalAggregate(AggregateOperator aggOp, MatrixValue newWithCorrection)
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//assert(aggOp.correctionExists);
		MatrixBlock newWithCor=checkType(newWithCorrection);
		KahanObject buffer=new KahanObject(0, 0);
		
		if(aggOp.correctionLocation==CorrectionLocationType.LASTROW)
		{			
			if( aggOp.increOp.fn instanceof KahanPlus )
			{
				LibMatrixAgg.aggregateBinaryMatrix(newWithCor, this, aggOp);
			}
			else
			{
				for(int r=0; r<rlen-1; r++)
					for(int c=0; c<clen; c++)
					{
						buffer._sum=this.quickGetValue(r, c);
						buffer._correction=this.quickGetValue(r+1, c);
						buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.quickGetValue(r, c), 
								newWithCor.quickGetValue(r+1, c));
						quickSetValue(r, c, buffer._sum);
						quickSetValue(r+1, c, buffer._correction);
					}
			}	
		}
		else if(aggOp.correctionLocation==CorrectionLocationType.LASTCOLUMN)
		{
			if(aggOp.increOp.fn instanceof Builtin 
			   && ( ((Builtin)(aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MAXINDEX 
			        || ((Builtin)(aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MININDEX) 
			        ){
				// *** HACK ALERT *** HACK ALERT *** HACK ALERT ***
				// rowIndexMax() and its siblings don't fit very well into the standard
				// aggregate framework. We (ab)use the "correction factor" argument to
				// hold the maximum value in each row/column.
			
				// The execute() method for this aggregate takes as its argument
				// two candidates for the highest value. Bookkeeping about
				// indexes (return column/row index with highest value, breaking
				// ties in favor of higher indexes) is handled in this function.
				// Note that both versions of incrementalAggregate() contain
				// very similar blocks of special-case code. If one block is
				// modified, the other needs to be changed to match.
				for(int r = 0; r < rlen; r++){
					double currMaxValue = quickGetValue(r, 1);
					long newMaxIndex = (long)newWithCor.quickGetValue(r, 0);
					double newMaxValue = newWithCor.quickGetValue(r, 1);
					double update = aggOp.increOp.fn.execute(newMaxValue, currMaxValue);
	
					if (2.0 == update) {
						// Return value of 2 ==> both values the same, break ties
						// in favor of higher index.
						long curMaxIndex = (long) quickGetValue(r,0);
						quickSetValue(r, 0, Math.max(curMaxIndex, newMaxIndex));
					} else if(1.0 == update){
						// Return value of 1 ==> new value is better; use its index
						quickSetValue(r, 0, newMaxIndex);
						quickSetValue(r, 1, newMaxValue);
					} else {
						// Other return value ==> current answer is best
					}
				}
				// *** END HACK ***
			}
			else
			{
				if(aggOp.increOp.fn instanceof KahanPlus)
				{
					LibMatrixAgg.aggregateBinaryMatrix(newWithCor, this, aggOp);
				}
				else
				{
					for(int r=0; r<rlen; r++)
						for(int c=0; c<clen-1; c++)
						{
							buffer._sum=this.quickGetValue(r, c);
							buffer._correction=this.quickGetValue(r, c+1);
							buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.quickGetValue(r, c), newWithCor.quickGetValue(r, c+1));
							quickSetValue(r, c, buffer._sum);
							quickSetValue(r, c+1, buffer._correction);
						}
				}
			}
		}/*else if(aggOp.correctionLocation==0)
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					//buffer._sum=this.getValue(r, c);
					//buffer._correction=0;
					//buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.getValue(r, c));
					setValue(r, c, this.getValue(r, c)+newWithCor.getValue(r, c));
				}
		}*/else if(aggOp.correctionLocation==CorrectionLocationType.LASTTWOROWS)
		{
			double n, n2, mu2;
			for(int r=0; r<rlen-2; r++)
				for(int c=0; c<clen; c++)
				{
					buffer._sum=this.quickGetValue(r, c);
					n=this.quickGetValue(r+1, c);
					buffer._correction=this.quickGetValue(r+2, c);
					mu2=newWithCor.quickGetValue(r, c);
					n2=newWithCor.quickGetValue(r+1, c);
					n=n+n2;
					double toadd=(mu2-buffer._sum)*n2/n;
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
					quickSetValue(r, c, buffer._sum);
					quickSetValue(r+1, c, n);
					quickSetValue(r+2, c, buffer._correction);
				}
			
		}else if(aggOp.correctionLocation==CorrectionLocationType.LASTTWOCOLUMNS)
		{
			double n, n2, mu2;
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen-2; c++)
				{
					buffer._sum=this.quickGetValue(r, c);
					n=this.quickGetValue(r, c+1);
					buffer._correction=this.quickGetValue(r, c+2);
					mu2=newWithCor.quickGetValue(r, c);
					n2=newWithCor.quickGetValue(r, c+1);
					n=n+n2;
					double toadd=(mu2-buffer._sum)*n2/n;
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
					quickSetValue(r, c, buffer._sum);
					quickSetValue(r, c+1, n);
					quickSetValue(r, c+2, buffer._correction);
				}
		}
		else
			throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
	}

	@Override
	public MatrixValue reorgOperations(ReorgOperator op, MatrixValue ret, int startRow, int startColumn, int length)
		throws DMLRuntimeException 
	{
		if ( !( op.fn instanceof SwapIndex || op.fn instanceof DiagIndex || op.fn instanceof SortIndex) )
			throw new DMLRuntimeException("the current reorgOperations cannot support: "+op.fn.getClass()+".");
		
		MatrixBlock result=checkType(ret);
		CellIndex tempCellIndex = new CellIndex(-1,-1);
		boolean reducedDim=op.fn.computeDimension(rlen, clen, tempCellIndex);
		boolean sps;
		if(reducedDim)
			sps = false;
		else if(op.fn.equals(DiagIndex.getDiagIndexFnObject()))
			sps = true;
		else
			sps = this.evalSparseFormatInMemory(true);
		
		if(result==null)
			result=new MatrixBlock(tempCellIndex.row, tempCellIndex.column, sps, this.nonZeros);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, sps, this.nonZeros);
		
		if( LibMatrixReorg.isSupportedReorgOperator(op) )
		{
			//SPECIAL case (operators with special performance requirements, 
			//or size-dependent special behavior)
			//currently supported opcodes: r', rdiag, rsort
			LibMatrixReorg.reorg(this, result, op);
		}
		else 
		{
			//GENERIC case (any reorg operator)
			CellIndex temp = new CellIndex(0, 0);
			if(sparse)
			{
				if(sparseRows!=null)
				{
					for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
					{
						if(sparseRows[r]==null) 
							continue;
						int[] cols=sparseRows[r].getIndexContainer();
						double[] values=sparseRows[r].getValueContainer();
						for(int i=0; i<sparseRows[r].size(); i++)
						{
							tempCellIndex.set(r, cols[i]);
							op.fn.execute(tempCellIndex, temp);
							result.appendValue(temp.row, temp.column, values[i]);
						}
					}
				}
			}
			else
			{
				if( denseBlock != null ) 
				{
					if( result.isInSparseFormat() ) //SPARSE<-DENSE
					{
						double[] a = denseBlock;
						for( int i=0, aix=0; i<rlen; i++ )
							for( int j=0; j<clen; j++, aix++ )
							{
								temp.set(i, j);
								op.fn.execute(temp, temp);
								result.appendValue(temp.row, temp.column, a[aix]);	
							}
					}
					else //DENSE<-DENSE
					{
						result.allocateDenseBlock();
						Arrays.fill(result.denseBlock, 0);
						double[] a = denseBlock;
						double[] c = result.denseBlock;
						int n = result.clen;
						
						for( int i=0, aix=0; i<rlen; i++ )
							for( int j=0; j<clen; j++, aix++ )
							{
								temp.set(i, j);
								op.fn.execute(temp, temp);
								c[temp.row*n+temp.column] = a[aix];	
							}
						result.nonZeros = nonZeros;
					}
				}
			}
		}
		
		return result;
	}
	
	/**
	 * 
	 * @param that
	 * @param ret
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public MatrixBlock appendOperations( MatrixBlock that, MatrixBlock ret ) 	
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		MatrixBlock result = checkType( ret );
		final int m = rlen;
		final int n = clen+that.clen;
		final long nnz = nonZeros+that.nonZeros;		
		boolean sp = evalSparseFormatInMemory(m, n, nnz);
		
		//init result matrix 
		if( result == null ) 
			result = new MatrixBlock(m, n, sp, nnz);
		else
			result.reset(m, n, sp, nnz);
		
		//core append operation
		//copy left and right input into output
		if( !result.sparse ) //DENSE
		{	
			result.copy(0, m-1, 0, clen-1, this, false);
			result.copy(0, m-1, clen, n-1, that, false);
		}
		else //SPARSE
		{
			//adjust sparse rows if required
			if( !this.isEmptyBlock(false) || !that.isEmptyBlock(false) )
				result.allocateSparseRowsBlock();
			result.appendToSparse(this, 0, 0);
			result.appendToSparse(that, 0, clen);
		}		
		result.nonZeros = nnz;
		
		return result;
	}

	/**
	 * 
	 * @param out
	 * @param tstype
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public MatrixValue transposeSelfMatrixMultOperations( MatrixBlock out, MMTSJType tstype ) 	
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		return transposeSelfMatrixMultOperations(out, tstype, 1);
	}
	
	/**
	 * 
	 * @param out
	 * @param tstype
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public MatrixValue transposeSelfMatrixMultOperations( MatrixBlock out, MMTSJType tstype, int k ) 	
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//check for transpose type
		if( !(tstype == MMTSJType.LEFT || tstype == MMTSJType.RIGHT) )
			throw new DMLRuntimeException("Invalid MMTSJ type '"+tstype.toString()+"'.");
		
		//setup meta data
		boolean leftTranspose = ( tstype == MMTSJType.LEFT );
		int dim = leftTranspose ? clen : rlen;
		
		//create output matrix block
		if( out == null )
			out = new MatrixBlock(dim, dim, false);
		else
			out.reset(dim, dim, false);
		
		//compute matrix mult
		if( k > 1 )
			LibMatrixMult.matrixMultTransposeSelf(this, out, leftTranspose, k);
		else
			LibMatrixMult.matrixMultTransposeSelf(this, out, leftTranspose);
		
		return out;
	}
	
	/**
	 * 
	 * @param v
	 * @param w
	 * @param out
	 * @param ctype
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public MatrixValue chainMatrixMultOperations( MatrixBlock v, MatrixBlock w, MatrixBlock out, ChainType ctype ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 	
	{
		return chainMatrixMultOperations(v, w, out, ctype, 1);
	}
	
	/**
	 * 
	 * @param v
	 * @param w
	 * @param out
	 * @param ctype
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public MatrixValue chainMatrixMultOperations( MatrixBlock v, MatrixBlock w, MatrixBlock out, ChainType ctype, int k ) 	
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//check for transpose type
		if( !(ctype == ChainType.XtXv || ctype == ChainType.XtwXv) )
			throw new DMLRuntimeException("Invalid mmchain type '"+ctype.toString()+"'.");
		
		//check for matching dimensions
		if( this.getNumColumns() != v.getNumRows() )
			throw new DMLRuntimeException("Dimensions mismatch on mmchain operation ("+this.getNumColumns()+" != "+v.getNumRows()+")");
		if( v!=null && v.getNumColumns() != 1 )
			throw new DMLRuntimeException("Invalid input vector (column vector expected, but ncol="+v.getNumColumns()+")");
		if( w!=null && w.getNumColumns() != 1 )
			throw new DMLRuntimeException("Invalid weight vector (column vector expected, but ncol="+w.getNumColumns()+")");
		
		//prepare result
		if( out != null )
			out.reset(clen, 1, false);
		else 
			out = new MatrixBlock(clen, 1, false);
		
		//compute matrix mult
		if( k > 1 )
			LibMatrixMult.matrixMultChain(this, v, w, out, ctype, k);
		else
			LibMatrixMult.matrixMultChain(this, v, w, out, ctype);
		
		return out;
	}
	
	/**
	 * 
	 * @param m1Val
	 * @param m2Val
	 * @param out1Val
	 * @param out2Val
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public void permutationMatrixMultOperations( MatrixValue m2Val, MatrixValue out1Val, MatrixValue out2Val ) 	
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		permutationMatrixMultOperations(m2Val, out1Val, out2Val, 1);
	}
	
	/**
	 * 
	 * @param m1Val
	 * @param m2Val
	 * @param out1Val
	 * @param out2Val
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public void permutationMatrixMultOperations( MatrixValue m2Val, MatrixValue out1Val, MatrixValue out2Val, int k ) 	
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//check input types and dimensions
		MatrixBlock m2 = checkType(m2Val);
		MatrixBlock ret1 = checkType(out1Val);
		MatrixBlock ret2 = checkType(out2Val);
		
		if(this.rlen!=m2.rlen)
			throw new RuntimeException("Dimensions do not match for permutation matrix multiplication ("+this.rlen+"!="+m2.rlen+").");

		//compute permutation matrix multiplication
		if (k > 1)
			LibMatrixMult.matrixMultPermute(this, m2, ret1, ret2, k);
		else
			LibMatrixMult.matrixMultPermute(this, m2, ret1, ret2);

	}
	
	/**
	 * Method to perform leftIndexing operation for a given lower and upper bounds in row and column dimensions.
	 * Updated matrix is returned as the output.
	 * 
	 * Operations to be performed: 
	 *   1) result=this; 
	 *   2) result[rowLower:rowUpper, colLower:colUpper] = rhsMatrix;
	 *  
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	public MatrixValue leftIndexingOperations(MatrixValue rhsMatrix, long rowLower, long rowUpper, 
			long colLower, long colUpper, MatrixValue ret, boolean inplace) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{	
		// Check the validity of bounds
		if ( rowLower < 1 || rowLower > getNumRows() || rowUpper < rowLower || rowUpper > getNumRows()
				|| colLower < 1 || colUpper > getNumColumns() || colUpper < colLower || colUpper > getNumColumns() ) {
			throw new DMLRuntimeException("Invalid values for matrix indexing: " +
					"["+rowLower+":"+rowUpper+"," + colLower+":"+colUpper+"] " +
							"must be within matrix dimensions ["+getNumRows()+","+getNumColumns()+"].");
		}
		
		if ( (rowUpper-rowLower+1) < rhsMatrix.getNumRows() || (colUpper-colLower+1) < rhsMatrix.getNumColumns()) {
			throw new DMLRuntimeException("Invalid values for matrix indexing: " +
					"dimensions of the source matrix ["+rhsMatrix.getNumRows()+"x" + rhsMatrix.getNumColumns() + "] " +
					"do not match the shape of the matrix specified by indices [" +
					rowLower +":" + rowUpper + ", " + colLower + ":" + colUpper + "].");
		}
		MatrixBlock result=checkType(ret);
		boolean sp = estimateSparsityOnLeftIndexing(rlen, clen, nonZeros, 
				     rhsMatrix.getNumRows(), rhsMatrix.getNumColumns(), rhsMatrix.getNonZeros());
		
		if( !inplace ) //general case
		{
			if(result==null)
				result=new MatrixBlock(rlen, clen, sp);
			else
				result.reset(rlen, clen, sp);
			result.copy(this, sp);
		}
		else //update in-place
			result = this;
		
		//NOTE conceptually we could directly use a zeroout and copy(..., false) but
		//     since this was factors slower, we still use a full copy and subsequently
		//     copy(..., true) - however, this can be changed in the future once we 
		//     improved the performance of zeroout.
		//result = (MatrixBlockDSM) zeroOutOperations(result, new IndexRange(rowLower,rowUpper, colLower, colUpper ), false);
		
		int rl = (int)rowLower-1;
		int ru = (int)rowUpper-1;
		int cl = (int)colLower-1;
		int cu = (int)colUpper-1;
		MatrixBlock src = (MatrixBlock)rhsMatrix;
		

		if(rl==ru && cl==cu) //specific case: cell update			
		{
			//copy single value and update nnz
			result.quickSetValue(rl, cl, src.quickGetValue(0, 0));
		}		
		else //general case
		{
			//copy submatrix into result
			result.copy(rl, ru, cl, cu, src, true);
		}

		return result;
	}
	
	/**
	 * Explicitly allow left indexing for scalars.
	 * 
	 * * Operations to be performed: 
	 *   1) result=this; 
	 *   2) result[row,column] = scalar.getDoubleValue();
	 * 
	 * @param scalar
	 * @param row
	 * @param col
	 * @param ret
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public MatrixValue leftIndexingOperations(ScalarObject scalar, long row, long col, MatrixValue ret, boolean inplace) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		MatrixBlock result=checkType(ret);		
		double inVal = scalar.getDoubleValue();
		boolean sp = estimateSparsityOnLeftIndexing(rlen, clen, nonZeros, 1, 1, (inVal!=0)?1:0);
		
		if( !inplace ) //general case
		{
			if(result==null)
				result=new MatrixBlock(rlen, clen, sp);
			else
				result.reset(rlen, clen, sp);
			result.copy(this, sp);
			
		}
		else //update in-place
			result = this;
		
		int rl = (int)row-1;
		int cl = (int)col-1;
		
		result.quickSetValue(rl, cl, inVal);
		return result;
	}
	
	/**
	 * Method to perform rangeReIndex operation for a given lower and upper bounds in row and column dimensions.
	 * Extracted submatrix is returned as "result".
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	public MatrixValue sliceOperations(long rowLower, long rowUpper, long colLower, long colUpper, MatrixValue ret) 
		throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		// check the validity of bounds
		if ( rowLower < 1 || rowLower > getNumRows() || rowUpper < rowLower || rowUpper > getNumRows()
				|| colLower < 1 || colUpper > getNumColumns() || colUpper < colLower || colUpper > getNumColumns() ) {
			throw new DMLRuntimeException("Invalid values for matrix indexing: " +
					"["+rowLower+":"+rowUpper+"," + colLower+":"+colUpper+"] " +
							"must be within matrix dimensions ["+getNumRows()+","+getNumColumns()+"]");
		}
		
		int rl = (int)rowLower-1;
		int ru = (int)rowUpper-1;
		int cl = (int)colLower-1;
		int cu = (int)colUpper-1;
		//System.out.println("  -- performing slide on [" + getNumRows() + "x" + getNumColumns() + "] with ["+rl+":"+ru+","+cl+":"+cu+"].");
		// Output matrix will have the same sparsity as that of the input matrix.
		// (assuming a uniform distribution of non-zeros in the input)
		MatrixBlock result=checkType(ret);
		long estnnz= (long) ((double)this.nonZeros/rlen/clen*(ru-rl+1)*(cu-cl+1));
		boolean result_sparsity = this.sparse && MatrixBlock.evalSparseFormatInMemory(ru-rl+1, cu-cl+1, estnnz);
		if(result==null)
			result=new MatrixBlock(ru-rl+1, cu-cl+1, result_sparsity, estnnz);
		else
			result.reset(ru-rl+1, cu-cl+1, result_sparsity, estnnz);
		
		// actual slice operation
		if( rowLower==1 && rowUpper==rlen && colLower==1 && colUpper==clen ) {
			// copy if entire matrix required
			result.copy( this );
		}
		else //general case
		{
			//core slicing operation (nnz maintained internally)
			if (sparse) 
				sliceSparse(rl, ru, cl, cu, result);
			else 
				sliceDense(rl, ru, cl, cu, result);
		}
		
		return result;
	}

	
	/**
	 * 
	 * @param rl
	 * @param ru
	 * @param cl
	 * @param cu
	 * @param dest
	 * @throws DMLRuntimeException 
	 */
	private void sliceSparse(int rl, int ru, int cl, int cu, MatrixBlock dest) 
		throws DMLRuntimeException
	{
		//check for early abort
		if( isEmptyBlock(false) ) 
			return;
		
		if( cl==cu ) //COLUMN VECTOR 
		{
			//note: always dense dest
			dest.allocateDenseBlock();
			for( int i=rl; i<=ru; i++ ) {
				SparseRow arow = sparseRows[i];
				if( arow != null && !arow.isEmpty() ) {
					double val = arow.get(cl);
					if( val != 0 ) {
						dest.denseBlock[i-rl] = val;
						dest.nonZeros++;
					}
				}
			}
		}
		else if( rl==ru && cl==0 && cu==clen-1 ) //ROW VECTOR 
		{
			//note: always sparse dest, but also works for dense
			dest.appendRow(0, sparseRows[rl]);
		}
		else //general case (sparse/dense dest)
		{
			for(int i=rl; i <= ru; i++) 
				if(sparseRows[i] != null && !sparseRows[i].isEmpty()) 
				{
					SparseRow arow = sparseRows[i];
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();
					int astart = (cl>0)?arow.searchIndexesFirstGTE(cl):0;
					if( astart != -1 )
						for( int j=astart; j<alen && aix[j] <= cu; j++ )
							dest.appendValue(i-rl, aix[j]-cl, avals[j]);	
				}
		}
	}
	
	/**
	 * 
	 * @param rl
	 * @param ru
	 * @param cl
	 * @param cu
	 * @param dest
	 * @throws DMLRuntimeException 
	 */
	private void sliceDense(int rl, int ru, int cl, int cu, MatrixBlock dest) 
		throws DMLRuntimeException
	{
		//ensure allocated input/output blocks
		if( denseBlock == null )
			return;
		dest.allocateDenseBlock();

		//indexing operation
		if( cl==cu ) //COLUMN INDEXING
		{
			if( clen==1 ) //vector -> vector
			{
				System.arraycopy(denseBlock, rl, dest.denseBlock, 0, ru-rl+1);
			}
			else //matrix -> vector
			{
				//IBM JVM bug (JDK7) causes crash for certain cl/cu values (e.g., divide by zero for 4) 
				//for( int i=rl*clen+cl, ix=0; i<=ru*clen+cu; i+=clen, ix++ )
				//	dest.denseBlock[ix] = denseBlock[i];
				int len = clen;
				for( int i=rl*len+cl, ix=0; i<=ru*len+cu; i+=len, ix++ )
					dest.denseBlock[ix] = denseBlock[i];
			}
		}
		else // GENERAL RANGE INDEXING
		{
			//IBM JVM bug (JDK7) causes crash for certain cl/cu values (e.g., divide by zero for 4) 
			//for(int i = rl, ix1 = rl*clen+cl, ix2=0; i <= ru; i++, ix1+=clen, ix2+=dest.clen) 
			//	System.arraycopy(denseBlock, ix1, dest.denseBlock, ix2, dest.clen);
			int len1 = clen;
			int len2 = dest.clen;
			for(int i = rl, ix1 = rl*len1+cl, ix2=0; i <= ru; i++, ix1+=len1, ix2+=len2) 
				System.arraycopy(denseBlock, ix1, dest.denseBlock, ix2, len2);
		}
		
		//compute nnz of output (not maintained due to native calls)
		dest.recomputeNonZeros();
	}
	
	public void sliceOperations(ArrayList<IndexedMatrixValue> outlist, IndexRange range, int rowCut, int colCut, 
			int normalBlockRowFactor, int normalBlockColFactor, int boundaryRlen, int boundaryClen)
	{
		MatrixBlock topleft=null, topright=null, bottomleft=null, bottomright=null;
		Iterator<IndexedMatrixValue> p=outlist.iterator();
		int blockRowFactor=normalBlockRowFactor, blockColFactor=normalBlockColFactor;
		if(rowCut>range.rowEnd)
			blockRowFactor=boundaryRlen;
		if(colCut>range.colEnd)
			blockColFactor=boundaryClen;
		
		int minrowcut=(int)Math.min(rowCut,range.rowEnd);
		int mincolcut=(int)Math.min(colCut, range.colEnd);
		int maxrowcut=(int)Math.max(rowCut, range.rowStart);
		int maxcolcut=(int)Math.max(colCut, range.colStart);
		
		if(range.rowStart<rowCut && range.colStart<colCut)
		{
			topleft=(MatrixBlock) p.next().getValue();
			//topleft.reset(blockRowFactor, blockColFactor, 
			//		checkSparcityOnSlide(rowCut-(int)range.rowStart, colCut-(int)range.colStart, blockRowFactor, blockColFactor));
			
			topleft.reset(blockRowFactor, blockColFactor, 
					estimateSparsityOnSlice(minrowcut-(int)range.rowStart, mincolcut-(int)range.colStart, blockRowFactor, blockColFactor));
		}
		if(range.rowStart<rowCut && range.colEnd>=colCut)
		{
			topright=(MatrixBlock) p.next().getValue();
			topright.reset(blockRowFactor, boundaryClen, 
					estimateSparsityOnSlice(minrowcut-(int)range.rowStart, (int)range.colEnd-maxcolcut+1, blockRowFactor, boundaryClen));
		}
		if(range.rowEnd>=rowCut && range.colStart<colCut)
		{
			bottomleft=(MatrixBlock) p.next().getValue();
			bottomleft.reset(boundaryRlen, blockColFactor, 
					estimateSparsityOnSlice((int)range.rowEnd-maxrowcut+1, mincolcut-(int)range.colStart, boundaryRlen, blockColFactor));
		}
		if(range.rowEnd>=rowCut && range.colEnd>=colCut)
		{
			bottomright=(MatrixBlock) p.next().getValue();
			bottomright.reset(boundaryRlen, boundaryClen, 
					estimateSparsityOnSlice((int)range.rowEnd-maxrowcut+1, (int)range.colEnd-maxcolcut+1, boundaryRlen, boundaryClen));
		}
		
		if(sparse)
		{
			if(sparseRows!=null)
			{
				int r=(int)range.rowStart;
				for(; r<Math.min(Math.min(rowCut, sparseRows.length), range.rowEnd+1); r++)
					sliceHelp(r, range, colCut, topleft, topright, normalBlockRowFactor-rowCut, normalBlockRowFactor, normalBlockColFactor);
				
				for(; r<=Math.min(range.rowEnd, sparseRows.length-1); r++)
					sliceHelp(r, range, colCut, bottomleft, bottomright, -rowCut, normalBlockRowFactor, normalBlockColFactor);
				//System.out.println("in: \n"+this);
				//System.out.println("outlist: \n"+outlist);
			}
		}else
		{
			if(denseBlock!=null)
			{
				int i=((int)range.rowStart)*clen;
				int r=(int) range.rowStart;
				for(; r<Math.min(rowCut, range.rowEnd+1); r++)
				{
					int c=(int) range.colStart;
					for(; c<Math.min(colCut, range.colEnd+1); c++)
						topleft.appendValue(r+normalBlockRowFactor-rowCut, c+normalBlockColFactor-colCut, denseBlock[i+c]);
					for(; c<=range.colEnd; c++)
						topright.appendValue(r+normalBlockRowFactor-rowCut, c-colCut, denseBlock[i+c]);
					i+=clen;
				}
				
				for(; r<=range.rowEnd; r++)
				{
					int c=(int) range.colStart;
					for(; c<Math.min(colCut, range.colEnd+1); c++)
						bottomleft.appendValue(r-rowCut, c+normalBlockColFactor-colCut, denseBlock[i+c]);
					for(; c<=range.colEnd; c++)
						bottomright.appendValue(r-rowCut, c-colCut, denseBlock[i+c]);
					i+=clen;
				}
			}
		}
	}
	
	private void sliceHelp(int r, IndexRange range, int colCut, MatrixBlock left, MatrixBlock right, int rowOffset, int normalBlockRowFactor, int normalBlockColFactor)
	{
		if(sparseRows[r]==null) 
			return;
		
		int[] cols=sparseRows[r].getIndexContainer();
		double[] values=sparseRows[r].getValueContainer();
		int start=sparseRows[r].searchIndexesFirstGTE((int)range.colStart);
		if(start<0) 
			return;
		int end=sparseRows[r].searchIndexesFirstLTE((int)range.colEnd);
		if(end<0 || start>end) 
			return;
		
		//actual slice operation
		for(int i=start; i<=end; i++) {
			if(cols[i]<colCut)
				left.appendValue(r+rowOffset, cols[i]+normalBlockColFactor-colCut, values[i]);
			else
				right.appendValue(r+rowOffset, cols[i]-colCut, values[i]);
		}
	}
	
	@Override
	//This the append operations for MR side
	//nextNCol is the number columns for the block right of block v2
	public void appendOperations(MatrixValue v2,
			ArrayList<IndexedMatrixValue> outlist, int blockRowFactor,
			int blockColFactor, boolean m2IsLast, int nextNCol)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		MatrixBlock m2=(MatrixBlock)v2;
		//System.out.println("second matrix: \n"+m2);
		Iterator<IndexedMatrixValue> p=outlist.iterator();
		if(this.clen==blockColFactor)
		{
			MatrixBlock first=(MatrixBlock) p.next().getValue();
			first.copy(this);
			MatrixBlock second=(MatrixBlock) p.next().getValue();
			second.copy(m2);
		}else
		{
			int ncol=Math.min(clen+m2.getNumColumns(), blockColFactor);
			int part=ncol-clen;
			MatrixBlock first=(MatrixBlock) p.next().getValue();
			first.reset(rlen, ncol, this.nonZeros+m2.getNonZeros()*part/m2.getNumColumns());
			
			//copy the first matrix
			if(this.sparse)
			{
				if(this.sparseRows!=null)
				{
					for(int i=0; i<Math.min(rlen, this.sparseRows.length); i++)
					{
						if(this.sparseRows[i]!=null)
							first.appendRow(i, this.sparseRows[i]);
					}
				}
			}else if(this.denseBlock!=null)
			{
				int sindx=0;
				for(int r=0; r<rlen; r++)
					for(int c=0; c<clen; c++)
					{
						first.appendValue(r, c, this.denseBlock[sindx]);
						sindx++;
					}
			}
			
			
			MatrixBlock second=null;
			
			if(part<m2.clen)
			{
				second=(MatrixBlock) p.next().getValue();
				if(m2IsLast)
					second.reset(m2.rlen, m2.clen-part, m2.sparse);
				else
					second.reset(m2.rlen, Math.min(m2.clen-part+nextNCol, blockColFactor), m2.sparse);
			}
			
			//copy the second
			if(m2.sparse)
			{
				if(m2.sparseRows!=null)
				{
					for(int i=0; i<Math.min(m2.rlen, m2.sparseRows.length); i++)
					{
						if(m2.sparseRows[i]!=null)
						{
							int[] indexContainer=m2.sparseRows[i].getIndexContainer();
							double[] valueContainer=m2.sparseRows[i].getValueContainer();
							for(int j=0; j<m2.sparseRows[i].size(); j++)
							{
								if(indexContainer[j]<part)
									first.appendValue(i, clen+indexContainer[j], valueContainer[j]);
								else
									second.appendValue(i, indexContainer[j]-part, valueContainer[j]);
							}
						}
					}
				}
			}else if(m2.denseBlock!=null)
			{
				int sindx=0;
				for(int r=0; r<m2.rlen; r++)
				{
					int c=0;
					for(; c<part; c++)
					{
						first.appendValue(r, clen+c, m2.denseBlock[sindx+c]);
					//	System.out.println("access "+(sindx+c));
					//	System.out.println("add first ("+r+", "+(clen+c)+"), "+m2.denseBlock[sindx+c]);
					}
					for(; c<m2.clen; c++)
					{
						second.appendValue(r, c-part, m2.denseBlock[sindx+c]);
					//	System.out.println("access "+(sindx+c));
					//	System.out.println("add second ("+r+", "+(c-part)+"), "+m2.denseBlock[sindx+c]);
					}
					sindx+=m2.clen;
				}
			}
		}
	}
	
	/**
	 * 
	 */
	public MatrixValue zeroOutOperations(MatrixValue result, IndexRange range, boolean complementary)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		checkType(result);
		double currentSparsity=(double)nonZeros/(double)rlen/(double)clen;
		double estimatedSps=currentSparsity*(double)(range.rowEnd-range.rowStart+1)
		                    *(double)(range.colEnd-range.colStart+1)/(double)rlen/(double)clen;
		if(!complementary)
			estimatedSps=currentSparsity-estimatedSps;
		
		boolean lsparse = evalSparseFormatInMemory(rlen, clen, (long)(estimatedSps*rlen*clen));
		
		if(result==null)
			result=new MatrixBlock(rlen, clen, lsparse, (int)(estimatedSps*rlen*clen));
		else
			result.reset(rlen, clen, lsparse, (int)(estimatedSps*rlen*clen));
		
		
		if(sparse)
		{
			if(sparseRows!=null)
			{
				if(!complementary)//if zero out
				{
					for(int r=0; r<Math.min((int)range.rowStart, sparseRows.length); r++)
						((MatrixBlock) result).appendRow(r, sparseRows[r]);
					for(int r=Math.min((int)range.rowEnd+1, sparseRows.length); r<Math.min(rlen, sparseRows.length); r++)
						((MatrixBlock) result).appendRow(r, sparseRows[r]);
				}
				for(int r=(int)range.rowStart; r<=Math.min(range.rowEnd, sparseRows.length-1); r++)
				{
					if(sparseRows[r]==null) 
						continue;
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					
					if(complementary)//if selection
					{
						int start=sparseRows[r].searchIndexesFirstGTE((int)range.colStart);
						if(start<0) continue;
						int end=sparseRows[r].searchIndexesFirstGT((int)range.colEnd);
						if(end<0 || start>end) 
							continue;
						
						for(int i=start; i<end; i++)
						{
							((MatrixBlock) result).appendValue(r, cols[i], values[i]);
						}
					}else
					{
						int start=sparseRows[r].searchIndexesFirstGTE((int)range.colStart);
						if(start<0) start=sparseRows[r].size();
						int end=sparseRows[r].searchIndexesFirstGT((int)range.colEnd);
						if(end<0) end=sparseRows[r].size();
						
						for(int i=0; i<start; i++)
						{
							((MatrixBlock) result).appendValue(r, cols[i], values[i]);
						}
						for(int i=end; i<sparseRows[r].size(); i++)
						{
							((MatrixBlock) result).appendValue(r, cols[i], values[i]);
						}
					}
				}
			}
		}else
		{
			if(denseBlock!=null)
			{
				if(complementary)//if selection
				{
					int offset=((int)range.rowStart)*clen;
					for(int r=(int) range.rowStart; r<=range.rowEnd; r++)
					{
						for(int c=(int) range.colStart; c<=range.colEnd; c++)
							((MatrixBlock) result).appendValue(r, c, denseBlock[offset+c]);
						offset+=clen;
					}
				}else
				{
					int offset=0;
					int r=0;
					for(; r<(int)range.rowStart; r++)
						for(int c=0; c<clen; c++, offset++)
							((MatrixBlock) result).appendValue(r, c, denseBlock[offset]);
					
					for(; r<=(int)range.rowEnd; r++)
					{
						for(int c=0; c<(int)range.colStart; c++)
							((MatrixBlock) result).appendValue(r, c, denseBlock[offset+c]);
						for(int c=(int)range.colEnd+1; c<clen; c++)
							((MatrixBlock) result).appendValue(r, c, denseBlock[offset+c]);
						offset+=clen;
					}
					
					for(; r<rlen; r++)
						for(int c=0; c<clen; c++, offset++)
							((MatrixBlock) result).appendValue(r, c, denseBlock[offset]);
				}
				
			}
		}
		return result;
	}
	
	public MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, 
			int blockingFactorRow, int blockingFactorCol, MatrixIndexes indexesIn)
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		return aggregateUnaryOperations(op, result, 
				blockingFactorRow, blockingFactorCol, indexesIn, false);
	}
	
	
	public MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, 
			int blockingFactorRow, int blockingFactorCol, MatrixIndexes indexesIn, boolean inCP) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		CellIndex tempCellIndex = new CellIndex(-1,-1);
		op.indexFn.computeDimension(rlen, clen, tempCellIndex);
		if(op.aggOp.correctionExists)
		{
			switch(op.aggOp.correctionLocation)
			{
				case LASTROW: 
					tempCellIndex.row++; 
					break;
				case LASTCOLUMN: 
					tempCellIndex.column++; 
					break;
				case LASTTWOROWS: 
					tempCellIndex.row+=2; 
					break;
				case LASTTWOCOLUMNS: 
					tempCellIndex.column+=2; 
					break;
				default:
					throw new DMLRuntimeException("unrecognized correctionLocation: "+op.aggOp.correctionLocation);	
			}
		}
		if(result==null)
			result=new MatrixBlock(tempCellIndex.row, tempCellIndex.column, false);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, false);
		
		MatrixBlock ret = (MatrixBlock) result;
		if( LibMatrixAgg.isSupportedUnaryAggregateOperator(op) ) {
			LibMatrixAgg.aggregateUnaryMatrix(this, ret, op);
			LibMatrixAgg.recomputeIndexes(ret, op, blockingFactorRow, blockingFactorCol, indexesIn);
		}
		else if(op.sparseSafe)
			sparseAggregateUnaryHelp(op, ret, blockingFactorRow, blockingFactorCol, indexesIn);
		else
			denseAggregateUnaryHelp(op, ret, blockingFactorRow, blockingFactorCol, indexesIn);
		
		if(op.aggOp.correctionExists && inCP)
			((MatrixBlock)result).dropLastRowsOrColums(op.aggOp.correctionLocation);
		
		return ret;
	}
	
	private void sparseAggregateUnaryHelp(AggregateUnaryOperator op, MatrixBlock result,
			int blockingFactorRow, int blockingFactorCol, MatrixIndexes indexesIn) throws DMLRuntimeException
	{
		//initialize result
		if(op.aggOp.initialValue!=0)
			result.resetDenseWithValue(result.rlen, result.clen, op.aggOp.initialValue);
		
		CellIndex tempCellIndex = new CellIndex(-1,-1);
		KahanObject buffer=new KahanObject(0,0);
		int r = 0, c = 0;
		
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(r=0; r<Math.min(rlen, sparseRows.length); r++)
				{
					if(sparseRows[r]==null) 
						continue;
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					for(int i=0; i<sparseRows[r].size(); i++)
					{
						tempCellIndex.set(r, cols[i]);
						op.indexFn.execute(tempCellIndex, tempCellIndex);
						incrementalAggregateUnaryHelp(op.aggOp, result, tempCellIndex.row, tempCellIndex.column, values[i], buffer);

					}
				}
			}
		}
		else
		{
			if(denseBlock!=null)
			{
				int limit=rlen*clen;
				for(int i=0; i<limit; i++)
				{
					r=i/clen;
					c=i%clen;
					tempCellIndex.set(r, c);
					op.indexFn.execute(tempCellIndex, tempCellIndex);
					incrementalAggregateUnaryHelp(op.aggOp, result, tempCellIndex.row, tempCellIndex.column, denseBlock[i], buffer);
				}
			}
		}
	}
	
	private void denseAggregateUnaryHelp(AggregateUnaryOperator op, MatrixBlock result,
			int blockingFactorRow, int blockingFactorCol, MatrixIndexes indexesIn) throws DMLRuntimeException
	{
		//initialize 
		if(op.aggOp.initialValue!=0)
			result.resetDenseWithValue(result.rlen, result.clen, op.aggOp.initialValue);
		
		CellIndex tempCellIndex = new CellIndex(-1,-1);
		KahanObject buffer=new KahanObject(0,0);
		
		for(int i=0; i<rlen; i++)
			for(int j=0; j<clen; j++)
			{
				tempCellIndex.set(i, j);
				op.indexFn.execute(tempCellIndex, tempCellIndex);

				if(op.aggOp.correctionExists
				   && op.aggOp.correctionLocation == CorrectionLocationType.LASTCOLUMN
				   && op.aggOp.increOp.fn instanceof Builtin 
				   && ( ((Builtin)(op.aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MAXINDEX
				        || ((Builtin)(op.aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MININDEX) 
				        ){
					double currMaxValue = result.quickGetValue(i, 1);
					long newMaxIndex = UtilFunctions.cellIndexCalculation(indexesIn.getColumnIndex(), blockingFactorCol, j);
					double newMaxValue = quickGetValue(i, j);
					double update = op.aggOp.increOp.fn.execute(newMaxValue, currMaxValue);
						   
					//System.out.println("currV="+currMaxValue+",newV="+newMaxValue+",newIX="+newMaxIndex+",update="+update);
					if(update == 1){
						result.quickSetValue(i, 0, newMaxIndex);
						result.quickSetValue(i, 1, newMaxValue);
					}
				}else
					incrementalAggregateUnaryHelp(op.aggOp, result, tempCellIndex.row, tempCellIndex.column, quickGetValue(i,j), buffer);
			}
	}
	
	
	private void incrementalAggregateUnaryHelp(AggregateOperator aggOp, MatrixBlock result, int row, int column, 
			double newvalue, KahanObject buffer) throws DMLRuntimeException
	{
		if(aggOp.correctionExists)
		{
			if(aggOp.correctionLocation==CorrectionLocationType.LASTROW || aggOp.correctionLocation==CorrectionLocationType.LASTCOLUMN)
			{
				int corRow=row, corCol=column;
				if(aggOp.correctionLocation==CorrectionLocationType.LASTROW)//extra row
					corRow++;
				else if(aggOp.correctionLocation==CorrectionLocationType.LASTCOLUMN)
					corCol++;
				else
					throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
				
				buffer._sum=result.quickGetValue(row, column);
				buffer._correction=result.quickGetValue(corRow, corCol);
				buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newvalue);
				result.quickSetValue(row, column, buffer._sum);
				result.quickSetValue(corRow, corCol, buffer._correction);
			}else if(aggOp.correctionLocation==CorrectionLocationType.NONE)
			{
				throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
			}else// for mean
			{
				int corRow=row, corCol=column;
				int countRow=row, countCol=column;
				if(aggOp.correctionLocation==CorrectionLocationType.LASTTWOROWS)
				{
					countRow++;
					corRow+=2;
				}
				else if(aggOp.correctionLocation==CorrectionLocationType.LASTTWOCOLUMNS)
				{
					countCol++;
					corCol+=2;
				}
				else
					throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
				buffer._sum=result.quickGetValue(row, column);
				buffer._correction=result.quickGetValue(corRow, corCol);
				double count=result.quickGetValue(countRow, countCol)+1.0;
				buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newvalue, count);
				result.quickSetValue(row, column, buffer._sum);
				result.quickSetValue(corRow, corCol, buffer._correction);
				result.quickSetValue(countRow, countCol, count);
			}
			
		}else
		{
			newvalue=aggOp.increOp.fn.execute(result.quickGetValue(row, column), newvalue);
			result.quickSetValue(row, column, newvalue);
		}
	}
	
	/**
	 * 
	 * @param correctionLocation
	 */
	public void dropLastRowsOrColums(CorrectionLocationType correctionLocation) 
	{
		//do nothing 
		if(   correctionLocation==CorrectionLocationType.NONE 
	       || correctionLocation==CorrectionLocationType.INVALID )
		{
			return;
		}
		
		//determine number of rows/cols to be removed
		int step = ( correctionLocation==CorrectionLocationType.LASTTWOROWS 
				    || correctionLocation==CorrectionLocationType.LASTTWOCOLUMNS) ? 2 : 1;
		
		//e.g., colSums, colMeans, colMaxs, colMeans
		if(   correctionLocation==CorrectionLocationType.LASTROW 
		   || correctionLocation==CorrectionLocationType.LASTTWOROWS )
		{
			if( sparse ) //SPARSE
			{
				if(sparseRows!=null)
					for(int i=1; i<=step; i++)
						if(sparseRows[rlen-i]!=null)
							this.nonZeros-=sparseRows[rlen-i].size();
			}
			else //DENSE
			{
				if(denseBlock!=null)
					for(int i=(rlen-step)*clen; i<rlen*clen; i++)
						if(denseBlock[i]!=0)
							this.nonZeros--;
			}
			
			//just need to shrink the dimension, the deleted rows won't be accessed
			rlen -= step;
		}
		
		//e.g., rowSums, rowsMeans, rowsMaxs, rowsMeans
		if(   correctionLocation==CorrectionLocationType.LASTCOLUMN 
		   || correctionLocation==CorrectionLocationType.LASTTWOCOLUMNS )
		{
			if(sparse) //SPARSE
			{
				if(sparseRows!=null)
				{
					for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
						if(sparseRows[r]!=null)
						{
							int newSize=sparseRows[r].searchIndexesFirstGTE(clen-step);
							if(newSize>=0)
							{
								this.nonZeros-=sparseRows[r].size()-newSize;
								sparseRows[r].truncate(newSize);
							}
						}
				}
			}
			else //DENSE
			{
				if(this.denseBlock!=null)
				{
					//the first row doesn't need to be copied
					int targetIndex=clen-step;
					int sourceOffset=clen;
					this.nonZeros=0;
					for(int i=0; i<targetIndex; i++)
						if(denseBlock[i]!=0)
							this.nonZeros++;
					
					//start from the 2nd row
					for(int r=1; r<rlen; r++)
					{
						for(int c=0; c<clen-step; c++)
						{
							if((denseBlock[targetIndex]=denseBlock[sourceOffset+c])!=0)
								this.nonZeros++;
							targetIndex++;
						}
						sourceOffset+=clen;
					}
				}
			}
			
			clen -= step;
		}
	}
		
	public CM_COV_Object cmOperations(CMOperator op) 
		throws DMLRuntimeException 
	{
		/* this._data must be a 1 dimensional vector */
		if ( this.getNumColumns() != 1) {
			throw new DMLRuntimeException("Central Moment can not be computed on [" 
					+ this.getNumRows() + "," + this.getNumColumns() + "] matrix.");
		}
		
		CM_COV_Object cmobj = new CM_COV_Object();
		int nzcount = 0;
		if(sparse && sparseRows!=null) //SPARSE
		{
			for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
			{
				if(sparseRows[r]==null) 
					continue;
				double[] values=sparseRows[r].getValueContainer();
				for(int i=0; i<sparseRows[r].size(); i++) {
					op.fn.execute(cmobj, values[i]);
					nzcount++;
				}
			}
			// account for zeros in the vector
			op.fn.execute(cmobj, 0.0, this.getNumRows()-nzcount);
		}
		else if(denseBlock!=null)  //DENSE
		{
			//always vector (see check above)
			for(int i=0; i<rlen; i++)
				op.fn.execute(cmobj, denseBlock[i]);
		}

		return cmobj;
	}
		
	public CM_COV_Object cmOperations(CMOperator op, MatrixBlock weights) 
		throws DMLRuntimeException 
	{
		/* this._data must be a 1 dimensional vector */
		if ( this.getNumColumns() != 1 || weights.getNumColumns() != 1) {
			throw new DMLRuntimeException("Central Moment can be computed only on 1-dimensional column matrices.");
		}
		if ( this.getNumRows() != weights.getNumRows() || this.getNumColumns() != weights.getNumColumns()) {
			throw new DMLRuntimeException("Covariance: Mismatching dimensions between input and weight matrices - " +
					"["+this.getNumRows()+","+this.getNumColumns() +"] != [" 
					+ weights.getNumRows() + "," + weights.getNumColumns() +"]");
		}
		
		CM_COV_Object cmobj = new CM_COV_Object();
		if (sparse && sparseRows!=null) //SPARSE
		{
			for(int i=0; i < rlen; i++) 
				op.fn.execute(cmobj, this.quickGetValue(i,0), weights.quickGetValue(i,0));
		}
		else if(denseBlock!=null) //DENSE
		{
			//always vectors (see check above)
			if( !weights.sparse )
			{
				//both dense vectors (default case)
				if(weights.denseBlock!=null)
					for( int i=0; i<rlen; i++ )
						op.fn.execute(cmobj, denseBlock[i], weights.denseBlock[i]);
			}
			else
			{
				for(int i=0; i<rlen; i++) 
					op.fn.execute(cmobj, denseBlock[i], weights.quickGetValue(i,0) );
			}
		}
		
		return cmobj;
	}
	
	public CM_COV_Object covOperations(COVOperator op, MatrixBlock that) 
		throws DMLRuntimeException 
	{
		/* this._data must be a 1 dimensional vector */
		if ( this.getNumColumns() != 1 || that.getNumColumns() != 1 ) {
			throw new DMLRuntimeException("Covariance can be computed only on 1-dimensional column matrices."); 
		}
		if ( this.getNumRows() != that.getNumRows() || this.getNumColumns() != that.getNumColumns()) {
			throw new DMLRuntimeException("Covariance: Mismatching input matrix dimensions - " +
					"["+this.getNumRows()+","+this.getNumColumns() +"] != [" 
					+ that.getNumRows() + "," + that.getNumColumns() +"]");
		}
		
		CM_COV_Object covobj = new CM_COV_Object();
		if(sparse && sparseRows!=null) //SPARSE
		{
			for(int i=0; i < rlen; i++ ) 
				op.fn.execute(covobj, this.quickGetValue(i,0), that.quickGetValue(i,0));
		}
		else if(denseBlock!=null) //DENSE
		{
			//always vectors (see check above)
			if( !that.sparse )
			{
				//both dense vectors (default case)
				if(that.denseBlock!=null)
					for( int i=0; i<rlen; i++ )
						op.fn.execute(covobj, denseBlock[i], that.denseBlock[i]);
			}
			else
			{
				for(int i=0; i<rlen; i++)
					op.fn.execute(covobj, denseBlock[i], that.quickGetValue(i,0));
			}
		}
		
		return covobj;
	}
	
	public CM_COV_Object covOperations(COVOperator op, MatrixBlock that, MatrixBlock weights) 
		throws DMLRuntimeException 
	{
		/* this._data must be a 1 dimensional vector */
		if ( this.getNumColumns() != 1 || that.getNumColumns() != 1 || weights.getNumColumns() != 1) {
			throw new DMLRuntimeException("Covariance can be computed only on 1-dimensional column matrices."); 
		}
		if ( this.getNumRows() != that.getNumRows() || this.getNumColumns() != that.getNumColumns()) {
			throw new DMLRuntimeException("Covariance: Mismatching input matrix dimensions - " +
					"["+this.getNumRows()+","+this.getNumColumns() +"] != [" 
					+ that.getNumRows() + "," + that.getNumColumns() +"]");
		}
		if ( this.getNumRows() != weights.getNumRows() || this.getNumColumns() != weights.getNumColumns()) {
			throw new DMLRuntimeException("Covariance: Mismatching dimensions between input and weight matrices - " +
					"["+this.getNumRows()+","+this.getNumColumns() +"] != [" 
					+ weights.getNumRows() + "," + weights.getNumColumns() +"]");
		}
		
		CM_COV_Object covobj = new CM_COV_Object();
		if(sparse && sparseRows!=null) //SPARSE
		{
			for(int i=0; i < rlen; i++ ) 
				op.fn.execute(covobj, this.quickGetValue(i,0), that.quickGetValue(i,0), weights.quickGetValue(i,0));
		}
		else if(denseBlock!=null) //DENSE
		{
			//always vectors (see check above)
			if( !that.sparse && !weights.sparse )
			{
				//all dense vectors (default case)
				if(that.denseBlock!=null)
					for( int i=0; i<rlen; i++ )
						op.fn.execute(covobj, denseBlock[i], that.denseBlock[i], weights.denseBlock[i]);
			}
			else
			{
				for(int i=0; i<rlen; i++)
					op.fn.execute(covobj, denseBlock[i], that.quickGetValue(i,0), weights.quickGetValue(i,0));
			}
		}
		
		return covobj;
	}

	public MatrixValue sortOperations(MatrixValue weights, MatrixValue result) throws DMLRuntimeException, DMLUnsupportedOperationException {
		boolean wtflag = (weights!=null);
		
		MatrixBlock wts= (weights == null ? null : checkType(weights));
		checkType(result);
		
		if ( getNumColumns() != 1 ) {
			throw new DMLRuntimeException("Invalid input dimensions (" + getNumRows() + "x" + getNumColumns() + ") to sort operation.");
		}
		if ( wts != null && wts.getNumColumns() != 1 ) {
			throw new DMLRuntimeException("Invalid weight dimensions (" + wts.getNumRows() + "x" + wts.getNumColumns() + ") to sort operation.");
		}
		
		// prepare result, currently always dense
		// #rows in temp matrix = 1 + #nnz in the input ( 1 is for the "zero" value)
		int dim1 = (int) (1+this.getNonZeros());
		if(result==null)
			result=new MatrixBlock(dim1, 2, false);
		else
			result.reset(dim1, 2, false);
		
		// Copy the input elements into a temporary array for sorting
		// First column is data and second column is weights
		// (since the inputs are vectors, they are likely dense - hence quickget is sufficient)
		MatrixBlock tdw = new MatrixBlock(dim1, 2, false);
		double d, w, zero_wt=0;
		int ind = 1;
		if( wtflag ) // w/ weights
		{
			for ( int i=0; i<rlen; i++ ) {
				d = quickGetValue(i,0);
				w = wts.quickGetValue(i,0);
				if ( d != 0 ) {
					tdw.quickSetValue(ind, 0, d);
					tdw.quickSetValue(ind, 1, w);
					ind++;
				}
				else
					zero_wt += w;
			}
		} 
		else //w/o weights
		{
			zero_wt = getNumRows() - getNonZeros();
			for( int i=0; i<rlen; i++ ) {
				d = quickGetValue(i,0);
				if( d != 0 ){
					tdw.quickSetValue(ind, 0, d);
					tdw.quickSetValue(ind, 1, 1);
					ind++;
				}
			}
		}
		tdw.quickSetValue(0, 0, 0.0);
		tdw.quickSetValue(0, 1, zero_wt); //num zeros in input
		
		// Sort td and tw based on values inside td (ascending sort), incl copy into result
		SortIndex sfn = SortIndex.getSortIndexFnObject(1, false, false);
		ReorgOperator rop = new ReorgOperator(sfn);
		LibMatrixReorg.reorg(tdw, (MatrixBlock)result, rop);
		
		return result;
	}
	
	public double interQuartileMean() throws DMLRuntimeException {
		double sum_wt = sumWeightForQuantile();
		
		double q25d = 0.25*sum_wt;
		double q75d = 0.75*sum_wt;
		
		int q25i = (int) Math.ceil(q25d);
		int q75i = (int) Math.ceil(q75d);
		
		// skip until (but excluding) q25
		int t = 0, i=-1;
		while(i<getNumRows() && t < q25i) {
			i++;
			//System.out.println("    " + i + ": " + quickGetValue(i,0) + "," + quickGetValue(i,1));
			t += quickGetValue(i,1);
		}
		// compute the portion of q25
		double runningSum = (t-q25d)*quickGetValue(i,0);
		
		// add until (including) q75
		while(i<getNumRows() && t < q75i) {
			i++;
			t += quickGetValue(i,1);
			runningSum += quickGetValue(i,0)*quickGetValue(i,1);
		}
		// subtract additional portion of q75
		runningSum -= (t-q75d)*quickGetValue(i,0);
		
		return runningSum/(sum_wt*0.5);
	}
	
	/**
	 * Computes the weighted interQuartileMean.
	 * The matrix block ("this" pointer) has two columns, in which the first column 
	 * refers to the data and second column denotes corresponding weights.
	 * 
	 * @return InterQuartileMean
	 * @throws DMLRuntimeException
	 */
	public double interQuartileMeanOLD() throws DMLRuntimeException {
		double sum_wt = sumWeightForQuantile();
		
		int fromPos = (int) Math.ceil(0.25*sum_wt);
		int toPos = (int) Math.ceil(0.75*sum_wt);
		int selectRange = toPos-fromPos; // range: (fromPos,toPos]
		
		if ( selectRange == 0 )
			return 0.0;
		
		int index, count=0;
		
		// The first row (0^th row) has value 0.
		// If it has a non-zero weight i.e., input data has zero values
		// then "index" must start from 0, otherwise we skip the first row 
		// and start with the next value in the data, which is in the 1st row.
		if ( quickGetValue(0,1) > 0 ) 
			index = 0;
		else
			index = 1;
		
		// keep scanning the weights, until we hit the required position <code>fromPos</code>
		while ( count < fromPos ) {
			count += quickGetValue(index,1);
			++index;
		}
		
		double runningSum; 
		double val;
		int wt, selectedCount;
		
		runningSum = (count-fromPos) * quickGetValue(index-1, 0);
		selectedCount = (count-fromPos);
		
		while(count <= toPos ) {
			val = quickGetValue(index,0);
			wt = (int) quickGetValue(index,1);
			
			runningSum += (val * Math.min(wt, selectRange-selectedCount));
			selectedCount += Math.min(wt, selectRange-selectedCount);
			count += wt;
			++index;
		}
		
		//System.out.println(fromPos + ", " + toPos + ": " + count + ", "+ runningSum + ", " + selectedCount);
		
		return runningSum/selectedCount;
	}
	
	public MatrixValue pickValues(MatrixValue quantiles, MatrixValue ret) 
		throws DMLUnsupportedOperationException, DMLRuntimeException {
	
		MatrixBlock qs=checkType(quantiles);
		
		if ( qs.clen != 1 ) {
			throw new DMLRuntimeException("Multiple quantiles can only be computed on a 1D matrix");
		}
		
		MatrixBlock output = checkType(ret);

		if(output==null)
			output=new MatrixBlock(qs.rlen, qs.clen, false); // resulting matrix is mostly likely be dense
		else
			output.reset(qs.rlen, qs.clen, false);
		
		for ( int i=0; i < qs.rlen; i++ ) {
			output.quickSetValue(i, 0, this.pickValue(qs.quickGetValue(i,0)) );
		}
		
		return output;
	}
	
	public double median() throws DMLRuntimeException {
		double sum_wt = sumWeightForQuantile();
		return pickValue(0.5, sum_wt%2==0);
	}
	
	public double pickValue(double quantile) throws DMLRuntimeException{
		return pickValue(quantile, false);
	}
	
	public double pickValue(double quantile, boolean average) 
		throws DMLRuntimeException 
	{
		double sum_wt = sumWeightForQuantile();
		
		// do averaging only if it is asked for; and sum_wt is even
		average = average && (sum_wt%2 == 0);
		
		int pos = (int) Math.ceil(quantile*sum_wt);
		
		int t = 0, i=-1;
		do {
			i++;
			t += quickGetValue(i,1);
		} while(t<pos && i < getNumRows());
		
		//System.out.println("values: " + quickGetValue(i,0) + "," + quickGetValue(i,1) + " -- " + quickGetValue(i+1,0) + "," +  quickGetValue(i+1,1));
		if ( quickGetValue(i,1) != 0 ) {
			// i^th value is present in the data set, simply return it
			if ( average ) {
				if(pos < t) {
					return quickGetValue(i,0);
				}
				if(quickGetValue(i+1,1) != 0)
					return (quickGetValue(i,0)+quickGetValue(i+1,0))/2;
				else
					// (i+1)^th value is 0. So, fetch (i+2)^th value
					return (quickGetValue(i,0)+quickGetValue(i+2,0))/2;
			}
			else 
				return quickGetValue(i, 0);
		}
		else {
			// i^th value is not present in the data set. 
			// It can only happen in the case where i^th value is 0.0; and 0.0 is not present in the data set (but introduced by sort).
			if ( i+1 < getNumRows() )
				// when 0.0 is not the last element in the sorted order
				return quickGetValue(i+1,0);
			else
				// when 0.0 is the last element in the sorted order (input data is all negative)
				return quickGetValue(i-1,0);
		}
	}
	
	/**
	 * In a given two column matrix, the second column denotes weights.
	 * This function computes the total weight
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 */
	private double sumWeightForQuantile() 
		throws DMLRuntimeException 
	{
		double sum_wt = 0;
		for (int i=0; i < getNumRows(); i++ )
			sum_wt += quickGetValue(i, 1);
		if ( Math.floor(sum_wt) < sum_wt ) {
			throw new DMLRuntimeException("Unexpected error while computing quantile -- weights must be integers.");
		}
		return sum_wt;
	}

	/**
	 * 
	 * @param m1Index
	 * @param m1Value
	 * @param m2Index
	 * @param m2Value
	 * @param result
	 * @param op
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public MatrixValue aggregateBinaryOperations(MatrixIndexes m1Index, MatrixValue m1Value, MatrixIndexes m2Index, MatrixValue m2Value, 
			MatrixValue result, AggregateBinaryOperator op ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		return aggregateBinaryOperations(m1Value, m2Value, result, op);
	}

	/**
	 * 
	 */
	public MatrixValue aggregateBinaryOperations(MatrixValue m1Value, MatrixValue m2Value, MatrixValue result, AggregateBinaryOperator op) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//check input types, dimensions, configuration
		MatrixBlock m1 = checkType(m1Value);
		MatrixBlock m2 = checkType(m2Value);
		MatrixBlock ret = checkType(result);
		if( m1.clen != m2.rlen ) {
			throw new RuntimeException("Dimensions do not match for matrix multiplication ("+m1.clen+"!="+m2.rlen+").");
		}
		if( !(op.binaryFn instanceof Multiply && op.aggOp.increOp.fn instanceof Plus) ) {
			throw new DMLRuntimeException("Unsupported binary aggregate operation: ("+op.binaryFn+", "+op.aggOp+").");
		}
			
		//setup meta data (dimensions, sparsity)
		int rl = m1.rlen;
		int cl = m2.clen;
		SparsityEstimate sp = estimateSparsityOnAggBinary(m1, m2, op);
		
		//create output matrix block
		if( ret==null )
			ret = new MatrixBlock(rl, cl, sp.sparse, sp.estimatedNonZeros);
		else
			ret.reset(rl, cl, sp.sparse, sp.estimatedNonZeros);
		
		//compute matrix multiplication (only supported binary aggregate operation)
		if( op.getNumThreads() > 1 )
			LibMatrixMult.matrixMult(m1, m2, ret, op.getNumThreads());
		else
			LibMatrixMult.matrixMult(m1, m2, ret);
		
		return ret;
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param m3
	 * @param op
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public ScalarObject aggregateTernaryOperations(MatrixBlock m1, MatrixBlock m2, MatrixBlock m3, AggregateBinaryOperator op) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//check input dimensions and operators
		if( m1.rlen!=m2.rlen || m2.rlen!=m3.rlen || m1.clen!=1 || m2.clen!=1 || m3.clen!=1 )
			throw new DMLRuntimeException("Invalid dimensions for aggregate tertiary ("+m1.rlen+"x"+m1.clen+", "+m2.rlen+"x"+m2.clen+", "+m3.rlen+"x"+m3.clen+").");
		if( !( op.aggOp.increOp.fn instanceof KahanPlus && op.binaryFn instanceof Multiply) )
			throw new DMLRuntimeException("Unsupported operator for aggregate tertiary operations.");
			
		//early abort if any block is empty
		if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) || m3.isEmptyBlock(false) )
			return new DoubleObject(0);
		
		//setup meta data (dimensions, sparsity)
		int rlen = m1.rlen;
		
		//compute block operations
		KahanObject kbuff = new KahanObject(0, 0);
		KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
		
		if( !m1.sparse && !m2.sparse && !m3.sparse ) //DENSE
		{
			double[] a = m1.denseBlock;
			double[] b = m2.denseBlock;
			double[] c = m3.denseBlock;
			
			for( int i=0; i<rlen; i++ ) {
				double val = a[i] * b[i] * c[i];
				kplus.execute2( kbuff, val );
			}
		}
		else //GENERAL CASE
		{
			for( int i=0; i<rlen; i++ ) {
				double val1 = m1.quickGetValue(i, 0);
				double val2 = m2.quickGetValue(i, 0);
				double val3 = m3.quickGetValue(i, 0);
				double val = val1 * val2 * val3;
				kplus.execute2( kbuff, val );
			}
		}
		
		//create output
		DoubleObject ret = new DoubleObject(kbuff._sum);
		
		return ret;	
	}
	
	/**
	 * Invocation from CP instructions. The aggregate is computed on the groups object
	 * against target and weights. 
	 * 
	 * Notes:
	 * * The computed number of groups is reused for multiple invocations with different target.
	 * * This implementation supports that the target is passed as column or row vector,
	 *   in case of row vectors we also use sparse-safe implementations for sparse safe
	 *   aggregation operators.
	 * 
	 * @param tgt
	 * @param wghts
	 * @param ret
	 * @param op
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public MatrixValue groupedAggOperations(MatrixValue tgt, MatrixValue wghts, MatrixValue ret, int ngroups, Operator op) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//setup input matrices
		// this <- groups
		MatrixBlock target = checkType(tgt);
		MatrixBlock weights = checkType(wghts);
		
		//check valid dimensions
		if( this.getNumColumns() != 1 || (weights!=null && weights.getNumColumns()!=1) )
			throw new DMLRuntimeException("groupedAggregate can only operate on 1-dimensional column matrices for groups and weights.");
		if( target.getNumColumns() != 1 && op instanceof CMOperator )
			throw new DMLRuntimeException("groupedAggregate can only operate on 1-dimensional column matrices for target (for this aggregation function).");
		if( target.getNumColumns() != 1 && target.getNumRows()!=1 )
			throw new DMLRuntimeException("groupedAggregate can only operate on 1-dimensional column or row matrix for target.");
		if( this.getNumRows() != Math.max(target.getNumRows(),target.getNumColumns()) || (weights != null && this.getNumRows() != weights.getNumRows()) ) 
			throw new DMLRuntimeException("groupedAggregate can only operate on matrices with equal dimensions.");
		
		// obtain numGroups from instruction, if provided
		if (ngroups > 0)
			numGroups = ngroups;
		
		// Determine the number of groups
		if( numGroups <= 0 ) //reuse if available
		{
			double min = this.min();
			double max = this.max();
			
			if ( min <= 0 )
				throw new DMLRuntimeException("Invalid value (" + min + ") encountered in 'groups' while computing groupedAggregate");
			if ( max <= 0 )
				throw new DMLRuntimeException("Invalid value (" + max + ") encountered in 'groups' while computing groupedAggregate.");
		
			numGroups = (int) max;
		}
	
		// Allocate result matrix
		MatrixBlock result = checkType(ret);
		boolean result_sparsity = estimateSparsityOnGroupedAgg(rlen, numGroups);
		if(result==null)
			result=new MatrixBlock(numGroups, 1, result_sparsity);
		else
			result.reset(numGroups, 1, result_sparsity);

		// Compute the result
		double w = 1; // default weight
		
		//CM operator for count, mean, variance
		//note: current support only for column vectors
		if(op instanceof CMOperator) {
			// initialize required objects for storing the result of CM operations
			CM cmFn = CM.getCMFnObject(((CMOperator) op).getAggOpType());
			CM_COV_Object[] cmValues = new CM_COV_Object[numGroups];
			for ( int i=0; i < numGroups; i++ )
				cmValues[i] = new CM_COV_Object();
			
			for ( int i=0; i < this.getNumRows(); i++ ) {
				int g = (int) this.quickGetValue(i, 0);
				if ( g > numGroups )
					continue;
				double d = target.quickGetValue(i,0);
				if ( weights != null )
					w = weights.quickGetValue(i,0);
				// cmValues is 0-indexed, whereas range of values for g = [1,numGroups]
				cmFn.execute(cmValues[g-1], d, w); 
			}
			
			// extract the required value from each CM_COV_Object
			for ( int i=0; i < numGroups; i++ )
				// result is 0-indexed, so is cmValues
				result.quickSetValue(i, 0, cmValues[i].getRequiredResult(op));
		}
		//Aggregate operator for sum (via kahan sum)
		//note: support for row/column vectors and dense/sparse
		else if( op instanceof AggregateOperator ) 
		{
			//the only aggregate operator that is supported here is sum,
			//furthermore, we always use KahanPlus and hence aggop.correctionExists is true
			
			AggregateOperator aggop = (AggregateOperator) op;
				
			//default case for aggregate(sum)
			groupedAggregateKahanPlus(target, weights, result, aggop);
		}
		else
			throw new DMLRuntimeException("Invalid operator (" + op + ") encountered while processing groupedAggregate.");
		
		return result;
	}
	

	/**
	 * This is a specific implementation for aggregate(fn="sum"), where we use KahanPlus for numerical
	 * stability. In contrast to other functions of aggregate, this implementation supports row and column
	 * vectors for target and exploits sparse representations since KahanPlus is sparse-safe.
	 * 
	 * @param target
	 * @param weights
	 * @param op
	 * @throws DMLRuntimeException 
	 */
	private void groupedAggregateKahanPlus( MatrixBlock target, MatrixBlock weights, MatrixBlock result, AggregateOperator aggop ) throws DMLRuntimeException
	{
		boolean rowVector = target.getNumColumns()>1;
		double w = 1; //default weight
		
		//skip empty blocks (sparse-safe operation)
		if( target.isEmptyBlock(false) ) 
			return;
		
		//init group buffers
		KahanObject[] buffer = new KahanObject[numGroups];
		for(int i=0; i < numGroups; i++ )
			buffer[i] = new KahanObject(aggop.initialValue, 0);
			
		if( rowVector ) //target is rowvector
		{	
			if( target.sparse ) //SPARSE target
			{
				if( target.sparseRows[0]!=null )
				{
					int len = target.sparseRows[0].size();
					int[] aix = target.sparseRows[0].getIndexContainer();
					double[] avals = target.sparseRows[0].getValueContainer();	
					for( int j=0; j<len; j++ ) //for each nnz
					{
						int g = (int) this.quickGetValue(aix[j], 0);		
						if ( g > numGroups )
							continue;
						if ( weights != null )
							w = weights.quickGetValue(aix[j],0);
						aggop.increOp.fn.execute(buffer[g-1], avals[j]*w);						
					}
				}
					
			}
			else //DENSE target
			{
				for ( int i=0; i < target.getNumColumns(); i++ ) {
					double d = target.denseBlock[ i ];
					if( d != 0 ) //sparse-safe
					{
						int g = (int) this.quickGetValue(i, 0);		
						if ( g > numGroups )
							continue;
						if ( weights != null )
							w = weights.quickGetValue(i,0);
						// buffer is 0-indexed, whereas range of values for g = [1,numGroups]
						aggop.increOp.fn.execute(buffer[g-1], d*w);
					}
				}
			}
		}
		else //column vector (always dense, but works for sparse as well)
		{
			for ( int i=0; i < this.getNumRows(); i++ ) 
			{
				double d = target.quickGetValue(i,0);
				if( d != 0 ) //sparse-safe
				{
					int g = (int) this.quickGetValue(i, 0);		
					if ( g > numGroups )
						continue;
					if ( weights != null )
						w = weights.quickGetValue(i,0);
					// buffer is 0-indexed, whereas range of values for g = [1,numGroups]
					aggop.increOp.fn.execute(buffer[g-1], d*w);
				}
			}
		}
		
		// extract the results from group buffers
		for ( int i=0; i < numGroups; i++ )
			result.quickSetValue(i, 0, buffer[i]._sum);
	}

	public MatrixValue removeEmptyOperations( MatrixValue ret, boolean rows )
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{	
		//check for empty inputs 
		//(the semantics of removeEmpty are that for an empty m-by-n matrix, the output 
		//is an empty 1-by-n or m-by-1 matrix because we dont allow matrices with dims 0)
		if( isEmptyBlock(false) ) {
			if( rows )
				ret.reset(1, clen, sparse);
			else //cols
				ret.reset(rlen, 1, sparse);	
			return ret;
		}
		
		MatrixBlock result = checkType(ret);
		
		if( rows )
			return removeEmptyRows(result);
		else //cols
			return removeEmptyColumns(result);
	}
	
	/**
	 * 
	 * @param ret
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private MatrixBlock removeEmptyRows(MatrixBlock ret) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{	
		final int m = rlen;
		final int n = clen;
		
		//Step 1: scan block and determine non-empty rows
		boolean[] flags = new boolean[ m ]; //false
		int rlen2 = 0; 
		if( sparse ) //SPARSE 
		{
			SparseRow[] a = sparseRows;
			
			for ( int i=0; i < m; i++ )
				if ( a[i] != null && !a[i].isEmpty() ) {
					flags[i] = true;
					rlen2++;
				}
		}
		else //DENSE
		{
			double[] a = denseBlock;
			
			for(int i=0, aix=0; i<m; i++, aix+=n) {
				for(int j=0; j<n; j++)
					if( a[aix+j] != 0 )
					{
						flags[i] = true;
						rlen2++;
						//early abort for current row
						break; 
					}
			}
		}

		//Step 2: reset result and copy rows
		//dense stays dense if correct input representation (but robust for any input), 
		//sparse might be dense/sparse
		rlen2 = Math.max(rlen2, 1); //ensure valid output
		boolean sp = evalSparseFormatInMemory(rlen2, n, nonZeros);
		ret.reset(rlen2, n, sp);
		
		if( sparse ) //* <- SPARSE
		{
			//note: output dense or sparse
			for( int i=0, cix=0; i<m; i++ )
				if( flags[i] )
					ret.appendRow(cix++, sparseRows[i]);
		}
		else if( !sparse && !ret.sparse )  //DENSE <- DENSE
		{
			ret.allocateDenseBlock();
			double[] a = denseBlock;
			double[] c = ret.denseBlock;
			
			for( int i=0, aix=0, cix=0; i<m; i++, aix+=n )
				if( flags[i] ) {
					System.arraycopy(a, aix, c, cix, n);
					cix += n; //target index
				}
		}
		else //SPARSE <- DENSE
		{
			ret.allocateSparseRowsBlock();
			double[] a = denseBlock;
			
			for( int i=0, aix=0, cix=0; i<m; i++, aix+=n )
				if( flags[i] ) {
					for( int j=0; j<n; j++ )
						ret.appendValue(cix, j, a[aix+j]);
					cix++;
				}
		}
		
		//check sparsity
		ret.nonZeros = this.nonZeros;
		ret.examSparsity();

		return ret;
	}
	
	/**
	 * 
	 * @param ret
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private MatrixBlock removeEmptyColumns(MatrixBlock ret) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		final int m = rlen;
		final int n = clen;
		
		//Step 1: scan block and determine non-empty columns 
		//(we optimized for cache-friendly behavior and hence don't do early abort)
		boolean[] flags = new boolean[ n ]; //false 
		
		if( sparse ) //SPARSE 
		{
			SparseRow[] a = sparseRows;
			
			for( int i=0; i<m; i++ ) 
				if ( a[i] != null && !a[i].isEmpty() ) {
					int alen = a[i].size();
					int[] aix = a[i].getIndexContainer();
					for( int j=0; j<alen; j++ )
						flags[ aix[j] ] = true;
				}
		}
		else //DENSE
		{
			double[] a = denseBlock;
			
			for(int i=0, aix=0; i<m; i++)
				for(int j=0; j<n; j++, aix++)
					if( a[aix] != 0 )
						flags[j] = true; 	
		}
		
		//Step 2: determine number of columns
		int clen2 = 0;
		for( int j=0; j<n; j++ )
			clen2 += flags[j] ? 1 : 0;
		
		//Step 3: create mapping of flags to target indexes
		int[] cix = new int[n];
		for( int j=0, pos=0; j<n; j++ )
			if( flags[j] )
				cix[j] = pos++;	
		
		//Step 3: reset result and copy cols
		//dense stays dense if correct input representation (but robust for any input), 
		// sparse might be dense/sparse
		clen2 = Math.max(clen2, 1); //ensure valid output
		boolean sp = evalSparseFormatInMemory(m, clen2, nonZeros);
		ret.reset(m, clen2, sp);
			
		if( sparse ) //* <- SPARSE 
		{
			//note: output dense or sparse
			SparseRow[] a = sparseRows;
			
			for( int i=0; i<m; i++ ) 
				if ( a[i] != null && !a[i].isEmpty() ) {
					int alen = a[i].size();
					int[] aix = a[i].getIndexContainer();
					double[] avals = a[i].getValueContainer();
					for( int j=0; j<alen; j++ )
						ret.appendValue(i, cix[aix[j]], avals[j]);
				}
		}
		else if( !sparse && !ret.sparse )  //DENSE <- DENSE
		{
			ret.allocateDenseBlock();
			double[] a = denseBlock;
			double[] c = ret.denseBlock;
			
			for(int i=0, aix=0, lcix=0; i<m; i++, lcix+=clen2)
				for(int j=0; j<n; j++, aix++)
					if( a[aix] != 0 )
						 c[ lcix+cix[j] ] = a[aix];	
		}
		else //SPARSE <- DENSE
		{
			ret.allocateSparseRowsBlock();
			double[] a = denseBlock;
			
			for(int i=0, aix=0; i<m; i++)
				for(int j=0; j<n; j++, aix++)
					if( a[aix] != 0 )
						 ret.appendValue(i, cix[j], a[aix]);	
		}
		
		//check sparsity
		ret.nonZeros = this.nonZeros;
		ret.examSparsity();
		
		return ret;
	}
	
	@Override
	public MatrixValue replaceOperations(MatrixValue result, double pattern, double replacement) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlock ret = checkType(result);
		examSparsity(); //ensure its in the right format
		ret.reset(rlen, clen, sparse);
		if( nonZeros == 0 && pattern != 0  )
			return ret; //early abort
		boolean NaNpattern = Double.isNaN(pattern);
		
		if( sparse ) //SPARSE
		{
			if( pattern != 0d ) //SPARSE <- SPARSE (sparse-safe)
			{
				ret.allocateSparseRowsBlock();
				SparseRow[] a = sparseRows;
				SparseRow[] c = ret.sparseRows;
				
				for( int i=0; i<rlen; i++ )
				{
					SparseRow arow = a[ i ];
					if( arow!=null && !arow.isEmpty() )
					{
						SparseRow crow = new SparseRow(arow.size());
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();
						for( int j=0; j<alen; j++ )
						{
							double val = avals[j];
							if( val== pattern || (NaNpattern && Double.isNaN(val)) )
								crow.append(aix[j], replacement);
							else
								crow.append(aix[j], val);
						}
						c[ i ] = crow;
					}
				}
			}
			else //DENSE <- SPARSE
			{
				ret.sparse = false;
				ret.allocateDenseBlock();	
				SparseRow[] a = sparseRows;
				double[] c = ret.denseBlock;
				
				//initialize with replacement (since all 0 values, see SPARSITY_TURN_POINT)
				Arrays.fill(c, replacement); 
				
				//overwrite with existing values (via scatter)
				if( a != null  ) //check for empty matrix
					for( int i=0, cix=0; i<rlen; i++, cix+=clen )
					{
						SparseRow arow = a[ i ];
						if( arow!=null && !arow.isEmpty() )
						{
							int alen = arow.size();
							int[] aix = arow.getIndexContainer();
							double[] avals = arow.getValueContainer();
							for( int j=0; j<alen; j++ )
								if( avals[ j ] != 0 )
									c[ cix+aix[j] ] = avals[ j ];
						}
					}
			}			
		}
		else //DENSE <- DENSE
		{
			int mn = ret.rlen * ret.clen;
			ret.allocateDenseBlock();
			double[] a = denseBlock;
			double[] c = ret.denseBlock;
			
			for( int i=0; i<mn; i++ ) 
			{
				double val = a[i];
				if( val== pattern || (NaNpattern && Double.isNaN(val)) )
					c[i] = replacement;
				else
					c[i] = a[i];
			}
		}
		
		ret.recomputeNonZeros();
		ret.examSparsity();
		
		return ret;
	}
	
	
	/**
	 *  D = ctable(A,v2,W)
	 *  this <- A; scalarThat <- v2; that2 <- W; result <- D
	 *  
	 * (i1,j1,v1) from input1 (this)
	 * (v2) from sclar_input2 (scalarThat)
	 * (i3,j3,w)  from input3 (that2)
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	@Override
	public void ternaryOperations(Operator op, double scalarThat,
			MatrixValue that2Val, CTableMap resultMap, MatrixBlock resultBlock)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		MatrixBlock that2 = checkType(that2Val);
		CTable ctable = CTable.getCTableFnObject();
		double v2 = scalarThat;
		
		//sparse-unsafe ctable execution
		//(because input values of 0 are invalid and have to result in errors) 
		if ( resultBlock == null ) {
			for( int i=0; i<rlen; i++ )
				for( int j=0; j<clen; j++ )
				{
					double v1 = this.quickGetValue(i, j);
					double w = that2.quickGetValue(i, j);
					ctable.execute(v1, v2, w, false, resultMap);
				}
		}
		else {
			for( int i=0; i<rlen; i++ )
				for( int j=0; j<clen; j++ )
				{
					double v1 = this.quickGetValue(i, j);
					double w = that2.quickGetValue(i, j);
					ctable.execute(v1, v2, w, false, resultBlock);
				}
			resultBlock.recomputeNonZeros();
		}
	}

	/**
	 *  D = ctable(A,v2,w)
	 *  this <- A; scalar_that <- v2; scalar_that2 <- w; result <- D
	 *  
	 * (i1,j1,v1) from input1 (this)
     * (v2) from sclar_input2 (scalarThat)
	 * (w)  from scalar_input3 (scalarThat2)
	 */
	@Override
	public void ternaryOperations(Operator op, double scalarThat,
			double scalarThat2, CTableMap resultMap, MatrixBlock resultBlock)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		CTable ctable = CTable.getCTableFnObject();
		double v2 = scalarThat;
		double w = scalarThat2;
		
		//sparse-unsafe ctable execution
		//(because input values of 0 are invalid and have to result in errors) 
		if ( resultBlock == null ) { 
			for( int i=0; i<rlen; i++ )
				for( int j=0; j<clen; j++ )
				{
					double v1 = this.quickGetValue(i, j);
					ctable.execute(v1, v2, w, false, resultMap);
				}
		}
		else {
			for( int i=0; i<rlen; i++ )
				for( int j=0; j<clen; j++ )
				{
					double v1 = this.quickGetValue(i, j);
					ctable.execute(v1, v2, w, false, resultBlock);
				}	
			resultBlock.recomputeNonZeros();
		}		
	}
	
	/**
	 * Specific ctable case of ctable(seq(...),X), where X is the only
	 * matrix input. The 'left' input parameter specifies if the seq appeared
	 * on the left, otherwise it appeared on the right.
	 * 
	 */
	@Override
	public void ternaryOperations(Operator op, MatrixIndexes ix1, double scalarThat,
			boolean left, int brlen, CTableMap resultMap, MatrixBlock resultBlock)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		CTable ctable = CTable.getCTableFnObject();
		double w = scalarThat;
		int offset = (int) ((ix1.getRowIndex()-1)*brlen); 
		
		//sparse-unsafe ctable execution
		//(because input values of 0 are invalid and have to result in errors) 
		if( resultBlock == null) {
			for( int i=0; i<rlen; i++ )
				for( int j=0; j<clen; j++ )
				{
					double v1 = this.quickGetValue(i, j);
					if( left )
						ctable.execute(offset+i+1, v1, w, false, resultMap);
					else
						ctable.execute(v1, offset+i+1, w, false, resultMap);
				}
		}
		else {
			for( int i=0; i<rlen; i++ )
				for( int j=0; j<clen; j++ )
				{
					double v1 = this.quickGetValue(i, j);
					if( left )
						ctable.execute(offset+i+1, v1, w, false, resultBlock);
					else
						ctable.execute(v1, offset+i+1, w, false, resultBlock);
				}
			resultBlock.recomputeNonZeros();
		}
	}

	/**
	 *  D = ctable(A,B,w)
	 *  this <- A; that <- B; scalar_that2 <- w; result <- D
	 *  
	 * (i1,j1,v1) from input1 (this)
	 * (i1,j1,v2) from input2 (that)
	 * (w)  from scalar_input3 (scalarThat2)
	 * 
	 * NOTE: This method supports both vectors and matrices. In case of matrices and ignoreZeros=true
	 * we can also use a sparse-safe implementation
	 */
	@Override
	public void ternaryOperations(Operator op, MatrixValue thatVal, double scalarThat2, boolean ignoreZeros,
			     CTableMap resultMap, MatrixBlock resultBlock)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		//setup ctable computation
		MatrixBlock that = checkType(thatVal);
		CTable ctable = CTable.getCTableFnObject();
		double w = scalarThat2;
		
		if( ignoreZeros //SPARSE-SAFE & SPARSE INPUTS
			&& this.sparse && that.sparse )
		{
			//note: only used if both inputs have aligned zeros, which
			//allows us to infer that the nnz both inputs are equivalent
			
			//early abort on empty blocks possible
			if( this.isEmptyBlock(false) && that.isEmptyBlock(false) )
				return;
			
			SparseRow[] a = this.sparseRows;
			SparseRow[] b = that.sparseRows;
			for( int i=0; i<rlen; i++ )
			{
				SparseRow arow = a[i];
				SparseRow brow = b[i];
				if( arow != null && !arow.isEmpty() )
				{
					int alen = arow.size();
					double[] avals = arow.getValueContainer();
					double[] bvals = brow.getValueContainer();
					
					if( resultBlock == null ) {
						for( int j=0; j<alen; j++ )
							ctable.execute(avals[j], bvals[j], w, ignoreZeros, resultMap);		
					}
					else {
						for( int j=0; j<alen; j++ )
							ctable.execute(avals[j], bvals[j], w, ignoreZeros, resultBlock);			
					}
				}
			}	
		}
		else //SPARSE-UNSAFE | GENERIC INPUTS
		{
			//sparse-unsafe ctable execution
			//(because input values of 0 are invalid and have to result in errors) 
			for( int i=0; i<rlen; i++ )
				for( int j=0; j<clen; j++ )
				{
					double v1 = this.quickGetValue(i, j);
					double v2 = that.quickGetValue(i, j);
					if( resultBlock == null )
						ctable.execute(v1, v2, w, ignoreZeros, resultMap);
					else
						ctable.execute(v1, v2, w, ignoreZeros, resultBlock);
				}
		}
		
		//maintain nnz (if necessary)
		if( resultBlock!=null )
			resultBlock.recomputeNonZeros();
	}
	
	/**
	 *  D = ctable(seq,A,w)
	 *  this <- seq; thatMatrix <- A; thatScalar <- w; result <- D
	 *  
	 * (i1,j1,v1) from input1 (this)
	 * (i1,j1,v2) from input2 (that)
	 * (w)  from scalar_input3 (scalarThat2)
	 */
	public void ternaryOperations(Operator op, MatrixValue thatMatrix, double thatScalar, MatrixBlock resultBlock)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		MatrixBlock that = checkType(thatMatrix);
		CTable ctable = CTable.getCTableFnObject();
		double w = thatScalar;
		
		//sparse-unsafe ctable execution
		//(because input values of 0 are invalid and have to result in errors) 		
		//resultBlock guaranteed to be allocated for ctableexpand
		//each row in resultBlock will be allocated and will contain exactly one value
		int maxCol = 0;
		for( int i=0; i<rlen; i++ ) {
			double v2 = that.quickGetValue(i, 0);
			maxCol = ctable.execute(i+1, v2, w, maxCol, resultBlock);
		}
		
		//update meta data (initially unknown number of columns)
		//note: nnz maintained in ctable (via quickset)
		resultBlock.clen = maxCol;
	}
	
	/**
	 *  D = ctable(A,B,W)
	 *  this <- A; that <- B; that2 <- W; result <- D
	 *  
	 * (i1,j1,v1) from input1 (this)
	 * (i1,j1,v2) from input2 (that)
	 * (i1,j1,w)  from input3 (that2)
	 */
	public void ternaryOperations(Operator op, MatrixValue thatVal, MatrixValue that2Val, CTableMap resultMap)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		ternaryOperations(op, thatVal, that2Val, resultMap, null);
	}
		
	@Override
	public void ternaryOperations(Operator op, MatrixValue thatVal, MatrixValue that2Val, CTableMap resultMap, MatrixBlock resultBlock)
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{	
		MatrixBlock that = checkType(thatVal);
		MatrixBlock that2 = checkType(that2Val);
		CTable ctable = CTable.getCTableFnObject();
		
		//sparse-unsafe ctable execution
		//(because input values of 0 are invalid and have to result in errors) 
		if(resultBlock == null) 
		{
			for( int i=0; i<rlen; i++ )
				for( int j=0; j<clen; j++ )
				{
					double v1 = this.quickGetValue(i, j);
					double v2 = that.quickGetValue(i, j);
					double w = that2.quickGetValue(i, j);
					ctable.execute(v1, v2, w, false, resultMap);
				}		
		}
		else 
		{
			for( int i=0; i<rlen; i++ )
				for( int j=0; j<clen; j++ )
				{
					double v1 = this.quickGetValue(i, j);
					double v2 = that.quickGetValue(i, j);
					double w = that2.quickGetValue(i, j);
					ctable.execute(v1, v2, w, false, resultBlock);
				}
			resultBlock.recomputeNonZeros();
		}
	}
	
	@Override
	public MatrixValue quaternaryOperations(Operator op, MatrixValue um, MatrixValue vm, MatrixValue wm, MatrixValue out, QuaternaryOperator qop)
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		return quaternaryOperations(op, um, vm, wm, out, qop, 1);
	}
	
	/**
	 * 
	 * @param op
	 * @param um
	 * @param vm
	 * @param wm
	 * @param out
	 * @param wt
	 * @param k
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public MatrixValue quaternaryOperations(Operator op, MatrixValue um, MatrixValue vm, MatrixValue wm, MatrixValue out, QuaternaryOperator qop, int k)
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//check input dimensions
		if( getNumRows() != um.getNumRows() )
			throw new DMLRuntimeException("Dimension mismatch rows on wsloss: "+getNumRows()+"!="+um.getNumRows());
		if( getNumColumns() != vm.getNumRows() )
			throw new DMLRuntimeException("Dimension mismatch columns on wsloss: "+getNumRows()+"!="+vm.getNumRows());
		
		//check input data types
		MatrixBlock X = this;
		MatrixBlock U = checkType(um);
		MatrixBlock V = checkType(vm);
		MatrixBlock R = checkType(out);
		
		//prepare intermediates and output
		if( qop.wtype1 != null )
			R.reset(1, 1, false);
		else
			R.reset(rlen, clen, sparse);
		
		//core block operation
		if( qop.wtype1 != null ) //wsloss
		{
			MatrixBlock W = (qop.wtype1!=WeightsType.NONE) ? checkType(wm) : null;
			if( k > 1 )
				LibMatrixMult.matrixMultWSLoss(X, U, V, W, R, qop.wtype1, k);
			else
				LibMatrixMult.matrixMultWSLoss(X, U, V, W, R, qop.wtype1);
		}
		else //wsigmoid
		{
			if( k > 1 )
				LibMatrixMult.matrixMultWSigmoid(X, U, V, R, qop.wtype2, k);
			else
				LibMatrixMult.matrixMultWSigmoid(X, U, V, R, qop.wtype2);
		}	
		
		return R;
	}
	
	////////
	// Data Generation Methods
	// (rand, sequence)
	
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
	public static MatrixBlock randOperations(int rows, int cols, double sparsity, double min, double max, String pdf, long seed) 
		throws DMLRuntimeException
	{
		DMLConfig conf = ConfigurationManager.getConfig();
		int blocksize = (conf!=null) ? ConfigurationManager.getConfig().getIntValue(DMLConfig.DEFAULT_BLOCK_SIZE)
				                     : DMLTranslator.DMLBlockSize;
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
	public static MatrixBlock randOperations(int rows, int cols, int rowsInBlock, int colsInBlock, double sparsity, double min, double max, String pdf, long seed) 
		throws DMLRuntimeException 
	{
		MatrixBlock out = new MatrixBlock();
		Well1024a bigrand = null;
		long[] nnzInBlock = null;

		//setup seeds and nnz per block
		if( !LibMatrixDatagen.isShortcutRandOperation(min, max, sparsity, pdf) ){
			bigrand = LibMatrixDatagen.setupSeedsForRand(seed);
			nnzInBlock = LibMatrixDatagen.computeNNZperBlock(rows, cols, rowsInBlock, colsInBlock, sparsity);
		}
		
		//generate rand data
		if ( pdf.equalsIgnoreCase(LibMatrixDatagen.RAND_PDF_NORMAL) ) {
			// for normally distributed values, min and max are specified as an invalid value NaN.
			out.randOperationsInPlace(pdf, rows, cols, rowsInBlock, colsInBlock, nnzInBlock, sparsity, Double.NaN, Double.NaN, bigrand, -1);
		}
		else {
			out.randOperationsInPlace(pdf, rows, cols, rowsInBlock, colsInBlock, nnzInBlock, sparsity, min, max, bigrand, -1);
		}
		
		return out;
	}
	
	/**
	 * Function to generate a matrix of random numbers. This is invoked both
	 * from CP as well as from MR. In case of CP, it generates an entire matrix
	 * block-by-block. A <code>bigrand</code> is passed so that block-level
	 * seeds are generated internally. In case of MR, it generates a single
	 * block for given block-level seed <code>bSeed</code>.
	 * 
	 * When pdf="uniform", cell values are drawn from uniform distribution in
	 * range <code>[min,max]</code>.
	 * 
	 * When pdf="normal", cell values are drawn from standard normal
	 * distribution N(0,1). The range of generated values will always be
	 * (-Inf,+Inf).
	 * 
	 * @param pdf
	 * @param rows
	 * @param cols
	 * @param rowsInBlock
	 * @param colsInBlock
	 * @param sparsity
	 * @param min
	 * @param max
	 * @param bigrand
	 * @param bSeed
	 * @return
	 * @throws DMLRuntimeException
	 */
	public MatrixBlock randOperationsInPlace(String pdf, int rows, int cols, int rowsInBlock, int colsInBlock, long[] nnzInBlock, double sparsity, double min, double max, Well1024a bigrand, long bSeed) 
		throws DMLRuntimeException
	{
		LibMatrixDatagen.generateRandomMatrix( this, pdf, rows, cols, rowsInBlock, colsInBlock, 
				                               nnzInBlock, sparsity, min, max, bigrand, bSeed );
		return this;
	}
	
	/**
	 * Method to generate a sequence according to the given parameters. The
	 * generated sequence is always in dense format.
	 * 
	 * Both end points specified <code>from</code> and <code>to</code> must be
	 * included in the generated sequence i.e., [from,to] both inclusive. Note
	 * that, <code>to</code> is included only if (to-from) is perfectly
	 * divisible by <code>incr</code>.
	 * 
	 * For example, seq(0,1,0.5) generates (0.0 0.5 1.0) 
	 *      whereas seq(0,1,0.6) generates (0.0 0.6) but not (0.0 0.6 1.0)
	 * 
	 * @param from
	 * @param to
	 * @param incr
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock seqOperations(double from, double to, double incr) 
		throws DMLRuntimeException 
	{
		MatrixBlock out = new MatrixBlock();
		LibMatrixDatagen.generateSequence( out, from, to, incr );
		
		return out;
	}
	
	/**
	 * 
	 * @param from
	 * @param to
	 * @param incr
	 * @return
	 * @throws DMLRuntimeException
	 */
	public MatrixBlock seqOperationsInPlace(double from, double to, double incr) 
		throws DMLRuntimeException 
	{
		LibMatrixDatagen.generateSequence( this, from, to, incr );
		
		return this;
	}
	
	/**
	 * 
	 * @param range
	 * @param size
	 * @param replace
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock sampleOperations(long range, int size, boolean replace) 
		throws DMLRuntimeException 
	{
		MatrixBlock out = new MatrixBlock();
		LibMatrixDatagen.generateSample( out, range, size, replace );
		
		return out;
	}
	
	////////
	// Misc methods
	
	private static MatrixBlock checkType(MatrixValue block) 
		throws RuntimeException
	{
		if( block!=null && !(block instanceof MatrixBlock))
			throw new RuntimeException("Unsupported matrix value: "+block.getClass().getSimpleName());
		return (MatrixBlock) block;
	}
	
	public void print()
	{
		System.out.println("sparse = "+sparse);
		if(!sparse)
			System.out.println("nonzeros = "+nonZeros);
		for(int i=0; i<rlen; i++)
		{
			for(int j=0; j<clen; j++)
			{
				System.out.print(quickGetValue(i, j)+"\t");
			}
			System.out.println();
		}
	}
	
	@Override
	public int compareTo(Object arg0) {
		throw new RuntimeException("CompareTo should never be called for matrix blocks.");
	}

	@Override
	public boolean equals(Object arg0) {
		MatrixBlock that = (MatrixBlock) arg0;
		
		if(this.rlen != that.rlen || this.clen != that.clen || this.nonZeros != that.nonZeros )
			return false;
		
		// TODO: this implementation is to be optimized for different block representations
		for(int i=0; i < this.rlen; i++) 
			for(int j=0; j < this.clen; j++ )
				if(this.getValue(i, j) != that.getValue(i,j)) 
					return false;
		
		return true;
	}
	
	@Override
	public int hashCode() {
		throw new RuntimeException("HashCode should never be called for matrix blocks.");
	}
	
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		
		sb.append("sparse? = ");
		sb.append(sparse);
		sb.append("\n");
		
		sb.append("nonzeros = ");
		sb.append(nonZeros);
		sb.append("\n");
		
		sb.append("size: ");
		sb.append(rlen);
		sb.append(" X ");
		sb.append(clen);
		sb.append("\n");
		
		if(sparse)
		{
			int len=0;
			if(sparseRows!=null)
				len = Math.min(rlen, sparseRows.length);
			int i=0;
			for(; i<len; i++)
			{
				sb.append("row +");
				sb.append(i);
				sb.append(": ");
				sb.append(sparseRows[i]);
				sb.append("\n");
			}
			for(; i<rlen; i++)
			{
				sb.append("row +");
				sb.append(i);
				sb.append(": null\n");
			}
		}
		else
		{
			if(denseBlock!=null)
			{
				for(int i=0, ix=0; i<rlen; i++, ix+=clen) {
					for(int j=0; j<clen; j++) {
						sb.append(this.denseBlock[ix+j]);
						sb.append("\t");
					}
					sb.append("\n");
				}
			}
		}
		
		return sb.toString();
	}


	///////////////////////////
	// Helper classes

	public static class SparsityEstimate
	{
		public long estimatedNonZeros=0;
		public boolean sparse=false;
		public SparsityEstimate(boolean sps, long nnzs)
		{
			sparse=sps;
			estimatedNonZeros=nnzs;
		}
		public SparsityEstimate(){}
	}
}
