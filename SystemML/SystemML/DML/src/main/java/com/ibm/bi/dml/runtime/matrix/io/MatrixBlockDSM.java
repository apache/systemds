/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.io;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.apache.commons.math.random.Well1024a;

import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.And;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.CTable;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.MaxIndex;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.COVOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;
import com.ibm.bi.dml.runtime.util.NormalPRNGenerator;
import com.ibm.bi.dml.runtime.util.PRNGenerator;
import com.ibm.bi.dml.runtime.util.UniformPRNGenerator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


public class MatrixBlockDSM extends MatrixValue
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//sparsity nnz threshold, based on practical experiments on space consumption and performance
	public static final double SPARCITY_TURN_POINT=0.4;
	//sparsity column threshold, based on initial capacity of sparse representation 
	public static final int SKINNY_MATRIX_TURN_POINT=4;
	//sparsity threshold for ultra-sparse matrix operations (40nnz in a 1kx1k block)
	public static final double ULTRA_SPARSITY_TURN_POINT = 0.00004; 
	
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
	protected int nonZeros   = 0;
	
	//matrix data (sparse or dense)
	protected double[] denseBlock    = null;
	protected SparseRow[] sparseRows = null;
		
	//sparse-block-specific attributes (allocation only)
	protected int estimatedNNzsPerRow = -1; 
		
	//ctable-specific attributes
	protected int maxrow;
	protected int maxcolumn;
	
	////////
	// Matrix Constructors
	//
	
	public MatrixBlockDSM()
	{
		rlen = 0;
		clen = 0;
		sparse = true;
		nonZeros = 0;
		maxrow = 0;
		maxcolumn = 0;
	}
	public MatrixBlockDSM(int rl, int cl, boolean sp)
	{
		rlen = rl;
		clen = cl;
		sparse = sp;
		nonZeros = 0;
		maxrow = 0;
		maxcolumn = 0;
	}
	
	public MatrixBlockDSM(int rl, int cl, boolean sp, int estnnzs)
	{
		this(rl, cl, sp);
		estimatedNNzsPerRow=(int)Math.ceil((double)estnnzs/(double)rl);	
	}
	
	public MatrixBlockDSM(MatrixBlockDSM that)
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
	
	public void reset(int estnnzs)
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
	}
	
	public void reset(int rl, int cl) {
		rlen=rl;
		clen=cl;
		nonZeros=0;
		reset();
	}
	
	public void reset(int rl, int cl, int estnnzs) {
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
	
	public void reset(int rl, int cl, boolean sp, int estnnzs)
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
		
		int limit=rlen*clen;
		if(denseBlock==null || denseBlock.length<limit)
			denseBlock=new double[limit];
		
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
	 */
	public void allocateDenseBlock() 
	{
		int limit=rlen*clen;
		if(denseBlock==null || denseBlock.length < limit )
			denseBlock=new double[limit];
		nonZeros = 0;
	}
	
	/**
	 * 
	 * @param r
	 */
	public void adjustSparseRows(int r)
	{
		if(sparseRows==null)
			sparseRows=new SparseRow[rlen];
		else if(sparseRows.length<=r)
		{
			SparseRow[] oldSparseRows=sparseRows;
			sparseRows=new SparseRow[rlen];
			for(int i=0; i<Math.min(oldSparseRows.length, rlen); i++) {
				sparseRows[i]=oldSparseRows[i];
			}
		}
	}
	
	
	/**
	 * This should be called only in the read and write functions for CP
	 * This function should be called before calling any setValueDenseUnsafe()
	 * 
	 * @param rl
	 * @param cl
	 */
	public void allocateDenseBlockUnsafe(int rl, int cl)
	{
		sparse=false;
		rlen=rl;
		clen=cl;
		int limit=rlen*clen;
		if(denseBlock==null || denseBlock.length<limit)
		{
			denseBlock=new double[limit];
		}else
			Arrays.fill(denseBlock, 0, limit, 0);
	}
	
	
	/**
	 * This should be called only in the read and write functions for CP
	 * This function should be called before calling any setValueSparseUnsafe() or appendValueSparseUnsafe()
	 
	 * @param rl
	 * @param cl
	 * @param estimatedmNNzs
	 */
	@Deprecated
	public void allocateSparseRowsUnsafe(int rl, int cl, int estimatedmNNzs)
	{
		sparse=true;
		rlen=rl;
		clen=cl;
		int nnzsPerRow=(int) Math.ceil((double)estimatedmNNzs/(double)rl);
		if(sparseRows!=null)
		{
			if(sparseRows.length>=rlen)
			{
				for(int i=0; i<rlen; i++)
				{
					if(sparseRows[i]==null)
						sparseRows[i]=new SparseRow(nnzsPerRow, cl);
					else
						sparseRows[i].reset(nnzsPerRow, cl);
				}
			}else
			{
				SparseRow[] temp=sparseRows;
				sparseRows=new SparseRow[rlen];
				int i=0;
				for(; i<temp.length; i++)
				{
					if(temp[i]!=null)
					{
						sparseRows[i]=temp[i];
						sparseRows[i].reset(nnzsPerRow, cl);
					}else
						sparseRows[i]=new SparseRow(nnzsPerRow, cl);
				}
				for(; i<rlen; i++)
					sparseRows[i]=new SparseRow(nnzsPerRow, cl);
			}
			
		}else
		{
			sparseRows=new SparseRow[rlen];
			for(int i=0; i<rlen; i++)
				sparseRows[i]=new SparseRow(nnzsPerRow, cl);
		}
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
	
	public int getNonZeros()
	{
		return nonZeros;
	}
	
	/**
	 * Returns the current representation (true for sparse).
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
	 * Returns the exact representation once it is written or
	 * exam sparsity is called.
	 * 
	 * @return
	 */
	public boolean isExactInSparseFormat()
	{
		long lrlen = (long) rlen;
		long lclen = (long) clen;
		long lnonZeros = (long) nonZeros;
			
		//ensure exact size estimates for write
		if( sparse || lnonZeros<(lrlen*lclen)*SPARCITY_TURN_POINT )
		{
			recomputeNonZeros();
			lnonZeros = (long) nonZeros;
		}
		
		return (lnonZeros<(lrlen*lclen)*SPARCITY_TURN_POINT && lclen>SKINNY_MATRIX_TURN_POINT);
	}
	
	/**
	 * 
	 * @param rlen
	 * @param clen
	 * @param nonZeros
	 * @return
	 */
	public static boolean isExactInSparseFormat(long rlen, long clen, long nonZeros)
	{		
		return (nonZeros<(rlen*clen)*SPARCITY_TURN_POINT && clen>SKINNY_MATRIX_TURN_POINT);
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
	
	public SparseCellIterator getSparseCellIterator()
	{
		if(!sparse)
			throw new RuntimeException("getSparseCellInterator should not be called for dense format");
		return new SparseCellIterator(rlen, sparseRows);
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
					if(sparseRows[r]==null) continue;
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
					if(sparseRows[r]==null) continue;
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
			adjustSparseRows(r);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(estimatedNNzsPerRow, clen);
			
			if(sparseRows[r].set(c, v))
				nonZeros++;
			
		}else
		{
			if(denseBlock==null && v==0.0)
				return;
			
			int limit=rlen*clen;
			if(denseBlock==null || denseBlock.length<limit)
			{
				denseBlock=new double[limit];
				//Arrays.fill(denseBlock, 0, limit, 0);
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
	 * If not, add a new cell with index (r,c)
	 */
	public void addValue(int r, int c, double v) {
		if(sparse)
		{
			adjustSparseRows(r);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(estimatedNNzsPerRow, clen);
			double curV=sparseRows[r].get(c);
			if(curV==0)
				nonZeros++;
			curV+=v;
			if(curV==0)
				nonZeros--;
			else
				sparseRows[r].set(c, curV);
			
		}
		else
		{
			int limit=rlen*clen;
			if(denseBlock==null)
			{
				denseBlock=new double[limit];
				Arrays.fill(denseBlock, 0, limit, 0);
			}
			
			int index=r*clen+c;
			if(denseBlock[index]==0)
				nonZeros++;
			denseBlock[index]+=v;
			if(denseBlock[index]==0)
				nonZeros--;
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
			adjustSparseRows(r);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(estimatedNNzsPerRow, clen);
			
			if(sparseRows[r].set(c, v))
				nonZeros++;
			
		}
		else
		{
			if(denseBlock==null && v==0.0)
				return;		
			int limit=rlen*clen;
			if(denseBlock==null || denseBlock.length<limit)
			{
				denseBlock=new double[limit];
				//Arrays.fill(denseBlock, 0, limit, 0);
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
	 * This can be only called when you know you have properly allocated spaces for a sparse representation
	 * and r and c are in the the range of the dimension
	 * Note: this function won't keep track of the nozeros	
	 */
	public void setValueSparseUnsafe(int r, int c, double v) 
	{
		sparseRows[r].set(c, v);		
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
			adjustSparseRows(r);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(estimatedNNzsPerRow, clen);
			/*else { 
				if (sparseRows[r].capacity()==0) {
					System.out.println(" ... !null: " + sparseRows[r].size() + ", " + sparseRows[r].capacity() + ", " + sparseRows[r].getValueContainer().length + ", " + sparseRows[r].estimatedNzs + ", " + sparseRows[r].maxNzs);
				}
			}*/
			sparseRows[r].append(c, v);
			nonZeros++;
		}
	}

	/**
	 * This can be only called when you know you have properly allocated spaces for a sparse representation
	 * and r and c are in the the range of the dimension
	 * Note: this function won't keep track of the nozeros
	 * This can only be called, when the caller knows the access pattern of the block
	 */
	public void appendValueSparseUnsafe(int r, int c, double v) 
	{
		sparseRows[r].append(c, v);		
	}
	
	public void appendRow(int r, SparseRow values)
	{
		if(values==null)
			return;
		if(sparse)
		{
			adjustSparseRows(r);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(values);
			else
				sparseRows[r].copy(values);
			nonZeros+=values.size();
			
		}else
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
	public void appendToSparse( MatrixBlockDSM that, int rowoffset, int coloffset ) 
	{
		if( that==null || that.isEmptyBlock(false) )
			return; //nothing to append
		
		//init sparse rows if necessary
		//adjustSparseRows(rlen-1);
		
		if( that.sparse ) //SPARSE <- SPARSE
		{
			for( int i=0; i<that.rlen; i++ )
			{
				SparseRow brow = that.sparseRows[i];
				if( brow!=null && brow.size()>0 )
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

	////////
	// basic block handling functions
	
	public void examSparsity() 
		throws DMLRuntimeException
	{
		double sp = ((double)nonZeros/rlen)/clen;
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if(sparse)
		{
			if(sp>=SPARCITY_TURN_POINT || clen<=SKINNY_MATRIX_TURN_POINT) {
				//System.out.println("Calling sparseToDense(): nz=" + nonZeros + ", rlen=" + rlen + ", clen=" + clen + ", sparsity = " + sp + ", spturn=" + SPARCITY_TURN_POINT );
				sparseToDense();
			}
		}
		else
		{
			if(sp<SPARCITY_TURN_POINT && clen>SKINNY_MATRIX_TURN_POINT) {
				//System.out.println("Calling denseToSparse(): nz=" + nonZeros + ", rlen=" + rlen + ", clen=" + clen + ", sparsity = " + sp + ", spturn=" + SPARCITY_TURN_POINT );
				denseToSparse();
			}
		}
	}	
	
	
	private void denseToSparse() 
	{	
		//LOG.info("**** denseToSparse: "+this.getNumRows()+"x"+this.getNumColumns()+"  nonZeros: "+this.nonZeros);
		sparse=true;
		adjustSparseRows(rlen-1);
		reset();
		if(denseBlock==null)
			return;
		int index=0;
		for(int r=0; r<rlen; r++)
		{
			for(int c=0; c<clen; c++)
			{
				if(denseBlock[index]!=0)
				{
					if(sparseRows[r]==null) //create sparse row only if required
						sparseRows[r]=new SparseRow(estimatedNNzsPerRow, clen);
					
					sparseRows[r].append(c, denseBlock[index]);
					nonZeros++;
				}
				index++;
			}
		}
				
		//cleanup dense block
		denseBlock = null;
	}
	
	private void sparseToDense() 
		throws DMLRuntimeException 
	{	
		//LOG.info("**** sparseToDense: "+this.getNumRows()+"x"+this.getNumColumns()+"  nonZeros: "+this.nonZeros);
		
		sparse=false;
		int limit=rlen*clen;
		if ( limit < 0 ) {
			throw new DMLRuntimeException("Unexpected error in sparseToDense().. limit < 0: " + rlen + ", " + clen + ", " + limit);
		}
		
		allocateDenseBlock();
		Arrays.fill(denseBlock, 0, limit, 0);
		nonZeros=0;
		
		if(sparseRows==null)
			return;
		
		for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
		{
			if(sparseRows[r]==null) continue;
			int[] cols=sparseRows[r].getIndexContainer();
			double[] values=sparseRows[r].getValueContainer();
			for(int i=0; i<sparseRows[r].size(); i++)
			{
				if(values[i]==0) continue;
				denseBlock[r*clen+cols[i]]=values[i];
				nonZeros++;
			}
			//sparseRows[r].setValueContainer(null);
			//sparseRows[r].setIndexContainer(null);
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
						if(sparseRows[i]!=null && sparseRows[i].size()>0)
							nnz+=sparseRows[i].size();	
				}
				else if( cl==cu ) //specific case: one column
				{
					for(int i=rl; i<rlimit; i++)
						if(sparseRows[i]!=null && sparseRows[i].size()>0)
							nnz += (sparseRows[i].get(cl)!=0) ? 1 : 0;
				}
				else //general case
				{
					int astart,aend;
					for(int i=rl; i<rlimit; i++)
						if(sparseRows[i]!=null && sparseRows[i].size()>0)
						{
							SparseRow arow = sparseRows[i];
							astart = arow.searchIndexesFirstGTE(cl);
							aend = arow.searchIndexesFirstGTE(cu);
							nnz += (astart!=-1) ? (aend-astart+1) : 0;
						}
				}
			}
		}else
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
	

	public void copy(MatrixValue thatValue) 
	{
		MatrixBlockDSM that;
		try {
			that = checkType(thatValue);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		}
		
		if( this == that ) //prevent data loss (e.g., on sparse-dense conversion)
			throw new RuntimeException( "Copy must not overwrite itself!" );
		
		this.rlen=that.rlen;
		this.clen=that.clen;
		this.sparse=checkRealSparsity(that);
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
	
	public void copy(MatrixValue thatValue, boolean sp) {
		MatrixBlockDSM that;
		try {
			that = checkType(thatValue);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		}	
		
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
	
	private void copySparseToSparse(MatrixBlockDSM that)
	{
		this.nonZeros=that.nonZeros;
		if(that.sparseRows==null)
		{
			resetSparse();
			return;
		}
	
		adjustSparseRows(Math.min(that.rlen, that.sparseRows.length)-1);
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
	
	private void copyDenseToDense(MatrixBlockDSM that)
	{
		this.nonZeros=that.nonZeros;
		
		if(that.denseBlock==null)
		{
			if(denseBlock!=null)
				Arrays.fill(denseBlock, 0);
			return;
		}
		int limit=rlen*clen;
		if(denseBlock==null || denseBlock.length<limit)
			denseBlock=new double[limit];
		System.arraycopy(that.denseBlock, 0, this.denseBlock, 0, limit);
	}
	
	private void copySparseToDense(MatrixBlockDSM that)
	{
		this.nonZeros=that.nonZeros;
		if(that.sparseRows==null)
		{
			if(denseBlock!=null)
				Arrays.fill(denseBlock, 0);
			return;
		}
		int limit=rlen*clen;
		if(denseBlock==null || denseBlock.length<limit)
			denseBlock=new double[limit];
		else
			Arrays.fill(denseBlock, 0, limit, 0);
		int start=0;
		for(int r=0; r<Math.min(that.sparseRows.length, rlen); r++, start+=clen)
		{
			if(that.sparseRows[r]==null) continue;
			double[] values=that.sparseRows[r].getValueContainer();
			int[] cols=that.sparseRows[r].getIndexContainer();
			for(int i=0; i<that.sparseRows[r].size(); i++)
			{
				denseBlock[start+cols[i]]=values[i];
			}
		}
	}
	
	private void copyDenseToSparse(MatrixBlockDSM that)
	{
		nonZeros = that.nonZeros;
		if(that.denseBlock==null)
		{
			resetSparse();
			return;
		}
		
		adjustSparseRows(rlen-1);
	
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
	 */
	public void copy(int rl, int ru, int cl, int cu, MatrixBlockDSM src, boolean awareDestNZ ) 
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

	private void copySparseToSparse(int rl, int ru, int cl, int cu, MatrixBlockDSM src, boolean awareDestNZ)
	{	
		//handle empty src and dest
		if(src.sparseRows==null)
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
			if( arow != null && arow.size()>0 )
			{
				alen = arow.size();
				aix = arow.getIndexContainer();
				avals = arow.getValueContainer();		
				
				if( sparseRows[rl+i] == null || sparseRows[rl+i].size()==0  )
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
							brow.deleteIndex(cl);
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
	
	private void copySparseToDense(int rl, int ru, int cl, int cu, MatrixBlockDSM src, boolean awareDestNZ)
	{	
		//handle empty src and dest
		if(src.sparseRows==null)
		{
			if( awareDestNZ && denseBlock != null ) {
				nonZeros -= (int)recomputeNonZeros(rl, ru, cl, cu);
				copyEmptyToDense(rl, ru, cl, cu);
			}
			return;		
		}
		if(denseBlock==null)
			allocateDenseBlock();
		else if( awareDestNZ )
		{
			nonZeros -= (int)recomputeNonZeros(rl, ru, cl, cu);
			copyEmptyToDense(rl, ru, cl, cu);
		}

		//copy values
		int alen;
		int[] aix;
		double[] avals;
		
		for( int i=0, ix=rl*clen; i<src.rlen; i++, ix+=clen )
		{	
			SparseRow arow = src.sparseRows[i];
			if( arow != null && arow.size()>0 )
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

	private void copyDenseToSparse(int rl, int ru, int cl, int cu, MatrixBlockDSM src, boolean awareDestNZ)
	{
		//handle empty src and dest
		if(src.denseBlock==null)
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
			if( sparseRows[rix]==null || sparseRows[rix].size()==0 )
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
						brow.deleteIndex(cl);
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
	
	private void copyDenseToDense(int rl, int ru, int cl, int cu, MatrixBlockDSM src, boolean awareDestNZ)
	{
		//handle empty src and dest
		if(src.denseBlock==null)
		{
			if( awareDestNZ && denseBlock != null ) {
				nonZeros -= (int)recomputeNonZeros(rl, ru, cl, cu);
				copyEmptyToDense(rl, ru, cl, cu);
			}
			return;		
		}
		if(denseBlock==null)
			allocateDenseBlock();
		//no need to clear for awareDestNZ since overwritten 
	
		if( awareDestNZ )
			nonZeros = nonZeros - (int)recomputeNonZeros(rl, ru, cl, cu) + src.nonZeros;
		
		//copy values
		int rowLen = cu-cl+1;				
		if(clen == src.clen) //optimization for equal width
			System.arraycopy(src.denseBlock, 0, denseBlock, rl*clen+cl, src.rlen*src.clen);
		else
			for( int i=0, ix1=0, ix2=rl*clen+cl; i<src.rlen; i++, ix1+=src.clen, ix2+=clen )
				System.arraycopy(src.denseBlock, ix1, denseBlock, ix2, rowLen);
	}
	
	
	public void copyRowArrayToDense(int destRow, double[] src, int sourceStart)
	{
		//handle empty dest
		if(denseBlock==null)
			allocateDenseBlock();
		//no need to clear for awareDestNZ since overwritten 
		System.arraycopy(src, sourceStart, denseBlock, destRow*clen, clen);
	}
	
	private void copyEmptyToSparse(int rl, int ru, int cl, int cu, boolean updateNNZ ) 
	{
		if( cl==cu ) //specific case: column vector
		{
			if( updateNNZ )
			{
				for( int i=rl; i<=ru; i++ )
					if( sparseRows[i] != null && sparseRows[i].size()>0 )
					{
						int lnnz = sparseRows[i].size();
						sparseRows[i].deleteIndex(cl);
						nonZeros += (sparseRows[i].size()-lnnz);
					}
			}
			else
			{
				for( int i=rl; i<=ru; i++ )
					if( sparseRows[i] != null && sparseRows[i].size()>0 )
						sparseRows[i].deleteIndex(cl);
			}
		}
		else
		{
			if( updateNNZ )
			{
				for( int i=rl; i<=ru; i++ )
					if( sparseRows[i] != null && sparseRows[i].size()>0 )
					{
						int lnnz = sparseRows[i].size();
						sparseRows[i].deleteIndexRange(cl, cu);
						nonZeros += (sparseRows[i].size()-lnnz);
					}						
			}
			else
			{
				for( int i=rl; i<=ru; i++ )
					if( sparseRows[i] != null && sparseRows[i].size()>0 )
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

	////////
	// Input/Output functions
	
	@Override
	public void readFields(DataInput in) 
		throws IOException 
	{
		rlen = in.readInt();
		clen = in.readInt();
		byte bformat = in.readByte();
		if( bformat<0 || bformat>=BlockType.values().length )
			throw new IOException("invalid format: '"+bformat+"' (need to be 0-"+BlockType.values().length+".");
						
		BlockType format=BlockType.values()[bformat];
		switch(format)
		{
			case ULTRA_SPARSE_BLOCK:
				sparse = true;
				cleanupBlock(true, true); //clean all
				readUltraSparseBlock(in);
				break;
			case SPARSE_BLOCK:
				sparse = true;
				cleanupBlock(true, false); //reuse sparse
				readSparseBlock(in);
				break;
			case DENSE_BLOCK:
				sparse = false;
				cleanupBlock(false, true); //reuse dense
				readDenseBlock(in);
				break;
			case EMPTY_BLOCK:
				sparse = true;
				cleanupBlock(true, true); //clean all
				nonZeros = 0;
				break;
		}
	}

	private void readDenseBlock(DataInput in) throws IOException {
		int limit=rlen*clen;
		if(denseBlock==null || denseBlock.length < limit )
			denseBlock=new double[limit];
		nonZeros=0;
		for(int i=0; i<limit; i++)
		{
			denseBlock[i]=in.readDouble();
			if(denseBlock[i]!=0)
				nonZeros++;
		}
	}
	
	private void readSparseBlock(DataInput in) throws IOException {
				
		adjustSparseRows(rlen-1);
		nonZeros=0;
		for(int r=0; r<rlen; r++)
		{
			int nr=in.readInt();
			nonZeros+=nr;
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
	
	private void readUltraSparseBlock(DataInput in) throws IOException 
	{	
		adjustSparseRows(rlen-1); //adjust to size
		resetSparse(); //reset all sparse rows
		
		//at least 1 nonZero, otherwise empty block
		nonZeros = in.readInt(); 
		
		for(int i=0; i<nonZeros; i++)
		{
			int r = in.readInt();
			int c = in.readInt();
			double val = in.readDouble();			
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(1,clen);
			sparseRows[r].append(c, val);
		}
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(rlen);
		out.writeInt(clen);
		
		if(sparse)
		{
			if(sparseRows==null || nonZeros==0) //MB or cond
				writeEmptyBlock(out);
			else if( nonZeros<rlen && nonZeros<((long)rlen)*((long)clen)*SPARCITY_TURN_POINT && clen>SKINNY_MATRIX_TURN_POINT )
				writeSparseToUltraSparse(out); //MB new write
			//if it should be dense, then write to the dense format
			else if(nonZeros>=((long)rlen)*((long)clen)*SPARCITY_TURN_POINT || clen<=SKINNY_MATRIX_TURN_POINT)
				writeSparseToDense(out);
			else
				writeSparseBlock(out);
		}else
		{
			if(denseBlock==null || nonZeros==0) //MB or cond
				writeEmptyBlock(out);
			//if it should be sparse
			else if( nonZeros<rlen && nonZeros<((long)rlen)*((long)clen)*SPARCITY_TURN_POINT && clen>SKINNY_MATRIX_TURN_POINT )
				writeDenseToUltraSparse(out);
			else if(nonZeros<((long)rlen)*((long)clen)*SPARCITY_TURN_POINT && clen>SKINNY_MATRIX_TURN_POINT)
				writeDenseToSparse(out);
			else
				writeDenseBlock(out);
		}
	}
	
	private void writeEmptyBlock(DataOutput out) throws IOException
	{
		//empty blocks do not need to materialize row information
		out.writeByte( BlockType.EMPTY_BLOCK.ordinal() );
	}
	
	private void writeDenseBlock(DataOutput out) throws IOException 
	{
		out.writeByte( BlockType.DENSE_BLOCK.ordinal() );
		
		int limit=rlen*clen;
		if( out instanceof MatrixBlockDSMDataOutput ) //fast serialize
			((MatrixBlockDSMDataOutput)out).writeDoubleArray(limit, denseBlock);
		else //general case (if fast serialize not supported)
			for(int i=0; i<limit; i++)
				out.writeDouble(denseBlock[i]);
	}
	
	private void writeSparseBlock(DataOutput out) throws IOException 
	{
		out.writeByte( BlockType.SPARSE_BLOCK.ordinal() );
		
		if( out instanceof MatrixBlockDSMDataOutput ) //fast serialize
			((MatrixBlockDSMDataOutput)out).writeSparseRows(rlen, sparseRows);
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
	
	private void writeSparseToUltraSparse(DataOutput out) throws IOException 
	{
		out.writeByte( BlockType.ULTRA_SPARSE_BLOCK.ordinal() );
		out.writeInt(nonZeros);

		for(int r=0;r<Math.min(rlen, sparseRows.length); r++)
			if(sparseRows[r]!=null && sparseRows[r].size()>0 )
			{
				int alen = sparseRows[r].size();
				int[] aix = sparseRows[r].getIndexContainer();
				double[] avals = sparseRows[r].getValueContainer();
				for(int j=0; j<alen; j++)
				{
					out.writeInt(r);
					out.writeInt(aix[j]);
					out.writeDouble(avals[j]);
				}
			}	
	}
	
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
				if( i<sparseRows.length && sparseRows[i]!=null && sparseRows[i].size()>0 )
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
		/* old version with binary search for each cell
		out.writeByte( BlockType.DENSE_BLOCK.ordinal() );
		for(int i=0; i<rlen; i++)
			for(int j=0; j<clen; j++)
				out.writeDouble(quickGetValue(i, j));
		*/
	}
	
	private void writeDenseToUltraSparse(DataOutput out) throws IOException 
	{
		out.writeByte( BlockType.ULTRA_SPARSE_BLOCK.ordinal() );
		out.writeInt(nonZeros);

		for(int r=0, ix=0; r<rlen; r++)
			for(int c=0; c<clen; c++, ix++)
				if( denseBlock[ix]!=0 )
				{
					out.writeInt(r);
					out.writeInt(c);
					out.writeDouble(denseBlock[ix]);
				}
	}
	
	private void writeDenseToSparse(DataOutput out) 
		throws IOException 
	{		
		out.writeByte( BlockType.SPARSE_BLOCK.ordinal() );
		
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
	 * NOTE: The used estimates must be kept consistent with the respective write functions. 
	 * 
	 * @return
	 */
	public long getExactSizeOnDisk()
	{
		long lrlen = (long) rlen;
		long lclen = (long) clen;
		long lnonZeros = (long) nonZeros;
		
		//ensure exact size estimates for write
		if( sparse || lnonZeros<(lrlen*lclen)*SPARCITY_TURN_POINT )
		{
			recomputeNonZeros();
			lnonZeros = (long) nonZeros;
		}
				
		//get exact size estimate (see write for the corresponding meaning)
		if(sparse)
		{
			if(sparseRows==null || nonZeros==0)
				return 9; //empty block
			else if( nonZeros<rlen && nonZeros<((long)rlen)*((long)clen)*SPARCITY_TURN_POINT && clen>SKINNY_MATRIX_TURN_POINT )
				return 4 + nonZeros*16 + 9; //ultra sparse block
			else if(lnonZeros>=(lrlen*lclen)*SPARCITY_TURN_POINT || lclen<=SKINNY_MATRIX_TURN_POINT)
				return lrlen*lclen*8 + 9;	//dense block
			else
				return lrlen*4 + lnonZeros*12 + 9; //sparse block
		}else
		{
			if(denseBlock==null || nonZeros==0)
				return 9; //empty block
			else if( nonZeros<rlen && nonZeros<((long)rlen)*((long)clen)*SPARCITY_TURN_POINT && clen>SKINNY_MATRIX_TURN_POINT )
				return 4 + nonZeros*16 + 9; //ultra sparse block
			else if(lnonZeros<(lrlen*lclen)*SPARCITY_TURN_POINT && lclen>SKINNY_MATRIX_TURN_POINT)
				return lrlen*4 + lnonZeros*12 + 9; //sparse block
			else
				return lrlen*lclen*8 + 9; //dense block
		}
	}
	
	////////
	// Estimates size and sparsity
	
	public static long estimateSize(long nrows, long ncols, double sparsity)
	{
		long size=44;//the basic variables and references sizes
		
		//determine sparse/dense representation
		boolean sparse=true;
		if(ncols<=SKINNY_MATRIX_TURN_POINT)
			sparse=false;
		else
			sparse= sparsity < SPARCITY_TURN_POINT;
		
		//estimate memory consumption for sparse/dense
		if(sparse)
		{
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
			size += (nrows-rlen) * 8; //empty rows
			
			//OLD ESTIMATE: 
			//int len = Math.max(SparseRow.initialCapacity, (int)Math.ceil(sparsity*ncols));
			//size += nrows * (28 + 12 * len );
		}
		else
		{
			size += nrows*ncols*8;
		}
		
		return size;
	}
		
	/**
	 * 
	 * @param lrlen
	 * @param lclen
	 * @param lnonZeros
	 * @param sparse
	 * @return
	 */
	public static long estimateSizeOnDisk( long lrlen, long lclen, long lnonZeros, boolean sparse )
	{		
		if(sparse)
		{
			return lrlen*4 + lnonZeros*12 + 9;	
		}
		else
		{
			return lrlen*lclen*8 + 9;
		}
	}
	
	public static SparsityEstimate estimateSparsityOnAggBinary(MatrixBlockDSM m1, MatrixBlockDSM m2, AggregateBinaryOperator op)
	{
		//NOTE: since MatrixMultLib always uses a dense intermediate output
		//with subsequent check for sparsity, we should always return a dense estimate.
		//Once, we support more aggregate binary operations, we need to change this.
		return new SparsityEstimate(false, m1.getNumRows()*m2.getNumRows());
	
		/*
		SparsityEstimate est=new SparsityEstimate();
		
		double m=m2.getNumColumns();
		
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if ( !op.sparseSafe || m <=SKINNY_MATRIX_TURN_POINT)
		{
			est.sparse=false;
		}
		else
		{
			double n=m1.getNumRows();
			double k=m1.getNumColumns();	
			double nz1=m1.getNonZeros();
			double nz2=m2.getNonZeros();
			double pq=nz1*nz2/n/k/k/m;
			double estimated= 1-Math.pow(1-pq, k);
			est.sparse=(estimated < SPARCITY_TURN_POINT);
			est.estimatedNonZeros=(int)(estimated*n*m);
		}
		return est;
		*/
	}
	
	private static SparsityEstimate estimateSparsityOnBinary(MatrixBlockDSM m1, MatrixBlockDSM m2, BinaryOperator op)
	{
		SparsityEstimate est=new SparsityEstimate();
		double m=m1.getNumColumns();
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide 
		if(!op.sparseSafe || m<=SKINNY_MATRIX_TURN_POINT)
		{
			est.sparse=false;
			return est;
		}
		
		double n=m1.getNumRows();
		double nz1=m1.getNonZeros();
		double nz2=m2.getNonZeros();
		
		double estimated=0;
		if(op.fn instanceof And || op.fn instanceof Multiply)//p*q
		{
			estimated=nz1/n/m*nz2/n/m;
			
		}else //1-(1-p)*(1-q)
		{
			estimated=1-(1-nz1/n/m)*(1-nz2/n/m);
		}
		est.sparse= (estimated<SPARCITY_TURN_POINT);
		est.estimatedNonZeros=(int)(estimated*n*m);
		
		return est;
	}
	
	private boolean estimateSparsityOnSlice(int selectRlen, int selectClen, int finalRlen, int finalClen)
	{
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if(finalClen<=SKINNY_MATRIX_TURN_POINT)
			return false;
		else
			return((double)nonZeros/(double)rlen/(double)clen*(double)selectRlen*(double)selectClen/(double)finalRlen/(double)finalClen<SPARCITY_TURN_POINT);
	}
	
	private boolean estimateSparsityOnLeftIndexing(long rlenm1, long clenm1, int nnzm1, int nnzm2)
	{
		boolean ret = (clenm1>SKINNY_MATRIX_TURN_POINT);
		long ennz = Math.min(rlenm1*clenm1, nnzm1+nnzm2);
		return (ret && (((double)ennz)/rlenm1/clenm1) < SPARCITY_TURN_POINT);
	}
	
	
	////////
	// Core block operations (called from instructions)
	
	public MatrixValue scalarOperations(ScalarOperator op, MatrixValue result) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlockDSM ret = checkType(result);
		
		// estimate the sparsity structure of result matrix
		boolean sp = this.sparse; // by default, we guess result.sparsity=input.sparsity
		if (!op.sparseSafe)
			sp = false; // if the operation is not sparse safe, then result will be in dense format
		
		if( ret==null )
			ret = new MatrixBlockDSM(rlen, clen, sp, this.nonZeros);
		else
			ret.reset(rlen, clen, sp, this.nonZeros);
		
		ret.copy(this, sp);
		
		if(op.sparseSafe)
			ret.sparseScalarOperationsInPlace(op);
		else
			ret.denseScalarOperationsInPlace(op);
		
		return ret;
	}
	
	public void scalarOperationsInPlace(ScalarOperator op) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(op.sparseSafe)
			this.sparseScalarOperationsInPlace(op);
		else
			this.denseScalarOperationsInPlace(op);
	}
	
	/**
	 * Note: only apply to non zero cells
	 * 
	 * @param op
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private void sparseScalarOperationsInPlace(ScalarOperator op) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(sparse)
		{
			if(sparseRows==null)
				return;
			nonZeros=0;
			for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
			{
				if(sparseRows[r]==null) continue;
				double[] values=sparseRows[r].getValueContainer();
				int[] cols=sparseRows[r].getIndexContainer();
				int pos=0;
				for(int i=0; i<sparseRows[r].size(); i++)
				{
					double v=op.executeScalar(values[i]);
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
		}else
		{
			//early abort possible since sparsesafe
			if(denseBlock==null) 
				return; 
			int limit=rlen*clen;
			nonZeros=0;
			for(int i=0; i<limit; i++)
			{
				denseBlock[i]=op.executeScalar(denseBlock[i]);
				if(denseBlock[i]!=0)
					nonZeros++;
			}
		}
	}
	
	private void denseScalarOperationsInPlace(ScalarOperator op) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		
		if( sparse ) //SPARSE MATRIX
		{
			double v;
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					v=op.executeScalar(quickGetValue(r, c));
					quickSetValue(r, c, v);
				}
		}
		else //DENSE MATRIX
		{
			//early abort not possible because not sparsesafe (e.g., A+7)
			if(denseBlock==null)
				allocateDenseBlock();
				
			int limit=rlen*clen;
			nonZeros=0;
			for(int i=0; i<limit; i++)
			{
				denseBlock[i]=op.executeScalar(denseBlock[i]);
				if(denseBlock[i]!=0)
					nonZeros++;
			}
		}
	}
		
	public MatrixValue unaryOperations(UnaryOperator op, MatrixValue result) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		checkType(result);
		
		// estimate the sparsity structure of result matrix
		boolean sp = this.sparse; // by default, we guess result.sparsity=input.sparsity
		if (!op.sparseSafe)
			sp = false; // if the operation is not sparse safe, then result will be in dense format
		
		if(result==null)
			result=new MatrixBlockDSM(rlen, clen, sp, this.nonZeros);
		result.copy(this);
		
		//core execution
		((MatrixBlockDSM)result).unaryOperationsInPlace(op);
		
		return result;
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
		if(sparse)
		{
			if(sparseRows==null)
				return;
			nonZeros=0;
			for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
			{
				if(sparseRows[r]==null) continue;
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
			//early abort possible since sparsesafe
			if(denseBlock==null)
				return;
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
	
	public MatrixValue binaryOperations(BinaryOperator op, MatrixValue thatValue, MatrixValue result) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlockDSM that=checkType(thatValue);
		checkType(result);
		if(this.rlen!=that.rlen || this.clen!=that.clen)
			throw new RuntimeException("block sizes are not matched for binary " +
					"cell operations: "+this.rlen+"*"+this.clen+" vs "+ that.rlen+"*"
					+that.clen);
		
		MatrixBlockDSM tmp = null;
		if(op.sparseSafe)
			tmp = sparseBinaryHelp(op, that, (MatrixBlockDSM)result);
		else
			tmp = denseBinaryHelp(op, that, (MatrixBlockDSM)result);
				
		return tmp;
	}
	
	private MatrixBlockDSM sparseBinaryHelp(BinaryOperator op, MatrixBlockDSM that, MatrixBlockDSM result) 
		throws DMLRuntimeException 
	{
		//+, -, (*)
		
		SparsityEstimate resultSparse=estimateSparsityOnBinary(this, that, op);
		if(result==null)
			result=new MatrixBlockDSM(rlen, clen, resultSparse.sparse, resultSparse.estimatedNonZeros);
		else
			result.reset(rlen, clen, resultSparse.sparse, resultSparse.estimatedNonZeros);
		
		if(this.sparse && that.sparse)
		{
			//special case, if both matrices are all 0s, just return
			if(this.sparseRows==null && that.sparseRows==null)
				return result;
			
			if(result.sparse)
				result.adjustSparseRows(result.rlen-1);
			if(this.sparseRows!=null)
				this.adjustSparseRows(rlen-1);
			if(that.sparseRows!=null)
				that.adjustSparseRows(that.rlen-1);
				
			if(this.sparseRows!=null && that.sparseRows!=null)
			{
				for(int r=0; r<rlen; r++)
				{
					if(this.sparseRows[r]==null && that.sparseRows[r]==null)
						continue;
					
					if(result.sparse)
					{
						int estimateSize=0;
						if(this.sparseRows[r]!=null)
							estimateSize+=this.sparseRows[r].size();
						if(that.sparseRows[r]!=null)
							estimateSize+=that.sparseRows[r].size();
						estimateSize=Math.min(clen, estimateSize);
						if(result.sparseRows[r]==null)
							result.sparseRows[r]=new SparseRow(estimateSize, result.clen);
						else if(result.sparseRows[r].capacity()<estimateSize)
							result.sparseRows[r].recap(estimateSize);
					}
					
					if(this.sparseRows[r]!=null && that.sparseRows[r]!=null)
					{
						mergeForSparseBinary(op, this.sparseRows[r].getValueContainer(), 
								this.sparseRows[r].getIndexContainer(), this.sparseRows[r].size(),
								that.sparseRows[r].getValueContainer(), 
								that.sparseRows[r].getIndexContainer(), that.sparseRows[r].size(), r, result);
						
					}else if(this.sparseRows[r]==null)
					{
						appendRightForSparseBinary(op, that.sparseRows[r].getValueContainer(), 
								that.sparseRows[r].getIndexContainer(), that.sparseRows[r].size(), 0, r, result);
					}else
					{
						appendLeftForSparseBinary(op, this.sparseRows[r].getValueContainer(), 
								this.sparseRows[r].getIndexContainer(), this.sparseRows[r].size(), 0, r, result);
					}
				}
			}else if(this.sparseRows==null)
			{
				for(int r=0; r<rlen; r++)
				{
					if(that.sparseRows[r]==null)
						continue;
					if(result.sparse)
					{
						if(result.sparseRows[r]==null)
							result.sparseRows[r]=new SparseRow(that.sparseRows[r].size(), result.clen);
						else if(result.sparseRows[r].capacity()<that.sparseRows[r].size())
							result.sparseRows[r].recap(that.sparseRows[r].size());
					}
					appendRightForSparseBinary(op, that.sparseRows[r].getValueContainer(), 
							that.sparseRows[r].getIndexContainer(), that.sparseRows[r].size(), 0, r, result);
				}
			}else
			{
				for(int r=0; r<rlen; r++)
				{
					if(this.sparseRows[r]==null)
						continue;
					if(result.sparse)
					{
						if(result.sparseRows[r]==null)
							result.sparseRows[r]=new SparseRow(this.sparseRows[r].size(), result.clen);
						else if(result.sparseRows[r].capacity()<this.sparseRows[r].size())
							result.sparseRows[r].recap(this.sparseRows[r].size());
					}
					appendLeftForSparseBinary(op, this.sparseRows[r].getValueContainer(), 
							this.sparseRows[r].getIndexContainer(), this.sparseRows[r].size(), 0, r, result);
				}
			}
		}
		else if( !result.sparse && (this.sparse || that.sparse) &&
				(op.fn instanceof Plus || op.fn instanceof Minus || 
				(op.fn instanceof Multiply && !that.sparse )))
		{
			//specific case in order to prevent binary search on sparse inputs (see quickget and quickset)
			result.allocateDenseBlock();
			int m = result.rlen;
			int n = result.clen;
			double[] c = result.denseBlock;
			
			//1) process left input: assignment
			int alen;
			int[] aix;
			double[] avals;
			
			if( this.sparse ) //SPARSE left
			{
				Arrays.fill(result.denseBlock, 0, result.denseBlock.length, 0); 
				
				if( this.sparseRows != null )
				{
					for( int i=0, ix=0; i<m; i++, ix+=n ) {
						SparseRow arow = this.sparseRows[i];
						if( arow != null && arow.size() > 0 )
						{
							alen = arow.size();
							aix = arow.getIndexContainer();
							avals = arow.getValueContainer();
							for(int k = 0; k < alen; k++) 
								c[ix+aix[k]] = avals[k];
						}
					}
				}
			}
			else //DENSE left
			{
				if( this.denseBlock!=null ) 
					System.arraycopy(this.denseBlock, 0, c, 0, m*n);
				else
					Arrays.fill(result.denseBlock, 0, result.denseBlock.length, 0); 
			}
			
			//2) process right input: op.fn (+,-,*), * only if dense
			if( that.sparse ) //SPARSE right
			{				
				if(that.sparseRows!=null)
				{
					for( int i=0, ix=0; i<m; i++, ix+=n ) {
						SparseRow arow = that.sparseRows[i];
						if( arow != null && arow.size() > 0 )
						{
							alen = arow.size();
							aix = arow.getIndexContainer();
							avals = arow.getValueContainer();
							for(int k = 0; k < alen; k++) 
								c[ix+aix[k]] = op.fn.execute(c[ix+aix[k]], avals[k]);
						}
					}	
				}
			}
			else //DENSE right
			{
				if( that.denseBlock!=null )
					for( int i=0; i<m*n; i++ )
						c[i] = op.fn.execute(c[i], that.denseBlock[i]);
				else if(op.fn instanceof Multiply)
					Arrays.fill(result.denseBlock, 0, result.denseBlock.length, 0); 
			}

			//3) recompute nnz
			result.recomputeNonZeros();
		}
		else if( !result.sparse && !this.sparse && !that.sparse && this.denseBlock!=null && that.denseBlock!=null )
		{
			result.allocateDenseBlock();
			int m = result.rlen;
			int n = result.clen;
			double[] c = result.denseBlock;
			
			//int nnz = 0;
			for( int i=0; i<m*n; i++ )
			{
				c[i] = op.fn.execute(this.denseBlock[i], that.denseBlock[i]);
				//HotSpot JVM bug causes crash in presence of NaNs 
				//nnz += (c[i]!=0)? 1 : 0;
				if( c[i] != 0 )
					result.nonZeros++;
			}
			//result.nonZeros = nnz;
		}
		else //generic case
		{
			double thisvalue, thatvalue, resultvalue;
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					thisvalue=this.quickGetValue(r, c);
					thatvalue=that.quickGetValue(r, c);
					if(thisvalue==0 && thatvalue==0)
						continue;
					resultvalue=op.fn.execute(thisvalue, thatvalue);
					result.appendValue(r, c, resultvalue);
				}
		}
		
		return result;
	}
	
	private MatrixBlockDSM denseBinaryHelp(BinaryOperator op, MatrixBlockDSM that, MatrixBlockDSM result) 
		throws DMLRuntimeException 
	{
		SparsityEstimate resultSparse=estimateSparsityOnBinary(this, that, op);
		if(result==null)
			result=new MatrixBlockDSM(rlen, clen, resultSparse.sparse, resultSparse.estimatedNonZeros);
		else
			result.reset(rlen, clen, resultSparse.sparse, resultSparse.estimatedNonZeros);
		
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				double v = op.fn.execute(this.quickGetValue(r, c), that.quickGetValue(r, c));
				result.appendValue(r, c, v);
			}
		
		return result;
	}
	
	/*
	 * like a merge sort
	 */
	private static void mergeForSparseBinary(BinaryOperator op, double[] values1, int[] cols1, int size1, 
				double[] values2, int[] cols2, int size2, int resultRow, MatrixBlockDSM result) 
		throws DMLRuntimeException
	{
		int p1=0, p2=0, column;
		double v;
		//merge
		while(p1<size1 && p2< size2)
		{
			if(cols1[p1]<cols2[p2])
			{
				v=op.fn.execute(values1[p1], 0);
				column=cols1[p1];
				p1++;
			}else if(cols1[p1]==cols2[p2])
			{
				v=op.fn.execute(values1[p1], values2[p2]);
				column=cols1[p1];
				p1++;
				p2++;
			}else
			{
				v=op.fn.execute(0, values2[p2]);
				column=cols2[p2];
				p2++;
			}
			result.appendValue(resultRow, column, v);	
		}
		
		//add left over
		appendLeftForSparseBinary(op, values1, cols1, size1, p1, resultRow, result);
		appendRightForSparseBinary(op, values2, cols2, size2, p2, resultRow, result);
	}
	
	private static void appendLeftForSparseBinary(BinaryOperator op, double[] values1, int[] cols1, int size1, 
				int pos, int resultRow, MatrixBlockDSM result) 
		throws DMLRuntimeException
	{
		for(int j=pos; j<size1; j++)
		{
			double v = op.fn.execute(values1[j], 0);
			result.appendValue(resultRow, cols1[j], v);
		}
	}
	
	private static void appendRightForSparseBinary(BinaryOperator op, double[] values2, int[] cols2, int size2, 
		int pos, int resultRow, MatrixBlockDSM result) throws DMLRuntimeException
	{
		for( int j=pos; j<size2; j++ )
		{
			double v = op.fn.execute(0, values2[j]);
			result.appendValue(resultRow, cols2[j], v);
		}
	}
	
	public void incrementalAggregate(AggregateOperator aggOp, MatrixValue correction, 
			MatrixValue newWithCorrection)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//assert(aggOp.correctionExists); 
		MatrixBlockDSM cor=checkType(correction);
		MatrixBlockDSM newWithCor=checkType(newWithCorrection);
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
			   && ((Builtin)(aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MAXINDEX ){
					for(int r=0; r<rlen; r++){
						double currMaxValue = cor.quickGetValue(r, 0);
						long newMaxIndex = (long)newWithCor.quickGetValue(r, 0);
						double newMaxValue = newWithCor.quickGetValue(r, 1);
						double update = aggOp.increOp.fn.execute(newMaxValue, currMaxValue);
						    
						if(update == 1){
							quickSetValue(r, 0, newMaxIndex);
							cor.quickSetValue(r, 0, newMaxValue);
						}
					}
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
				MatrixAggLib.aggregateBinaryMatrix(newWithCor, this, cor);
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
						if( brow != null && brow.size() > 0 ) 
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
		MatrixBlockDSM newWithCor=checkType(newWithCorrection);
		KahanObject buffer=new KahanObject(0, 0);
		
		if(aggOp.correctionLocation==CorrectionLocationType.LASTROW)
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
			
		}else if(aggOp.correctionLocation==CorrectionLocationType.LASTCOLUMN)
		{
			if(aggOp.increOp.fn instanceof Builtin 
			   && ((Builtin)(aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MAXINDEX ){
				for(int r = 0; r < rlen; r++){
					double currMaxValue = quickGetValue(r, 1);
					long newMaxIndex = (long)newWithCor.quickGetValue(r, 0);
					double newMaxValue = newWithCor.quickGetValue(r, 1);
					double update = aggOp.increOp.fn.execute(newMaxValue, currMaxValue);
					
					if(update == 1){
						quickSetValue(r, 0, newMaxIndex);
						quickSetValue(r, 1, newMaxValue);
					}
				}
			}else{
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
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		if (!(op.fn.equals(SwapIndex.getSwapIndexFnObject()) || op.fn.equals(MaxIndex.getMaxIndexFnObject())))
			throw new DMLRuntimeException("the current reorgOperations cannot support: "+op.fn.getClass()+".");
		
		MatrixBlockDSM result=checkType(ret);
		CellIndex tempCellIndex = new CellIndex(-1,-1);
		boolean reducedDim=op.fn.computeDimension(rlen, clen, tempCellIndex);
		boolean sps;
		if(reducedDim)
			sps = false;
		else if(op.fn.equals(MaxIndex.getMaxIndexFnObject()))
			sps = true;
		else
			sps = checkRealSparsity(this, true);
		
		if(result==null)
			result=new MatrixBlockDSM(tempCellIndex.row, tempCellIndex.column, sps, this.nonZeros);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, sps, this.nonZeros);
		
		if( MatrixReorgLib.isSupportedReorgOperator(op) )
		{
			//SPECIAL case (operators with special performance requirements, 
			//or size-dependent special behavior)
			MatrixReorgLib.reorg(this, result, op);
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
						if(sparseRows[r]==null) continue;
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
	public MatrixBlockDSM appendOperations( MatrixBlockDSM that, MatrixBlockDSM ret ) 	
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		MatrixBlockDSM result = checkType( ret );
		final int m = rlen;
		final int n = clen+that.clen;
		final int nnz = nonZeros+that.nonZeros;		
		boolean sp = checkRealSparsity(m, n, nnz);
		
		//init result matrix 
		if( result == null ) 
			result = new MatrixBlockDSM(m, n, sp, nnz);
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
			result.adjustSparseRows(rlen-1);
			result.appendToSparse(this, 0, 0);
			result.appendToSparse(that, 0, clen);
		}		
		result.nonZeros = nnz;
		
		return result;
	}
	
	@Deprecated
	public MatrixValue appendOperations(ReorgOperator op, MatrixValue ret, int startRow, int startColumn, int length) 	
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		MatrixBlockDSM result=checkType(ret);
		CellIndex tempCellIndex = new CellIndex(-1,-1);
		boolean reducedDim=op.fn.computeDimension(rlen, clen, tempCellIndex);
		boolean sps;
		if(reducedDim)
			sps=false;
		else
			sps=checkRealSparsity(this);
			
		if(result==null)
			result=new MatrixBlockDSM(tempCellIndex.row, tempCellIndex.column, sps, this.nonZeros);
		else if(result.getNumRows()==0 && result.getNumColumns()==0)
			result.reset(tempCellIndex.row, tempCellIndex.column, sps, this.nonZeros);
		
		CellIndex temp = new CellIndex(0, 0);
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
				{
					if(sparseRows[r]==null) continue;
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
		}else
		{
			if(denseBlock!=null)
			{
				int limit=rlen*clen;
				int r,c;
				for(int i=0; i<limit; i++)
				{
					r=i/clen;
					c=i%clen;
					temp.set(r, c);
					op.fn.execute(temp, temp);
					result.appendValue(temp.row, temp.column, denseBlock[i]);
				}
			}
		}
		
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
	public MatrixValue transposeSelfMatrixMultOperations( MatrixBlockDSM out, MMTSJType tstype ) 	
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//check for transpose type
		if( !(tstype == MMTSJType.LEFT || tstype == MMTSJType.RIGHT) )
			throw new DMLRuntimeException("Invalid MMTSJ type '"+tstype+"'.");
		
		//compute matrix mult
		boolean leftTranspose = ( tstype == MMTSJType.LEFT );
		MatrixMultLib.matrixMultTransposeSelf(this, out, leftTranspose);
		
		return out;
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
			long colLower, long colUpper, MatrixValue ret) throws DMLRuntimeException, DMLUnsupportedOperationException {
		
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
		MatrixBlockDSM result=checkType(ret);
		boolean sp = estimateSparsityOnLeftIndexing(rlen, clen, nonZeros, rhsMatrix.getNonZeros());
		if(result==null)
			result=new MatrixBlockDSM(rlen, clen, sp);
		else
			result.reset(rlen, clen, sp);
		result.copy(this, sp);
		
		//NOTE conceptually we could directly use a zeroout and copy(..., false) but
		//     since this was factors slower, we still use a full copy and subsequently
		//     copy(..., true) - however, this can be changed in the future once we 
		//     improved the performance of zeroout.
		//result = (MatrixBlockDSM) zeroOutOperations(result, new IndexRange(rowLower,rowUpper, colLower, colUpper ), false);
		
		int rl = (int)rowLower-1;
		int ru = (int)rowUpper-1;
		int cl = (int)colLower-1;
		int cu = (int)colUpper-1;
		MatrixBlockDSM src = (MatrixBlockDSM)rhsMatrix;
		

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
	public MatrixValue leftIndexingOperations(ScalarObject scalar, long row, long col, MatrixValue ret) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		MatrixBlockDSM result=checkType(ret);
		if(result==null)
			result=new MatrixBlockDSM(this);
		else 
			result.copy(this);
		
		int rl = (int)row-1;
		int cl = (int)col-1;
		
		result.quickSetValue(rl, cl, scalar.getDoubleValue());
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
		boolean result_sparsity = this.sparse && ((cu-cl+1)>SKINNY_MATRIX_TURN_POINT);
		MatrixBlockDSM result=checkType(ret);
		int estnnzs=(int) ((double)this.nonZeros/rlen/clen*(ru-rl+1)*(cu-cl+1));
		if(result==null)
			result=new MatrixBlockDSM(ru-rl+1, cu-cl+1, result_sparsity, estnnzs);
		else
			result.reset(ru-rl+1, cu-cl+1, result_sparsity, estnnzs);
		
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
	 */
	private void sliceSparse(int rl, int ru, int cl, int cu, MatrixBlockDSM dest)
	{
		if ( sparseRows == null ) 
			return;
		
		if( cl==cu ) //specific case: column vector (always dense)
		{
			dest.allocateDenseBlock();
			double val;
			for( int i=rl, ix=0; i<=ru; i++, ix++ )
				if( sparseRows[i] != null && sparseRows[i].size()>0 )
					if( (val = sparseRows[i].get(cl)) != 0 )
					{
						dest.denseBlock[ix] = val;
						dest.nonZeros++;
					}
		}
		else //general case (sparse and dense)
		{
			for(int i=rl; i <= ru; i++) 
				if(sparseRows[i] != null && sparseRows[i].size()>0) 
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
	 */
	private void sliceDense(int rl, int ru, int cl, int cu, MatrixBlockDSM dest)
	{
		if( denseBlock == null )
			return;
		dest.allocateDenseBlock();
		
		if( cl==cu ) //specific case: column vector 
		{
			if( clen==1 ) //vector -> vector
				System.arraycopy(denseBlock, rl, dest.denseBlock, 0, ru-rl+1);
			else //matrix -> vector
				for( int i=rl*clen+cl, ix=0; i<=ru*clen+cu; i+=clen, ix++ )
					dest.denseBlock[ix] = denseBlock[i];
		}
		else //general case (dense)
		{
			for(int i = rl, ix1 = rl*clen+cl, ix2=0; i <= ru; i++, ix1+=clen, ix2+=dest.clen) 
				System.arraycopy(denseBlock, ix1, dest.denseBlock, ix2, dest.clen);
		}
		
		dest.recomputeNonZeros();
	}
	
	public void sliceOperations(ArrayList<IndexedMatrixValue> outlist, IndexRange range, int rowCut, int colCut, 
			int normalBlockRowFactor, int normalBlockColFactor, int boundaryRlen, int boundaryClen)
	{
		MatrixBlockDSM topleft=null, topright=null, bottomleft=null, bottomright=null;
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
			topleft=(MatrixBlockDSM) p.next().getValue();
			//topleft.reset(blockRowFactor, blockColFactor, 
			//		checkSparcityOnSlide(rowCut-(int)range.rowStart, colCut-(int)range.colStart, blockRowFactor, blockColFactor));
			
			topleft.reset(blockRowFactor, blockColFactor, 
					estimateSparsityOnSlice(minrowcut-(int)range.rowStart, mincolcut-(int)range.colStart, blockRowFactor, blockColFactor));
		}
		if(range.rowStart<rowCut && range.colEnd>=colCut)
		{
			topright=(MatrixBlockDSM) p.next().getValue();
			topright.reset(blockRowFactor, boundaryClen, 
					estimateSparsityOnSlice(minrowcut-(int)range.rowStart, (int)range.colEnd-maxcolcut+1, blockRowFactor, boundaryClen));
		}
		if(range.rowEnd>=rowCut && range.colStart<colCut)
		{
			bottomleft=(MatrixBlockDSM) p.next().getValue();
			bottomleft.reset(boundaryRlen, blockColFactor, 
					estimateSparsityOnSlice((int)range.rowEnd-maxrowcut+1, mincolcut-(int)range.colStart, boundaryRlen, blockColFactor));
		}
		if(range.rowEnd>=rowCut && range.colEnd>=colCut)
		{
			bottomright=(MatrixBlockDSM) p.next().getValue();
			bottomright.reset(boundaryRlen, boundaryClen, 
					estimateSparsityOnSlice((int)range.rowEnd-maxrowcut+1, (int)range.colEnd-maxcolcut+1, boundaryRlen, boundaryClen));
			//System.out.println("bottomright size: "+bottomright.rlen+" X "+bottomright.clen);
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
	
	private void sliceHelp(int r, IndexRange range, int colCut, MatrixBlockDSM left, MatrixBlockDSM right, int rowOffset, int normalBlockRowFactor, int normalBlockColFactor)
	{
	//	if(left==null || right==null)
	//		throw new RuntimeException("left = "+left+", and right = "+right);
		if(sparseRows[r]==null) return;
		//System.out.println("row "+r+"\t"+sparseRows[r]);
		int[] cols=sparseRows[r].getIndexContainer();
		double[] values=sparseRows[r].getValueContainer();
		int start=sparseRows[r].searchIndexesFirstGTE((int)range.colStart);
		//System.out.println("start: "+start);
		if(start<0) return;
		int end=sparseRows[r].searchIndexesFirstLTE((int)range.colEnd);
		//System.out.println("end: "+end);
		if(end<0 || start>end) return;
		for(int i=start; i<=end; i++)
		{
			if(cols[i]<colCut)
				left.appendValue(r+rowOffset, cols[i]+normalBlockColFactor-colCut, values[i]);
			else
				right.appendValue(r+rowOffset, cols[i]-colCut, values[i]);
		//	System.out.println("set "+r+", "+cols[i]+": "+values[i]);
		}
	}
	
	@Override
	//This the append operations for MR side
	//nextNCol is the number columns for the block right of block v2
	public void appendOperations(MatrixValue v2,
			ArrayList<IndexedMatrixValue> outlist, int blockRowFactor,
			int blockColFactor, boolean m2IsLast, int nextNCol)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		MatrixBlockDSM m2=(MatrixBlockDSM)v2;
		//System.out.println("second matrix: \n"+m2);
		Iterator<IndexedMatrixValue> p=outlist.iterator();
		if(this.clen==blockColFactor)
		{
			MatrixBlockDSM first=(MatrixBlockDSM) p.next().getValue();
			first.copy(this);
			MatrixBlockDSM second=(MatrixBlockDSM) p.next().getValue();
			second.copy(m2);
		}else
		{
			int ncol=Math.min(clen+m2.getNumColumns(), blockColFactor);
			int part=ncol-clen;
			MatrixBlockDSM first=(MatrixBlockDSM) p.next().getValue();
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
			
			
			MatrixBlockDSM second=null;
			
			if(part<m2.clen)
			{
				second=(MatrixBlockDSM) p.next().getValue();
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
	
	public MatrixValue zeroOutOperations(MatrixValue result, IndexRange range, boolean complementary)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		checkType(result);
		boolean sps;
		double currentSparsity=(double)nonZeros/(double)rlen/(double)clen;
		double estimatedSps=currentSparsity*(double)(range.rowEnd-range.rowStart+1)
		*(double)(range.colEnd-range.colStart+1)/(double)rlen/(double)clen;
		if(!complementary)
			estimatedSps=currentSparsity-estimatedSps;
		if(estimatedSps< SPARCITY_TURN_POINT)
			sps=true;
		else sps=false;
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if(clen<=SKINNY_MATRIX_TURN_POINT)
			sps=false;
			
		if(result==null)
			result=new MatrixBlockDSM(rlen, clen, sps, (int)(estimatedSps*rlen*clen));
		else
			result.reset(rlen, clen, sps, (int)(estimatedSps*rlen*clen));
		
		
		if(sparse)
		{
			if(sparseRows!=null)
			{
				if(!complementary)//if zero out
				{
					for(int r=0; r<Math.min((int)range.rowStart, sparseRows.length); r++)
						((MatrixBlockDSM) result).appendRow(r, sparseRows[r]);
					for(int r=Math.min((int)range.rowEnd+1, sparseRows.length); r<Math.min(rlen, sparseRows.length); r++)
						((MatrixBlockDSM) result).appendRow(r, sparseRows[r]);
				}
				for(int r=(int)range.rowStart; r<=Math.min(range.rowEnd, sparseRows.length-1); r++)
				{
					if(sparseRows[r]==null) continue;
					//System.out.println("row "+r+"\t"+sparseRows[r]);
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					
					if(complementary)//if selection
					{
						int start=sparseRows[r].searchIndexesFirstGTE((int)range.colStart);
						//System.out.println("start: "+start);
						if(start<0) continue;
						int end=sparseRows[r].searchIndexesFirstGT((int)range.colEnd);
						//System.out.println("end: "+end);
						if(end<0 || start>end) continue;
						
						for(int i=start; i<end; i++)
						{
							((MatrixBlockDSM) result).appendValue(r, cols[i], values[i]);
						//	System.out.println("set "+r+", "+cols[i]+": "+values[i]);
						}
					}else
					{
						int start=sparseRows[r].searchIndexesFirstGTE((int)range.colStart);
						//System.out.println("start: "+start);
						if(start<0) start=sparseRows[r].size();
						int end=sparseRows[r].searchIndexesFirstGT((int)range.colEnd);
						//System.out.println("end: "+end);
						if(end<0) end=sparseRows[r].size();
						
				/*		if(r==999)
						{
							System.out.println("----------------------");
							System.out.println("range: "+range);
							System.out.println("row: "+sparseRows[r]);
							System.out.println("start: "+start);
							System.out.println("end: "+end);
						}
				*/		
						for(int i=0; i<start; i++)
						{
							((MatrixBlockDSM) result).appendValue(r, cols[i], values[i]);
					//		if(r==999) System.out.println("append ("+r+", "+cols[i]+"): "+values[i]);
						}
						for(int i=end; i<sparseRows[r].size(); i++)
						{
							((MatrixBlockDSM) result).appendValue(r, cols[i], values[i]);
					//		if(r==999) System.out.println("append ("+r+", "+cols[i]+"): "+values[i]);
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
							((MatrixBlockDSM) result).appendValue(r, c, denseBlock[offset+c]);
						offset+=clen;
					}
				}else
				{
					int offset=0;
					int r=0;
					for(; r<(int)range.rowStart; r++)
						for(int c=0; c<clen; c++, offset++)
							((MatrixBlockDSM) result).appendValue(r, c, denseBlock[offset]);
					
					for(; r<=(int)range.rowEnd; r++)
					{
						for(int c=0; c<(int)range.colStart; c++)
							((MatrixBlockDSM) result).appendValue(r, c, denseBlock[offset+c]);
						for(int c=(int)range.colEnd+1; c<clen; c++)
							((MatrixBlockDSM) result).appendValue(r, c, denseBlock[offset+c]);
						offset+=clen;
					}
					
					for(; r<rlen; r++)
						for(int c=0; c<clen; c++, offset++)
							((MatrixBlockDSM) result).appendValue(r, c, denseBlock[offset]);
				}
				
			}
		}
		//System.out.println("zeroout in:\n"+this);
		//System.out.println("zeroout result:\n"+result);
		return result;
	}
	
	//This function is not really used
/*	public void zeroOutOperationsInPlace(IndexRange range, boolean complementary)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//do not change the format of the block
		if(sparse)
		{
			if(sparseRows==null) return;
			
			if(complementary)//if selection, need to remove unwanted rows
			{
				for(int r=0; r<Math.min((int)range.rowStart, sparseRows.length); r++)
					if(sparseRows[r]!=null)
					{
						nonZeros-=sparseRows[r].size();
						sparseRows[r].reset();
					}
				for(int r=Math.min((int)range.rowEnd+1, sparseRows.length-1); r<Math.min(rlen, sparseRows.length); r++)
					if(sparseRows[r]!=null)
					{
						nonZeros-=sparseRows[r].size();
						sparseRows[r].reset();
					}
			}
			
			for(int r=(int)range.rowStart; r<=Math.min(range.rowEnd, sparseRows.length-1); r++)
			{
				if(sparseRows[r]==null) continue;
				int oldsize=sparseRows[r].size();
				if(complementary)//if selection
					sparseRows[r].deleteIndexComplementaryRange((int)range.colStart, (int)range.rowEnd);
				else //if zeroout
					sparseRows[r].deleteIndexRange((int)range.colStart, (int)range.rowEnd);
				nonZeros-=(oldsize-sparseRows[r].size());
			}
			
		}else
		{		
			if(denseBlock==null) return;
			int start=(int)range.rowStart*clen;
			
			if(complementary)//if selection, need to remove unwanted rows
			{
				nonZeros=0;
				Arrays.fill(denseBlock, 0, start, 0);
				Arrays.fill(denseBlock, ((int)range.rowEnd+1)*clen, rlen*clen, 0);
				for(int r=(int) range.rowStart; r<=range.rowEnd; r++)
				{
					Arrays.fill(denseBlock, start, start+(int) range.colStart, 0);
					Arrays.fill(denseBlock, start+(int)range.colEnd+1, start+clen, 0);
					for(int c=(int) range.colStart; c<=range.colEnd; c++)
						if(denseBlock[start+c]!=0)
							nonZeros++;		
					start+=clen;
				}
			}else
			{
				for(int r=(int) range.rowStart; r<=range.rowEnd; r++)
				{
					for(int c=(int) range.colStart; c<=range.colEnd; c++)
						if(denseBlock[start+c]!=0)
							nonZeros--;		
					Arrays.fill(denseBlock, start+(int) range.colStart, start+(int)range.colEnd+1, 0);
					start+=clen;
				}
			}
		}
	}*/
	
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
			case LASTROW: tempCellIndex.row++; break;
			case LASTCOLUMN: tempCellIndex.column++; break;
			case LASTTWOROWS: tempCellIndex.row+=2; break;
			case LASTTWOCOLUMNS: tempCellIndex.column+=2; break;
			default:
				throw new DMLRuntimeException("unrecognized correctionLocation: "+op.aggOp.correctionLocation);	
			}
		}
		if(result==null)
			result=new MatrixBlockDSM(tempCellIndex.row, tempCellIndex.column, false);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, false);
		
		MatrixBlockDSM ret = (MatrixBlockDSM) result;
		if( MatrixAggLib.isSupportedUnaryAggregateOperator(op) ) {
			MatrixAggLib.aggregateUnaryMatrix(this, ret, op);
			MatrixAggLib.recomputeIndexes(ret, op, blockingFactorRow, blockingFactorCol, indexesIn);
		}
		else if(op.sparseSafe)
			sparseAggregateUnaryHelp(op, ret, blockingFactorRow, blockingFactorCol, indexesIn);
		else
			denseAggregateUnaryHelp(op, ret, blockingFactorRow, blockingFactorCol, indexesIn);
		
		if(op.aggOp.correctionExists && inCP)
			((MatrixBlockDSM)result).dropLastRowsOrColums(op.aggOp.correctionLocation);
		
		return ret;
	}
	
	private void sparseAggregateUnaryHelp(AggregateUnaryOperator op, MatrixBlockDSM result,
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
					if(sparseRows[r]==null) continue;
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
	
	private void denseAggregateUnaryHelp(AggregateUnaryOperator op, MatrixBlockDSM result,
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
				   && ((Builtin)(op.aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MAXINDEX ){
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
	
	
	private void incrementalAggregateUnaryHelp(AggregateOperator aggOp, MatrixBlockDSM result, int row, int column, 
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
	private void dropLastRowsOrColums(CorrectionLocationType correctionLocation) 
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
				if(sparseRows[r]==null) continue;
				//int[] cols=sparseRows[r].getIndexContainer();
				double[] values=sparseRows[r].getValueContainer();
				for(int i=0; i<sparseRows[r].size(); i++)
				{
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
		
	public CM_COV_Object cmOperations(CMOperator op, MatrixBlockDSM weights) 
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

/*				
			int zerocount = 0, zerorows=0, nzrows=0;
				for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
				{
					// This matrix has only a single column
					if(sparseRows[r]==null) {
						zerocount += weights.getValue(r,0);
						zerorows++;
					}
					//int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					//x = sparseRows[r].size();
					if ( sparseRows[r].size() == 0 ) 
						zerorows++;
					for(int i=0; i<sparseRows[r].size(); i++) {
						//op.fn.execute(cmobj, values[i], weights.getValue(r,0));
						nzrows++;
					}
				}
				System.out.println("--> total="+this.getNumRows() + ", nzrows=" + nzrows + ", zerorows="+zerorows+"... zerocount="+zerocount);
				// account for zeros in the vector
				//op.fn.execute(cmobj, 0.0, zerocount);
*/		}
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
	
	public CM_COV_Object covOperations(COVOperator op, MatrixBlockDSM that) 
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
	
	public CM_COV_Object covOperations(COVOperator op, MatrixBlockDSM that, MatrixBlockDSM weights) 
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
		
		MatrixBlockDSM wts= (weights == null ? null : checkType(weights));
		checkType(result);
		
		if ( getNumColumns() != 1 ) {
			throw new DMLRuntimeException("Invalid input dimensions (" + getNumRows() + "x" + getNumColumns() + ") to sort operation.");
		}
		if ( wts != null && wts.getNumColumns() != 1 ) {
			throw new DMLRuntimeException("Invalid weight dimensions (" + wts.getNumRows() + "x" + wts.getNumColumns() + ") to sort operation.");
		}
		
		// Copy the input elements into a temporary array for sorting
		// #rows in temp matrix = 1 + #nnz in the input ( 1 is for the "zero" value)
		int dim1 = 1+this.getNonZeros();
		// First column is data and second column is weights
		double[][] tdw = new double[dim1][2]; 
		
		double d, w, zero_wt=0;
		if ( wtflag ) {
			for ( int r=0, ind=1; r < getNumRows(); r++ ) {
				d = quickGetValue(r,0);
				w = wts.quickGetValue(r,0);
				if ( d != 0 ) {
					tdw[ind][0] = d;
					tdw[ind][1] = w;
					ind++;
				}
				else
					zero_wt += w;
			}
			tdw[0][0] = 0.0;
			tdw[0][1] = zero_wt;
		} 
		else {
			tdw[0][0] = 0.0;
			tdw[0][1] = getNumRows() - getNonZeros(); // number of zeros in the input data
			
			int ind = 1;
			if(sparse) {
				if(sparseRows!=null) {
					for(int r=0; r<Math.min(rlen, sparseRows.length); r++) {
						if(sparseRows[r]==null) continue;
						//int[] cols=sparseRows[r].getIndexContainer();
						double[] values=sparseRows[r].getValueContainer();
						for(int i=0; i<sparseRows[r].size(); i++) {
							tdw[ind][0] = values[i];
							tdw[ind][1] = 1;
							ind++;
						}
					}
				}
			}
			else {
				if(denseBlock!=null) {
					int limit=rlen*clen;
					for(int i=0; i<limit; i++) {
						// copy only non-zero values
						if ( denseBlock[i] != 0.0 ) {
							tdw[ind][0] = denseBlock[i];
							tdw[ind][1] = 1;
							ind++;
						}
					}
				}
			}
		}
		
		// Sort td and tw based on values inside td (ascending sort)
		Arrays.sort(tdw, new Comparator<double[]>(){
			@Override
			public int compare(double[] arg0, double[] arg1) {
				return (arg0[0] < arg1[0] ? -1 : (arg0[0] == arg1[0] ? 0 : 1));
			}} 
		);
		
		// Copy the output from sort into "result"
		// result is always dense (currently)
		if(result==null)
			result=new MatrixBlockDSM(dim1, 2, false);
		else
			result.reset(dim1, 2, false);
		((MatrixBlockDSM) result).init(tdw, dim1, 2);
		
		return result;
	}
	
	/**
	 * Computes the weighted interQuartileMean.
	 * The matrix block ("this" pointer) has two columns, in which the first column 
	 * refers to the data and second column denotes corresponding weights.
	 * 
	 * @return InterQuartileMean
	 * @throws DMLRuntimeException
	 */
	public double interQuartileMean() throws DMLRuntimeException {
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
	
		MatrixBlockDSM qs=checkType(quantiles);
		
		if ( qs.clen != 1 ) {
			throw new DMLRuntimeException("Multiple quantiles can only be computed on a 1D matrix");
		}
		
		MatrixBlockDSM output = checkType(ret);

		if(output==null)
			output=new MatrixBlockDSM(qs.rlen, qs.clen, false); // resulting matrix is mostly likely be dense
		else
			output.reset(qs.rlen, qs.clen, false);
		
		for ( int i=0; i < qs.rlen; i++ ) {
			output.quickSetValue(i, 0, this.pickValue(qs.quickGetValue(i,0)) );
		}
		
		return output;
	}
	
	public double pickValue(double quantile) 
		throws DMLRuntimeException 
	{
		double sum_wt = sumWeightForQuantile();
		
		int pos = (int) Math.ceil(quantile*sum_wt);
		
		int t = 0, i=-1;
		do {
			i++;
			t += quickGetValue(i,1);
		} while(t<pos && i < getNumRows());
		
		return quickGetValue(i,0);
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
		if ( (int)sum_wt != sum_wt ) {
			throw new DMLRuntimeException("Unexpected error while computing quantile -- weights must be integers.");
		}
		return sum_wt;
	}
	
	public MatrixValue aggregateBinaryOperations(MatrixValue m1Value, MatrixValue m2Value, 
			MatrixValue result, AggregateBinaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlockDSM m1=checkType(m1Value);
		MatrixBlockDSM m2=checkType(m2Value);
		checkType(result);
		if(m1.clen!=m2.rlen)
			throw new RuntimeException("dimensions do not match for matrix multiplication ("+m1.clen+"!="+m2.rlen+")");
		int rl=m1.rlen;
		int cl=m2.clen;
		SparsityEstimate sp=estimateSparsityOnAggBinary(m1, m2, op);
		if(result==null)
			result=new MatrixBlockDSM(rl, cl, sp.sparse, sp.estimatedNonZeros);//m1.sparse&&m2.sparse);
		else
			result.reset(rl, cl, sp.sparse, sp.estimatedNonZeros);//m1.sparse&&m2.sparse);
		
		if(op.sparseSafe)
			sparseAggregateBinaryHelp(m1, m2, (MatrixBlockDSM)result, op);
		else
			aggBinSparseUnsafe(m1, m2, (MatrixBlockDSM)result, op);

		return result;
	}
	
	public MatrixValue aggregateBinaryOperations(MatrixIndexes m1Index, MatrixValue m1Value, MatrixIndexes m2Index, MatrixValue m2Value, 
			MatrixValue result, AggregateBinaryOperator op, boolean partialMult) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlockDSM m1=checkType(m1Value);
		MatrixBlockDSM m2=checkType(m2Value);
		checkType(result);

		if (  partialMult ) {
			// check if matrix block's row-column range falls within the whole vector's row-column range 
		}
		else if(m1.clen!=m2.rlen)
			throw new RuntimeException("dimensions do not match for matrix multiplication ("+m1.clen+"!="+m2.rlen+")");

		int rl, cl;
		SparsityEstimate sp;
		if ( partialMult ) {
			// TODO: avoid this code!!
			rl = m1.rlen;
			cl = 1;
			sp = new SparsityEstimate(false,m1.rlen);
		}
		else {
			rl=m1.rlen;
			cl=m2.clen;
			sp = estimateSparsityOnAggBinary(m1, m2, op);
		}
		
		if(result==null)
			result=new MatrixBlockDSM(rl, cl, sp.sparse, sp.estimatedNonZeros);//m1.sparse&&m2.sparse);
		else
			result.reset(rl, cl, sp.sparse, sp.estimatedNonZeros);//m1.sparse&&m2.sparse);
		
		if(op.sparseSafe)
			sparseAggregateBinaryHelp(m1Index, m1, m2Index, m2, (MatrixBlockDSM)result, op, partialMult);
		else
			aggBinSparseUnsafe(m1, m2, (MatrixBlockDSM)result, op);
		return result;
	}

	private static void sparseAggregateBinaryHelp(MatrixIndexes m1Index, MatrixBlockDSM m1, MatrixIndexes m2Index, MatrixBlockDSM m2, 
			MatrixBlockDSM result, AggregateBinaryOperator op, boolean partialMult) throws DMLRuntimeException 
	{
		//matrix multiplication
		if(op.binaryFn instanceof Multiply && op.aggOp.increOp.fn instanceof Plus && !partialMult)
		{
			MatrixMultLib.matrixMult(m1, m2, result);
		}
		else
		{
			if(!m1.sparse && m2 != null && !m2.sparse )
				aggBinDenseDense(m1, m2, result, op);
			else if(m1.sparse && m2 != null && m2.sparse)
				aggBinSparseSparse(m1, m2, result, op);
			else if(m1.sparse)
				aggBinSparseDense(m1Index, m1, m2Index, m2, result, op, partialMult);
			else
				aggBinDenseSparse(m1, m2, result, op);
		}
	}
	
	private static void sparseAggregateBinaryHelp(MatrixBlockDSM m1, MatrixBlockDSM m2, 
			MatrixBlockDSM result, AggregateBinaryOperator op) throws DMLRuntimeException 
	{
		if(op.binaryFn instanceof Multiply && op.aggOp.increOp.fn instanceof Plus )
		{
			MatrixMultLib.matrixMult(m1, m2, result);
		}
		else
		{
			if(!m1.sparse && !m2.sparse)
				aggBinDenseDense(m1, m2, result, op);
			else if(m1.sparse && m2.sparse)
				aggBinSparseSparse(m1, m2, result, op);
			else if(m1.sparse)
				aggBinSparseDense(m1, m2, result, op);
			else
				aggBinDenseSparse(m1, m2, result, op);
		}
	}
	
	/**
	 * to perform aggregateBinary when both matrices are dense
	 */
	private static void aggBinDenseDense(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM result, AggregateBinaryOperator op) throws DMLRuntimeException
	{
		int j, l, i, aIndex, bIndex;
		double temp;
		double v;
		double[] a = m1.getDenseArray();
		double[] b = m2.getDenseArray();
		if(a==null || b==null)
			return;
		
		for(l = 0; l < m1.clen; l++)
		{
			aIndex = l;
			//cIndex = 0;
			for(i = 0; i < m1.rlen; i++)
			{
				// aIndex = l + i * m1clen
				temp = a[aIndex];
			
				bIndex = l * m1.rlen;
				for(j = 0; j < m2.clen; j++)
				{
					// cIndex = i * m1.rlen + j
					// bIndex = l * m1.rlen + j
					v = op.aggOp.increOp.fn.execute(result.quickGetValue(i, j), op.binaryFn.execute(temp, b[bIndex]));
					result.quickSetValue(i, j, v);
					//cIndex++;
					bIndex++;
				}
				
				aIndex += m1.clen;
			}
		}
	}
	
	/*
	 * to perform aggregateBinary when the first matrix is dense and the second is sparse
	 */
	private static void aggBinDenseSparse(MatrixBlockDSM m1, MatrixBlockDSM m2,
			MatrixBlockDSM result, AggregateBinaryOperator op) 
		throws DMLRuntimeException 
	{
		if(m2.sparseRows==null)
			return;

		for(int k=0; k<Math.min(m2.rlen, m2.sparseRows.length); k++)
		{
			if(m2.sparseRows[k]==null) continue;
			int[] cols=m2.sparseRows[k].getIndexContainer();
			double[] values=m2.sparseRows[k].getValueContainer();
			for(int p=0; p<m2.sparseRows[k].size(); p++)
			{
				int j=cols[p];
				for(int i=0; i<m1.rlen; i++)
				{
					double old=result.quickGetValue(i, j);
					double aik=m1.quickGetValue(i, k);
					double addValue=op.binaryFn.execute(aik, values[p]);
					double newvalue=op.aggOp.increOp.fn.execute(old, addValue);
					result.quickSetValue(i, j, newvalue);
				}
			}
		}
	}
	
	private static void aggBinSparseDense(MatrixIndexes m1Index,
			MatrixBlockDSM m1, MatrixIndexes m2Index, MatrixBlockDSM m2,
			MatrixBlockDSM result, AggregateBinaryOperator op,
			boolean partialMult) throws DMLRuntimeException {
		if (m1.sparseRows == null)
			return;

		//System.out.println("#*#*#**# in special aggBinSparseDense ... ");
		int end_l, incrA = 0, incrB = 0;

		// l varies from 0..end_l, where the upper bound end_l is determined by
		// the Matrix Input (not the vector input)
		if (partialMult == false) {
			// both A and B are matrices
			end_l = m1.clen;
		} else {
			if (m2.isVector()) {
				// A is a matrix and B is a vector
				incrB = (int) UtilFunctions.cellIndexCalculation(m1Index
						.getColumnIndex(), 1000, 0) - 1; // matrixB.l goes from
															// incr+start_l to
															// incr+end_l
				end_l = incrB + m1.clen;
			} else if (m1.isVector()) {
				// A is a vector and B is a matrix
				incrA = (int) UtilFunctions.cellIndexCalculation(m2Index
						.getRowIndex(), 1000, 0) - 1; // matrixA.l goes from
														// incr+start_l to
														// incr+end_l
				end_l = incrA + m2.rlen;
			} else
				throw new RuntimeException(
						"Unexpected case in matrixMult w/ partialMult");
		}

		/*System.out.println("m1: [" + m1.rlen + "," + m1.clen + "]   m2: ["
				+ m2.rlen + "," + m2.clen + "]  incrA: " + incrA + " incrB: "
				+ incrB + ", end_l " + end_l);*/

		for (int i = 0; i < Math.min(m1.rlen, m1.sparseRows.length); i++) {
			if (m1.sparseRows[i] == null)
				continue;
			int[] cols = m1.sparseRows[i].getIndexContainer();
			double[] values = m1.sparseRows[i].getValueContainer();
			for (int j = 0; j < m2.clen; j++) {
				double aij = 0;

				/*int p = 0;
				if (partialMult) {
					// this code is executed when m1 is a vector and m2 is a
					// matrix
					while (m1.isVector() && cols[p] < incrA)
						p++;
				}*/
				
				// when m1 and m2 are matrices : incrA=0 & end_l=m1.clen (#cols
				// in whole m1)
				// when m1 is matrix & m2 is vector: incrA=0 &
				// end_l=m1.sparseRows[i].size()
				// when m1 is vector & m2 is matrix: incrA=based on
				// m2Index.rowIndex & end_l=incrA+m2.rlen (#rows in m2's block)
				for (int p = incrA; p < m1.sparseRows[i].size()
						&& cols[p] < end_l; p++) {
					int k = cols[p];
					double addValue = op.binaryFn.execute(values[p], m2
							.quickGetValue(k + incrB, j));
					aij = op.aggOp.increOp.fn.execute(aij, addValue);
				}
				result.appendValue(i, j, aij);
			}
		}
	}

	/*
	 * to perform aggregateBinary when the first matrix is sparse and the second is dense
	 */
	private static void aggBinSparseDense(MatrixBlockDSM m1, MatrixBlockDSM m2,
			MatrixBlockDSM result, AggregateBinaryOperator op) 
		throws DMLRuntimeException
	{
		if(m1.sparseRows==null)
			return;

		for(int i=0; i<Math.min(m1.rlen, m1.sparseRows.length); i++)
		{
			if(m1.sparseRows[i]==null) continue;
			int[] cols=m1.sparseRows[i].getIndexContainer();
			double[] values=m1.sparseRows[i].getValueContainer();
			for(int j=0; j<m2.clen; j++)
			{
				double aij=0;
				for(int p=0; p<m1.sparseRows[i].size(); p++)
				{
					int k=cols[p];
					double addValue=op.binaryFn.execute(values[p], m2.quickGetValue(k, j));
					aij=op.aggOp.increOp.fn.execute(aij, addValue);
				}
				result.appendValue(i, j, aij);
			}
			
		}
	}
	
	/*
	 * to perform aggregateBinary when both matrices are sparse
	 */	
	private static void aggBinSparseSparse(MatrixBlockDSM m1, MatrixBlockDSM m2,
			MatrixBlockDSM result, AggregateBinaryOperator op) 
		throws DMLRuntimeException 
	{
		if(m1.sparseRows==null || m2.sparseRows==null)
			return;
		//double[] cache=null;
		TreeMap<Integer, Double> cache=null;
		if(result.isInSparseFormat())
		{
			//cache=new double[m2.getNumColumns()];
			cache=new TreeMap<Integer, Double>();
		}
		for(int i=0; i<Math.min(m1.rlen, m1.sparseRows.length); i++)
		{
			if(m1.sparseRows[i]==null) continue;
			int[] cols1=m1.sparseRows[i].getIndexContainer();
			double[] values1=m1.sparseRows[i].getValueContainer();
			for(int p=0; p<m1.sparseRows[i].size(); p++)
			{
				int k=cols1[p];
				if(m2.sparseRows[k]==null) continue;
				int[] cols2=m2.sparseRows[k].getIndexContainer();
				double[] values2=m2.sparseRows[k].getValueContainer();
				for(int q=0; q<m2.sparseRows[k].size(); q++)
				{
					int j=cols2[q];
					double addValue=op.binaryFn.execute(values1[p], values2[q]);
					if(result.isInSparseFormat())
					{
						//cache[j]=op.aggOp.increOp.fn.execute(cache[j], addValue);
						Double old=cache.get(j);
						if(old==null)
							old=0.0;
						cache.put(j, op.aggOp.increOp.fn.execute(old, addValue));
					}else
					{
						double old=result.quickGetValue(i, j);
						double newvalue=op.aggOp.increOp.fn.execute(old, addValue);
						result.quickSetValue(i, j, newvalue);
					}	
				}
			}
			
			if(result.isInSparseFormat())
			{
				/*for(int j=0; j<cache.length; j++)
				{
					if(cache[j]!=0)
					{
						result.appendValue(i, j, cache[j]);
						cache[j]=0;
					}
				}*/
				for(Entry<Integer, Double> e: cache.entrySet())
				{
					result.appendValue(i, e.getKey(), e.getValue());
				}
				cache.clear();
			}
		}
	}
	
	private static void aggBinSparseUnsafe(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM result, 
			AggregateBinaryOperator op) throws DMLRuntimeException
	{
		for(int i=0; i<m1.rlen; i++)
			for(int j=0; j<m2.clen; j++)
			{
				double aggValue=op.aggOp.initialValue;
				for(int k=0; k<m1.clen; k++)
				{
					double aik=m1.quickGetValue(i, k);
					double bkj=m2.quickGetValue(k, j);
					double addValue=op.binaryFn.execute(aik, bkj);
					aggValue=op.aggOp.increOp.fn.execute(aggValue, addValue);
				}
				result.appendValue(i, j, aggValue);
			}
	}
		
	public MatrixValue groupedAggOperations(MatrixValue tgt, MatrixValue wghts, MatrixValue ret, Operator op) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		// this <- groups
		MatrixBlockDSM target=checkType(tgt);
		MatrixBlockDSM weights=checkType(wghts);
		
		if (this.getNumColumns() != 1 || target.getNumColumns() != 1 || (weights!=null && weights.getNumColumns()!=1) )
			throw new DMLRuntimeException("groupedAggregate can only operate on 1-dimensional column matrices.");
		if ( this.getNumRows() != target.getNumRows() || (weights != null && this.getNumRows() != weights.getNumRows()) ) 
			throw new DMLRuntimeException("groupedAggregate can only operate on matrices with equal dimensions.");
		
		// Determine the number of groups
		double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
		double d;
		if ( sparse ) {
			for ( int i=0; i < sparseRows.length; i++ ) {
				if ( sparseRows[i] == null)
					continue;
				double[] values = sparseRows[i].getValueContainer();
				for ( int j=0; j < sparseRows[i].size(); j++ ) {
					d = values[j];
					min = (d < min ? d : min);
					max = (d > max ? d : max);
				}
			}
		}
		else {
			for ( int i=0; i < denseBlock.length; i++ ) {
				d = denseBlock[i];
				min = (d < min ? d : min);
				max = (d > max ? d : max);
			}
		}
		if ( min <= 0 )
			throw new DMLRuntimeException("Invalid value (" + min + ") encountered in \"groups\" while computing groupedAggregate");
		if ( max <= 0 )
			throw new DMLRuntimeException("Invalid value (" + max + ") encountered in \"groups\" while computing groupedAggregate.");
		int numGroups = (int) max;
	
		MatrixBlockDSM result=checkType(ret);
		
		// Allocate memory to hold the result
		boolean result_sparsity = false; // it is likely that resulting matrix is dense
		if(result==null)
			result=new MatrixBlockDSM(numGroups, 1, result_sparsity);
		else
			result.reset(numGroups, 1, result_sparsity);

		// Compute the result
		int g;
		double w = 1; // default weight
		if(op instanceof CMOperator) {
			// initialize required objects for storing the result of CM operations
			CM cmFn = CM.getCMFnObject(((CMOperator) op).getAggOpType());
			CM_COV_Object[] cmValues = new CM_COV_Object[numGroups];
			for ( int i=0; i < numGroups; i++ )
				cmValues[i] = new CM_COV_Object();
			
			for ( int i=0; i < this.getNumRows(); i++ ) {
				g = (int) this.quickGetValue(i, 0);
				d = target.quickGetValue(i,0);
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
		else if(op instanceof AggregateOperator) {
			AggregateOperator aggop=(AggregateOperator) op;
				
			if(aggop.correctionExists) {
				
				KahanObject[] buffer = new KahanObject[numGroups];
				for(int i=0; i < numGroups; i++ )
					buffer[i] = new KahanObject(aggop.initialValue, 0);
				
				for ( int i=0; i < this.getNumRows(); i++ ) {
					g = (int) this.quickGetValue(i, 0);
					d = target.quickGetValue(i,0);
					if ( weights != null )
						w = weights.quickGetValue(i,0);
					// buffer is 0-indexed, whereas range of values for g = [1,numGroups]
					aggop.increOp.fn.execute(buffer[g-1], d*w);
				}

				// extract the required value from each KahanObject
				for ( int i=0; i < numGroups; i++ )
					// result is 0-indexed, so is buffer
					result.quickSetValue(i, 0, buffer[i]._sum);
			}
			else {
				for ( int i=0; i < numGroups; i++ )
					result.quickSetValue(i, 0, aggop.initialValue);
				
				double v;
				for ( int i=0; i < this.getNumRows(); i++ ) {
					g = (int) this.quickGetValue(i, 0);
					d = target.quickGetValue(i,0);
					if ( weights != null )
						w = weights.quickGetValue(i, 0);
					// buffer is 0-indexed, whereas range of values for g = [1,numGroups]
					v = aggop.increOp.fn.execute(result.getValue(g-1,1), d*w);
					result.quickSetValue(g-1, 0, v);
				}
			}
			
		}else
			throw new DMLRuntimeException("Invalid operator (" + op + ") encountered while processing groupedAggregate.");
		
		return result;
	}

	public MatrixValue removeEmptyOperations( MatrixValue ret, boolean rows )
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//check for empty inputs
		if( nonZeros==0 ) {
			ret.reset(0, 0, false);
			return ret;
		}
		
		MatrixBlockDSM result = checkType(ret);
		
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
	private MatrixBlockDSM removeEmptyRows(MatrixBlockDSM ret) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{	
		//scan block and determine empty rows
		int rlen2 = 0;
		boolean[] flags = new boolean[ rlen ]; 
		if( sparse ) 
		{
			for ( int i=0; i < sparseRows.length; i++ ) {
				if ( sparseRows[i] != null &&  sparseRows[i].size() > 0 )
				{
					flags[i] = false;
					rlen2++;
				}
				else
					flags[i] = true;
			}
		}
		else 
		{
			for(int i=0; i<rlen; i++) {
				flags[i] = true;
				int index = i*clen;
				for(int j=0; j<clen; j++)
					if( denseBlock[index++] != 0 )
					{
						flags[i] = false;
						rlen2++;
						break; //early abort for current row
					}
			}
		}

		//reset result and copy rows
		ret.reset(rlen2, clen, sparse);
		int rindex = 0;
		for( int i=0; i<rlen; i++ )
			if( !flags[i] )
			{
				//copy row to result
				if(sparse)
				{
					ret.appendRow(rindex, sparseRows[i]);
				}
				else
				{
					if( ret.denseBlock==null )
						ret.denseBlock = new double[ rlen2*clen ];
					int index1 = i*clen;
					int index2 = rindex*clen;
					for(int j=0; j<clen; j++)
					{
						if( denseBlock[index1] != 0 )
						{
							ret.denseBlock[index2] = denseBlock[index1];
							ret.nonZeros++;
						}
						index1++;
						index2++;
					}
				}
				rindex++;
			}
		
		
		//check sparsity
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
	private MatrixBlockDSM removeEmptyColumns(MatrixBlockDSM ret) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//scan block and determine empty cols
		int clen2 = 0;
		boolean[] flags = new boolean[ clen ]; 
		
		for(int j=0; j<clen; j++) {
			flags[j] = true;
			for(int i=0; i<rlen; i++) {
				double value = quickGetValue(i, j);
				if( value != 0 )
				{
					flags[j] = false;
					clen2++;
					break; //early abort for current col
				}
			}
		}

		//reset result and copy rows
		ret.reset(rlen, clen2, sparse);
		
		int cindex = 0;
		for( int j=0; j<clen; j++ )
			if( !flags[j] )
			{
				//copy col to result
				for( int i=0; i<rlen; i++ )
				{
					double value = quickGetValue(i, j);
					if( value != 0 )
						ret.quickSetValue(i, cindex, value);
				}
				
				cindex++;
			}
		
		//check sparsity
		ret.examSparsity();
		
		return ret;
	}
	
	@Override
	public MatrixValue replaceOperations(MatrixValue result, double pattern, double replacement) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlockDSM ret = checkType(result);
		examSparsity(); //ensure its in the right format
		ret.reset(rlen, clen, sparse);
		if( nonZeros == 0 && pattern != 0  )
			return ret; //early abort
		boolean NaNpattern = Double.isNaN(pattern);
		
		if( sparse ) //SPARSE
		{
			if( pattern !=0d ) //SPARSE <- SPARSE (sparse-safe)
			{
				ret.adjustSparseRows(ret.rlen-1);
				SparseRow[] a = sparseRows;
				SparseRow[] c = ret.sparseRows;
				
				for( int i=0; i<rlen; i++ )
				{
					SparseRow arow = a[ i ];
					if( arow!=null && arow.size()>0 )
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
						if( arow!=null && arow.size()>0 )
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
	 */
	@Override
	public void tertiaryOperations(Operator op, double scalarThat,
			MatrixValue that2Val, HashMap<MatrixIndexes, Double> ctableResult)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		MatrixBlockDSM that2 = checkType(that2Val);
		CTable ctable = CTable.getCTableFnObject();
		double v2 = scalarThat;
		
		//sparse-unsafe ctable execution
		//(because input values of 0 are invalid and have to result in errors) 
		for( int i=0; i<rlen; i++ )
			for( int j=0; j<clen; j++ )
			{
				double v1 = this.quickGetValue(i, j);
				double w = that2.quickGetValue(i, j);
				ctable.execute(v1, v2, w, ctableResult);
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
	public void tertiaryOperations(Operator op, double scalarThat,
			double scalarThat2, HashMap<MatrixIndexes, Double> ctableResult)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		CTable ctable = CTable.getCTableFnObject();
		double v2 = scalarThat;
		double w = scalarThat2;
		
		//sparse-unsafe ctable execution
		//(because input values of 0 are invalid and have to result in errors) 
		for( int i=0; i<rlen; i++ )
			for( int j=0; j<clen; j++ )
			{
				double v1 = this.quickGetValue(i, j);
				ctable.execute(v1, v2, w, ctableResult);
			}		
	}
	
	/**
	 * Specific ctable case of ctable(seq(...),X), where X is the only
	 * matrix input. The 'left' input parameter specifies if the seq appeared
	 * on the left, otherwise it appeared on the right.
	 * 
	 */
	@Override
	public void tertiaryOperations(Operator op, MatrixIndexes ix1, double scalarThat,
			boolean left, int brlen, HashMap<MatrixIndexes, Double> ctableResult)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		CTable ctable = CTable.getCTableFnObject();
		double w = scalarThat;
		int offset = (int) ((ix1.getRowIndex()-1)*brlen); 
		
		//sparse-unsafe ctable execution
		//(because input values of 0 are invalid and have to result in errors) 
		for( int i=0; i<rlen; i++ )
			for( int j=0; j<clen; j++ )
			{
				double v1 = this.quickGetValue(i, j);
				if( left )
					ctable.execute(offset+i+1, v1, w, ctableResult);
				else
					ctable.execute(v1, offset+i+1, w, ctableResult);
			}
	}

	/**
	 *  D = ctable(A,B,w)
	 *  this <- A; that <- B; scalar_that2 <- w; result <- D
	 *  
	 * (i1,j1,v1) from input1 (this)
	 * (i1,j1,v2) from input2 (that)
	 * (w)  from scalar_input3 (scalarThat2)
	 */
	@Override
	public void tertiaryOperations(Operator op, MatrixValue thatVal,
			double scalarThat2, HashMap<MatrixIndexes, Double> ctableResult)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		MatrixBlockDSM that = checkType(thatVal);
		CTable ctable = CTable.getCTableFnObject();
		double w = scalarThat2;
		
		//sparse-unsafe ctable execution
		//(because input values of 0 are invalid and have to result in errors) 
		for( int i=0; i<rlen; i++ )
			for( int j=0; j<clen; j++ )
			{
				double v1 = this.quickGetValue(i, j);
				double v2 = that.quickGetValue(i, j);
				ctable.execute(v1, v2, w, ctableResult);
			}
	}
	
	/**
	 *  D = ctable(A,B,W)
	 *  this <- A; that <- B; that2 <- W; result <- D
	 *  
	 * (i1,j1,v1) from input1 (this)
	 * (i1,j1,v2) from input2 (that)
	 * (i1,j1,w)  from input3 (that2)
	 */
	public void tertiaryOperations(Operator op, MatrixValue thatVal, MatrixValue that2Val, HashMap<MatrixIndexes, Double> ctableResult)
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{	
		MatrixBlockDSM that = checkType(thatVal);
		MatrixBlockDSM that2 = checkType(that2Val);
		CTable ctable = CTable.getCTableFnObject();
		
		//sparse-unsafe ctable execution
		//(because input values of 0 are invalid and have to result in errors) 
		for( int i=0; i<rlen; i++ )
			for( int j=0; j<clen; j++ )
			{
				double v1 = this.quickGetValue(i, j);
				double v2 = that.quickGetValue(i, j);
				double w = that2.quickGetValue(i, j);
				ctable.execute(v1, v2, w, ctableResult);
			}		
	}

	public void binaryOperationsInPlace(BinaryOperator op, MatrixValue thatValue) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlockDSM that=checkType(thatValue);
		if(this.rlen!=that.rlen || this.clen!=that.clen)
			throw new RuntimeException("block sizes are not matched for binary " +
					"cell operations: "+this.rlen+"*"+this.clen+" vs "+ that.rlen+"*"
					+that.clen);
	//	System.out.println("-- this:\n"+this);
	//	System.out.println("-- that:\n"+that);
		if(op.sparseSafe)
			sparseBinaryInPlaceHelp(op, that);
		else
			denseBinaryInPlaceHelp(op, that);
	//	System.out.println("-- this (result):\n"+this);
	}

	private void sparseBinaryInPlaceHelp(BinaryOperator op, MatrixBlockDSM that) throws DMLRuntimeException 
	{
		SparsityEstimate resultSparse=estimateSparsityOnBinary(this, that, op);
		if(resultSparse.sparse && !this.sparse)
			denseToSparse();
		else if(!resultSparse.sparse && this.sparse)
			sparseToDense();
		
		if(this.sparse && that.sparse)
		{
			//special case, if both matrices are all 0s, just return
			if(this.sparseRows==null && that.sparseRows==null)
				return;
			
			if(this.sparseRows!=null)
				adjustSparseRows(rlen-1);
			if(that.sparseRows!=null)
				that.adjustSparseRows(rlen-1);
			
			if(this.sparseRows!=null && that.sparseRows!=null)
			{
				for(int r=0; r<rlen; r++)
				{
					if(this.sparseRows[r]==null && that.sparseRows[r]==null)
						continue;
					
					if(that.sparseRows[r]==null)
					{
						double[] values=this.sparseRows[r].getValueContainer();
						for(int i=0; i<this.sparseRows[r].size(); i++)
							values[i]=op.fn.execute(values[i], 0);
					}else
					{
						int estimateSize=0;
						if(this.sparseRows[r]!=null)
							estimateSize+=this.sparseRows[r].size();
						if(that.sparseRows[r]!=null)
							estimateSize+=that.sparseRows[r].size();
						estimateSize=Math.min(clen, estimateSize);
						
						//temp
						SparseRow thisRow=this.sparseRows[r];
						this.sparseRows[r]=new SparseRow(estimateSize, clen);
						
						if(thisRow!=null)
						{
							nonZeros-=thisRow.size();
							mergeForSparseBinary(op, thisRow.getValueContainer(), 
									thisRow.getIndexContainer(), thisRow.size(),
									that.sparseRows[r].getValueContainer(), 
									that.sparseRows[r].getIndexContainer(), that.sparseRows[r].size(), r, this);
							
						}else
						{
							appendRightForSparseBinary(op, that.sparseRows[r].getValueContainer(), 
									that.sparseRows[r].getIndexContainer(), that.sparseRows[r].size(), 0, r, this);
						}
					}
				}	
			}
			else if(this.sparseRows==null)
			{
				this.sparseRows=new SparseRow[rlen];
				for(int r=0; r<rlen; r++)
				{
					SparseRow brow = that.sparseRows[r];
					if( brow!=null && brow.size()>0 )
					{
						this.sparseRows[r] = new SparseRow( brow.size(), clen );
						appendRightForSparseBinary(op, brow.getValueContainer(), brow.getIndexContainer(), brow.size(), 0, r, this);
					}
				}				
			}
			else //that.sparseRows==null
			{
				for(int r=0; r<rlen; r++)
				{
					SparseRow arow = this.sparseRows[r];
					if( arow!=null && arow.size()>0 )
						appendLeftForSparseBinary(op, arow.getValueContainer(), arow.getIndexContainer(), arow.size(), 0, r, this);
				}
			}
		}else
		{
			double thisvalue, thatvalue, resultvalue;
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					thisvalue=this.quickGetValue(r, c);
					thatvalue=that.quickGetValue(r, c);
					resultvalue=op.fn.execute(thisvalue, thatvalue);
					this.quickSetValue(r, c, resultvalue);
				}	
		}
	}
	
	private void denseBinaryInPlaceHelp(BinaryOperator op, MatrixBlockDSM that) throws DMLRuntimeException 
	{
		SparsityEstimate resultSparse=estimateSparsityOnBinary(this, that, op);
		if(resultSparse.sparse && !this.sparse)
			denseToSparse();
		else if(!resultSparse.sparse && this.sparse)
			sparseToDense();
		
		double v;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				v=op.fn.execute(this.quickGetValue(r, c), that.quickGetValue(r, c));
				quickSetValue(r, c, v);
			}
	}
	
	////////////////////////////////////////////////////////////////////////////////
/*	public MatrixBlockDSM getRandomDenseMatrix_normal(int rows, int cols, long seed)
	{
		Random random=new Random(seed);
		this.allocateDenseBlock();
		
		RandNPair pair = new RandNPair();
		int index = 0;
		while ( index < rows*cols ) {
			pair.compute(random);
			this.denseBlock[index++] = pair.getFirst();
			if ( index < rows*cols )
				this.denseBlock[index++] = pair.getSecond();
		}
		this.updateNonZeros();
		return this;
	}
	
	public MatrixBlockDSM getRandomSparseMatrix_normal(int rows, int cols, double sparsity, long seed)
	{
		double val;
		Random random = new Random(System.currentTimeMillis());
		RandN rn = new RandN(seed);
		
		this.sparseRows=new SparseRow[rows];
		for(int i=0; i<rows; i++)
		{
			this.sparseRows[i]=new SparseRow();	
			for(int j=0; j<cols; j++)
			{
				if(random.nextDouble()>sparsity)
					continue;
				val = rn.nextDouble();
				this.sparseRows[i].append(j, val );
			}
		}
		this.updateNonZeros();
		return this;
	}
	
*/	
	
	////////
	// Data Generation Methods
	// (rand, sequence)
	
	/**
	 * Function to generate a matrix of random numbers. This is invoked from maptasks of 
	 * DataGen MR job.  The parameter <code>seed</code> denotes the block-level seed.
	 * 
	 * @param pdf
	 * @param rows
	 * @param cols
	 * @param rowsInBlock
	 * @param colsInBlock
	 * @param sparsity
	 * @param min
	 * @param max
	 * @param seed
	 * @return
	 * @throws DMLRuntimeException
	 */
	public MatrixBlockDSM getRandomMatrix(String pdf, int rows, int cols, int rowsInBlock, int colsInBlock, double sparsity, double min, double max, long seed) throws DMLRuntimeException
	{
		return getRandomMatrix(pdf, rows, cols, rowsInBlock, colsInBlock, sparsity, min, max, null, seed);
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
	public MatrixBlockDSM getRandomMatrix(String pdf, int rows, int cols, int rowsInBlock, int colsInBlock, double sparsity, double min, double max, Well1024a bigrand, long bSeed) throws DMLRuntimeException
	{
		// Setup Pseudo Random Number Generator for cell values based on 'pdf'.
		PRNGenerator valuePRNG = null;
		if ( pdf.equalsIgnoreCase("uniform")) 
			valuePRNG = new UniformPRNGenerator();
		else if ( pdf.equalsIgnoreCase("normal"))
			valuePRNG = new NormalPRNGenerator();
		else
			throw new DMLRuntimeException("Unsupported distribution function for Rand: " + pdf);
		
		/*
		 * Setup min and max for distributions other than "uniform". Min and Max
		 * are set up in such a way that the usual logic of
		 * (max-min)*prng.nextDouble() is still valid. This is done primarily to
		 * share the same code across different distributions.
		 */
		if ( pdf.equalsIgnoreCase("normal") ) {
			min=0;
			max=1;
		}
		
		// Determine the sparsity of output matrix
		sparse = (sparsity < SPARCITY_TURN_POINT);
		if(cols<=SKINNY_MATRIX_TURN_POINT) {
			sparse=false;
		}
		this.reset(rows, cols, sparse);
		
		// Special case shortcuts for efficiency
		if ( pdf.equalsIgnoreCase("uniform")) {
			//specific cases for efficiency
			if ( min == 0.0 && max == 0.0 ) { //all zeros
				// nothing to do here
				return this;
			} 
			else if( !sparse && sparsity==1.0d && min == max ) //equal values
			{
				allocateDenseBlock();
				Arrays.fill(denseBlock, 0, rlen*clen, min);
				nonZeros = rlen*clen;
				return this;
			}
		}
		
		// Allocate memory
		if ( sparse ) {
			sparseRows = new SparseRow[rows];
			//note: individual sparse rows are allocated on demand,
			//for consistentcy with memory estimates and prevent OOMs.
		}
		else {
			this.allocateDenseBlock();
		}

		double range = max - min;

		int nrb = (int) Math.ceil((double)rows/rowsInBlock);
		int ncb = (int) Math.ceil((double)cols/colsInBlock);
		int blockrows, blockcols, rowoffset, coloffset;
		int blocknnz;
		// loop throught row-block indices
		for(int rbi=0; rbi < nrb; rbi++) {
			blockrows = (rbi == nrb-1 ? (rows-rbi*rowsInBlock) : rowsInBlock);
			rowoffset = rbi*rowsInBlock;
			
			// loop throught column-block indices
			for(int cbj=0; cbj < ncb; cbj++) {
				blockcols = (cbj == ncb-1 ? (cols-cbj*colsInBlock) : colsInBlock);
				coloffset = cbj*colsInBlock;
				
				// Generate a block (rbi,cbj) 
				
				// select the appropriate block-level seed
				long seed = -1;
				if ( bigrand == null ) {
					// case of MR: simply use the passed-in value
					seed = bSeed;
				}
				else {
					// case of CP: generate a block-level seed from matrix-level Well1024a seed
					seed = bigrand.nextLong();
				}
				// Initialize the PRNGenerator for cell values
				valuePRNG.init(seed);
				
				// Initialize the PRNGenerator for determining cells that contain a non-zero value
				// Note that, "pdf" parameter applies only to cell values and the individual cells 
				// are always selected uniformly at random.
				UniformPRNGenerator nnzPRNG = new UniformPRNGenerator(seed);
				
				// block-level sparsity, which may differ from overall sparsity in the matrix.
				boolean localSparse = sparse && !(blockcols<=SKINNY_MATRIX_TURN_POINT);
				
				if ( localSparse ) {
					blocknnz = (int) Math.ceil((blockrows*sparsity)*blockcols);
					for(int ind=0; ind<blocknnz; ind++) {
						int i = nnzPRNG.nextInt(blockrows);
						int j = nnzPRNG.nextInt(blockcols);
						double v = nnzPRNG.nextDouble();
						if( sparseRows[rowoffset+i]==null )
							sparseRows[rowoffset+i]=new SparseRow(estimatedNNzsPerRow, clen);
						sparseRows[rowoffset+i].set(coloffset+j, v);
					}
				}
				else {
					if (sparsity == 1.0) {
						for(int ii=0; ii < blockrows; ii++) {
							for(int jj=0, index = ((ii+rowoffset)*cols)+coloffset; jj < blockcols; jj++, index++) {
								double val = min + (range * valuePRNG.nextDouble());
								this.denseBlock[index] = val;
							}
						}
					}
					else {
						if ( sparse ) {
							/* This case evaluated only when this function is invoked from CP. 
							 * In this case:
							 *     sparse=true -> entire matrix is in sparse format and hence denseBlock=null
							 *     localSparse=true -> local block is dense, and hence on MR side a denseBlock will be allocated
							 * i.e., we need to generate data in a dense-style but set values in sparseRows
							 * 
							 */
							// In this case, entire matrix is in sparse format but the current block is dense
							for(int ii=0; ii < blockrows; ii++) {
								for(int jj=0; jj < blockcols; jj++) {
									if(nnzPRNG.nextDouble() <= sparsity) {
										double val = min + (range * valuePRNG.nextDouble());
										if( sparseRows[ii+rowoffset]==null )
											sparseRows[ii+rowoffset]=new SparseRow(estimatedNNzsPerRow, clen);
										sparseRows[ii+rowoffset].set(jj+coloffset, val);
									}
								}
							}
						}
						else {
							for(int ii=0; ii < blockrows; ii++) {
								for(int jj=0, index = ((ii+rowoffset)*cols)+coloffset; jj < blockcols; jj++, index++) {
									if(nnzPRNG.nextDouble() <= sparsity) {
										double val = min + (range * valuePRNG.nextDouble());
										this.denseBlock[index] = val;
									}
								}
							}
						}
					}
				} // sparse or dense 
			} // cbj
		} // rbi
		
		recomputeNonZeros();
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
	public MatrixBlockDSM getSequence(double from, double to, double incr) throws DMLRuntimeException {
		boolean neg = (from > to);
		if (neg != (incr < 0))
			throw new DMLRuntimeException("Wrong sign for the increment in a call to seq()");
		
		//System.out.println(System.nanoTime() + ": begin of seq()");
		int rows = 1 + (int)Math.floor((to-from)/incr);
		int cols = 1;
		sparse = false; // sequence matrix is always dense
		this.reset(rows, cols, sparse);
		
		this.allocateDenseBlock();
		
		//System.out.println(System.nanoTime() + ": MatrixBlockDSM.seq(): seq("+from+","+to+","+incr+") rows = " + rows);
		
		this.denseBlock[0] = from;
		for(int i=1; i < rows; i++) {
			from += incr;
			this.denseBlock[i] = from;
		}
		recomputeNonZeros();
		//System.out.println(System.nanoTime() + ": end of seq()");
		return this;
	}

	////////
	// Misc methods
		
	private static MatrixBlockDSM checkType(MatrixValue block) throws DMLUnsupportedOperationException
	{
		if( block!=null && !(block instanceof MatrixBlockDSM))
			throw new DMLUnsupportedOperationException("the Matrix Value is not MatrixBlockDSM!");
		return (MatrixBlockDSM) block;
	}
	
	private static boolean checkRealSparsity(long rlen, long clen, long nnz)
	{
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if( clen<=SKINNY_MATRIX_TURN_POINT )
			return false;
		else
			return (((double)nnz)/rlen/clen < SPARCITY_TURN_POINT);
	}
	
	private static boolean checkRealSparsity(MatrixBlockDSM m)
	{
		return checkRealSparsity( m, false );
	}
	
	private static boolean checkRealSparsity(MatrixBlockDSM m, boolean transpose)
	{
		int lrlen = (transpose) ? m.clen : m.rlen;
		int lclen = (transpose) ? m.rlen : m.clen;
		int lnnz = m.getNonZeros();
		
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if(lclen<=SKINNY_MATRIX_TURN_POINT)
			return false;
		else
			return (((double)lnnz)/lrlen/lclen < SPARCITY_TURN_POINT);
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
	public int compareTo(Object arg0) 
	{
		// don't compare blocks
		return 0;
	}

	@Override
	public String toString()
	{
		String ret="sparse? = "+sparse+"\n" ;
		ret+="nonzeros = "+nonZeros+"\n";
		ret+="size: "+rlen+" X "+clen+"\n";
		boolean toprint=false;
		if(!toprint)
			return "sparse? = "+sparse+"\nnonzeros = "+nonZeros+"\nsize: "+rlen+" X "+clen+"\n";
		if(sparse)
		{
			int len=0;
			if(sparseRows!=null)
				len=Math.min(rlen, sparseRows.length);
			int i=0;
			for(; i<len; i++)
			{
				ret+="row +"+i+": "+sparseRows[i]+"\n";
				if(sparseRows[i]!=null)
				{
					for(int j=0; j<sparseRows[i].size(); j++)
						if(sparseRows[i].getValueContainer()[j]!=0.0)
							toprint=true;
				}
			}
			for(; i<rlen; i++)
			{
				ret+="row +"+i+": null\n";
			}
		}else
		{
			if(denseBlock!=null)
			{
				int start=0;
				for(int i=0; i<rlen; i++)
				{
					for(int j=0; j<clen; j++)
					{
						ret+=this.denseBlock[start+j]+"\t";
						if(this.denseBlock[start+j]!=0.0)
							toprint=true;
					}
					ret+="\n";
					start+=clen;
				}
			}
		}
		return ret;
	}


	///////////////////////////
	// Helper classes

	public static class SparsityEstimate
	{
		public int estimatedNonZeros=0;
		public boolean sparse=false;
		public SparsityEstimate(boolean sps, int nnzs)
		{
			sparse=sps;
			estimatedNonZeros=nnzs;
		}
		public SparsityEstimate(){}
	}

	public static class IJV
	{
		public int i=-1;
		public int j=-1;
		public double v=0;
		public IJV()
		{}
		public IJV(int i, int j, double v)
		{
			set(i, j, v);
		}
		public void set(int i, int j, double v)
		{
			this.i=i;
			this.j=j;
			this.v=v;
		}
		public String toString()
		{
			return "("+i+", "+j+"): "+v;
		}
	}
	
	public static class SparseCellIterator implements Iterator<IJV>
	{
		private int rlen=0;
		private SparseRow[] sparseRows=null;
		private int curRow=-1;
		private int curColIndex=-1;
		private int[] colIndexes=null;
		private double[] values=null;
		private boolean nothingLeft=false;
		private IJV retijv=new IJV();
		
		public SparseCellIterator(int nrows, SparseRow[] mtx)
		{
			rlen=nrows;
			sparseRows=mtx;
			curRow=0;
			
			if(sparseRows==null)
				nothingLeft=true;
			else
				findNextNonZeroRow();
		}
		
		private void findNextNonZeroRow() {
			while(curRow<Math.min(rlen, sparseRows.length) && (sparseRows[curRow]==null || sparseRows[curRow].size()==0))
				curRow++;
			if(curRow>=Math.min(rlen, sparseRows.length))
				nothingLeft=true;
			else
			{
				curColIndex=0;
				colIndexes=sparseRows[curRow].getIndexContainer();
				values=sparseRows[curRow].getValueContainer();
			}
		}
		
		@Override
		public boolean hasNext() {
			if(nothingLeft)
				return false;
			else
				return true;
		}

		@Override
		public IJV next() {
			retijv.set(curRow, colIndexes[curColIndex], values[curColIndex]);
			curColIndex++;
			if(curColIndex>=sparseRows[curRow].size())
			{
				curRow++;
				findNextNonZeroRow();
			}
			return retijv;
		}

		@Override
		public void remove() {
			throw new RuntimeException("SparseCellIterator.remove should not be called!");
			
		}		
	}
}
