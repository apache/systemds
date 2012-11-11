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
import java.util.Random;
import java.util.TreeMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.MaxIndex;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.COVOperator;
import com.ibm.bi.dml.runtime.matrix.operators.LeftScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;
import com.ibm.bi.dml.runtime.util.RandN;
import com.ibm.bi.dml.runtime.util.RandNPair;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.And;


public class MatrixBlockDSM extends MatrixValue{

	public static final double SPARCITY_TURN_POINT=0.4;//based on practial experiments on space consumption and performance
	//protected static final Log LOG = LogFactory.getLog(MatrixBlock1D.class);
	
	public static final int SKINNY_MATRIX_TURN_POINT=4;//based on math
	
	private int rlen;
	private int clen;
	private boolean sparse;
	
	protected int maxrow, maxcolumn;
	protected double[] denseBlock=null;
	protected SparseRow[] sparseRows=null;
	protected int nonZeros=0;
	
	protected int estimatedNNzsPerRow=-1;//only used for allocation purpose for sparse representation
	
	/**
	 * Computes the size of this {@link MatrixBlockDSM} object in main memory,
	 * in bytes, as precisely as possible.  Used for caching purposes.
	 * 
	 * @return the size of this object in bytes
	 */
	//NOTE: not required anymore
	/*public long getObjectSizeInMemory ()
	{
		long all_size = 32;
		if (denseBlock != null)
			all_size += denseBlock.length * 8;
		if (sparseRows != null)
		{
			for (int i = 0; i < sparseRows.length; i++) {
				if ( sparseRows[i] != null )
					all_size += 8 + sparseRows [i].getObjectSizeInMemory ();
			}
		}
		return all_size;
	}*/
	
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
			//account for sparsity and initial capacity
			int len = Math.max(SparseRow.initialCapacity, (int)Math.ceil(sparsity*ncols));
			size += nrows * (28 + 12 * len );
			
			//size += (Math.ceil(sparsity*ncols)*12+28)*nrows;
		}
		else
		{
			size += nrows*ncols*8;
		}
		
		return size;
	}
	
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
	
	public static SparsityEstimate checkSparcityOnAggBinary(MatrixBlockDSM m1, MatrixBlockDSM m2, AggregateBinaryOperator op)
	{
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
	}
	
	/*private boolean isVector() {
		return (this.getNumRows() == 1 || this.getNumColumns() == 1);
	}*/
	
	private static SparsityEstimate checkSparcityOnBinary(MatrixBlockDSM m1, MatrixBlockDSM m2, BinaryOperator op)
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
	
	private static boolean checkRealSparcity(MatrixBlockDSM m)
	{
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if(m.getNumColumns()<=SKINNY_MATRIX_TURN_POINT)
			return false;
		else
			return ( (double)m.getNonZeros()/(double)m.getNumRows()/(double)m.getNumColumns() < SPARCITY_TURN_POINT);
	}
	
	public MatrixBlockDSM()
	{
		rlen=0;
		clen=0;
		sparse=true;
		nonZeros=0;
		maxrow = maxcolumn = 0;
	}
	public MatrixBlockDSM(int rl, int cl, boolean sp)
	{
		rlen=rl;
		clen=cl;
		sparse=sp;
		nonZeros=0;
		maxrow = maxcolumn = 0;
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
	
	public void init(double[][] arr, int r, int c) throws DMLRuntimeException {
		/* This method is designed only for dense representation */
		if ( sparse )
			throw new DMLRuntimeException("MatrixBlockDSM.init() can be invoked only on matrices with dense representation.");
		
		if ( denseBlock == null 
				|| (denseBlock != null && denseBlock.length<rlen*clen) ) {
			denseBlock=null;
			denseBlock = new double[r*c];
		}
		
		int ind = 0;
		double lvalue = -1;
		for(int i=0; i < r; i++) {
			for(int j=0; j < c; j++) {
				lvalue = arr[i][j];
				denseBlock[ind++] = lvalue;
				if( lvalue!=0 )
					nonZeros++;
			}
		}
		maxrow = r;
		maxcolumn = c;
	}
	
	public MatrixBlockDSM(HashMap<CellIndex, Double> map) {
		// compute dimensions from the map
		int nrows=0, ncols=0;
		for (CellIndex index : map.keySet()) {
			nrows = (nrows < index.row ? index.row : nrows);
			ncols = (ncols < index.column ? index.column : ncols);
		}
		
		rlen = nrows;
		clen = ncols;
		sparse = true;
		nonZeros = 0;
		maxrow = nrows;
		maxcolumn = ncols;
		estimatedNNzsPerRow=(int)Math.ceil((double)map.size()/(double)rlen);
		
		for (CellIndex index : map.keySet()) {
			double d  = map.get(index).doubleValue();
			if ( d > 0 ) {
				this.quickSetValue(index.row-1, index.column-1, d);
				//nonZeros++;
			}
		}
	}
	
	public int getNumRows()
	{
		return rlen;
	}
	
	public int getNumColumns()
	{
		return clen;
	}
	
	// Return the maximum row encountered WITHIN the current block
	public int getMaxRow() {
		if (!sparse) 
			return getNumRows();
		else {
			return maxrow;
		}
	}
	
	// Return the maximum column encountered WITHIN the current block
	public int getMaxColumn() {
		if (!sparse) 
			return getNumColumns();
		else {
			return maxcolumn;
		}
	}
	
	public void setMaxRow(int _r) {
		maxrow = _r;
	}
	
	public void setMaxColumn(int _c) {
		maxcolumn = _c;
	}
	
	// NOTE: setNumRows() and setNumColumns() are used only in tertiaryInstruction (for contingency tables)
	public void setNumRows(int _r) {
		rlen = _r;
	}
	
	public void setNumColumns(int _c) {
		clen = _c;
	}
	
	public void print()
	{
		System.out.println("spathanks" +
				"rse? = "+sparse);
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
	
	public boolean isInSparseFormat()
	{
		return sparse;
	}
	
	private void resetSparse()
	{
		if(sparseRows!=null)
		{
			for(int i=0; i<Math.min(rlen, sparseRows.length); i++)
				if(sparseRows[i]!=null)
					sparseRows[i].reset(estimatedNNzsPerRow, clen);
		}
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
	
	public void reset()
	{
		reset(-rlen);
	}
	
	public void reset(int rl, int cl, int estnnzs) {
		rlen=rl;
		clen=cl;
		nonZeros=0;
		reset(estnnzs);
	}
	
	public void reset(int rl, int cl) {
		rlen=rl;
		clen=cl;
		nonZeros=0;
		reset();
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
	
	
	public void resetDenseWithValue(int rl, int cl, double v) {
		
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
	
	public void examSparsity() throws DMLRuntimeException
	{
		double sp = ((double)nonZeros/rlen)/clen;
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if(sparse)
		{
			if(sp>SPARCITY_TURN_POINT || clen<=SKINNY_MATRIX_TURN_POINT) {
				//System.out.println("Calling sparseToDense(): nz=" + nonZeros + ", rlen=" + rlen + ", clen=" + clen + ", sparsity = " + sp + ", spturn=" + SPARCITY_TURN_POINT );
				sparseToDense();
			}
		}else
		{
			if(sp<SPARCITY_TURN_POINT && clen>SKINNY_MATRIX_TURN_POINT) {
				//System.out.println("Calling denseToSparse(): nz=" + nonZeros + ", rlen=" + rlen + ", clen=" + clen + ", sparsity = " + sp + ", spturn=" + SPARCITY_TURN_POINT );
				denseToSparse();
			}
		}
	}
	
	public void cleanUp()
	{
		if(sparse && denseBlock!=null)
			denseBlock=null;
		else if(!sparse && sparseRows!=null)
			sparseRows=null;
			
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
		this.nonZeros=that.nonZeros;
		if(that.denseBlock==null)
		{
			resetSparse();
			return;
		}
		
		adjustSparseRows(rlen-1);
	
		int n=0;
		for(int r=0; r<rlen; r++)
		{
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(estimatedNNzsPerRow, clen);
			else
				sparseRows[r].reset(estimatedNNzsPerRow, clen);
			
			for(int c=0; c<clen; c++)
			{
				if(that.denseBlock[n]!=0)
					sparseRows[r].append(c, that.denseBlock[n]);
				n++;
			}
		}
	}
	
	public void copy(MatrixValue thatValue, boolean sp) {
		MatrixBlockDSM that;
		try {
			that = checkType(thatValue);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		}		
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
	
	public void copy(MatrixValue thatValue) 
	{
		MatrixBlockDSM that;
		try {
			that = checkType(thatValue);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		}		
		this.rlen=that.rlen;
		this.clen=that.clen;
		this.sparse=checkRealSparcity(that);
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
	
	public double[] getDenseArray()
	{
		if(sparse)
			return null;
		return denseBlock;
	}
	
	@Deprecated
	public HashMap<CellIndex, Double> getSparseMap()
	{
		if(!sparse || sparseRows==null)
			return null;
		HashMap<CellIndex, Double> map=new HashMap<CellIndex, Double>(nonZeros);
		for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
		{
			if(sparseRows[r]==null) continue;
			double[] values=sparseRows[r].getValueContainer();
			int[] cols=sparseRows[r].getIndexContainer();
			for(int i=0; i<sparseRows[r].size(); i++)
				map.put(new CellIndex(r, cols[i]), values[i]);
		}
		return map;
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
	
	public SparseCellIterator getSparseCellIterator()
	{
		if(!sparse)
			throw new RuntimeException("getSparseCellInterator should not be called for dense format");
		return new SparseCellIterator(rlen, sparseRows);
	}
	
	public int getNonZeros()
	{
		return nonZeros;
	}
	
	//only apply to non zero cells
	public void sparseScalarOperationsInPlace(ScalarOperator op) 
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
	
	//only apply to non zero cells
	public void sparseUnaryOperationsInPlace(UnaryOperator op) 
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
			
		}else
		{
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
	
	public void denseScalarOperationsInPlace(ScalarOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		double v;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				v=op.executeScalar(quickGetValue(r, c));
				quickSetValue(r, c, v);
			}	
	}
	
	public void denseUnaryOperationsInPlace(UnaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		double v;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				v=op.fn.execute(quickGetValue(r, c));
				quickSetValue(r, c, v);
			}	
	}
	
	private static MatrixBlockDSM checkType(MatrixValue block) throws DMLUnsupportedOperationException
	{
		if( block!=null && !(block instanceof MatrixBlockDSM))
			throw new DMLUnsupportedOperationException("the Matrix Value is not MatrixBlockDSM!");
		return (MatrixBlockDSM) block;
	}

	
	public MatrixValue scalarOperations(ScalarOperator op, MatrixValue result) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		checkType(result);
		
		// estimate the sparsity structure of result matrix
		boolean sp = this.sparse; // by default, we guess result.sparsity=input.sparsity
		if (!op.sparseSafe)
			sp = false; // if the operation is not sparse safe, then result will be in dense format
		
		if(result==null)
			result=new MatrixBlockDSM(rlen, clen, sp, this.nonZeros);
		else
			result.reset(rlen, clen, sp, this.nonZeros);
		
		result.copy(this, sp);
		
		if(op.sparseSafe)
			((MatrixBlockDSM)result).sparseScalarOperationsInPlace(op);
		else
			((MatrixBlockDSM)result).denseScalarOperationsInPlace(op);
		return result;
	}
	
	public void scalarOperationsInPlace(ScalarOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(op.sparseSafe)
			this.sparseScalarOperationsInPlace(op);
		else
			this.denseScalarOperationsInPlace(op);
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
		
		if(op.sparseSafe)
			((MatrixBlockDSM)result).sparseUnaryOperationsInPlace(op);
		else
			((MatrixBlockDSM)result).denseUnaryOperationsInPlace(op);
		return result;
	}
	
	public void unaryOperationsInPlace(UnaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(op.sparseSafe)
			this.sparseUnaryOperationsInPlace(op);
		else
			this.denseUnaryOperationsInPlace(op);
	}
	
	private MatrixBlockDSM denseBinaryHelp(BinaryOperator op, MatrixBlockDSM that, MatrixBlockDSM result) 
	throws DMLRuntimeException 
	{
		SparsityEstimate resultSparse=checkSparcityOnBinary(this, that, op);
		if(result==null)
			result=new MatrixBlockDSM(rlen, clen, resultSparse.sparse, resultSparse.estimatedNonZeros);
		else
			result.reset(rlen, clen, resultSparse.sparse, resultSparse.estimatedNonZeros);
		
		//double st = System.nanoTime();
		double v;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				v=op.fn.execute(this.quickGetValue(r, c), that.quickGetValue(r, c));
				result.appendValue(r, c, v);
			}
		//double en = System.nanoTime();
		//System.out.println("denseBinaryHelp()-new: " + (en-st)/Math.pow(10, 9) + " sec.");
		
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
			int startPosition, int resultRow, MatrixBlockDSM result) 
	throws DMLRuntimeException
	{
		int column;
		double v;
		int p1=startPosition;
		//take care of left over
		while(p1<size1)
		{
			v=op.fn.execute(values1[p1], 0);
			column=cols1[p1];
			p1++;	
			result.appendValue(resultRow, column, v);
		}
	}
	
	private static void appendRightForSparseBinary(BinaryOperator op, double[] values2, int[] cols2, int size2, 
			int startPosition, int resultRow, MatrixBlockDSM result) throws DMLRuntimeException
	{
		int column;
		double v;
		int p2=startPosition;
		while(p2<size2)
		{
			v=op.fn.execute(0, values2[p2]);
			column=cols2[p2];
			p2++;
			result.appendValue(resultRow, column, v);
		}
	}
	
	private MatrixBlockDSM sparseBinaryHelp(BinaryOperator op, MatrixBlockDSM that, MatrixBlockDSM result) 
	throws DMLRuntimeException 
	{
		SparsityEstimate resultSparse=checkSparcityOnBinary(this, that, op);
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
		else
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
	//	System.out.println("-- input 1: \n"+this.toString());
	//	System.out.println("-- input 2: \n"+that.toString());
	//	System.out.println("~~ output: \n"+result);
		return result;
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
		
		if(op.sparseSafe)
			return sparseBinaryHelp(op, that, (MatrixBlockDSM)result);
		else
			return denseBinaryHelp(op, that, (MatrixBlockDSM)result);
		
	}
	
	
	
	
	public void incrementalAggregate(AggregateOperator aggOp, MatrixValue correction, 
			MatrixValue newWithCorrection)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		assert(aggOp.correctionExists);
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
		}else if(aggOp.correctionLocation==CorrectionLocationType.NONE)
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
		}else if(aggOp.correctionLocation==CorrectionLocationType.LASTTWOROWS)
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
		assert(aggOp.correctionExists);
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

	//allocate space if sparseRows[r] doesnot exist
	private void adjustSparseRows(int r)
	{
		if(sparseRows==null)
			sparseRows=new SparseRow[rlen];
		else if(sparseRows.length<=r)
		{
			SparseRow[] oldSparseRows=sparseRows;
			sparseRows=new SparseRow[rlen];
			for(int i=0; i<Math.min(oldSparseRows.length, rlen); i++) {
				sparseRows[i]=oldSparseRows[i];
				//System.out.println(i + " " + oldSparseRows.length + " " + rlen);
			}
		}
		
	//	if(sparseRows[r]==null)
	//		sparseRows[r]=new SparseRow();
	}
	@Override
	/*
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
			
		}else
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
	
	public void updateNonZeros() {
		nonZeros = (int)computeNonZeros();
	}
	public long computeNonZeros() {
		long nnz = 0;
		if ( sparse ) {
			if ( sparseRows != null ) {
				double []values = null;
				for(int r=0; r < sparseRows.length; r++ ) {
					if ( sparseRows[r] == null )
						continue;
					values = sparseRows[r].getValueContainer();
					for (int i=0; i<sparseRows[r].size(); i++) {
						if ( values[i] != 0 )
							++nnz;
					}
				}
			}
		}
		else {
			if ( denseBlock != null ) {
				for ( int i=0; i < denseBlock.length; i++ ) {
					if( denseBlock[i] != 0 ) 
						++nnz;
				}
			}
		}
		return nnz;
	}
	public void setDenseValue(int r, int c, double v) {
		denseBlock[r*clen+c] = v;
		/*
		int index=r*clen+c;
		if(denseBlock[index]==0)
			nonZeros++;
		denseBlock[index]=v;
		if(v==0)
			nonZeros--;
		*/
	}
	
	@Override
	public void setValue(int r, int c, double v) {
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
	
	public void quickSetValue(int r, int c, double v) {
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
	
	
	/*
	 * append value is only used when values are appended at the end of each row for the sparse representation
	 * This can only be called, when the caller knows the access pattern of the block
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
	
	@Override
	public void setValue(CellIndex index, double v) {
		setValue(index.row, index.column, v);
	}
	
	@Override
	public double getValue(int r, int c) {
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
	
	public double quickGetValue(int r, int c) {
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
	public void getCellValues(Collection<Double> ret) {
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
	public void getCellValues(Map<Double, Integer> ret) {
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
	public void readFields(DataInput in) throws IOException {
		rlen=in.readInt();
		clen=in.readInt();
		sparse=in.readBoolean();
		if(sparse)
			readSparseBlock(in);
		else
			readDenseBlock(in);
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
		
		this.adjustSparseRows(rlen-1);
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
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(rlen);
		out.writeInt(clen);
		
		if(sparse)
		{
			if(sparseRows==null)
				writeEmptyBlock(out);
			//if it should be dense, then write to the dense format
			else if(nonZeros>((long)rlen)*((long)clen)*SPARCITY_TURN_POINT || clen<=SKINNY_MATRIX_TURN_POINT)
				writeSparseToDense(out);
			else
				writeSparseBlock(out);
		}else
		{
			if(denseBlock==null)
				writeEmptyBlock(out);
			//if it should be sparse
			else if(nonZeros<((long)rlen)*((long)clen)*SPARCITY_TURN_POINT && clen>SKINNY_MATRIX_TURN_POINT)
				writeDenseToSparse(out);
			else
				writeDenseBlock(out);
		}
	}
	
	private void writeEmptyBlock(DataOutput out) throws IOException
	{
		out.writeBoolean(true);
		for(int r=0; r<rlen; r++)
			out.writeInt(0);
	}
	
	private void writeDenseBlock(DataOutput out) throws IOException {
		out.writeBoolean(sparse);
		int limit=rlen*clen;
		for(int i=0; i<limit; i++)
			out.writeDouble(denseBlock[i]);
	}
	
	private void writeSparseBlock(DataOutput out) throws IOException {
		out.writeBoolean(sparse);
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
	
	private void writeSparseToDense(DataOutput out) throws IOException {
		out.writeBoolean(false);
		for(int i=0; i<rlen; i++)
			for(int j=0; j<clen; j++)
				out.writeDouble(quickGetValue(i, j));
	}
	
	private void writeDenseToSparse(DataOutput out) throws IOException {
		
		if(denseBlock==null)
		{
			writeEmptyBlock(out);
			return;
		}
		
		out.writeBoolean(true);
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
//		if(num!=nonZeros)
//			throw new IOException("nonZeros = "+nonZeros+", but should be "+num);
	}
	
	@Override
	public int compareTo(Object arg0) {
		// don't compare blocks
		return 0;
	}

	@Override
	public MatrixValue reorgOperations(ReorgOperator op, MatrixValue ret,
			int startRow, int startColumn, int length)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		if (!(op.fn.equals(SwapIndex.getSwapIndexFnObject()) || op.fn.equals(MaxIndex.getMaxIndexFnObject())))
			throw new DMLRuntimeException("the current reorgOperations cannot support "
					+op.fn.getClass()+", needs to examine whether appendValue is applicable!");
	//	long time=System.currentTimeMillis();
		MatrixBlockDSM result=checkType(ret);
		boolean reducedDim=op.fn.computeDimension(rlen, clen, tempCellIndex);
		boolean sps;
		if(reducedDim)
			sps=false;
		else if(op.fn.equals(MaxIndex.getMaxIndexFnObject())) {
			sps=true;
		}
		else
			sps=checkRealSparcity(this);
			
		if(result==null)
			result=new MatrixBlockDSM(tempCellIndex.row, tempCellIndex.column, sps, this.nonZeros);
		else
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
		
	//	time=System.currentTimeMillis()-time;
	//	System.out.println("------------- transpose ------------");
	//	System.out.println(time+"\t"+result.getCapacity()+"\t"+result.getNonZeros());
		return result;
	}
	
	public int getCapacity()
	{
		if(sparse)
		{
			if(sparseRows==null)
				return 0;
			int size=0;
			for(int r=0; r<Math.min(sparseRows.length, getNumRows()); r++)
				if(sparseRows[r]!=null)
					size+=sparseRows[r].capacity();
			return size;
		}else
		{
			if(denseBlock==null)
				return 0;
			return denseBlock.length;
		}
	}
	
	//TODO MB: remove row, col, len because not used (after MatrixObject is removed)
	public MatrixValue appendOperations(ReorgOperator op, MatrixValue ret, 
										int startRow, int startColumn, int length) 	
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		MatrixBlockDSM result=checkType(ret);
		boolean reducedDim=op.fn.computeDimension(rlen, clen, tempCellIndex);
		boolean sps;
		if(reducedDim)
			sps=false;
		else
			sps=checkRealSparcity(this);
			
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

	private void slideHelp(int r, IndexRange range, int colCut, MatrixBlockDSM left, MatrixBlockDSM right, int rowOffset, int normalBlockRowFactor, int normalBlockColFactor)
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
	
	private boolean checkSparcityOnSlide(int selectRlen, int selectClen, int finalRlen, int finalClen)
	{
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if(finalClen<=SKINNY_MATRIX_TURN_POINT)
			return false;
		else
			return((double)nonZeros/(double)rlen/(double)clen*(double)selectRlen*(double)selectClen/(double)finalRlen/(double)finalClen<SPARCITY_TURN_POINT);
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
		if(result==null)
			result=new MatrixBlockDSM(this);
		else {
			//result.reset(ru-rl+1, cu-cl+1, result_sparsity);
			result.copy(this);
		}
		
		// TODO: following implementation can be significantly improved by looking at the representations of "this" and "rhsmatrix"
		double d = 0;
		for ( int i=0, res_i=(int)rowLower-1; i < rhsMatrix.getNumRows(); i++, res_i++ ) {
			for ( int j=0, res_j=(int)colLower-1; j < rhsMatrix.getNumColumns(); j++, res_j++ ) {
				d = ((MatrixBlockDSM) rhsMatrix).quickGetValue(i,j);
				result.quickSetValue(res_i, res_j, d);
			}
		}
		
		return result;
	}
	
	/**
	 * Method to perform rangeReIndex operation for a given lower and upper bounds in row and column dimensions.
	 * Extracted submatrix is returned as "result".
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	public MatrixValue slideOperations(long rowLower, long rowUpper, long colLower, long colUpper, MatrixValue ret) 
	throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		// Check the validity of bounds
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
		boolean result_sparsity = this.sparse;
		MatrixBlockDSM result=checkType(ret);
		int estnnzs=(int) ((double)this.nonZeros/rlen/clen*(ru-rl+1)*(cu-cl+1));
		if(result==null)
			result=new MatrixBlockDSM(ru-rl+1, cu-cl+1, result_sparsity, estnnzs);
		else
			result.reset(ru-rl+1, cu-cl+1, result_sparsity, estnnzs);
		
		if (sparse) {
			if ( sparseRows != null ) {
				for(int r=rl; r <= Math.min(ru,getNumRows()); r++) {
					if(sparseRows[r] != null) {
						int[] cols=sparseRows[r].getIndexContainer();
						double[] values=sparseRows[r].getValueContainer();
						int j=0;
						while(j < sparseRows[r].size() && cols[j] < cl )
							j++;
						while(j < sparseRows[r].size() && cols[j] <= Math.min(cu,getNumColumns())) {
							((MatrixBlockDSM)result).appendValue(r-rl, cols[j]-cl, values[j]);
							j++;
						}
					}
				}
			}
		}
		else {
			if(denseBlock!=null)
			{
				int i = rl*clen;
				for(int r = rl; r <= Math.min(ru,getNumRows()); r++) {
					for(int c = cl; c <= Math.min(cu, getNumColumns()); c++) {
						result.quickSetValue(r-rl, c-cl, denseBlock[i+c]);
					}
					i+=clen;
				}
			}
		}
		return result;
	}
	
	
	
	public void slideOperations(ArrayList<IndexedMatrixValue> outlist, IndexRange range, int rowCut, int colCut, 
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
					checkSparcityOnSlide(minrowcut-(int)range.rowStart, mincolcut-(int)range.colStart, blockRowFactor, blockColFactor));
		}
		if(range.rowStart<rowCut && range.colEnd>=colCut)
		{
			topright=(MatrixBlockDSM) p.next().getValue();
			topright.reset(blockRowFactor, boundaryClen, 
					checkSparcityOnSlide(minrowcut-(int)range.rowStart, (int)range.colEnd-maxcolcut+1, blockRowFactor, boundaryClen));
		}
		if(range.rowEnd>=rowCut && range.colStart<colCut)
		{
			bottomleft=(MatrixBlockDSM) p.next().getValue();
			bottomleft.reset(boundaryRlen, blockColFactor, 
					checkSparcityOnSlide((int)range.rowEnd-maxrowcut+1, mincolcut-(int)range.colStart, boundaryRlen, blockColFactor));
		}
		if(range.rowEnd>=rowCut && range.colEnd>=colCut)
		{
			bottomright=(MatrixBlockDSM) p.next().getValue();
			bottomright.reset(boundaryRlen, boundaryClen, 
					checkSparcityOnSlide((int)range.rowEnd-maxrowcut+1, (int)range.colEnd-maxcolcut+1, boundaryRlen, boundaryClen));
			//System.out.println("bottomright size: "+bottomright.rlen+" X "+bottomright.clen);
		}
		
		if(sparse)
		{
			if(sparseRows!=null)
			{
				int r=(int)range.rowStart;
				for(; r<Math.min(Math.min(rowCut, sparseRows.length), range.rowEnd+1); r++)
					slideHelp(r, range, colCut, topleft, topright, normalBlockRowFactor-rowCut, normalBlockRowFactor, normalBlockColFactor);
				
				for(; r<=Math.min(range.rowEnd, sparseRows.length-1); r++)
					slideHelp(r, range, colCut, bottomleft, bottomright, -rowCut, normalBlockRowFactor, normalBlockColFactor);
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

	private void traceHelp(AggregateUnaryOperator op, MatrixBlockDSM result, 
			int blockingFactorRow, int blockingFactorCol, MatrixIndexes indexesIn) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//test whether this block contains any cell in the diag
		long topRow=UtilFunctions.cellIndexCalculation(indexesIn.getRowIndex(), blockingFactorRow, 0);
		long bottomRow=UtilFunctions.cellIndexCalculation(indexesIn.getRowIndex(), blockingFactorRow, this.rlen-1);
		long leftColumn=UtilFunctions.cellIndexCalculation(indexesIn.getColumnIndex(), blockingFactorCol, 0);
		long rightColumn=UtilFunctions.cellIndexCalculation(indexesIn.getColumnIndex(), blockingFactorCol, this.clen-1);
		
		long start=Math.max(topRow, leftColumn);
		long end=Math.min(bottomRow, rightColumn);
		
		if(start>end)
			return;
		
		if(op.aggOp.correctionExists)
		{
			KahanObject buffer=new KahanObject(0,0);
			for(long i=start; i<=end; i++)
			{
				buffer=(KahanObject) op.aggOp.increOp.fn.execute(buffer, 
						quickGetValue(UtilFunctions.cellInBlockCalculation(i, blockingFactorRow), UtilFunctions.cellInBlockCalculation(i, blockingFactorCol)));
			}
			result.quickSetValue(0, 0, buffer._sum);
			if(op.aggOp.correctionLocation==CorrectionLocationType.LASTROW)//extra row
				result.quickSetValue(1, 0, buffer._correction);
			else if(op.aggOp.correctionLocation==CorrectionLocationType.LASTCOLUMN)
				result.quickSetValue(0, 1, buffer._correction);
			else
				throw new DMLRuntimeException("unrecognized correctionLocation: "+op.aggOp.correctionLocation);
		}else
		{
			double newv=0;
			for(long i=start; i<=end; i++)
			{
				newv+=op.aggOp.increOp.fn.execute(newv,
						quickGetValue(UtilFunctions.cellInBlockCalculation(i, blockingFactorRow), UtilFunctions.cellInBlockCalculation(i, blockingFactorCol)));
			}
			result.quickSetValue(0, 0, newv);
		}
	}
		
	//change to a column vector
	private void diagM2VHelp(AggregateUnaryOperator op, MatrixBlockDSM result, 
			int blockingFactorRow, int blockingFactorCol, MatrixIndexes indexesIn) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//test whether this block contains any cell in the diag
		long topRow=UtilFunctions.cellIndexCalculation(indexesIn.getRowIndex(), blockingFactorRow, 0);
		long bottomRow=UtilFunctions.cellIndexCalculation(indexesIn.getRowIndex(), blockingFactorRow, this.rlen-1);
		long leftColumn=UtilFunctions.cellIndexCalculation(indexesIn.getColumnIndex(), blockingFactorCol, 0);
		long rightColumn=UtilFunctions.cellIndexCalculation(indexesIn.getColumnIndex(), blockingFactorCol, this.clen-1);
		
		long start=Math.max(topRow, leftColumn);
		long end=Math.min(bottomRow, rightColumn);
		
		if(start>end)
			return;
		
		for(long i=start; i<=end; i++)
		{
			int cellRow=UtilFunctions.cellInBlockCalculation(i, blockingFactorRow);
			int cellCol=UtilFunctions.cellInBlockCalculation(i, blockingFactorCol);
			result.appendValue(cellRow, 0, quickGetValue(cellRow, cellCol));
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
				double toadd=(newvalue-buffer._sum)/count;
				buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
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
	
	private void sparseAggregateUnaryHelp(AggregateUnaryOperator op, MatrixBlockDSM result,
			int blockingFactorRow, int blockingFactorCol, MatrixIndexes indexesIn) throws DMLRuntimeException
	{
		//initialize result
		if(op.aggOp.initialValue!=0)
			result.resetDenseWithValue(result.rlen, result.clen, op.aggOp.initialValue);
		
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
						result.tempCellIndex.set(r, cols[i]);
						op.indexFn.execute(result.tempCellIndex, result.tempCellIndex);
						incrementalAggregateUnaryHelp(op.aggOp, result, result.tempCellIndex.row, result.tempCellIndex.column, values[i], buffer);

					}
				}
			}
		}else
		{
			if(denseBlock!=null)
			{
				int limit=rlen*clen;
				for(int i=0; i<limit; i++)
				{
					r=i/clen;
					c=i%clen;
					result.tempCellIndex.set(r, c);
					op.indexFn.execute(result.tempCellIndex, result.tempCellIndex);
					incrementalAggregateUnaryHelp(op.aggOp, result, result.tempCellIndex.row, result.tempCellIndex.column, denseBlock[i], buffer);
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
		
		KahanObject buffer=new KahanObject(0,0);
		for(int i=0; i<rlen; i++)
			for(int j=0; j<clen; j++)
			{
				result.tempCellIndex.set(i, j);
				op.indexFn.execute(result.tempCellIndex, result.tempCellIndex);

				if(op.aggOp.correctionExists
				   && op.aggOp.correctionLocation == CorrectionLocationType.LASTCOLUMN
				   && op.aggOp.increOp.fn instanceof Builtin 
				   && ((Builtin)(op.aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MAXINDEX ){
					double currMaxValue = result.quickGetValue(i, 1);
					long newMaxIndex = UtilFunctions.cellIndexCalculation(indexesIn.getColumnIndex(), maxcolumn, j);
					double newMaxValue = quickGetValue(i, j);
					double update = op.aggOp.increOp.fn.execute(newMaxValue, currMaxValue);
						    
					if(update == 1){
						result.quickSetValue(i, 0, newMaxIndex);
						result.quickSetValue(i, 1, newMaxValue);
					}
				}else
					incrementalAggregateUnaryHelp(op.aggOp, result, result.tempCellIndex.row, result.tempCellIndex.column, quickGetValue(i,j), buffer);
			}
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
		/*	
			if(op.aggOp.correctionLocation==1)
				tempCellIndex.row++;
			else if(op.aggOp.correctionLocation==2)
				tempCellIndex.column++;
			else
				throw new DMLRuntimeException("unrecognized correctionLocation: "+op.aggOp.correctionLocation);	*/
		}
		if(result==null)
			result=new MatrixBlockDSM(tempCellIndex.row, tempCellIndex.column, false);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, false);
		
		//TODO: this code is hack to support trace, and should be removed when selection is supported
		if(op.isTrace)
			traceHelp(op, (MatrixBlockDSM)result, blockingFactorRow, blockingFactorCol, indexesIn);
		else if(op.isDiagM2V)
			diagM2VHelp(op, (MatrixBlockDSM)result, blockingFactorRow, blockingFactorCol, indexesIn);
		else if(op.sparseSafe)
			sparseAggregateUnaryHelp(op, (MatrixBlockDSM)result, blockingFactorRow, blockingFactorCol, indexesIn);
		else
			denseAggregateUnaryHelp(op, (MatrixBlockDSM)result, blockingFactorRow, blockingFactorCol, indexesIn);
		
		if(op.aggOp.correctionExists && inCP)
			((MatrixBlockDSM)result).dropLastRowsOrColums(op.aggOp.correctionLocation);
		return result;
	}
	
	public CM_COV_Object cmOperations(CMOperator op) throws DMLRuntimeException {
		/* this._data must be a 1 dimensional vector */
		if ( this.getNumColumns() != 1) {
			throw new DMLRuntimeException("Central Moment can not be computed on [" 
					+ this.getNumRows() + "," + this.getNumColumns() + "] matrix.");
		}
		CM_COV_Object cmobj = new CM_COV_Object();
		int nzcount = 0;
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
				{
					if(sparseRows[r]==null) continue;
					//int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					for(int i=0; i<sparseRows[r].size(); i++)
					{
						op.fn.execute(cmobj, values[i], 1.0);
						nzcount++;
					}
				}
				// account for zeros in the vector
				op.fn.execute(cmobj, 0.0, this.getNumRows()-nzcount);
			}
		}
		else {
			if(denseBlock!=null)
			{
				int limit=rlen*clen;
				for(int i=0; i<limit; i++)
				{
					op.fn.execute(cmobj, denseBlock[i], 1.0);
				}
			}
		}
		return cmobj;
	}
	
	public CM_COV_Object cmOperations(CMOperator op, MatrixBlockDSM weights) throws DMLRuntimeException {
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
		if (sparse) {
			if(sparseRows!=null)
			{
				for(int r=0; r < this.getNumRows(); r++) {
					op.fn.execute(cmobj, this.quickGetValue(r,0), weights.quickGetValue(r,0));
				}
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
*/			}
		}
		else {
			if(denseBlock!=null)
			{
				int limit=rlen*clen, r, c;
				for(int i=0; i<limit; i++) {
					r=i/clen;
					c=i%clen;
					op.fn.execute(cmobj, denseBlock[i], weights.quickGetValue(r,c) );
				}
			}
		}
		return cmobj;
	}
	
	public CM_COV_Object covOperations(COVOperator op, MatrixBlockDSM that) throws DMLRuntimeException {
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
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r < this.getNumRows(); r++ ) {
					op.fn.execute(covobj, this.quickGetValue(r,0), that.quickGetValue(r,0), 1.0);
				}
			}
		}
		else {
			if(denseBlock!=null)
			{
				int limit=rlen*clen, r, c;
				for(int i=0; i<limit; i++)
				{
					r=i/clen;
					c=i%clen;
					op.fn.execute(covobj, denseBlock[i], that.quickGetValue(r,c), 1.0);
				}
			}
		}
		return covobj;
	}
	
	public CM_COV_Object covOperations(COVOperator op, MatrixBlockDSM that, MatrixBlockDSM weights) throws DMLRuntimeException {
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
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r < this.getNumRows(); r++ ) {
					op.fn.execute(covobj, this.quickGetValue(r,0), that.quickGetValue(r,0), weights.quickGetValue(r,0));
				}
			}
		}
		else {
			if(denseBlock!=null)
			{
				int limit=rlen*clen, r, c;
				for(int i=0; i<limit; i++)
				{
					r=i/clen;
					c=i%clen;
					op.fn.execute(covobj, denseBlock[i], that.quickGetValue(r,c), weights.quickGetValue(r,c));
				}
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
	
	// In a given two column matrix, the second column denotes weights.
	// This function computes the total weight
	private double sumWeightForQuantile() throws DMLRuntimeException {
		double sum_wt = 0;
		for (int i=0; i < getNumRows(); i++ )
			sum_wt += quickGetValue(i, 1);
		if ( (int)sum_wt != sum_wt ) {
			throw new DMLRuntimeException("Unexpected error while computing quantile -- weights must be integers.");
		}
		return sum_wt;
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
	
	public double pickValue(double quantile) throws DMLRuntimeException {
		double sum_wt = sumWeightForQuantile();
		
		int pos = (int) Math.ceil(quantile*sum_wt);
		
		int t = 0, i=-1;
		do {
			i++;
			t += quickGetValue(i,1);
		} while(t<pos && i < getNumRows());
		
		if ( sum_wt%2 == 1 ) {
			// sum_wt is odd
			return quickGetValue(i,0);
		}
		else {
			// sum_wt is even
			if ( pos+1 <= t ) {
				return quickGetValue(i,0);
			}
			else {
				return (quickGetValue(i,0)+quickGetValue(i+1,0))/2;
			}
		}
	}
	
	
	
	public void dropLastRowsOrColums(CorrectionLocationType correctionLocation) {
		
		if(correctionLocation==CorrectionLocationType.NONE || correctionLocation==CorrectionLocationType.INVALID)
			return;
		
		int step=1;
		if(correctionLocation==CorrectionLocationType.LASTTWOROWS || correctionLocation==CorrectionLocationType.LASTTWOCOLUMNS)
			step=2;
		
		if(correctionLocation==CorrectionLocationType.LASTROW || correctionLocation==CorrectionLocationType.LASTTWOROWS)
		{
			
			if(sparse)
			{
				if(sparseRows!=null)
				{
					for(int i=1; i<=step; i++)
					{
						if(sparseRows[rlen-i]!=null)
							this.nonZeros-=sparseRows[rlen-i].size();
					}
				}
			}else
			{
				if(denseBlock!=null)
				{
					for(int i=rlen*(clen-step); i<rlen*clen; i++)
					{
						if(denseBlock[i]!=0)
							this.nonZeros--;
					}
				}
			}
			//just need to shrink the dimension, the deleted rows won't be accessed
			this.rlen-=step;
		}
		
		if(correctionLocation==CorrectionLocationType.LASTCOLUMN || correctionLocation==CorrectionLocationType.LASTTWOCOLUMNS)
		{
			if(sparse)
			{
				if(sparseRows!=null)
				{
					for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
					{
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
			}else
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
			this.clen-=step;
		}
	}
	
	private static void sparseAggregateBinaryHelp(MatrixBlockDSM m1, MatrixBlockDSM m2, 
			MatrixBlockDSM result, AggregateBinaryOperator op) throws DMLRuntimeException 
	{
		if(!m1.sparse && !m2.sparse)
			aggBinDense(m1, m2, result, op);
		else if(m1.sparse && m2.sparse)
			aggBinSparse(m1, m2, result, op);
		else if(m1.sparse)
			aggBinSparseDense(m1, m2, result, op);
		else
			aggBinDenseSparse(m1, m2, result, op);
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
		SparsityEstimate sp=checkSparcityOnAggBinary(m1, m2, op);
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
	
	public void allocateDenseBlock() {
		int limit=rlen*clen;
		if(denseBlock==null || denseBlock.length < limit )
			denseBlock=new double[limit];
		nonZeros = 0;
	}
	
	/*private static void aggBinDenseSparse_2(MatrixBlockDSM m1, MatrixBlockDSM m2,
			MatrixBlockDSM result, AggregateBinaryOperator op) throws DMLRuntimeException 
	{
		if(m2.sparseRows==null)
			return;
		
		if ( result.sparse ) {
			throw new DMLRuntimeException("this case is not implemented yet.");
		}
		else {
			long begin, st, m1Time=0, mmultTime=0, resultTime=0, nnzTime=0, totalTime=0; 
			
			result.allocateDenseBlock();
			
			begin = st = System.currentTimeMillis();
			//m1.inplaceTranspose();
			//m1Time = System.currentTimeMillis() - st; 
			
			//st = System.currentTimeMillis();
			int[] cols=null;
			double[] values=null;
			double aik, bkj;
			for(int k=0; k<Math.min(m2.rlen, m2.sparseRows.length); k++) {
				if(m2.sparseRows[k]==null) continue;
				cols=m2.sparseRows[k].getIndexContainer();
				values=m2.sparseRows[k].getValueContainer();
				for(int p=0; p<m2.sparseRows[k].size(); p++) {
					int j=cols[p];
					for(int i=0; i<m1.rlen; i++) {
						//double old=result.getValue(i, j);
						aik= m1.denseBlock[i*m1.clen+k];
						bkj = values[p];
						
						// actual_result[i,j] will be @ result[j,i] 
						result.denseBlock[j*result.rlen+i] += (aik*bkj);
						//double addValue=op.binaryFn.execute(aik, values[p]);
						//double newvalue=op.aggOp.increOp.fn.execute(old, addValue);
						//result.setValue(i, j, newvalue);
					}
				}
			}
			mmultTime = System.currentTimeMillis()-st;
			
			// Convert from column-major to row-major
			st = System.currentTimeMillis();
			double [] temp = new double[result.clen*result.rlen];
			for(int j=0; j < result.clen; j++) {
				for(int i=0; i < result.rlen; i++) {
					//int x = j*result.clen+i;
					//int y = i*result.clen+j;
					//System.out.println(x + "<-" + y);
					temp[i*result.clen+j] = result.denseBlock[j*result.rlen+i];
				}
			}
			result.denseBlock = null;
			result.denseBlock = temp;
			temp = null;
			resultTime = System.currentTimeMillis() - st;
			
			st = System.currentTimeMillis();
			result.updateNonZeros();
			nnzTime = System.currentTimeMillis() - st;
			
			totalTime = System.currentTimeMillis() - begin;
			
			System.out.println("denseSparse:\t" + m1Time + "\t" + mmultTime + "\t" + resultTime + "\t" + nnzTime + "\t" + totalTime);
		}
		
	}*/
	
	/*
	 * to perform aggregateBinary when the first matrix is dense and the second is sparse
	 */
	private static void aggBinDenseSparse(MatrixBlockDSM m1, MatrixBlockDSM m2,
			MatrixBlockDSM result, AggregateBinaryOperator op) throws DMLRuntimeException 
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
	
	/*
	 * to perform aggregateBinary when the first matrix is sparse and the second is dense
	 */
	private static void aggBinSparseDense(MatrixBlockDSM m1, MatrixBlockDSM m2,
			MatrixBlockDSM result, AggregateBinaryOperator op) throws DMLRuntimeException
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
	
	public static void aggBinSparse(MatrixBlockDSM m1, MatrixBlockDSM m2,
			MatrixBlockDSM result, AggregateBinaryOperator op) throws DMLRuntimeException 
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
	
	/**
	 * <p>
	 * 	Performs a dense-dense matrix multiplication using a modified algorithm and
	 * 	stores the result in the resulting matrix.<br />
	 *	The result of the matrix multiplication is again a dense matrix.
	 * </p>
	 * 
	 * @param matrixA first matrix
	 * @param matrixB second matrix
	 * @param resultMatrix result matrix
	 * @throws IllegalArgumentException if the matrixes are of wrong format
	 */
	public static void matrixMult(MatrixBlockDSM matrixA, MatrixBlockDSM matrixB,
			MatrixBlockDSM resultMatrix)
	{
	/*	if(matrixA.sparse || matrixB.sparse || resultMatrix.sparse)
			throw new IllegalArgumentException("only dense matrixes are allowed");
		if(resultMatrix.rlen != matrixA.rlen || resultMatrix.clen != matrixB.clen)
			throw new IllegalArgumentException("result matrix has wrong size");
	*/	
		int l, i, j, aIndex, bIndex, cIndex;
		double temp;
		double[] a = matrixA.getDenseArray();
		double[] b = matrixB.getDenseArray();
		if(a==null || b==null)
			return;
		if(resultMatrix.denseBlock==null)
			resultMatrix.denseBlock = new double[resultMatrix.rlen * resultMatrix.clen];
		Arrays.fill(resultMatrix.denseBlock, 0, resultMatrix.denseBlock.length, 0);
		double[] c=resultMatrix.denseBlock;
		int m = matrixA.rlen;
		int n = matrixB.clen;
		int k = matrixA.clen;
		
		int nnzs=0;
		for(l = 0; l < k; l++)
		{
			aIndex = l;
			cIndex = 0;
			for(i = 0; i < m; i++)
			{
				// aIndex = i * k + l => a[i, l]
				temp = a[aIndex];
				if(temp != 0)
				{
					bIndex = l * n;
					for(j = 0; j < n; j++)
					{
						// bIndex = l * n + j => b[l, j]
						// cIndex = i * n + j => c[i, j]
						if(c[cIndex]==0)
							nnzs++;
						c[cIndex] = c[cIndex] + temp * b[bIndex];
						if(c[cIndex]==0)
							nnzs--;
						cIndex++;
						bIndex++;
					}
				}else
					cIndex+=n;
				aIndex += k;
			}
		}
		resultMatrix.nonZeros=nnzs;
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
	/*
	 * to perform aggregateBinary when both matrices are dense
	 */
	private static void aggBinDense(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM result, AggregateBinaryOperator op) throws DMLRuntimeException
	{
		if(op.binaryFn instanceof Multiply && (op.aggOp.increOp.fn instanceof Plus) && !result.sparse)
		{
			matrixMult(m1, m2, result);
		} else
		{
			int j, l, i, cIndex, bIndex, aIndex;
			double temp;
			double v;
			double[] a = m1.getDenseArray();
			double[] b = m2.getDenseArray();
			if(a==null || b==null)
				return;
			
			for(l = 0; l < m1.clen; l++)
			{
				aIndex = l;
				cIndex = 0;
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
						cIndex++;
						bIndex++;
					}
					
					aIndex += m1.clen;
				}
			}
		}
	}
	
	public MatrixValue groupedAggOperations(MatrixValue tgt, MatrixValue wghts, MatrixValue ret, Operator op) 
	throws DMLRuntimeException, DMLUnsupportedOperationException {
		// this <- groups
		MatrixBlockDSM target=checkType(tgt);
		MatrixBlockDSM weights=checkType(wghts);
		
		if (this.getNumColumns() != 1 || target.getNumColumns() != 1 || (weights!=null && weights.getNumColumns()!=1) )
			throw new DMLRuntimeException("groupedAggregate can only operate on 1-dimensional column matrices.");
		if ( this.getNumRows() != target.getNumRows() || (weights != null && this.getNumRows() != weights.getNumRows()) ) 
			throw new DMLRuntimeException("groupedAggregate can only operate on matrices with equal dimensions.");
		
		// Determine the number of groups
		double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
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
			CM cmFn = CM.getCMFnObject();
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

	
	/**
	 * 
	 * @param ret
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public MatrixValue removeEmptyRows(MatrixValue ret) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//check for empty inputs
		if( nonZeros==0 ) {
			ret.reset(0, 0, false);
			return ret;
		}
		
		MatrixBlockDSM result = checkType(ret);
		
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
		result.reset(rlen2, clen, sparse);
		int rindex = 0;
		for( int i=0; i<rlen; i++ )
			if( !flags[i] )
			{
				//copy row to result
				if(sparse)
				{
					result.appendRow(rindex, sparseRows[i]);
				}
				else
				{
					if( result.denseBlock==null )
						result.denseBlock = new double[ rlen2*clen ];
					int index1 = i*clen;
					int index2 = rindex*clen;
					for(int j=0; j<clen; j++)
					{
						if( denseBlock[index1] != 0 )
						{
							result.denseBlock[index2] = denseBlock[index1];
							result.nonZeros++;
						}
						index1++;
						index2++;
					}
				}
				rindex++;
			}
		
		
		//check sparsity
		result.examSparsity();
		
		return result;
	}
	
	/**
	 * 
	 * @param ret
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public MatrixValue removeEmptyColumns(MatrixValue ret) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//check for empty inputs
		if( nonZeros==0 ) {
			ret.reset(0, 0, false);
			return ret;
		}
		
		MatrixBlockDSM result = checkType(ret);
		
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
		result.reset(rlen, clen2, sparse);
		
		int cindex = 0;
		for( int j=0; j<clen; j++ )
			if( !flags[j] )
			{
				//copy col to result
				for( int i=0; i<rlen; i++ )
				{
					double value = quickGetValue(i, j);
					if( value != 0 )
						result.quickSetValue(i, cindex, value);
				}
				
				cindex++;
			}
		
		//check sparsity
		result.examSparsity();
		
		return result;
	}
	
	
	
	@Override
	/*
	 *  D = ctable(A,v2,W)
	 *  this <- A; scalarThat <- v2; that2 <- W; result <- D
	 */
	public void tertiaryOperations(Operator op, double scalarThat,
			MatrixValue that2Val, HashMap<CellIndex, Double> ctableResult)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		/*
		 * (i1,j1,v1) from input1 (this)
		 * (v2) from sclar_input2 (scalarThat)
		 * (i3,j3,w)  from input3 (that2)
		 */
		
		MatrixBlockDSM that2 = checkType(that2Val);
		
		double v1;
		double v2 = scalarThat;
		double w;
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
				{
					if ( sparseRows[r] == null )
						continue;
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					for(int i=0; i<sparseRows[r].size(); i++)
					{
						// output (v1,v2,w)
						v1 = values[i];
						w = that2.quickGetValue(r, cols[i]);
						updateCtable(v1, v2, w, ctableResult);
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
					v1 = this.quickGetValue(r, c);
					w = that2.quickGetValue(r, c);
					updateCtable(v1, v2, w, ctableResult);
				}
			}
			
		}
		
	}

	/*
	 *  D = ctable(A,v2,w)
	 *  this <- A; scalar_that <- v2; scalar_that2 <- w; result <- D
	 */
	@Override
	public void tertiaryOperations(Operator op, double scalarThat,
			double scalarThat2, HashMap<CellIndex, Double> ctableResult)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		/*
		 * (i1,j1,v1) from input1 (this)
		 * (v2) from sclar_input2 (scalarThat)
		 * (w)  from scalar_input3 (scalarThat2)
		 */
		
		double v1;
		double v2 = scalarThat;
		double w = scalarThat2;
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
				{
					if ( sparseRows[r] == null )
						continue;
					//int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					for(int i=0; i<sparseRows[r].size(); i++)
					{
						// output (v1,v2,w)
						v1 = values[i];
						updateCtable(v1, v2, w, ctableResult);
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
					v1 = this.quickGetValue(r, c);
					updateCtable(v1, v2, w, ctableResult);
				}
			}
			
		}
		
	}

	/*
	 *  D = ctable(A,B,w)
	 *  this <- A; that <- B; scalar_that2 <- w; result <- D
	 */
	@Override
	public void tertiaryOperations(Operator op, MatrixValue thatVal,
			double scalarThat2, HashMap<CellIndex, Double> ctableResult)
			throws DMLUnsupportedOperationException, DMLRuntimeException {

		/*
		 * (i1,j1,v1) from input1 (this)
		 * (i1,j1,v2) from input2 (that)
		 * (w)  from scalar_input3 (scalarThat2)
		 */
		
		MatrixBlockDSM that = checkType(thatVal);
		
		double v1, v2;
		double w = scalarThat2;
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
				{
					if ( sparseRows[r] == null )
						continue;
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					for(int i=0; i<sparseRows[r].size(); i++)
					{
						// output (v1,v2,w)
						v1 = values[i];
						v2 = that.quickGetValue(r, cols[i]);
						updateCtable(v1, v2, w, ctableResult);
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
					v1 = this.quickGetValue(r, c);
					v2 = that.quickGetValue(r, c);
					updateCtable(v1, v2, w, ctableResult);
				}
			}
			
		}
	}
	
	/*
	 *  D = ctable(A,B,W)
	 *  this <- A; that <- B; that2 <- W; result <- D
	 */
	public void tertiaryOperations(Operator op, MatrixValue thatVal, MatrixValue that2Val, HashMap<CellIndex, Double> ctableResult)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{	
		/*
		 * (i1,j1,v1) from input1 (this)
		 * (i1,j1,v2) from input2 (that)
		 * (i1,j1,w)  from input3 (that2)
		 */
		
		MatrixBlockDSM that = checkType(thatVal);
		MatrixBlockDSM that2 = checkType(that2Val);
		
		double v1, v2, w;
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r<Math.min(rlen, sparseRows.length); r++)
				{
					if ( sparseRows[r] == null )
						continue;
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					for(int i=0; i<sparseRows[r].size(); i++)
					{
						// output (v1,v2,w)
						v1 = values[i];
						v2 = that.quickGetValue(r, cols[i]);
						w = that2.quickGetValue(r, cols[i]);
						updateCtable(v1, v2, w, ctableResult);
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
					v1 = this.quickGetValue(r, c);
					v2 = that.quickGetValue(r, c);
					w = that2.quickGetValue(r, c);
					updateCtable(v1, v2, w, ctableResult);
				}
			}
			
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
	
	public void denseToSparse() {
		
		//LOG.info("**** denseToSparse: "+this.getNumRows()+"x"+this.getNumColumns()+"  nonZeros: "+this.nonZeros);
		sparse=true;
		adjustSparseRows(rlen-1);
		reset();
		if(denseBlock==null)
			return;
		int index=0;
		for(int r=0; r<rlen; r++)
		{
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(estimatedNNzsPerRow, clen);
			
			for(int c=0; c<clen; c++)
			{
				if(denseBlock[index]!=0)
				{
					sparseRows[r].append(c, denseBlock[index]);
					nonZeros++;
				}
				index++;
			}
		}
	}
	public void sparseToDense() throws DMLRuntimeException {
		
		//LOG.info("**** sparseToDense: "+this.getNumRows()+"x"+this.getNumColumns()+"  nonZeros: "+this.nonZeros);
		
		sparse=false;
		int limit=rlen*clen;
		if ( limit < 0 ) {
			throw new DMLRuntimeException("Unexpected error in sparseToDense().. limit < 0: " + rlen + ", " + clen + ", " + limit);
		}
		if(denseBlock==null || denseBlock.length < limit )
			denseBlock=new double[limit];
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
		
	}
	
	private void denseBinaryInPlaceHelp(BinaryOperator op, MatrixBlockDSM that) throws DMLRuntimeException 
	{
		SparsityEstimate resultSparse=checkSparcityOnBinary(this, that, op);
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
	
	private void sparseBinaryInPlaceHelp(BinaryOperator op, MatrixBlockDSM that) throws DMLRuntimeException 
	{
		SparsityEstimate resultSparse=checkSparcityOnBinary(this, that, op);
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
			}else if(this.sparseRows==null)
			{
				this.sparseRows=new SparseRow[rlen];
				for(int r=0; r<rlen; r++)
				{
					if(that.sparseRows[r]==null)
						continue;
					
					this.sparseRows[r]=new SparseRow(that.sparseRows[r].size(), clen);
					appendRightForSparseBinary(op, that.sparseRows[r].getValueContainer(), 
							that.sparseRows[r].getIndexContainer(), that.sparseRows[r].size(), 0, r, this);
				}
				
			}else
			{
				for(int r=0; r<rlen; r++)
				{
					if(this.sparseRows[r]==null)
						continue;
					appendLeftForSparseBinary(op, this.sparseRows[r].getValueContainer(), 
							this.sparseRows[r].getIndexContainer(), this.sparseRows[r].size(), 0, r, this);
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
	
*/	/**
	 * Generates a matrix of random numbers from uniform distribution U[0,1).
	 */
	public MatrixBlockDSM getRandomSparseMatrix(int rows, int cols, double sparsity, double min, double max, long seed)
	{
		sparse = (sparsity < SPARCITY_TURN_POINT);
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if(cols<=SKINNY_MATRIX_TURN_POINT)
			sparse=false;
		this.reset(rows, cols, sparse);
		
		if ( min == 0.0 && max == 0.0 ) {
			// nothing to do here
			return this;
		}

		double val;
		Random ru = new Random(seed); // for actual values
		Random random=new Random(ru.nextLong()); // for checking sparsity
		double range = max - min;

		if ( sparse ) {
			// sparse representation
			this.sparseRows=new SparseRow[rows];
			for(int i=0; i<rows; i++) {
				this.sparseRows[i]=new SparseRow(estimatedNNzsPerRow, clen);	
				for(int j=0; j<cols; j++) {
					if(random.nextDouble() <= sparsity) {
						val = min + (range * ru.nextDouble());
						this.sparseRows[i].append(j, val );
					}
				}
			}
		}
		else {
			// dense representation
			this.allocateDenseBlock();
			if ( sparsity == 1.0 ) {
				// special handling for the case sparsity=1.0, for better efficiency
				for(int index=0; index < rows*cols; index++) {
					val = min + (range * ru.nextDouble());
					this.denseBlock[index] = val;
				}
			}
			else {
				for(int index=0; index < rows*cols; index++) {
					if(random.nextDouble() <= sparsity) {
						val = min + (range * ru.nextDouble());
						this.denseBlock[index] = val;
					}
				}
			}
		}
		
		this.updateNonZeros();
		return this;
	}
	
	/**
	 * Generates a matrix of random numbers from standard normal distribution N(0,1).
	 * Unlike in <code>getRandomSparseMatrix()</code>, parameters <code>min</code> and
	 * <code>max</code> are not relevant for this function, since the range of values
	 * in N(0,1) is (-Inf,+Inf).
	 * 
	 */
	public MatrixBlockDSM getNormalRandomSparseMatrix(int rows, int cols, double sparsity, long seed)
	{
		sparse = (sparsity < SPARCITY_TURN_POINT);
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if(cols<=SKINNY_MATRIX_TURN_POINT)
			sparse=false;
		this.reset(rows, cols, sparse);

		double val;
		Random ru = new Random(seed); // for generating actual values, // uniform generator, used internally within RandNPair 
		Random random=new Random(ru.nextLong()); // for checking the sparsity
		
		if ( sparse ) {
			// sparse representation
			RandN rn = new RandN(ru);
			this.sparseRows=new SparseRow[rows];
			for(int i=0; i<rows; i++) {
				this.sparseRows[i]=new SparseRow(estimatedNNzsPerRow, clen);	
				for(int j=0; j<cols; j++) {
					if(random.nextDouble()>sparsity)
						continue;
					val = rn.nextDouble();
					this.sparseRows[i].append(j, val );
				}
			}
		}
		else {
			// dense representation
			this.allocateDenseBlock();
			if ( sparsity == 1.0 ) {
				// special handling for the case sparsity=1.0, for better efficiency
				
				//Random ru = new Random(seed); // uniform generator, used internally within RandNPair 
				RandNPair pair = new RandNPair();
				int index = 0;
				while ( index < rows*cols ) {
					pair.compute(ru);
					this.denseBlock[index++] = pair.getFirst();
					if ( index < rows*cols )
						this.denseBlock[index++] = pair.getSecond();
				}
			}
			else {
				RandN rn = new RandN(seed);
				for(int index=0; index < rows*cols; index++) {
					if(random.nextDouble() <= sparsity) {
						this.denseBlock[index] = rn.nextDouble();
					}
				}
			}
		}
		
		this.updateNonZeros();
		return this;
	}

	public static MatrixBlockDSM getRandomSparseMatrix(int rows, int cols, double sparsity, long seed)
	{
		Random random=new Random(seed);
		MatrixBlockDSM m=new MatrixBlockDSM(rows, cols, true, (int)(sparsity*rows*cols));
		m.sparseRows=new SparseRow[rows];
		double val;
		for(int i=0; i<rows; i++)
		{
			m.sparseRows[i]=new SparseRow(m.estimatedNNzsPerRow, m.clen);	
			for(int j=0; j<cols; j++)
			{
				if(random.nextDouble()>sparsity)
					continue;
				val = random.nextDouble();
				m.sparseRows[i].append(j, val); 
				//if ( val != 0)
				//	m.nonZeros++;
			}
		}
		m.updateNonZeros();
		return m;
	}
	
	public static MatrixBlock1D getRandomSparseMatrix1D(int rows, int cols, double sparsity, long seed)
	{
		Random random=new Random(seed);
		MatrixBlock1D m=new MatrixBlock1D(rows, cols, true);
		for(int i=0; i<rows; i++)
		{
			for(int j=0; j<cols; j++)
			{
				if(random.nextDouble()>sparsity)
					continue;
				m.addValue(i, j, random.nextDouble());
			}
		}
		return m;
	}
	
	public String toString()
	{
		String ret="sparse? = "+sparse+"\n" ;
		ret+="nonzeros = "+nonZeros+"\n";
		ret+="size: "+rlen+" X "+clen+"\n";
		boolean toprint=true;
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
	
	
	/*
	 * Recalculate nonzeros
	 */
	
	public void recomputeNonZeros()
	{
		nonZeros=0;
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int i=0; i<Math.min(rlen, sparseRows.length); i++)
					if(sparseRows[i]!=null)
						nonZeros+=sparseRows[i].size();
			}
		}else
		{
			if(denseBlock!=null)
			{
				int limit=rlen*clen;
				for(int i=0; i<limit; i++)
					if(denseBlock[i]!=0)
						nonZeros++;
			}
		}
	}
	
	/*
	 * this should be called only in the read and write functions for CP
	 * This function should be called before calling any setValueDenseUnsafe()
	 */
	public void spaceAllocForDenseUnsafe(int rl, int cl)
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
	
	
	/*
	 * this should be called only in the read and write functions for CP
	 * This function should be called before calling any setValueSparseUnsafe() or appendValueSparseUnsafe()
	 */
	public void spaceAllocForSparseUnsafe(int rl, int cl, int estimatedmNNzs)
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
	
	/*
	 * This can be only called when you know you have properly allocated spaces for a sparse representation
	 * and r and c are in the the range of the dimension
	 * Note: this function won't keep track of the nozeros
	 */
	
	public void setValueSparseUnsafe(int r, int c, double v) {
		sparseRows[r].set(c, v);		
	}
	
	/*
	 * This can be only called when you know you have properly allocated spaces for a sparse representation
	 * and r and c are in the the range of the dimension
	 * Note: this function won't keep track of the nozeros
	 * This can only be called, when the caller knows the access pattern of the block
	 */
	
	public void appendValueSparseUnsafe(int r, int c, double v) {
		sparseRows[r].append(c, v);		
	}
	
	/*
	 * This can be only called when you know you have properly allocated spaces for a dense representation
	 * and r and c are in the the range of the dimension
	 * Note: this function won't keep track of the nozeros
	 */
	
	public void setValueDenseUnsafe(int r, int c, double v) {
		denseBlock[r*clen+c]=v;		
	}
	
	
	public double getValueSparseUnsafe(int r, int c) {
		if(sparseRows==null || sparseRows.length<=r || sparseRows[r]==null)
			return 0;
		return sparseRows[r].get(c);
		
	}
	
	public double getValueDenseUnsafe(int r, int c) {
		if(denseBlock==null)
			return 0;
		return denseBlock[r*clen+c]; 
	}
	
	public static void testUnsafeSetNGet(int r, int c)
	{
		MatrixBlockDSM b1=new MatrixBlockDSM();
		b1.spaceAllocForDenseUnsafe(r, c);
		Random rand=new Random();
		int n=rand.nextInt(r*c);
		for(int i=0; i<n; i++)
			b1.setValueDenseUnsafe(rand.nextInt(r), rand.nextInt(c), rand.nextDouble());
		b1.recomputeNonZeros();
		System.out.println("~~~~~~~~~~~~\nb1\n");
		System.out.println(b1);
		
		MatrixBlockDSM b2=new MatrixBlockDSM();
		b2.spaceAllocForSparseUnsafe(r, c, SparseRow.initialCapacity*r);
		for(int i=0; i<r; i++)
			for(int j=0; j<c; j++)
				b2.setValueSparseUnsafe(i, j, b1.getValueDenseUnsafe(i, j));
		b2.recomputeNonZeros();
		System.out.println("~~~~~~~~~~~~\nb2\n");
		System.out.println(b2);
		
		//append is better than set, if you control the order of set
		MatrixBlockDSM b3=new MatrixBlockDSM();
		b3.spaceAllocForSparseUnsafe(r, c, SparseRow.initialCapacity*r);
		for(int i=0; i<r; i++)
			for(int j=0; j<c; j++)
				b3.appendValueSparseUnsafe(i, j, b2.getValueSparseUnsafe(i, j));
		b3.recomputeNonZeros();
		System.out.println("~~~~~~~~~~~~\nb3\n");
		System.out.println(b3);
		if(!equal(b1, b2) || !equal(b1, b3))
			System.err.println("the results are not equal!");
		
		
		
		if(b1.getNonZeros()!=b2.getNonZeros() || b2.getNonZeros()!=b3.getNonZeros())
			System.err.println("non zeros do not match!");
	}
	
	public static boolean equal(MatrixBlockDSM m1, MatrixBlockDSM m2)
	{
		boolean ret=true;
		for(int i=0; i<m1.getNumRows(); i++)
			for(int j=0; j<m1.getNumColumns(); j++)
				if(Math.abs(m1.quickGetValue(i, j)-m2.quickGetValue(i, j))>0.0000000001)
				{
					System.out.println(m1.quickGetValue(i, j)+" vs "+m2.quickGetValue(i, j)+":"+ (Math.abs(m1.quickGetValue(i, j)-m2.quickGetValue(i, j))));
					ret=false;
				}
		return ret;
	}
	
	public static boolean equal(MatrixBlock1D m1, MatrixBlockDSM m2)
	{
		boolean ret=true;
		for(int i=0; i<m1.getNumRows(); i++)
			for(int j=0; j<m1.getNumColumns(); j++)
				if(Math.abs(m1.getValue(i, j)-m2.quickGetValue(i, j))>0.0000000001)
				{
					System.out.println(m1.getValue(i, j)+" vs "+m2.getValue(i, j)+":"+ (Math.abs(m1.getValue(i, j)-m2.quickGetValue(i, j))));
					ret=false;
				}
		return ret;
	}
	static class Factory1D implements ObjectFactory
	{
		int rows, cols;
		double sparsity;
		public Factory1D(int rows, int cols, double sparsity) {
			this.rows=rows;
			this.cols=cols;
			this.sparsity=sparsity;
		}

		public Object makeObject() {
			
			return getRandomSparseMatrix1D(rows, cols, sparsity, 1);
		}
	}
	
	static class FactoryDSM implements ObjectFactory
	{
		int rows, cols;
		double sparsity;
		public FactoryDSM(int rows, int cols, double sparsity) {
			this.rows=rows;
			this.cols=cols;
			this.sparsity=sparsity;
		}

		public Object makeObject() {
			
			return getRandomSparseMatrix(rows, cols, sparsity, 1);
		}
		
	}
	
	public static void printResults(String info, long oldtime, long newtime)
	{
	//	System.out.println(info+((double)oldtime/(double)newtime));
		System.out.println(((double)oldtime/(double)newtime));
	}
	
	public static void onerun(int rows, int cols, double sparsity, int runs) throws Exception
	{
//		MemoryTestBench bench=new MemoryTestBench();
//		bench.showMemoryUsage(new Factory1D(rows, cols, sparsity));
//		bench.showMemoryUsage(new FactoryDSM(rows, cols, sparsity));
		System.out.println("-----------------------------------------");
//		System.out.println("rows: "+rows+", cols: "+cols+", sparsity: "+sparsity+", runs: "+runs);
		System.out.println(sparsity);
		MatrixBlock1D m_old=getRandomSparseMatrix1D(rows, cols, sparsity, 1);
		//m_old.examSparsity();
		MatrixBlock1D m_old2=getRandomSparseMatrix1D(rows, cols, sparsity, 2);
		//m_old2.examSparsity();
		MatrixBlock1D m_old3=new MatrixBlock1D(rows, cols, true);
		//System.out.println(m_old);
		MatrixBlockDSM m_new=getRandomSparseMatrix(rows, cols, sparsity, 1);
		//m_new.examSparsity();
		MatrixBlockDSM m_new2=getRandomSparseMatrix(rows, cols, sparsity, 2);
	//	m_new2.examSparsity();
		MatrixBlockDSM m_new3=new MatrixBlockDSM(rows, cols, true);
	//	System.out.println(m_new);
		long start, oldtime, newtime;
		//Operator op;
		
		UnaryOperator op=new UnaryOperator(Builtin.getBuiltinFnObject("round"));
/*		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_old.unaryOperationsInPlace(op);
		oldtime=System.nanoTime()-start;
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_new.unaryOperationsInPlace(op);
		newtime=System.nanoTime()-start;
		if(!equal(m_old, m_new))
			System.err.println("result doesn't match!");
		printResults("unary inplace op: ", oldtime, newtime);
	//	System.out.println("sparsity of m_mew: "+m_new.isInSparseFormat()+"\t sparsity of m_old: "+m_old.isInSparseFormat());
*/
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_old.unaryOperations(op, m_old3);
		oldtime=System.nanoTime()-start;
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_new.unaryOperations(op, m_new3);
		newtime=System.nanoTime()-start;
		if(!equal(m_old3, m_new3))
			System.err.println("result doesn't match!");
		//System.out.println("unary op: "+oldtime+", "+newtime+", "+((double)oldtime/(double)newtime));
		printResults("unary op: ", oldtime, newtime);
//		System.out.println("sparsity of m_mew: "+m_new.isInSparseFormat()+"\t sparsity of m_old: "+m_old.isInSparseFormat());
 	
		LeftScalarOperator op1=new LeftScalarOperator(Multiply.getMultiplyFnObject(), 2);
/*		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_old.scalarOperationsInPlace(op1);
		oldtime=System.nanoTime()-start;
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_new.scalarOperationsInPlace(op1);
		newtime=System.nanoTime()-start;
		if(!equal(m_old, m_new))
			System.err.println("result doesn't match!");
		printResults("scalar inplace op: ", oldtime, newtime);
//		System.out.println("sparsity of m_mew: "+m_new.isInSparseFormat()+"\t sparsity of m_old: "+m_old.isInSparseFormat());
*/
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_old.scalarOperations(op1, m_old3);
		oldtime=System.nanoTime()-start;
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_new.scalarOperations(op1, m_new3);
		newtime=System.nanoTime()-start;
		if(!equal(m_old3, m_new3))
			System.err.println("result doesn't match!");
	//	System.out.println("scalar op: "+oldtime+", "+newtime+", "+((double)oldtime/(double)newtime));
		printResults("scalar op: ", oldtime, newtime);
	//	System.out.println("sparsity of m_mew: "+m_new.isInSparseFormat()+"\t sparsity of m_old: "+m_old.isInSparseFormat());
		
		BinaryOperator op11=new BinaryOperator(Plus.getPlusFnObject());
		
/*		start=System.nanoTime();
		for(int i=0; i<runs; i++)
		{	
			long begin=System.nanoTime();
			m_old.binaryOperationsInPlace(op11, m_old2);
			System.out.println("sparsity of m_mew: "+m_new.isInSparseFormat()+"\t sparsity of m_old: "+m_old.isInSparseFormat());
		//	System.out.println(System.nanoTime()-begin);
		}
		oldtime=System.nanoTime()-start;
		start=System.nanoTime();
	//	System.out.println("~~~");
		for(int i=0; i<runs; i++)
		{
			long begin=System.nanoTime();
			m_new.binaryOperationsInPlace(op11, m_new2);
			System.out.println("sparsity of m_mew: "+m_new.isInSparseFormat()+"\t sparsity of m_old: "+m_old.isInSparseFormat());
		//	System.out.println(System.nanoTime()-begin);
		}
		newtime=System.nanoTime()-start;
		if(!equal(m_old, m_new))
			System.err.println("result doesn't match!");
		//System.out.println("binary op: "+oldtime+", "+newtime+", "+((double)oldtime/(double)newtime));
		printResults("binary op inplace: ", oldtime, newtime);
		System.out.println("sparsity of m_mew: "+m_new.isInSparseFormat()+"\t sparsity of m_old: "+m_old.isInSparseFormat());
*/		
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
		{
		//	long begin=System.nanoTime();
			m_old.binaryOperations(op11, m_old2, m_old3);
	//		System.out.println(System.nanoTime()-begin);
		}
		oldtime=System.nanoTime()-start;
	//	System.out.println("~~~");
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
		{
		//	long begin=System.nanoTime();
			m_new.binaryOperations(op11, m_new2, m_new3);
	//		System.out.println(System.nanoTime()-begin);
		}
		newtime=System.nanoTime()-start;
		if(!equal(m_old3, m_new3))
			System.err.println("result doesn't match!");
		//System.out.println("binary op: "+oldtime+", "+newtime+", "+((double)oldtime/(double)newtime));
		printResults("binary op: ", oldtime, newtime);
//		System.out.println("sparsity of m_mew: "+m_new.isInSparseFormat()+"\t sparsity of m_old: "+m_old.isInSparseFormat());

		ReorgOperator op12=new ReorgOperator(SwapIndex.getSwapIndexFnObject());
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_old.reorgOperations(op12, m_old3, 0, 0, m_old.getNumRows());
		oldtime=System.nanoTime()-start;
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_new.reorgOperations(op12, m_new3, 0, 0, m_old.getNumRows());
		newtime=System.nanoTime()-start;
		if(!equal(m_old3, m_new3))
			System.err.println("result doesn't match!");
		//System.out.println("unary op: "+oldtime+", "+newtime+", "+((double)oldtime/(double)newtime));
		printResults("reorg op: ", oldtime, newtime);
//		System.out.println("sparsity of m_mew: "+m_new.isInSparseFormat()+"\t sparsity of m_old: "+m_old.isInSparseFormat());
	
/*		AggregateBinaryOperator op13=new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), new AggregateOperator(0, Plus.getPlusFnObject()));
		
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_old.aggregateBinaryOperations(m_old, m_old2, m_old3, op13);
		oldtime=System.nanoTime()-start;
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_new.aggregateBinaryOperations(m_new, m_new2, m_new3, op13);
		newtime=System.nanoTime()-start;
		if(!equal(m_old3, m_new3))
			System.err.println("result doesn't match!");
		//System.out.println("binary op: "+oldtime+", "+newtime+", "+((double)oldtime/(double)newtime));
		printResults("aggregate binary op: ", oldtime, newtime);
//		System.out.println("sparsity of m_mew: "+m_new.isInSparseFormat()+"\t sparsity of m_old: "+m_old.isInSparseFormat());
*/		
/*		AggregateUnaryOperator op14=new AggregateUnaryOperator(new AggregateOperator(0, Plus.getPlusFnObject()), ReduceAll.getReduceAllFnObject());
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_old.aggregateUnaryOperations(op14, m_old3, m_old.getNumRows(), m_old.getNumColumns(), new MatrixIndexes(1, 1));
		oldtime=System.nanoTime()-start;
		start=System.nanoTime();
		for(int i=0; i<runs; i++)
			m_new.aggregateUnaryOperations(op14, m_new3, m_old.getNumRows(), m_old.getNumColumns(), new MatrixIndexes(1, 1));
		newtime=System.nanoTime()-start;
		if(!equal(m_old3, m_new3))
			System.err.println("result doesn't match!");
	//	System.out.println("scalar op: "+oldtime+", "+newtime+", "+((double)oldtime/(double)newtime));
		printResults("aggregate unary op: ", oldtime, newtime);
	//	System.out.println("sparsity of m_mew: "+m_new.isInSparseFormat()+"\t sparsity of m_old: "+m_old.isInSparseFormat());
*/

	}
	
	public static void testSelection(int rows, int cols, double sparsity) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlockDSM m=getRandomSparseMatrix(rows, cols, sparsity, 1);
		m.examSparsity();
		System.out.println(m);
		//MatrixBlockDSM result=new MatrixBlockDSM(rows, cols, true);
		//m.selectOperations(result, new IndexRange(3, rows-3, 3, cols-3));
		
		ArrayList<IndexedMatrixValue> results=new ArrayList<IndexedMatrixValue>();
		for(int i=0; i<4; i++)
			results.add(new IndexedMatrixValue(MatrixBlockDSM.class));
		m.slideOperations(results, new IndexRange(3, rows-3, 3, cols-3), 5, 5, 10, 10, 6, 6);
		for(IndexedMatrixValue r: results)
			System.out.println("----------------\n\n"+r);
	}
	
	public static void testGetSparseCellInterator()
	{
		int rows=50;
		int cols=10;
		double sparsity=0.05;
		MatrixBlockDSM m=getRandomSparseMatrix(rows, cols, sparsity, 1);
		System.out.println(m);
		SparseCellIterator iter=m.getSparseCellIterator();
		while(iter.hasNext())
			System.out.println(iter.next());
	}
	
	public static void  main(String[] args) throws Exception
	{
		long rows=7525500;
		long cols=rows;
		double sp = (double)1/rows;
		System.out.println("rows " + rows + ", cols " + cols + ", sp " + sp + ": " + (estimateSize(rows,cols,sp)/1024/1024));
		
		/*MatrixBlockDSM m=getRandomSparseMatrix(10, 10, 0.5, 1);
		System.out.println("~~~~~~~~~~~~");
		System.out.println(m);
		MatrixBlockDSM m2=new MatrixBlock();
		m2.copy(m);
		System.out.println("~~~~~~~~~~~~");
		System.out.println(m2);
		m.reset(20, 20, true, 200);
		m.setValue(1, 19, 119);
		m.setValue(19, 19, 1919);
		
		System.out.println("~~~~~~~~~~~~");
		System.out.println(m);*/
		
		//testGetSparseCellInterator();
		//testUnsafeSetNGet(10, 10);
		
	//	int rows=10, cols=10, runs=10;
		/*double[] sparsities=new double[]{0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1};
		for(double sparsity: sparsities)
			onerun(rows, cols, sparsity, runs);
			*/
		//testSelection(10, 10, 1);
		
	/*	double sparsity=0.5;
	//	MatrixBlockDSM m=getRandomSparseMatrix(rows, cols, sparsity, 1);
		MatrixBlockDSM m=new MatrixBlockDSM(rows, cols, false);
		//m.examSparsity();
		System.out.println("~~~~~~~~~~~~");
		System.out.println(m);
		m.dropLastRowsOrColums(CorrectionLocationType.LASTROW);
		System.out.println("~~~~~~~~~~~~");
		System.out.println(m);
		m.dropLastRowsOrColums(CorrectionLocationType.LASTCOLUMN);
		System.out.println("~~~~~~~~~~~~");
		System.out.println(m);
		m.dropLastRowsOrColums(CorrectionLocationType.LASTTWOROWS);
		System.out.println("~~~~~~~~~~~~");
		System.out.println(m);
		m.dropLastRowsOrColums(CorrectionLocationType.LASTTWOCOLUMNS);
		System.out.println("~~~~~~~~~~~~");
		System.out.println(m);
*/
		
		/*
		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		
		MatrixBlockDSM m1=getRandomDenseMatrix(2, 4, 1);
		System.out.println(m1.toString());
		
		MatrixBlockDSM m2=getRandomSparseMatrix(4,3,0.3,1);
		System.out.println(m2.toString());
		
		MatrixBlockDSM result = (MatrixBlockDSM) m1.aggregateBinaryOperations(m1, m2, new MatrixBlock(), aggbin);
		
		System.out.println(result.toString());
		
		// -------------------------------------------------
		//DMLScript.READ_AS_SPARSE = Boolean.parseBoolean(args[0]);
		String Vfile = args[0];
		String Wfile = args[1];
		int numD = Integer.parseInt(args[2]);
		int numW = Integer.parseInt(args[3]);
		int numT = Integer.parseInt(args[4]);
		
		MatrixBlockDSM W = DataConverter.readMatrixFromHDFS(Wfile, InputInfo.BinaryBlockInputInfo, numD, numT, 1000, 1000);
		MatrixBlockDSM V = DataConverter.readMatrixFromHDFS(Vfile, InputInfo.BinaryBlockInputInfo, numD, numW, 1000, 1000);
		
		long st = System.currentTimeMillis();
		MatrixBlockDSM tW = new MatrixBlockDSM(numT, numD, W.sparse);
		W.reorgOperations(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), tW, 0, 0, 0);
		long txTime = System.currentTimeMillis() - st;
		
		MatrixBlockDSM tWV = new MatrixBlockDSM(numT, numW, false);
		st = System.currentTimeMillis();
		tW.aggregateBinaryOperations(tW, V, tWV, aggbin);
		long mmultTime = System.currentTimeMillis()-st;
		System.out.println("    Transpose " + txTime + ", MMult " + mmultTime);
		
		//m.examSparsity();
		W = null;
		V = null;
		tW = null;
		tWV = null;
		*/
	}

	public void addDummyZeroValue() {
		/*		if ( sparse ) {
					if ( sparseBlock == null )
						sparseBlock = new HashMap<CellIndex,Double>(); 
					sparseBlock.put(new CellIndex(0, 0), 0.0);
				}
				else {
					try {
						throw new DMLRuntimeException("Unexpected.. ");
					} catch (DMLRuntimeException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
		*/	}

}
