package dml.runtime.matrix.io;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.Map.Entry;

import dml.runtime.functionobjects.Builtin;
import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.Plus;
import dml.runtime.functionobjects.SwapIndex;
import dml.runtime.instructions.CPInstructions.KahanObject;
import dml.runtime.instructions.MRInstructions.SelectInstruction.IndexRange;
import dml.runtime.matrix.operators.AggregateBinaryOperator;
import dml.runtime.matrix.operators.AggregateOperator;
import dml.runtime.matrix.operators.AggregateUnaryOperator;
import dml.runtime.matrix.operators.BinaryOperator;
import dml.runtime.matrix.operators.LeftScalarOperator;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.ReorgOperator;
import dml.runtime.matrix.operators.ScalarOperator;
import dml.runtime.matrix.operators.UnaryOperator;
import dml.runtime.test.ObjectFactory;
import dml.runtime.util.UtilFunctions;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class MatrixBlockDSM extends MatrixValue{

	//protected static final Log LOG = LogFactory.getLog(MatrixBlock1D.class);
	private int rlen;
	private int clen;
	private int maxrow, maxcolumn;
	private boolean sparse;
	private double[] denseBlock=null;
	private int nonZeros=0;
	public static final double SPARCITY_TURN_POINT=0.4;
	
	private SparseRow[] sparseRows=null;
	
	
	public static boolean checkSparcityOnAggBinary(MatrixBlockDSM m1, MatrixBlockDSM m2)
	{
		double n=m1.getNumRows();
		double k=m1.getNumColumns();
		double m=m2.getNumColumns();
		double nz1=m1.getNonZeros();
		double nz2=m2.getNonZeros();
		double pq=nz1*nz2/n/k/k/m;
	//	double estimated= 1-Math.pow(1-pq, k);
		return ( 1-Math.pow(1-pq, k) < SPARCITY_TURN_POINT );
	}
	
	private static boolean checkSparcityOnBinary(MatrixBlockDSM m1, MatrixBlockDSM m2)
	{
		double n=m1.getNumRows();
		double m=m1.getNumColumns();
		double nz1=m1.getNonZeros();
		double nz2=m2.getNonZeros();
		//1-(1-p)*(1-q)
	//	double estimated=1- (1-nz1/n/m)*(1-nz2/n/m);
		return ( 1- (1-nz1/n/m)*(1-nz2/n/m) < SPARCITY_TURN_POINT);
		
	}
	
	private static boolean checkRealSparcity(MatrixBlockDSM m)
	{
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
	
	public MatrixBlockDSM(MatrixBlockDSM that)
	{
		this.copy(that);
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
				System.out.print(getValue(i, j)+"\t");
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
			for(int i=0; i<sparseRows.length; i++)
				if(sparseRows[i]!=null)
					sparseRows[i].reset();
		}
	}
	public void reset()
	{
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
	
	public void reset(int rl, int cl, boolean sp)
	{
		sparse=sp;
		reset(rl, cl);
	}
	
	public void resetDenseWithValue(int rl, int cl, double v) {
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
	
	public void examSparsity()
	{
		if(sparse)
		{
			if(nonZeros>rlen*clen*SPARCITY_TURN_POINT)
				sparseToDense();
		}else
		{
			if(nonZeros<rlen*clen*SPARCITY_TURN_POINT)
				denseToSparse();
		}
	}
	
	private void copySparseToSparse(MatrixBlockDSM that)
	{
		this.nonZeros=that.nonZeros;
		if(that.sparseRows==null)
		{
			resetSparse();
			return;
		}
	
		adjustSparseRows(that.sparseRows.length);
		for(int i=0; i<that.sparseRows.length; i++)
		{
			if(that.sparseRows[i]!=null)
			{
				if(sparseRows[i]==null)
					sparseRows[i]=new SparseRow(that.sparseRows[i]);
				else
					sparseRows[i].copy(that.sparseRows[i]);
			}else if(this.sparseRows[i]!=null)
				this.sparseRows[i].reset();
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
		for(int r=0; r<that.sparseRows.length; r++, start+=clen)
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
				sparseRows[r]=new SparseRow();
			else
				sparseRows[r].reset();
			
			for(int c=0; c<clen; c++)
			{
				if(that.denseBlock[n]!=0)
					sparseRows[r].append(c, that.denseBlock[n]);
				n++;
			}
		}
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
	
	//TODO: this function is used in many places, but may not be the right api to expose sparse cells.
	public HashMap<CellIndex, Double> getSparseMap()
	{
		if(!sparse || sparseRows==null)
			return null;
		HashMap<CellIndex, Double> map=new HashMap<CellIndex, Double>(nonZeros);
		for(int r=0; r<sparseRows.length; r++)
		{
			if(sparseRows[r]==null) continue;
			double[] values=sparseRows[r].getValueContainer();
			int[] cols=sparseRows[r].getIndexContainer();
			for(int i=0; i<sparseRows[r].size(); i++)
				map.put(new CellIndex(r, cols[i]), values[i]);
		}
		return map;
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
			for(int r=0; r<sparseRows.length; r++)
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
			for(int r=0; r<sparseRows.length; r++)
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
				v=op.executeScalar(getValue(r, c));
				setValue(r, c, v);
			}	
	}
	
	public void denseUnaryOperationsInPlace(UnaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		double v;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				v=op.fn.execute(getValue(r, c));
				setValue(r, c, v);
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
		if(result==null)
			result=new MatrixBlockDSM(rlen, clen, sparse);
		result.copy(this);
		
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
		if(result==null)
			result=new MatrixBlockDSM(rlen, clen, sparse);
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
		boolean resultSparse=checkSparcityOnBinary(this, that);
		if(result==null)
			result=new MatrixBlockDSM(rlen, clen, resultSparse);
		else
			result.reset(rlen, clen, resultSparse);
		
		//double st = System.nanoTime();
		double v;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				v=op.fn.execute(this.getValue(r, c), that.getValue(r, c));
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
		boolean resultSparse=checkSparcityOnBinary(this, that);
		if(result==null)
			result=new MatrixBlockDSM(rlen, clen, resultSparse);
		else
			result.reset(rlen, clen, resultSparse);
		
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
							result.sparseRows[r]=new SparseRow(estimateSize);
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
							result.sparseRows[r]=new SparseRow(that.sparseRows[r].size());
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
							result.sparseRows[r]=new SparseRow(this.sparseRows[r].size());
						else if(result.sparseRows[r].capacity()<that.sparseRows[r].size())
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
					thisvalue=this.getValue(r, c);
					thatvalue=that.getValue(r, c);
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
		
		if(aggOp.correctionLocation==1)
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					buffer._sum=this.getValue(r, c);
					buffer._correction=cor.getValue(0, c);
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.getValue(r, c), 
							newWithCor.getValue(r+1, c));
					setValue(r, c, buffer._sum);
					cor.setValue(0, c, buffer._correction);
				}
			
		}else if(aggOp.correctionLocation==2)
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					buffer._sum=this.getValue(r, c);
					buffer._correction=cor.getValue(r, 0);;
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.getValue(r, c), 
							newWithCor.getValue(r, c+1));
					setValue(r, c, buffer._sum);
					cor.setValue(r, 0, buffer._correction);
				}
		}else if(aggOp.correctionLocation==0)
		{
			
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					buffer._sum=this.getValue(r, c);
					buffer._correction=cor.getValue(r, c);
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.getValue(r, c));
					setValue(r, c, buffer._sum);
					cor.setValue(r, c, buffer._correction);
				}
		}else if(aggOp.correctionLocation==3)
		{
			double n, n2, mu2;
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					buffer._sum=this.getValue(r, c);
					n=cor.getValue(0, c);
					buffer._correction=cor.getValue(1, c);
					mu2=newWithCor.getValue(r, c);
					n2=newWithCor.getValue(r+1, c);
					n=n+n2;
					double toadd=(mu2-buffer._sum)*n2/n;
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
					setValue(r, c, buffer._sum);
					cor.setValue(0, c, n);
					cor.setValue(1, c, buffer._correction);
				}
			
		}else if(aggOp.correctionLocation==4)
		{
			double n, n2, mu2;
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					buffer._sum=this.getValue(r, c);
					n=cor.getValue(r, 0);
					buffer._correction=cor.getValue(r, 1);
					mu2=newWithCor.getValue(r, c);
					n2=newWithCor.getValue(r, c+1);
					n=n+n2;
					double toadd=(mu2-buffer._sum)*n2/n;
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
					setValue(r, c, buffer._sum);
					cor.setValue(r, 0, n);
					cor.setValue(r, 1, buffer._correction);
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
		
		if(aggOp.correctionLocation==1)
		{
			for(int r=0; r<rlen-1; r++)
				for(int c=0; c<clen; c++)
				{
					buffer._sum=this.getValue(r, c);
					buffer._correction=this.getValue(r+1, c);
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.getValue(r, c), 
							newWithCor.getValue(r+1, c));
					setValue(r, c, buffer._sum);
					setValue(r+1, c, buffer._correction);
				}
			
		}else if(aggOp.correctionLocation==2)
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen-1; c++)
				{
					buffer._sum=this.getValue(r, c);
					buffer._correction=this.getValue(r, c+1);
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.getValue(r, c), 
							newWithCor.getValue(r, c+1));
					setValue(r, c, buffer._sum);
					setValue(r, c+1, buffer._correction);
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
		}*/else if(aggOp.correctionLocation==3)
		{
			double n, n2, mu2;
			for(int r=0; r<rlen-2; r++)
				for(int c=0; c<clen; c++)
				{
					buffer._sum=this.getValue(r, c);
					n=this.getValue(r+1, c);
					buffer._correction=this.getValue(r+2, c);
					mu2=newWithCor.getValue(r, c);
					n2=newWithCor.getValue(r+1, c);
					n=n+n2;
					double toadd=(mu2-buffer._sum)*n2/n;
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
					setValue(r, c, buffer._sum);
					setValue(r+1, c, n);
					setValue(r+2, c, buffer._correction);
				}
			
		}else if(aggOp.correctionLocation==4)
		{
			double n, n2, mu2;
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen-2; c++)
				{
					buffer._sum=this.getValue(r, c);
					n=this.getValue(r, c+1);
					buffer._correction=this.getValue(r, c+2);
					mu2=newWithCor.getValue(r, c);
					n2=newWithCor.getValue(r, c+1);
					n=n+n2;
					double toadd=(mu2-buffer._sum)*n2/n;
					buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
					setValue(r, c, buffer._sum);
					setValue(r, c+1, n);
					setValue(r, c+2, buffer._correction);
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
			for(int i=0; i<oldSparseRows.length; i++)
				sparseRows[i]=oldSparseRows[i];
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
				sparseRows[r]=new SparseRow();
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
	
	@Override
	public void setValue(int r, int c, double v) {
		if(r>rlen || c > clen)
			throw new RuntimeException("indexes ("+r+","+c+") out of range ("+rlen+","+clen+")");
		if(sparse)
		{
			adjustSparseRows(r);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow();
			
			if(sparseRows[r].set(c, v))
				nonZeros++;
			
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
			setValue(r, c, v);
		else
		{
			adjustSparseRows(r);
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow();
			sparseRows[r].append(c, v);
			nonZeros++;
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
			Double d=sparseRows[r].get(c);
			if(d!=null)
				return d;
			else
				return 0;
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
				for(int r=0; r<sparseRows.length; r++)
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
				for(int r=0; r<sparseRows.length; r++)
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
					sparseRows[r].reset();
				continue;
			}
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow(nr);
			else
				sparseRows[r].reset();
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
			else if(nonZeros>rlen*clen*SPARCITY_TURN_POINT)
				writeSparseToDense(out);
			else
				writeSparseBlock(out);
		}else
		{
			if(denseBlock==null)
				writeEmptyBlock(out);
			//if it should be sparse
			else if(nonZeros<rlen*clen*SPARCITY_TURN_POINT)
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
		for(;r<sparseRows.length; r++)
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
				out.writeDouble(getValue(i, j));
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
	public MatrixValue reorgOperations(ReorgOperator op, MatrixValue result,
			int startRow, int startColumn, int length)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		checkType(result);
		boolean reducedDim=op.fn.computeDimension(rlen, clen, tempCellIndex);
		boolean sps;
		if(reducedDim)
			sps=false;
		else
			sps=checkRealSparcity(this);
			
		if(result==null)
			result=new MatrixBlockDSM(tempCellIndex.row, tempCellIndex.column, sps);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, sps);
		
		CellIndex temp = new CellIndex(0, 0);
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r<sparseRows.length; r++)
				{
					if(sparseRows[r]==null) continue;
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					for(int i=0; i<sparseRows[r].size(); i++)
					{
						tempCellIndex.set(r, cols[i]);
						op.fn.execute(tempCellIndex, temp);
						result.setValue(temp.row, temp.column, values[i]);
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
					result.setValue(temp.row, temp.column, denseBlock[i]);
				}
			}
		}
		
		return result;
	}

	@Override
	public MatrixValue selectOperations(MatrixValue result, IndexRange range)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		checkType(result);
		boolean sps;
		if((double)nonZeros/(double)rlen/(double)clen*(double)(range.rowEnd-range.rowStart+1)*(double)(range.colEnd-range.colStart+1)
				/(double)rlen/(double)clen< SPARCITY_TURN_POINT)
			sps=true;
		else sps=false;
			
		if(result==null)
			result=new MatrixBlockDSM(tempCellIndex.row, tempCellIndex.column, sps);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, sps);
		
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=(int)range.rowStart; r<=range.rowEnd; r++)
				{
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					int start=sparseRows[r].searchIndexesFirstGTE((int)range.colStart);
					int end=sparseRows[r].searchIndexesFirstLTE((int)range.colEnd);
					for(int i=start; i<=end; i++)	
						result.setValue(r, cols[i], values[i]);
				}
			}
		}else
		{
			if(denseBlock!=null)
			{
				int i=0;
				for(int r=(int) range.rowStart; r<=range.rowEnd; r++)
				{
					for(int c=(int) range.colStart; c<=range.colEnd; c++)
						result.setValue(r, c, denseBlock[i+c]);
					i+=clen;
				}
			}
		}
		
		return result;
	}

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
						getValue(UtilFunctions.cellInBlockCalculation(i, blockingFactorRow), UtilFunctions.cellInBlockCalculation(i, blockingFactorCol)));
			}
			result.setValue(0, 0, buffer._sum);
			if(op.aggOp.correctionLocation==1)//extra row
				result.setValue(1, 0, buffer._correction);
			else if(op.aggOp.correctionLocation==2)
				result.setValue(0, 1, buffer._correction);
			else
				throw new DMLRuntimeException("unrecognized correctionLocation: "+op.aggOp.correctionLocation);
		}else
		{
			double newv=0;
			for(long i=start; i<=end; i++)
			{
				newv+=op.aggOp.increOp.fn.execute(newv,
						getValue(UtilFunctions.cellInBlockCalculation(i, blockingFactorRow), UtilFunctions.cellInBlockCalculation(i, blockingFactorCol)));
			}
			result.setValue(0, 0, newv);
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
			result.setValue(cellRow, 0, getValue(cellRow, cellCol));
		}
	}
	
	private void incrementalAggregateUnaryHelp(AggregateOperator aggOp, MatrixBlockDSM result, int row, int column, 
			double newvalue, KahanObject buffer) throws DMLRuntimeException
	{
		if(aggOp.correctionExists)
		{
			if(aggOp.correctionLocation==1 || aggOp.correctionLocation==2)
			{
				int corRow=row, corCol=column;
				if(aggOp.correctionLocation==1)//extra row
					corRow++;
				else if(aggOp.correctionLocation==2)
					corCol++;
				else
					throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
				
				buffer._sum=result.getValue(row, column);
				buffer._correction=result.getValue(corRow, corCol);
				buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newvalue);
				result.setValue(row, column, buffer._sum);
				result.setValue(corRow, corCol, buffer._correction);
			}else if(aggOp.correctionLocation==0)
			{
				throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
			}else// for mean
			{
				int corRow=row, corCol=column;
				int countRow=row, countCol=column;
				if(aggOp.correctionLocation==3)//extra row
				{
					countRow++;
					corRow+=2;
				}
				else if(aggOp.correctionLocation==4)
				{
					countCol++;
					corCol+=2;
				}
				else
					throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
				buffer._sum=result.getValue(row, column);
				buffer._correction=result.getValue(corRow, corCol);
				double count=result.getValue(countRow, countCol)+1.0;
				double toadd=(newvalue-buffer._sum)/count;
				buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
				result.setValue(row, column, buffer._sum);
				result.setValue(corRow, corCol, buffer._correction);
				result.setValue(countRow, countCol, count);
			}
			
		}else
		{
			newvalue=aggOp.increOp.fn.execute(result.getValue(row, column), newvalue);
			result.setValue(row, column, newvalue);
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
				for(r=0; r<sparseRows.length; r++)
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
				incrementalAggregateUnaryHelp(op.aggOp, result, result.tempCellIndex.row, result.tempCellIndex.column, getValue(i,j), buffer);
			}
	}
	
	public MatrixValue aggregateUnaryOperations(AggregateUnaryOperator op, MatrixValue result, 
			int blockingFactorRow, int blockingFactorCol, MatrixIndexes indexesIn) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		op.indexFn.computeDimension(rlen, clen, tempCellIndex);
		if(op.aggOp.correctionExists)
		{
			switch(op.aggOp.correctionLocation)
			{
			case 1: tempCellIndex.row++; break;
			case 2: tempCellIndex.column++; break;
			case 3: tempCellIndex.row+=2; break;
			case 4: tempCellIndex.column+=2; break;
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
		
		return result;
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
			throw new RuntimeException("dimenstions do not match for matrix multiplication");
		int rl=m1.rlen;
		int cl=m2.clen;
		boolean sp=checkSparcityOnAggBinary(m1, m2);
		if(result==null)
			result=new MatrixBlockDSM(rl, cl, sp);//m1.sparse&&m2.sparse);
		else
			result.reset(rl, cl, sp);//m1.sparse&&m2.sparse);
		
		if(op.sparseSafe)
			sparseAggregateBinaryHelp(m1, m2, (MatrixBlockDSM)result, op);
		else
			aggBinSparseUnsafe(m1, m2, (MatrixBlockDSM)result, op);
		return result;
	}
	
	
	/*
	 * to perform aggregateBinary when the first matrix is dense and the second is sparse
	 */
	private static void aggBinDenseSparse(MatrixBlockDSM m1, MatrixBlockDSM m2,
			MatrixBlockDSM result, AggregateBinaryOperator op) throws DMLRuntimeException 
	{
		if(m2.sparseRows==null)
			return;
		
		for(int k=0; k<m2.sparseRows.length; k++)
		{
			if(m2.sparseRows[k]==null) continue;
			int[] cols=m2.sparseRows[k].getIndexContainer();
			double[] values=m2.sparseRows[k].getValueContainer();
			for(int p=0; p<m2.sparseRows[k].size(); p++)
			{
				int j=cols[p];
				for(int i=0; i<m1.rlen; i++)
				{
					double old=result.getValue(i, j);
					double aik=m1.getValue(i, k);
					double addValue=op.binaryFn.execute(aik, values[p]);
					double newvalue=op.aggOp.increOp.fn.execute(old, addValue);
					result.setValue(i, j, newvalue);
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
		
		for(int i=0; i<m1.sparseRows.length; i++)
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
					double addValue=op.binaryFn.execute(values[p], m2.getValue(k, j));
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
		for(int i=0; i<m1.sparseRows.length; i++)
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
						double old=result.getValue(i, j);
						double newvalue=op.aggOp.increOp.fn.execute(old, addValue);
						result.setValue(i, j, newvalue);
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
	 * @author schnetter
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
					double aik=m1.getValue(i, k);
					double bkj=m2.getValue(k, j);
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
						v = op.aggOp.increOp.fn.execute(result.getValue(i, j), op.binaryFn.execute(temp, b[bIndex]));
						result.setValue(i, j, v);
						cIndex++;
						bIndex++;
					}
					
					aIndex += m1.clen;
				}
			}
		}
	}
	
	@Override
	/*
	 *  D = ctable(A,v2,W)
	 *  this <- A; scalarThat <- v2; that2 <- W; result <- D
	 */
	public void tertiaryOperations(Operator op, double scalarThat,
			MatrixValue that2, HashMap<CellIndex, Double> ctableResult)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		/*
		 * (i1,j1,v1) from input1 (this)
		 * (v2) from sclar_input2 (scalarThat)
		 * (i3,j3,w)  from input3 (that2)
		 */
		
		double v1;
		double v2 = scalarThat;
		double w;
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r<sparseRows.length; r++)
				{
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					for(int i=0; i<sparseRows[r].size(); i++)
					{
						// output (v1,v2,w)
						v1 = values[i];
						w = that2.getValue(r, cols[i]);
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
					v1 = this.getValue(r, c);
					w = that2.getValue(r, c);
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
				for(int r=0; r<sparseRows.length; r++)
				{
					int[] cols=sparseRows[r].getIndexContainer();
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
					v1 = this.getValue(r, c);
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
	public void tertiaryOperations(Operator op, MatrixValue that,
			double scalarThat2, HashMap<CellIndex, Double> ctableResult)
			throws DMLUnsupportedOperationException, DMLRuntimeException {

		/*
		 * (i1,j1,v1) from input1 (this)
		 * (i1,j1,v2) from input2 (that)
		 * (w)  from scalar_input3 (scalarThat2)
		 */
		
		double v1, v2;
		double w = scalarThat2;
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r<sparseRows.length; r++)
				{
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					for(int i=0; i<sparseRows[r].size(); i++)
					{
						// output (v1,v2,w)
						v1 = values[i];
						v2 = that.getValue(r, cols[i]);
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
					v1 = this.getValue(r, c);
					v2 = that.getValue(r, c);
					updateCtable(v1, v2, w, ctableResult);
				}
			}
			
		}
	}
	
	/*
	 *  D = ctable(A,B,W)
	 *  this <- A; that <- B; that2 <- W; result <- D
	 */
	public void tertiaryOperations(Operator op, MatrixValue that, MatrixValue that2, HashMap<CellIndex, Double> ctableResult)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{	
		/*
		 * (i1,j1,v1) from input1 (this)
		 * (i1,j1,v2) from input2 (that)
		 * (i1,j1,w)  from input3 (that2)
		 */
		
		double v1, v2, w;
		if(sparse)
		{
			if(sparseRows!=null)
			{
				for(int r=0; r<sparseRows.length; r++)
				{
					int[] cols=sparseRows[r].getIndexContainer();
					double[] values=sparseRows[r].getValueContainer();
					for(int i=0; i<sparseRows[r].size(); i++)
					{
						// output (v1,v2,w)
						v1 = values[i];
						v2 = that.getValue(r, cols[i]);
						w = that2.getValue(r, cols[i]);
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
					v1 = this.getValue(r, c);
					v2 = that.getValue(r, c);
					w = that2.getValue(r, c);
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
		if(denseBlock==null)
			return;
		int index=0;
		for(int r=0; r<rlen; r++)
		{
			if(sparseRows[r]==null)
				sparseRows[r]=new SparseRow();
			else
				sparseRows[r].reset();
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
	public void sparseToDense() {
		
		//LOG.info("**** sparseToDense: "+this.getNumRows()+"x"+this.getNumColumns()+"  nonZeros: "+this.nonZeros);
		sparse=false;
		int limit=rlen*clen;
		if(denseBlock==null || denseBlock.length < limit )
			denseBlock=new double[limit];
		Arrays.fill(denseBlock, 0, limit, 0);
		nonZeros=0;
		
		if(sparseRows==null)
			return;
		
		for(int r=0; r<sparseRows.length; r++)
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
		}
	}
	
	private void denseBinaryInPlaceHelp(BinaryOperator op, MatrixBlockDSM that) throws DMLRuntimeException 
	{
		boolean resultSparse=checkSparcityOnBinary(this, that);
		if(resultSparse && !this.sparse)
			denseToSparse();
		else if(!resultSparse && this.sparse)
			sparseToDense();
		
		double v;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				v=op.fn.execute(this.getValue(r, c), that.getValue(r, c));
				setValue(r, c, v);
			}
	}
	
	private void sparseBinaryInPlaceHelp(BinaryOperator op, MatrixBlockDSM that) throws DMLRuntimeException 
	{
		boolean resultSparse=checkSparcityOnBinary(this, that);
		if(resultSparse && !this.sparse)
			denseToSparse();
		else if(!resultSparse && this.sparse)
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
						this.sparseRows[r]=new SparseRow(estimateSize);
						
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
					
					this.sparseRows[r]=new SparseRow(that.sparseRows[r].size());
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
					thisvalue=this.getValue(r, c);
					thatvalue=that.getValue(r, c);
					resultvalue=op.fn.execute(thisvalue, thatvalue);
					this.setValue(r, c, resultvalue);
				}	
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////
	public static MatrixBlockDSM getRandomSparseMatrix(int rows, int cols, double sparsity, long seed)
	{
		Random random=new Random(seed);
		MatrixBlockDSM m=new MatrixBlockDSM(rows, cols, true);
		m.sparseRows=new SparseRow[rows];
		for(int i=0; i<rows; i++)
		{
			m.sparseRows[i]=new SparseRow();	
			for(int j=0; j<cols; j++)
			{
				if(random.nextDouble()>sparsity)
					continue;
				m.sparseRows[i].append(j, random.nextDouble());
				m.nonZeros++;
			}
		}
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
		if(!sparse)
			ret+="nonzeros = "+nonZeros+"\n";
		if(sparse)
		{
			int len=0;
			if(sparseRows!=null)
				len=sparseRows.length;
			int i=0;
			for(; i<len; i++)
			{
				ret+="row +"+i+": "+sparseRows[i]+"\n";
			}
			for(; i<rlen; i++)
			{
				ret+="row +"+i+": null\n";
			}
		}else
		{
			int start=0;
			for(int i=0; i<rlen; i++)
			{
				for(int j=0; j<clen; j++)
				{
					ret+=this.denseBlock[start+j]+"\t";
				}
				ret+="\n";
				start+=clen;
			}
		}
		
		return ret;
	}
	
	public static boolean equal(MatrixBlock1D m1, MatrixBlockDSM m2)
	{
		boolean ret=true;
		for(int i=0; i<m1.getNumRows(); i++)
			for(int j=0; j<m1.getNumColumns(); j++)
				if(Math.abs(m1.getValue(i, j)-m2.getValue(i, j))>0.0000000001)
				{
					System.out.println(m1.getValue(i, j)+" vs "+m2.getValue(i, j)+":"+ (Math.abs(m1.getValue(i, j)-m2.getValue(i, j))));
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
	
	public static void  main(String[] args) throws Exception
	{
		int rows=1000, cols=1000, runs=10;
		double[] sparsities=new double[]{0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1};
		for(double sparsity: sparsities)
			onerun(rows, cols, sparsity, runs);
	}

}
