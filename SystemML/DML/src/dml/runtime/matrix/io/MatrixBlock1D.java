package dml.runtime.matrix.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Vector;
import java.util.Map.Entry;

import dml.runtime.matrix.operators.AggregateBinaryOperator;
import dml.runtime.matrix.operators.AggregateOperator;
import dml.runtime.matrix.operators.AggregateUnaryOperator;
import dml.runtime.matrix.operators.BinaryOperator;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.ReorgOperator;
import dml.runtime.matrix.operators.ScalarOperator;
import dml.runtime.matrix.operators.UnaryOperator;
import dml.runtime.util.UtilFunctions;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;
import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.Plus;
import dml.runtime.instructions.CPInstructions.KahanObject;
import dml.runtime.instructions.MRInstructions.SelectInstruction.IndexRange;

public class MatrixBlock1D extends MatrixValue{

	//protected static final Log LOG = LogFactory.getLog(MatrixBlock1D.class);
	private int rlen;
	private int clen;
	private int maxrow, maxcolumn;
	private boolean sparse;
	private double[] denseBlock=null;
	private int nonZeros=0;
	public static final double SPARCITY_TURN_POINT=0.4;
	//private static final int THRESHOLD_K= (int)(1/(1+Math.log(1+SPARCITY_TURN_POINT)/Math.log(1-SPARCITY_TURN_POINT)));	
	private HashMap<CellIndex, Double> sparseBlock=null;
	
	public static boolean checkSparcityOnAggBinary(MatrixBlock1D m1, MatrixBlock1D m2)
	{
		double n=m1.getNumRows();
		double k=m1.getNumColumns();
		double m=m2.getNumColumns();
		double nz1=m1.getNonZeros();
		double nz2=m2.getNonZeros();
		double pq=nz1*nz2/n/k/k/m;
		return ( 1-Math.pow(1-pq, k) < SPARCITY_TURN_POINT );
	}
	
	private static boolean checkSparcityOnBinary(MatrixBlock1D m1, MatrixBlock1D m2)
	{
		double n=m1.getNumRows();
		double m=m1.getNumColumns();
		double nz1=m1.getNonZeros();
		double nz2=m2.getNonZeros();
		//1-(1-p)*(1-q)
		double estimated=1- (1-nz1/n/m)*(1-nz2/n/m);
		return ( 1- (1-nz1/n/m)*(1-nz2/n/m) < SPARCITY_TURN_POINT);
		
		
	}
	
	private static boolean checkRealSparcity(MatrixBlock1D m)
	{
		return ( (double)m.getNonZeros()/(double)m.getNumRows()/(double)m.getNumColumns() < SPARCITY_TURN_POINT);
	}
	
	public MatrixBlock1D()
	{
		rlen=0;
		clen=0;
		sparse=true;
		nonZeros=0;
		maxrow = maxcolumn = 0;
	}
	public MatrixBlock1D(int rl, int cl, boolean sp)
	{
		rlen=rl;
		clen=cl;
		sparse=sp;
		nonZeros=0;
		maxrow = maxcolumn = 0;
	}
	
	public MatrixBlock1D(MatrixBlock1D that)
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
	
	public boolean isInSparseFormat()
	{
		return sparse;
	}
	
	public void reset()
	{
		if(sparse)
		{
			if(sparseBlock!=null)
				sparseBlock.clear();
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
			nonZeros=0;
		}
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
		if(sparseBlock!=null)
			sparseBlock.clear();
		
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
	
	private void copySparseToSparse(MatrixBlock1D that)
	{
		this.nonZeros=that.nonZeros;
		if(that.sparseBlock==null)
		{
			if(sparseBlock!=null)
				sparseBlock.clear();
			return;
		}
		
		if(sparseBlock==null)
			sparseBlock=new HashMap<CellIndex, Double>(that.sparseBlock);
		else
		{
			sparseBlock.clear();
			sparseBlock.putAll(that.sparseBlock);
		}
	}
	
	private void copyDenseToDense(MatrixBlock1D that)
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
	
	private void copySparseToDense(MatrixBlock1D that)
	{
		//LOG.info("**** copySparseToDense: "+that.getNumRows()+"x"+that.getNumColumns()+"  nonZeros: "+that.nonZeros);
		this.nonZeros=0;//set value will increase the nonZeros
		if(that.sparseBlock==null)
		{
			if(denseBlock!=null)
				Arrays.fill(denseBlock, 0);
			return;
		}
		int limit=rlen*clen;
		if(denseBlock==null || denseBlock.length<limit)
			denseBlock=new double[limit];
		Arrays.fill(denseBlock, 0, limit, 0);
		for(Entry<CellIndex, Double> e: that.sparseBlock.entrySet())
			setValue(e.getKey(), e.getValue());
	}
	
	private void copyDenseToSparse(MatrixBlock1D that)
	{
		//LOG.info("**** copyDenseToSparse: "+that.getNumRows()+"x"+that.getNumColumns()+"  nonZeros: "+that.nonZeros);
		this.nonZeros=0;//set value will increase the nonZeros
		if(that.denseBlock==null)
		{
			if(sparseBlock!=null)
				sparseBlock.clear();
			return;
		}
		
		if(sparseBlock==null)
			sparseBlock=new HashMap<CellIndex, Double>(that.nonZeros);
		else
			sparseBlock.clear();
		
		int n=0;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				if(that.denseBlock[n]!=0)
					setValue(r, c, that.denseBlock[n]);
				n++;
			}
	}
	
	public void copy(MatrixValue thatValue) 
	{
		MatrixBlock1D that;
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
	
	public HashMap<CellIndex, Double> getSparseMap()
	{
		if(!sparse)
			return null;
		return sparseBlock;
	}
	
	public int getNonZeros()
	{
		if(sparse)
		{
			if(sparseBlock!=null)
				return sparseBlock.size();
			else
				return 0;
		}
		else
		{
			if(denseBlock!=null)
				return nonZeros;
			else
				return 0;
		}
	}
	
	//only apply to non zero cells
	public void sparseScalarOperationsInPlace(ScalarOperator op) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(sparse)
		{
			if(sparseBlock==null)
				return;
			Iterator<Entry<CellIndex, Double>> iter=sparseBlock.entrySet().iterator();
			while(iter.hasNext())
			{
				Entry<CellIndex, Double> e=iter.next();
				double v=op.executeScalar(e.getValue());
				if(v==0)
					iter.remove();
				else
					e.setValue(new Double(v));
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
	public void sparseUnaryOperationsInPlace(UnaryOperator op) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(sparse)
		{
			if(sparseBlock==null)
				return;
			Iterator<Entry<CellIndex, Double>> iter=sparseBlock.entrySet().iterator();
			while(iter.hasNext())
			{
				Entry<CellIndex, Double> e=iter.next();
				double v=op.fn.execute(e.getValue());
				if(v==0)
					iter.remove();
				else
					e.setValue(new Double(v));
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
	

	public void denseScalarOperationsInPlace(ScalarOperator op) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		double v;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				v=op.executeScalar(getValue(r, c));
				setValue(r, c, v);
			}	
	}
	
	public void denseUnaryOperationsInPlace(UnaryOperator op) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		double v;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				v=op.fn.execute(getValue(r, c));
				setValue(r, c, v);
			}	
	}
	
	private static MatrixBlock1D checkType(MatrixValue block) throws DMLUnsupportedOperationException
	{
		if( block!=null && !(block instanceof MatrixBlock1D))
			throw new DMLUnsupportedOperationException("the Matrix Value is not MatrixBlock1D!");
		return (MatrixBlock1D) block;
	}

	
	public MatrixValue scalarOperations(ScalarOperator op, MatrixValue result) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		checkType(result);
		if(result==null)
			result=new MatrixBlock1D(rlen, clen, sparse);
		result.copy(this);
		
		if(op.sparseSafe)
			((MatrixBlock1D)result).sparseScalarOperationsInPlace(op);
		else
			((MatrixBlock1D)result).denseScalarOperationsInPlace(op);
		return result;
	}
	
	public void scalarOperationsInPlace(ScalarOperator op) throws DMLUnsupportedOperationException, DMLRuntimeException
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
			result=new MatrixBlock1D(rlen, clen, sparse);
		result.copy(this);
		
		if(op.sparseSafe)
			((MatrixBlock1D)result).sparseUnaryOperationsInPlace(op);
		else
			((MatrixBlock1D)result).denseUnaryOperationsInPlace(op);
		return result;
	}
	
	public void unaryOperationsInPlace(UnaryOperator op) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(op.sparseSafe)
			this.sparseUnaryOperationsInPlace(op);
		else
			this.denseUnaryOperationsInPlace(op);
	}
	
	private MatrixBlock1D denseBinaryHelp(BinaryOperator op, MatrixBlock1D that, MatrixBlock1D result) throws DMLRuntimeException 
	{
		boolean resultSparse=checkSparcityOnBinary(this, that);
		if(result==null)
			result=new MatrixBlock1D(rlen, clen, resultSparse);
		else
			result.reset(rlen, clen, resultSparse);
		
		//double st = System.nanoTime();
		double v;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++)
			{
				v=op.fn.execute(this.getValue(r, c), that.getValue(r, c));
				result.setValue(r, c, v);
			}
		//double en = System.nanoTime();
		//System.out.println("denseBinaryHelp()-new: " + (en-st)/Math.pow(10, 9) + " sec.");
		
		return result;
	}
	
	private MatrixBlock1D sparseBinaryHelp(BinaryOperator op, MatrixBlock1D that, MatrixBlock1D result) throws DMLRuntimeException 
	{
		boolean resultSparse=checkSparcityOnBinary(this, that);
		if(result==null)
			result=new MatrixBlock1D(rlen, clen, resultSparse);
		else
			result.reset(rlen, clen, resultSparse);
		
		if(this.sparse && that.sparse)
		{
			//special case, if both matrices are all 0s, just return
			if(this.sparseBlock==null && that.sparseBlock==null)
				return result;
			
			if(result.sparseBlock==null)
				result.sparseBlock=new HashMap<CellIndex, Double>();
			
			Iterator<Entry<CellIndex, Double>> iter;
			//first, go through the first matrix, and apply operaitons on the nonzero cells
			if(this.sparseBlock!=null)
			{
				iter=this.sparseBlock.entrySet().iterator();
				while(iter.hasNext())
				{
					Entry<CellIndex, Double> e=iter.next();
					if(e.getValue()==0)
						continue;
					double v=op.fn.execute(e.getValue(), that.getValue(e.getKey()));
					if(v!=0)
						result.setValue(e.getKey(), v);
				}
			}
			
			//now, go through the second matrix, and apply operations on the ones 
			//that do not have a corresponding cell in the first matrix
			if(that.sparseBlock!=null)
			{
				iter=that.sparseBlock.entrySet().iterator();
				while(iter.hasNext())
				{
					Entry<CellIndex, Double> e=iter.next();
					
					if(e.getValue()==0)
						continue;
					
					if(this.getValue(e.getKey())!=0)
						continue;
					
					double v=op.fn.execute(0, e.getValue());
					if(v!=0)
						result.setValue(e.getKey(), v);
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
					if(thisvalue==0 && thatvalue==0)
						continue;
					resultvalue=op.fn.execute(thisvalue, thatvalue);
					result.setValue(r, c, resultvalue);
				}
		}
		
		return result;
	}
	
	public MatrixValue binaryOperations(BinaryOperator op, MatrixValue thatValue, MatrixValue result) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlock1D that=checkType(thatValue);
		checkType(result);
		if(this.rlen!=that.rlen || this.clen!=that.clen)
			throw new RuntimeException("block sizes are not matched for binary " +
					"cell operations: "+this.rlen+"*"+this.clen+" vs "+ that.rlen+"*"
					+that.clen);
		
		if(op.sparseSafe)
			return sparseBinaryHelp(op, that, (MatrixBlock1D)result);
		else
			return denseBinaryHelp(op, that, (MatrixBlock1D)result);
	}
	
	public void incrementalAggregate(AggregateOperator aggOp, MatrixValue correction, 
			MatrixValue newWithCorrection)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		assert(aggOp.correctionExists);
		MatrixBlock1D cor=checkType(correction);
		MatrixBlock1D newWithCor=checkType(newWithCorrection);
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
		}else if(aggOp.correctionLocation == 5)
                {
		    for(int r=0; r<rlen; r++)
			{
			    double currMaxValue = cor.getValue(r, 0);
			    long newMaxIndex = (long)newWithCor.getValue(r, 0);
			    double newMaxValue = newWithCor.getValue(r, 1);
			    double update = aggOp.increOp.fn.execute(newMaxValue, currMaxValue);
			    
			    if(update == 1){
				setValue(r, 0, newMaxIndex);
				cor.setValue(r, 0, newMaxValue);
			    }
			}
		}else
			throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
	}
	
	
	public void incrementalAggregate(AggregateOperator aggOp, MatrixValue newWithCorrection)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		assert(aggOp.correctionExists);
		MatrixBlock1D newWithCor=checkType(newWithCorrection);
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
		}else if(aggOp.correctionLocation == 5)
		{
		    for(int r = 0; r < rlen; r++){
			double currMaxValue = getValue(r, 1);
			long newMaxIndex = (long)newWithCor.getValue(r, 0);
			double newMaxValue = newWithCor.getValue(r, 1);
			double update = aggOp.increOp.fn.execute(newMaxValue, currMaxValue);
			
			if(update == 1){
			    setValue(r, 0, newMaxIndex);
			    setValue(r, 1, newMaxValue);
			}
		    }
		}else
			throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
	}
	
	
	public void binaryOperationsInPlace(BinaryOperator op, MatrixValue thatValue) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixBlock1D that=checkType(thatValue);
		if(this.rlen!=that.rlen || this.clen!=that.clen)
			throw new RuntimeException("block sizes are not matched for binary " +
					"cell operations: "+this.rlen+"*"+this.clen+" vs "+ that.rlen+"*"
					+that.clen);
		
		if(op.sparseSafe)
			sparseBinaryInPlaceHelp(op, that);
		else
			denseBinaryInPlaceHelp(op, that);
	}
	
	private void denseBinaryInPlaceHelp(BinaryOperator op, MatrixBlock1D that) throws DMLRuntimeException 
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
	
	private void sparseBinaryInPlaceHelp(BinaryOperator op, MatrixBlock1D that) throws DMLRuntimeException 
	{
		boolean resultSparse=checkSparcityOnBinary(this, that);
		if(resultSparse && !this.sparse)
			denseToSparse();
		else if(!resultSparse && this.sparse)
			sparseToDense();
		
		if(this.sparse && that.sparse)
		{
			//special case, if both matrices are all 0s, just return
			if(this.sparseBlock==null && that.sparseBlock==null)
				return;
			
			if(this.sparseBlock==null)
				this.sparseBlock=new HashMap<CellIndex, Double>();
			
			HashSet<CellIndex> allIndexes=new HashSet<CellIndex> ();
			allIndexes.addAll(this.sparseBlock.keySet());
			if(that.sparseBlock!=null)
				allIndexes.addAll(that.sparseBlock.keySet());
			
			double thisvalue, thatvalue, resultvalue;
			for(CellIndex index: allIndexes)
			{
				thisvalue=this.getValue(index);
				thatvalue=that.getValue(index);
				resultvalue=op.fn.execute(thisvalue, thatvalue);
				this.setValue(index, resultvalue);
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
	
	public MatrixValue reorgOperations(ReorgOperator op, MatrixValue result,
			int startRow, int startColumn, int length)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{	
		checkType(result);
		boolean reducedDim=op.fn.computeDimension(rlen, clen, tempCellIndex);
		boolean sps;
		if(reducedDim)
			sps=false;
		else
			sps=checkRealSparcity(this);
			
		if(result==null)
			result=new MatrixBlock1D(tempCellIndex.row, tempCellIndex.column, sps);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, sps);
		
		//TODO: right now cannot handle matrix diag
	/*	if(op==Reorg.SupportedOperation.REORG_MATRIX_DIAG)
		{
			
			for(int i=0; i<length; i++)
				result.setValue(startRow+i, 0, getValue(startRow+i, startColumn+i));
			return result;
		}*/
		
		CellIndex temp = new CellIndex(0, 0);
		if(sparse)
		{
			if(sparseBlock!=null)
			{
				for(Entry<CellIndex, Double> e: sparseBlock.entrySet())
				{
					op.fn.execute(e.getKey(), temp);
					result.setValue(temp.row, temp.column, e.getValue());
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
	

	private void updateCtable(double v1, double v2, double w, MatrixValue result) throws DMLRuntimeException {
		int _row, _col;
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v1) || Double.isNaN(v2) || Double.isNaN(w) ) {
			return;
		}
		else {
			_row = (int)v1;
			_col = (int)v2;
			
			if ( _row <= 0 || _col <= 0 ) {
				throw new DMLRuntimeException("Erroneous input while computing the contingency table (one of the value <= zero).");
			} 
			
			result.addValue(_row, _col, w);
			
			if ( _row > result.getMaxRow() ) 
				result.setMaxRow(_row);
			if ( _col > result.getMaxColumn() )
				result.setMaxColumn(_col);
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
			if(sparseBlock!=null)
			{
				for(Entry<CellIndex, Double> e: sparseBlock.entrySet())
				{
					// output (v1,v2,w)
					v1 = e.getValue();
					w = that2.getValue(e.getKey().row, e.getKey().column);
					updateCtable(v1, v2, w, ctableResult);
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
			if(sparseBlock!=null)
			{
				for(Entry<CellIndex, Double> e: sparseBlock.entrySet())
				{
					// output (v1,v2,w)
					v1 = e.getValue();
					updateCtable(v1, v2, w, ctableResult);
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
			if(sparseBlock!=null)
			{
				for(Entry<CellIndex, Double> e: sparseBlock.entrySet())
				{
					// output (v1,v2,w)
					v1 = e.getValue();
					v2 = that.getValue(e.getKey().row, e.getKey().column);
					updateCtable(v1, v2, w, ctableResult);
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
			if(sparseBlock!=null)
			{
				for(Entry<CellIndex, Double> e: sparseBlock.entrySet())
				{
					// output (v1,v2,w)
					v1 = e.getValue();
					v2 = that.getValue(e.getKey().row, e.getKey().column);
					w = that2.getValue(e.getKey().row, e.getKey().column);
					updateCtable(v1, v2, w, ctableResult);
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
	
	
	private void traceHelp(AggregateUnaryOperator op, MatrixBlock1D result, 
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
	private void diagM2VHelp(AggregateUnaryOperator op, MatrixBlock1D result, 
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

/*
 * blockingFactorRow and blockingFactorCol are the blocking factor for a matrix (1000x1000), they are only used for trace,
 * */
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
			case 5: tempCellIndex.column++; break; 
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
			result=new MatrixBlock1D(tempCellIndex.row, tempCellIndex.column, false);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, false);
		
		//TODO: this code is hack to support trace, and should be removed when selection is supported
		if(op.isTrace)
			traceHelp(op, (MatrixBlock1D)result, blockingFactorRow, blockingFactorCol, indexesIn);
		else if(op.isDiagM2V)
			diagM2VHelp(op, (MatrixBlock1D)result, blockingFactorRow, blockingFactorCol, indexesIn);
		else if(op.sparseSafe)
			sparseAggregateUnaryHelp(op, (MatrixBlock1D)result, blockingFactorRow, blockingFactorCol, indexesIn);
		else
			denseAggregateUnaryHelp(op, (MatrixBlock1D)result, blockingFactorRow, blockingFactorCol, indexesIn);
		
		return result;
	}
	
	private void incrementalAggregateUnaryHelp(AggregateOperator aggOp, MatrixBlock1D result, CellIndex index, 
			double newvalue, KahanObject buffer) throws DMLRuntimeException
	{
		if(aggOp.correctionExists)
		{
			if(aggOp.correctionLocation==1 || aggOp.correctionLocation==2)
			{
				int corRow=index.row, corCol=index.column;
				if(aggOp.correctionLocation==1)//extra row
					corRow++;
				else if(aggOp.correctionLocation==2)
					corCol++;
				else
					throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
				
				buffer._sum=result.getValue(index);
				buffer._correction=result.getValue(corRow, corCol);
				buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newvalue);
				result.setValue(index.row, index.column, buffer._sum);
				result.setValue(corRow, corCol, buffer._correction);
			}else if(aggOp.correctionLocation==0)
			{
				throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
			}else// for mean
			{
				int corRow=index.row, corCol=index.column;
				int countRow=index.row, countCol=index.column;
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
				buffer._sum=result.getValue(index);
				buffer._correction=result.getValue(corRow, corCol);
				double count=result.getValue(countRow, countCol)+1.0;
				double toadd=(newvalue-buffer._sum)/count;
				buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
				result.setValue(index.row, index.column, buffer._sum);
				result.setValue(corRow, corCol, buffer._correction);
				result.setValue(countRow, countCol, count);
			}
			
		}else
		{
			newvalue=aggOp.increOp.fn.execute(result.getValue(index), newvalue);
			result.setValue(index.row, index.column, newvalue);
		}
	}
	
	private void sparseAggregateUnaryHelp(AggregateUnaryOperator op, MatrixBlock1D result,
			int blockingFactorRow, int blockingFactorCol, MatrixIndexes indexesIn) throws DMLRuntimeException
	{
		//initialize result
		if(op.aggOp.initialValue!=0)
			result.resetDenseWithValue(result.rlen, result.clen, op.aggOp.initialValue);
		
		KahanObject buffer=new KahanObject(0,0);
		int r = 0, c = 0;
		if(sparse)
		{
			if(sparseBlock!=null)
			{
				for(Entry<CellIndex, Double> e: sparseBlock.entrySet())
				{
					op.indexFn.execute(e.getKey(), result.tempCellIndex);
					incrementalAggregateUnaryHelp(op.aggOp, result, result.tempCellIndex, e.getValue(), buffer);
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
					incrementalAggregateUnaryHelp(op.aggOp, result, result.tempCellIndex, denseBlock[i], buffer);
				}
			}
		}
	}
	
	private void denseAggregateUnaryHelp(AggregateUnaryOperator op, MatrixBlock1D result,
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
				
				if(op.aggOp.correctionExists && op.aggOp.correctionLocation == 5){
				    double currMaxValue = result.getValue(i, 1);
				    long newMaxIndex = UtilFunctions.cellIndexCalculation(indexesIn.getColumnIndex(), maxcolumn, j);
				    double newMaxValue = getValue(i, j);
				    double update = op.aggOp.increOp.fn.execute(newMaxValue, currMaxValue);
				    
				    if(update == 1){
					result.setValue(i, 0, newMaxIndex);
					result.setValue(i, 1, newMaxValue);
				    }
				}else
				    incrementalAggregateUnaryHelp(op.aggOp, result, result.tempCellIndex, getValue(i,j), buffer);
			}
	}
	
	private static void sparseAggregateBinaryHelp(MatrixBlock1D m1, MatrixBlock1D m2, 
			MatrixBlock1D result, AggregateBinaryOperator op) throws DMLRuntimeException 
	{
		//TODO: right now only support matrix multiplication
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
		MatrixBlock1D m1=checkType(m1Value);
		MatrixBlock1D m2=checkType(m2Value);
		checkType(result);
		if(m1.clen!=m2.rlen)
			throw new RuntimeException("dimenstions do not match for matrix multiplication");
		int rl=m1.rlen;
		int cl=m2.clen;
		boolean sp=checkSparcityOnAggBinary(m1, m2);
		if(result==null)
			result=new MatrixBlock1D(rl, cl, sp);//m1.sparse&&m2.sparse);
		else
			result.reset(rl, cl, sp);//m1.sparse&&m2.sparse);
		
		if(op.sparseSafe)
			sparseAggregateBinaryHelp(m1, m2, (MatrixBlock1D)result, op);
		else
			aggBinSparseUnsafe(m1, m2, (MatrixBlock1D)result, op);
		return result;
	}
	
	
	/*
	 * to perform aggregateBinary when the first matrix is dense and the second is sparse
	 */
	private static void aggBinDenseSparse(MatrixBlock1D m1, MatrixBlock1D m2,
			MatrixBlock1D result, AggregateBinaryOperator op) throws DMLRuntimeException 
	{

		
		if(m2.sparseBlock==null)
			return;
		
		for(Entry<CellIndex, Double> e: m2.sparseBlock.entrySet())
		{
			int k=e.getKey().row;
			int j=e.getKey().column;
			for(int i=0; i<m1.rlen; i++)
			{
				double old=result.getValue(i, j);
				double aik=m1.getValue(i, k);
				double addValue=op.binaryFn.execute(aik, e.getValue());
				double newvalue=op.aggOp.increOp.fn.execute(old, addValue);
				result.setValue(i, j, newvalue);
			}
		}
	}
	
	/*
	 * to perform aggregateBinary when the first matrix is sparse and the second is dense
	 */
	private static void aggBinSparseDense(MatrixBlock1D m1, MatrixBlock1D m2,
			MatrixBlock1D result, AggregateBinaryOperator op) throws DMLRuntimeException
	{
		
		if(m1.sparseBlock==null)
			return;
		
		for(Entry<CellIndex, Double> e: m1.sparseBlock.entrySet())
		{
			int i=e.getKey().row;
			int k=e.getKey().column;
			for(int j=0; j<m2.clen; j++)
			{
				double old=result.getValue(i, j);
				double bkj=m2.getValue(k, j);
				double addValue=op.binaryFn.execute(e.getValue(), bkj);
				double newvalue=op.aggOp.increOp.fn.execute(old, addValue);
				result.setValue(i, j, newvalue);
			}
		}
	}
	
	static class Remain{
		public int remainIndex;
		public double value;
		public Remain(int indx, double v)
		{
			remainIndex=indx;
			value=v;
		}
	}
	
	/*
	 * to perform aggregateBinary when both matrices are sparse
	 */
	
	public static void aggBinSparse(MatrixBlock1D m1, MatrixBlock1D m2,
			MatrixBlock1D result, AggregateBinaryOperator op) throws DMLRuntimeException 
	{
		//hash-based join
		HashMap<Integer, Vector<Remain>> map=new HashMap<Integer, Vector<Remain>>();
		
		double st1, en1, innerTime=0;
		//go through the second matrix, put its nonzero cells to the hash map
		if(m2.sparseBlock!=null)
		{
			for(Entry<CellIndex, Double> e: m2.sparseBlock.entrySet())
			{
				int k=e.getKey().row;
				Vector<Remain> vec=map.get(k);
				if(vec==null)
				{
					vec=new Vector<Remain>();
					map.put(k, vec);
				}
				vec.add(new Remain(e.getKey().column, e.getValue()));
			}
		}
		
		//go through the first matrix, and perform join
		if(m1.sparseBlock!=null)
		{
			st1 = System.nanoTime();
			for(Entry<CellIndex, Double> e: m1.sparseBlock.entrySet())
			{
				int i=e.getKey().row;
				int k=e.getKey().column;
				Vector<Remain> vec=map.get(k);
				if(vec==null)
					continue;
				for(Remain rm: vec)
				{
					int j=rm.remainIndex;
					double old=result.getValue(i, j);
					double addValue=op.binaryFn.execute(e.getValue(), rm.value);
					double newvalue=op.aggOp.increOp.fn.execute(old, addValue);
					result.setValue(i, j, newvalue);
				}
			}
			en1 = System.nanoTime();
			innerTime = innerTime + ((en1-st1)/Math.pow(10, 9));
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
	public static void matrixMult(MatrixBlock1D matrixA, MatrixBlock1D matrixB,
			MatrixBlock1D resultMatrix)
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
	
	private static void aggBinSparseUnsafe(MatrixBlock1D m1, MatrixBlock1D m2, MatrixBlock1D result, 
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
				result.setValue(i, j, aggValue);
			}
	}
	/*
	 * to perform aggregateBinary when both matrices are dense
	 */
	private static void aggBinDense(MatrixBlock1D m1, MatrixBlock1D m2, MatrixBlock1D result, AggregateBinaryOperator op) throws DMLRuntimeException
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
	
	public void setValue(int r, int c, double v)
	{
		if(r>rlen || c > clen)
			throw new RuntimeException("indexes ("+r+","+c+") out of range ("+rlen+","+clen+")");
		if(sparse)
		{
			if(sparseBlock==null)
				sparseBlock=new HashMap<CellIndex, Double>();
			
			if(v!=0)
				sparseBlock.put(new CellIndex(r,c), v);
			else
				sparseBlock.remove(new CellIndex(r,c));
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
	
	public void setValue(CellIndex index, double v)
	{
		if(index.row>rlen || index.column > clen)
			throw new RuntimeException("indexes ("+index.row+","+index.column+") out of range ("+rlen+","+clen+")");
		if(sparse)
		{
			if(sparseBlock==null)
				sparseBlock=new HashMap<CellIndex, Double>();
			
			if(v!=0)
				sparseBlock.put(index, v);
			else
				sparseBlock.remove(index);
		}else
		{
			int limit=rlen*clen;
			if(denseBlock==null)
			{
				denseBlock=new double[limit];
				Arrays.fill(denseBlock, 0, limit, 0);
			}
			
			int ind=index.row*clen+index.column;
			if(denseBlock[ind]==0)
				nonZeros++;
			denseBlock[ind]=v;
			if(v==0)
				nonZeros--;
		}
	}
	
	/*
	 * If (r,c) \in Block, add v to existing cell
	 * If not, add a new cell with index (r,c)
	 */
	public void addValue(int r, int c, double v)
	{
		// Since this method is also used to construct matrices whose dimensions are unknown 
		// at compilation time, boundary checks on r and c must not be performed
		Double d;
		if(sparse)
		{
			if(sparseBlock==null)
				sparseBlock=new HashMap<CellIndex, Double>();
			
			tempCellIndex.set(r,c);
			if ( (d = sparseBlock.get(tempCellIndex)) == null )
				sparseBlock.put(new CellIndex(r,c), v);
			else {
				if (d.doubleValue()+v != 0 )
					sparseBlock.put(tempCellIndex, d.doubleValue() + v);
				else
					sparseBlock.remove(tempCellIndex);
			}
		}else
		{
			if ( v != 0 ) {
				int limit=rlen*clen;
				if(denseBlock==null)
				{
					denseBlock=new double[limit];
					Arrays.fill(denseBlock, 0, limit, 0);
				}
				
				int index=r*clen+c;
				
				if ( denseBlock[index] == 0 ) 
					nonZeros++;
				denseBlock[index] += v;
				if(denseBlock[index] == 0)
					nonZeros--;
			}
		}
	}
	
	public double getValue(int r, int c)
	{
		if(r>rlen || c > clen)
			throw new RuntimeException("indexes ("+r+","+c+") out of range ("+rlen+","+clen+")");
		
		if(sparse)
		{
			if(sparseBlock==null)
				return 0;
			tempCellIndex.set(r,c);
			Double d=sparseBlock.get(tempCellIndex);
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
	
	private double getValue(CellIndex index) {
		if(index.row>rlen || index.column > clen)
			throw new RuntimeException("indexes ("+index.row+","+index.column+") out of range ("+rlen+","+clen+")");
		
		if(sparse)
		{
			if(sparseBlock==null)
				return 0;
			Double d=sparseBlock.get(index);
			if(d!=null)
				return d;
			else
				return 0;
		}else
		{
			if(denseBlock==null)
				return 0;
			return denseBlock[index.row*clen+index.column]; 
		}
	}
	
	public void examSparsity()
	{
		if(sparse)
		{
			if(sparseBlock.size()>rlen*clen*SPARCITY_TURN_POINT)
				sparseToDense();
		}else
		{
			if(nonZeros<rlen*clen*SPARCITY_TURN_POINT)
				denseToSparse();
		}
	}
		
	public void denseToSparse() {
		
		//LOG.info("**** denseToSparse: "+this.getNumRows()+"x"+this.getNumColumns()+"  nonZeros: "+this.nonZeros);
		
		if(sparseBlock==null)
			sparseBlock=new HashMap<CellIndex, Double>(nonZeros);
		else
			sparseBlock.clear();
		
		if(denseBlock==null)
			return;
		
		int limit=rlen*clen;
		int r, c; double v;
		for(int i=0; i<limit; i++)
		{
			v=denseBlock[i];
			if(v==0)
				continue;
			r=i/clen;
			c=i%clen;
			sparseBlock.put(new CellIndex(r,c), v);
		}
		sparse=true;
	}
	public void sparseToDense() {
		
		//LOG.info("**** sparseToDense: "+this.getNumRows()+"x"+this.getNumColumns()+"  nonZeros: "+this.nonZeros);
		
		int limit=rlen*clen;
		if(denseBlock==null || denseBlock.length < limit )
			denseBlock=new double[limit];
		Arrays.fill(denseBlock, 0, limit, 0);
		nonZeros=0;
		
		if(sparseBlock==null)
			return;
		
		int r, c; double v;
		for(Entry<CellIndex, Double> e: sparseBlock.entrySet())
		{
			v=e.getValue();
			if(v==0)
				continue;
			
			r=e.getKey().row;
			c=e.getKey().column;
			denseBlock[r*clen+c]=v;
			nonZeros++;
		}
		sparse=false;
		sparseBlock=null;
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
	//	Arrays.fill(denseBlock, 0, limit, 0);
		
		nonZeros=0;
		for(int i=0; i<limit; i++)
		{
			denseBlock[i]=in.readDouble();
			if(denseBlock[i]!=0)
				nonZeros++;
		}
	}
	
	private void readSparseBlock(DataInput in) throws IOException {
		
		int size=in.readInt();
		if(sparseBlock==null)
			sparseBlock=new HashMap<CellIndex, Double>(size);
		else
			sparseBlock.clear();
		
		int r, c; double v;
		for(int i=0; i<size; i++)
		{
			r=in.readInt();
			c=in.readInt();
			v=in.readDouble();
			if(v!=0)
				sparseBlock.put(new CellIndex(r, c), new Double(v));
		}
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(rlen);
		out.writeInt(clen);
		
		if(sparse)
		{
			if(sparseBlock==null)
				writeEmptyBlock(out);
			//if it should be dense, then write to the dense format
			else if(sparseBlock.size()>rlen*clen*SPARCITY_TURN_POINT)
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
		out.writeInt(sparseBlock.size());
		for(Entry<CellIndex, Double> e: sparseBlock.entrySet())
		{
			if(e.getValue()==0)
				continue;
			
			out.writeInt(e.getKey().row);
			out.writeInt(e.getKey().column);
			out.writeDouble(e.getValue());
		}
	}
	
	private void writeSparseToDense(DataOutput out) throws IOException {
		out.writeBoolean(false);
		for(int i=0; i<rlen; i++)
			for(int j=0; j<clen; j++)
				out.writeDouble(getValue(i, j));
	}
	
	private void writeDenseToSparse(DataOutput out) throws IOException {
		out.writeBoolean(true);
		out.writeInt(nonZeros);
		int limit=rlen*clen;
		int num=0;
		if(denseBlock==null)
			return;
		for(int i=0; i<limit; i++)
		{
			if(denseBlock[i]!=0)
			{
				out.writeInt(i/clen);
				out.writeInt(i%clen);
				out.writeDouble(denseBlock[i]);
				num++;
			}
		}
		if(num!=nonZeros)
			throw new IOException("nonZeros = "+nonZeros+", but should be "+num);
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
	
	public String toString()
	{
		String ret="sparse? = "+sparse+"\n" ;
		if(!sparse)
			ret+="nonzeros = "+nonZeros+"\n";
		for(int i=0; i<rlen; i++)
		{
			for(int j=0; j<clen; j++)
			{
				ret+=getValue(i, j)+"\t";
			}
			ret+="\n";
		}
		return ret;
	}
	
	public static void main(String[] args)
	{
		
		int n=10;
		int k=10000;
		int m=1000;
		int nz1=100000;
		int nz2=(int) (0.001*k*m);
		double pq=(double)(nz1*nz2)/(double)n/(double)k/(double)k/(double)m;
		System.out.println( SPARCITY_TURN_POINT );
		System.out.println( pq );
		System.out.println( (1-Math.pow(1-pq, k) )*n*m );
		System.out.println( 1-Math.pow(1-pq, k) < SPARCITY_TURN_POINT );
		
		
		double d=0;
		if(d==0)
			System.out.println(d+"=0");
		else
			System.out.println(d+"!=0");
		
		boolean sparse1=false, sparse23=true;
		boolean sparseSafe=false;
		
		MatrixBlock1D m1=new MatrixBlock1D(3, 2, sparse1);
		m1.setValue(0, 0, 1);
		m1.setValue(0, 1, 2);
		//m1.setValue(1, 0, 3);
		//m1.setValue(1, 1, 4);
		//m1.setValue(2, 0, 5);
		//m1.setValue(2, 1, 6);
		System.out.println("matrix m1: ");
		m1.print();
		
		MatrixBlock1D m2=new MatrixBlock1D(3, 2, sparse23);
		m2.setValue(0, 0, 6);
		m2.setValue(0, 1, 5);
		//m2.setValue(1, 0, 4);
		//m2.setValue(1, 1, 3);
		//m2.setValue(2, 0, 2);
		//m2.setValue(2, 1, 1);
		System.out.println("matrix m2: ");
		m2.print();
		
		
		MatrixBlock1D m3=new MatrixBlock1D(2, 3, sparse23);
		m3.setValue(0, 0, 6);
		m3.setValue(0, 1, 5);
	//	m3.setValue(0, 2, 4);
	//	m3.setValue(1, 0, 3);
	//	m3.setValue(1, 1, 2);
	//	m3.setValue(1, 2, 1);
		System.out.println("matrix m3:");
		m3.print();
	
		
		MatrixBlock1D m4=new MatrixBlock1D();
		try {
/*			System.out.println("--------------------------------");
			System.out.println("m4=m1*2");
			m4=(MatrixBlock1D) m1.scalarOperations(Scalar.SupportedOperation.SCALAR_MULTIPLICATION, 2, m4, sparseSafe);
			m4.print();
			
			System.out.println("--------------------------------");
			System.out.println("m4=m4*2");
			m4.scalarOperationsInPlace(Scalar.SupportedOperation.SCALAR_MULTIPLICATION, 2, sparseSafe);
			m4.print();
			*/
			/*System.out.println("--------------------------------");
			System.out.println("m4=m1*m2");
			//m4=(MatrixBlock1D) m1.binaryOperations(Binary.SupportedOperation.BINARY_ADDITION, m2, m4, sparseSafe);
			m4=(MatrixBlock1D) m1.binaryOperations(new Multiply(), m2, m4, sparseSafe);
			m4.print();*/
			
		/*	System.out.println("--------------------------------");
			System.out.println("m1=m1+m2");
			m1.binaryOperationsInPlace(Binary.SupportedOperation.BINARY_ADDITION, m2, sparseSafe);
			m1.print();
			
			System.out.println("--------------------------------");
			System.out.println("m4=col_sum(m1)");
			//m4=(MatrixBlock1D) m1.aggregateUnaryOperations(AggregateUnary.SupportedOperation.AGU_COLUMN_SUM, m4, sparseSafe);
			m4.print();
			
			System.out.println("--------------------------------");
			System.out.println("m4=m1 %*% m3");
			m4=(MatrixBlock1D) m1.aggregateBinaryOperations(m1, m3, m4, AggregateBinary.SupportedOperation.AGB_MMULT, sparseSafe);
			m4.print();*/
					
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public void getCellValues(Collection<Double> ret) {
		int limit=rlen*clen;
		if(sparse)
		{
			if(sparseBlock==null)
			{
				for(int i=0; i<limit; i++)
					ret.add(0.0);
			}else
			{
				ret.addAll(sparseBlock.values());
				for(int i=0; i<limit-sparseBlock.size(); i++)
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
			if(sparseBlock==null)
			{
				ret.put(0.0, limit);
			}else
			{
				for(Double v: sparseBlock.values())
				{
					Integer old=ret.get(v);
					if(old!=null)
						ret.put(v, old+1);
					else
						ret.put(v, 1);
				}
				Integer old=ret.get(0.0);
				if(old!=null)
					ret.put(0.0, old+limit-sparseBlock.size());
				else
					ret.put(0.0, limit-sparseBlock.size());
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
	public int compareTo(Object o) {
		//TODO don't compare the blocks
		return 0;
	}

	@Override
	public MatrixValue selectOperations(MatrixValue result, IndexRange range)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		checkType(result);
		boolean sps;
		if(nonZeros/rlen/clen*(range.rowEnd-range.rowStart+1)*(range.colEnd-range.colStart+1)/rlen/clen< SPARCITY_TURN_POINT)
			sps=true;
		else sps=false;
			
		if(result==null)
			result=new MatrixBlock1D(tempCellIndex.row, tempCellIndex.column, sps);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, sps);
		
		if(sparse)
		{
			if(sparseBlock!=null)
			{
				for(Entry<CellIndex, Double> e: sparseBlock.entrySet())
				{
					if(UtilFunctions.isIn(e.getKey().row, range.rowStart, range.rowEnd)
					   && UtilFunctions.isIn(e.getKey().column, range.colStart, range.colEnd))
						result.setValue(e.getKey().row, e.getKey().column, e.getValue());
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
}
