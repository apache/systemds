/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.io;

import java.util.Arrays;

import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.Builtin.BuiltinFunctionCode;
import com.ibm.bi.dml.runtime.functionobjects.IndexFunction;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.Mean;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.ReduceAll;
import com.ibm.bi.dml.runtime.functionobjects.ReduceCol;
import com.ibm.bi.dml.runtime.functionobjects.ReduceRow;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

/**
 * MB:
 * Library for matrix aggregations including ak+, uak+ for all
 * combinations of dense and sparse representations, and corrections.
 * Those are performance-critical operations because they are used
 * on combiners/reducers of important operations like tsmm, mvmult,
 * indexing, but also basic sum/min/max/mean, row*, col*, etc. Specific
 * handling is especially required for all non sparse-safe operations
 * in order to prevent unnecessary worse asymptotic behavior.
 *
 * This library currently covers the following opcodes:
 * ak+, uak+, uark+, uack+, uamin, uarmin, uacmin, uamax, uarmax, uacmax,
 * ua*, uamean, uarmean, uacmean, uarimax.
 * 
 * TODO next opcode extensions: a+, trace, diagM2V, row/colindexmax
 * TODO low level optimization (potential 3x, sum non-conclusive yet)
 */
public class MatrixAggLib 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	////////////////////////////////
	// public matrix agg interface
	////////////////////////////////
	
	public static final boolean LOW_LEVEL_OPTIMIZATION = true;
	public static final boolean NAN_AWARENESS = false;
	
	private enum AggType {
		KAHAN_SUM,
		MIN,
		MAX,
		MEAN,
		MAX_INDEX,
		PROD,
		INVALID,
	}

	
	/**
	 * Core incremental matrix aggregate (ak+) as used in tsmm, cpmm,
	 * append, indexing, etc. Note that we try to keep the current 
	 * aggVal and aggCorr in dense format in order to allow efficient
	 * access according to the dense/sparse input. 
	 * 
	 * 
	 * @param in input matrix
	 * @param aggVal current aggregate values (in/out)
	 * @param aggCorr current aggregate correction (in/out)
	 * @throws DMLRuntimeException 
	 */
	public static void aggregateBinaryMatrix(MatrixBlockDSM in, MatrixBlockDSM aggVal, MatrixBlockDSM aggCorr) 
		throws DMLRuntimeException
	{	
		//Timing time = new Timing(true);
		
		if(!in.sparse && !aggVal.sparse && !aggCorr.sparse)
			aggregateBinaryMatrixDense(in, aggVal, aggCorr);
		else if(in.sparse && !aggVal.sparse && !aggCorr.sparse)
			aggregateBinaryMatrixSparse(in, aggVal, aggCorr);
		else
			aggregateBinaryMatrixGeneric(in, aggVal, aggCorr);
		
		//System.out.println("agg ("+in.rlen+","+in.clen+","+in.sparse+") in "+time.stop()+"ms.");
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param vFn
	 * @param ixFn
	 * @throws DMLRuntimeException
	 */
	public static void aggregateUnaryMatrix(MatrixBlockDSM in, MatrixBlockDSM out, AggregateUnaryOperator uaop) 
		throws DMLRuntimeException
	{
		//Timing time = new Timing(true);
		
		AggType aggtype = getAggType(uaop);
		if( !in.sparse )
			aggregateUnaryMatrixDense(in, out, aggtype, uaop.aggOp.increOp.fn, uaop.indexFn);
		else
			aggregateUnaryMatrixSparse(in, out, aggtype, uaop.aggOp.increOp.fn, uaop.indexFn);
		
		//System.out.println("uagg ("+in.rlen+","+in.clen+","+in.sparse+") in "+time.stop()+"ms.");
	}
	
	/**
	 * 
	 * @param op
	 * @return
	 */
	public static boolean isSupportedUnaryAggregateOperator( AggregateUnaryOperator op )
	{
		AggType type = getAggType( op );
		return (type != AggType.INVALID);
	}
	
	/**
	 * Recompute outputs (e.g., maxindex) according to block indexes from MR.
	 * TODO: this should not be part of block operations but of the MR instruction.
	 * 
	 * @param out
	 * @param op
	 * @param brlen
	 * @param bclen
	 * @param ix
	 */
	public static void recomputeIndexes( MatrixBlockDSM out, AggregateUnaryOperator op, int brlen, int bclen, MatrixIndexes ix )
	{
		AggType type = getAggType(op);
		if( type == AggType.MAX_INDEX && ix.getColumnIndex()!=1 )
		{
			int m = out.rlen;
			double[] c = out.getDenseArray();
			for( int i=0, cix=0; i<m; i++, cix+=2 )
				c[cix] = UtilFunctions.cellIndexCalculation(ix.getColumnIndex(), bclen, (int)c[cix]-1);
		}
	}
	
	/**
	 * 
	 * @param op
	 * @return
	 */
	private static AggType getAggType( AggregateUnaryOperator op )
	{
		ValueFunction vfn = op.aggOp.increOp.fn;
		IndexFunction ifn = op.indexFn;
		
		//(kahan) sum 
		if( vfn instanceof KahanPlus 
			&& (op.aggOp.correctionLocation == CorrectionLocationType.LASTCOLUMN ||op.aggOp.correctionLocation == CorrectionLocationType.LASTROW ) 
			&& (ifn instanceof ReduceAll || ifn instanceof ReduceCol || ifn instanceof ReduceRow) )
		{
			return AggType.KAHAN_SUM;
		}
		
		//mean 
		if( vfn instanceof Mean 
			&& (op.aggOp.correctionLocation == CorrectionLocationType.LASTTWOCOLUMNS || op.aggOp.correctionLocation == CorrectionLocationType.LASTTWOROWS ) 
			&& (ifn instanceof ReduceAll || ifn instanceof ReduceCol || ifn instanceof ReduceRow) )
		{
			return AggType.MEAN;
		}
		
		//prod
		if( vfn instanceof Multiply && ifn instanceof ReduceAll )
		{
			return AggType.PROD;
		}
		
		//min / max
		if( vfn instanceof Builtin &&
		    (ifn instanceof ReduceAll || ifn instanceof ReduceCol || ifn instanceof ReduceRow) )
		{
			BuiltinFunctionCode bfcode = ((Builtin)vfn).bFunc;
			switch( bfcode ){
				case MAX: return AggType.MAX;
				case MIN: return AggType.MIN;
				case MAXINDEX: return AggType.MAX_INDEX;
			}
		}
		
		return AggType.INVALID;
	}

	/**
	 * 
	 * @param in
	 * @param aggVal
	 * @param aggCorr
	 * @throws DMLRuntimeException
	 */
	private static void aggregateBinaryMatrixDense(MatrixBlockDSM in, MatrixBlockDSM aggVal, MatrixBlockDSM aggCorr) 
			throws DMLRuntimeException
	{
		if( in.denseBlock==null )
			return;
		
		//allocate output arrays (if required)
		aggVal.allocateDenseBlock(); //should always stay in dense
		aggCorr.allocateDenseBlock(); //should always stay in dense
		
		double[] a = in.getDenseArray();
		double[] c = aggVal.getDenseArray();
		double[] cc = aggCorr.getDenseArray();
		
		KahanObject buffer1 = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		final int len = Math.min(a.length, in.rlen*in.clen);
		
		int nnzC = 0;
		int nnzCC = 0;
		
		for( int i=0; i<len; i++ )
		{
			buffer1._sum        = c[i];
			buffer1._correction = cc[i];
			akplus.execute2(buffer1, a[i]);
			c[i]  = buffer1._sum;
			cc[i] = buffer1._correction;
			nnzC += (buffer1._sum!=0)?1:0;
			nnzCC += (buffer1._correction!=0)?1:0;
		}
		
		aggVal.nonZeros = nnzC;
		aggCorr.nonZeros = nnzCC;
	}

	/**
	 * 
	 * @param in
	 * @param aggVal
	 * @param aggCorr
	 * @throws DMLRuntimeException
	 */
	private static void aggregateBinaryMatrixSparse(MatrixBlockDSM in, MatrixBlockDSM aggVal, MatrixBlockDSM aggCorr) 
			throws DMLRuntimeException
	{
		if( in.sparseRows==null )
			return;
		
		//allocate output arrays (if required)
		aggVal.allocateDenseBlock(); //should always stay in dense
		aggCorr.allocateDenseBlock(); //should always stay in dense
		
		SparseRow[] a = in.getSparseRows();
		double[] c = aggVal.getDenseArray();
		double[] cc = aggCorr.getDenseArray();
		
		KahanObject buffer1 = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		final int m = in.rlen;
		final int n = in.clen;
		final int rlen = Math.min(a.length, m);
		
		int nnzC = 0;
		int nnzCC = 0;
		
		for( int i=0, cix=0; i<rlen; i++, cix+=n )
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()<0 );
			{
				int alen = arow.size();
				int[] aix = arow.getIndexContainer();
				double[] avals = arow.getValueContainer();
				
				for( int j=0; j<alen; j++ )
				{
					int ix = cix+aix[j];
					buffer1._sum        = c[ix];
					buffer1._correction = cc[ix];
					akplus.execute2(buffer1, avals[j]);
					c[ix]  = buffer1._sum;
					cc[ix] = buffer1._correction;
					nnzC += (buffer1._sum!=0)?1:0;
					nnzCC += (buffer1._correction!=0)?1:0;
				}
			}
		}
		
		aggVal.nonZeros = nnzC;
		aggCorr.nonZeros = nnzCC;
	}
	
	/**
	 * 
	 * @param in
	 * @param aggVal
	 * @param aggCorr
	 * @throws DMLRuntimeException
	 */
	private static void aggregateBinaryMatrixGeneric(MatrixBlockDSM in, MatrixBlockDSM aggVal, MatrixBlockDSM aggCorr) 
		throws DMLRuntimeException
	{	
		final int m = in.rlen;
		final int n = in.clen;
		
		KahanObject buffer = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		//incl implicit nnz maintenance
		for(int i=0; i<m; i++)
			for(int j=0; j<n; j++)
			{
				buffer._sum = aggVal.quickGetValue(i, j);
				buffer._correction = aggCorr.quickGetValue(i, j);
				akplus.execute(buffer, in.quickGetValue(i, j));
				aggVal.quickSetValue(i, j, buffer._sum);
				aggCorr.quickSetValue(i, j, buffer._correction);
			}
		
		aggVal.examSparsity();
		aggCorr.examSparsity();
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param vFn
	 * @param ixFn
	 * @throws DMLRuntimeException
	 */
	private static void aggregateUnaryMatrixDense(MatrixBlockDSM in, MatrixBlockDSM out, AggType optype, ValueFunction vFn, IndexFunction ixFn) 
			throws DMLRuntimeException
	{
		if( in.denseBlock==null )
			return;
		
		final int m = in.rlen;
		final int n = in.clen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		
		//allocate output arrays (if required)
		out.reset(m2, n2, false); //always dense
		out.allocateDenseBlock();
		
		double[] a = in.getDenseArray();
		double[] c = out.getDenseArray();		
		
		switch( optype )
		{
			case KAHAN_SUM: //SUM via k+
			{
				KahanObject kbuff = new KahanObject(0, 0);
				
				if( ixFn instanceof ReduceAll ) // SUM
					d_uakp(a, c, m, n, kbuff, (KahanPlus)vFn);
				else if( ixFn instanceof ReduceCol ) //ROWSUM
					d_uarkp(a, c, m, n, kbuff, (KahanPlus)vFn);
				else if( ixFn instanceof ReduceRow ) //COLSUM
					d_uackp(a, c, m, n, kbuff, (KahanPlus)vFn);
				break;
			}
			case MAX: 
			case MIN: //MAX/MIN
			{
				double init = Double.MAX_VALUE * ((optype==AggType.MAX)?-1:1);
				
				if( ixFn instanceof ReduceAll ) // MIN/MAX
					d_uamxx(a, c, m, n, init, (Builtin)vFn);
				else if( ixFn instanceof ReduceCol ) //ROWMIN/ROWMAX
					d_uarmxx(a, c, m, n, init, (Builtin)vFn);
				else if( ixFn instanceof ReduceRow ) //COLMIN/COLMAX
					d_uacmxx(a, c, m, n, init, (Builtin)vFn);
				
				break;
			}
			case MAX_INDEX:
			{
				double init = -Double.MAX_VALUE;
				
				if( ixFn instanceof ReduceCol ) //ROWINDEXMAX
					d_uarimxx(a, c, m, n, init, (Builtin)vFn);
				break;
			}
			case MEAN: //MEAN
			{
				KahanObject kbuff = new KahanObject(0, 0);
				
				if( ixFn instanceof ReduceAll ) // MEAN
					d_uamean(a, c, m, n, kbuff, (Mean)vFn);
				else if( ixFn instanceof ReduceCol ) //ROWMEAN
					d_uarmean(a, c, m, n, kbuff, (Mean)vFn);
				else if( ixFn instanceof ReduceRow ) //COLMEAN
					d_uacmean(a, c, m, n, kbuff, (Mean)vFn);
				
				break;
			}
			case PROD: //PROD 
			{
				if( ixFn instanceof ReduceAll ) // PROD
					d_uam(a, c, m, n );
				break;
			}
		}
		
		//cleanup output and change representation (if necessary)
		out.recomputeNonZeros();
		out.examSparsity();
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param vFn
	 * @param ixFn
	 * @throws DMLRuntimeException
	 */
	private static void aggregateUnaryMatrixSparse(MatrixBlockDSM in, MatrixBlockDSM out, AggType optype, ValueFunction vFn, IndexFunction ixFn) 
			throws DMLRuntimeException
	{
		if( in.sparseRows==null )
			return;
		
		final int m = in.rlen;
		final int n = in.clen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		
		//allocate output arrays (if required)
		out.reset(m2, n2, false); //always dense
		out.allocateDenseBlock();
		
		SparseRow[] a = in.getSparseRows();
		double[] c = out.getDenseArray();
		
		switch( optype )
		{
			case KAHAN_SUM: //SUM via k+
			{
				KahanObject kbuff = new KahanObject(0, 0);
				
				if( ixFn instanceof ReduceAll ) // SUM
					s_uakp(a, c, m, n, kbuff, (KahanPlus)vFn);
				else if( ixFn instanceof ReduceCol ) //ROWSUM
					s_uarkp(a, c, m, n, kbuff, (KahanPlus)vFn);
				else if( ixFn instanceof ReduceRow ) //COLSUM
					s_uackp(a, c, m, n, kbuff, (KahanPlus)vFn);
				break;
			}
			case MIN:
			case MAX: //MAX/MIN
			{
				double init = Double.MAX_VALUE * ((optype==AggType.MAX)?-1:1);
				
				if( ixFn instanceof ReduceAll ) // MIN/MAX
					s_uamxx(a, c, m, n, init, (Builtin)vFn);
				else if( ixFn instanceof ReduceCol ) //ROWMIN/ROWMAX
					s_uarmxx(a, c, m, n, init, (Builtin)vFn);
				else if( ixFn instanceof ReduceRow ) //COLMIN/COLMAX
					s_uacmxx(a, c, m, n, init, (Builtin)vFn);
				break;
			}
			case MAX_INDEX:
			{
				double init = -Double.MAX_VALUE;
				
				if( ixFn instanceof ReduceCol ) //ROWINDEXMAX
					s_uarimxx(a, c, m, n, init, (Builtin)vFn);
				break;
			}
			case MEAN:
			{
				KahanObject kbuff = new KahanObject(0, 0);
				
				if( ixFn instanceof ReduceAll ) // MEAN
					s_uamean(a, c, m, n, kbuff, (Mean)vFn);
				else if( ixFn instanceof ReduceCol ) //ROWMEAN
					s_uarmean(a, c, m, n, kbuff, (Mean)vFn);
				else if( ixFn instanceof ReduceRow ) //COLMEAN
					s_uacmean(a, c, m, n, kbuff, (Mean)vFn);
				
				break;
			}
			case PROD: //PROD 
			{
				if( ixFn instanceof ReduceAll ) // PROD
					s_uam(a, c, m, n );
				break;
			}
		}
		
		//cleanup output and change representation (if necessary)
		out.recomputeNonZeros();
		out.examSparsity();
	}

	
	////////////////////////////////////////////
	// core aggregation functions             //
	////////////////////////////////////////////

	/**
	 * SUM, opcode: uak+, dense input. 
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void d_uakp( double[] a, double[] c, int m, int n, KahanObject kbuff, KahanPlus kplus )
	{
		int len = Math.min(m*n, a.length);
		sum( a, 0, len, kbuff, kplus );		
		c[0] = kbuff._sum;
		c[1] = kbuff._correction;	
	}
	
	/**
	 * ROWSUM, opcode: uark+, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void d_uarkp( double[] a, double[] c, int m, int n, KahanObject kbuff, KahanPlus kplus ) 
	{
		for( int i=0, aix=0, cix=0; i<m; i++, aix+=n, cix+=2 )
		{
			kbuff.set(0, 0); //reset buffer
			sum( a, aix, n, kbuff, kplus );
			c[cix+0] = kbuff._sum;
			c[cix+1] = kbuff._correction;		
		}
	}
	
	/**
	 * COLSUM, opcode: uack+, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void d_uackp( double[] a, double[] c, int m, int n, KahanObject kbuff, KahanPlus kplus ) 
	{
		for( int i=0, aix=0; i<m; i++, aix+=n )
			sumAgg( a, c, aix, 0, n, kbuff, kplus );
	}
	
	/**
	 * MIN/MAX, opcode: uamin/uamax, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void d_uamxx( double[] a, double[] c, int m, int n, double init, Builtin builtin )
	{
		int len = Math.min(m*n, a.length);
		c[0] = builtin(a, 0, init, len, builtin);
	}
	
	/**
	 * ROWMIN/ROWMAX, opcode: uarmin/uarmax, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param builtin
	 */
	private static void d_uarmxx( double[] a, double[] c, int m, int n, double init, Builtin builtin )
	{
		for( int i=0, aix=0; i<m; i++, aix+=n )
			c[i] = builtin(a, aix, init, n, builtin);
	}
	
	/**
	 * COLMIN/COLMAX, opcode: uacmin/uacmax, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param builtin
	 */
	private static void d_uacmxx( double[] a, double[] c, int m, int n, double init, Builtin builtin )
	{
		//init output (base for incremental agg)
		Arrays.fill(c, init);
		
		//execute builtin aggregate
		for( int i=0, aix=0; i<m; i++, aix+=n )
			builtinAgg( a, c, aix, n, builtin );
	}

	/**
	 * ROWINDEXMAX, opcode: uarimax, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param init
	 * @param builtin
	 */
	private static void d_uarimxx( double[] a, double[] c, int m, int n, double init, Builtin builtin )
	{
		for( int i=0, aix=0, cix=0; i<m; i++, aix+=n, cix+=2 )
		{
			int maxindex = indexmax(a, aix, init, n, builtin);
			c[cix+0] = (double)maxindex + 1;
			c[cix+1] = a[aix+maxindex]; //max value
		}
	}
	
	/**
	 * MEAN, opcode: uamean, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void d_uamean( double[] a, double[] c, int m, int n, KahanObject kbuff, Mean kmean )
	{
		int len = Math.min(m*n, a.length);
		mean(a, 0, len, 0, kbuff, kmean);
		c[0] = kbuff._sum;
		c[1] = len;
		c[2] = kbuff._correction;
	}
	
	/**
	 * ROWMEAN, opcode: uarmean, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void d_uarmean( double[] a, double[] c, int m, int n, KahanObject kbuff, Mean kmean )
	{
		for( int i=0, aix=0, cix=0; i<m; i++, aix+=n, cix+=3 )
		{
			kbuff.set(0, 0); //reset buffer
			mean(a, aix, n, 0, kbuff, kmean);
			c[cix+0] = kbuff._sum;
			c[cix+1] = n;
			c[cix+2] = kbuff._correction;	
		}
	}
	
	/**
	 * COLMEAN, opcode: uacmean, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param builtin
	 */
	private static void d_uacmean( double[] a, double[] c, int m, int n, KahanObject kbuff, Mean kmean )
	{
		//init output (base for incremental agg)
		Arrays.fill(c, 0); //including counts
		
		//execute builtin aggregate
		for( int i=0, aix=0; i<m; i++, aix+=n )
			meanAgg( a, c, aix, 0, n, kbuff, kmean );
	}
	
	
	/**
	 * PROD, opcode: ua*, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 */
	private static void d_uam( double[] a, double[] c, int m, int n )
	{
		int len = Math.min(m*n, a.length);
		c[0] = product( a, len );	
	}
	
	
	/**
	 * SUM, opcode: uak+, sparse input. 
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void s_uakp( SparseRow[] a, double[] c, int m, int n, KahanObject kbuff, KahanPlus kplus )
	{
		for( int i=0; i<m; i++ )
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				sum(avals, 0, alen, kbuff, kplus);
			}
		}
		c[0] = kbuff._sum;
		c[1] = kbuff._correction;	
	}
	
	/**
	 * ROWSUM, opcode: uark+, sparse input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void s_uarkp( SparseRow[] a, double[] c, int m, int n, KahanObject kbuff, KahanPlus kplus ) 
	{
		//init result (for empty rows)
		Arrays.fill(c, 0); 
		
		//compute row aggregates
		for( int i=0, cix=0; i<m; i++, cix+=2 )
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				kbuff.set(0, 0); //reset buffer
				sum( avals, 0, alen, kbuff, kplus );
				c[cix+0] = kbuff._sum;
				c[cix+1] = kbuff._correction;			
			}
		}
	}
	
	/**
	 * COLSUM, opcode: uack+, sparse input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void s_uackp( SparseRow[] a, double[] c, int m, int n, KahanObject kbuff, KahanPlus kplus ) 
	{
		//init result (for empty columns)
		Arrays.fill(c, 0); 
				
		//compute column aggregates
		for( int i=0; i<m; i++ )
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				int[] aix = arow.getIndexContainer();
				sumAgg( avals, c, aix, alen, n, kbuff, kplus );
			}
		}
	}
	
	/**
	 * MIN/MAX, opcode: uamin/uamax, sparse input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param init
	 * @param builtin
	 */
	private static void s_uamxx( SparseRow[] a, double[] c, int m, int n, double init, Builtin builtin )
	{
		double ret = init; //keep init val
		for( int i=0; i<m; i++ )
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				double lval = builtin(avals, 0, init, alen, builtin);
				ret = builtin.execute2(ret, lval);
			}
		
			//correction (not sparse-safe)
			if( arow==null || arow.size()<m )
				ret = builtin.execute2(ret, 0); 
		}
		c[0] = ret; 
	}
	
	/**
	 * ROWMIN/ROWMAX, opcode: uarmin/uarmax, sparse input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param init
	 * @param builtin
	 */
	private static void s_uarmxx( SparseRow[] a, double[] c, int m, int n, double init, Builtin builtin ) 
	{
		//init result (for empty rows)
		Arrays.fill(c, init); //not sparse-safe
		
		for( int i=0; i<m; i++ )
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				c[ i ] = builtin(avals, 0, init, alen, builtin);
			}
			//correction (not sparse-safe)
			if( arow==null || arow.size()<n )
				c[ i ] = builtin.execute2(c[ i ], 0); 
		}
	}
	
	/**
	 * COLMIN/COLMAX, opcode: uacmin/uacmax, sparse input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param init
	 * @param builtin
	 */
	private static void s_uacmxx( SparseRow[] a, double[] c, int m, int n, double init, Builtin builtin ) 
	{
		//init output (base for incremental agg)
		Arrays.fill(c, init);
		
		//init count arrays (helper, see correction)
		int[] cnt = new int[ n ]; 
		Arrays.fill(cnt, 0); //init count array
		
		//compute column aggregates min/max
		for( int i=0; i<m; i++ )
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				int[] aix = arow.getIndexContainer();
				builtinAgg( avals, c, aix, alen, builtin );
				countAgg( avals, cnt, aix, alen );
			}
		}
		
		//correction (not sparse-safe)
		//note: we need to determine if there are only nnz in a column
		// in order to know if for example a colMax of -8 is true or need
		// to be replaced with a 0 because there was a missing nonzero. 
		for( int i=0; i<n; i++ )
			if( cnt[i] < m ) //no dense column
				c[i] = builtin.execute2(c[i], 0);	
	}

	private static void s_uarimxx( SparseRow[] a, double[] c, int m, int n, double init, Builtin builtin ) 
	{
		for( int i=0, cix=0; i<m; i++, cix+=2 )
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				int[] aix = arow.getIndexContainer();
				int maxindex = indexmax(avals, 0, init, alen, builtin);
				c[cix+0] = (double)aix[maxindex] + 1;
				c[cix+1] = avals[maxindex]; //max value
				
				//correction (not sparse-safe)	
				if(alen < n && (builtin.execute2( 0, c[cix+1] ) == 1))
				{
					int ix = n-1; //find last 0 value
					for( int j=alen-1; j>=0; j--, ix-- )
						if( aix[j]!=ix )
							break;
					c[cix+0] = ix + 1; //max index (last)
					c[cix+1] = 0; //max value
				}
			}
			else //if( arow==null )
			{
				//correction (not sparse-safe)	
				c[cix+0] = n; //max index (last)
				c[cix+1] = 0; //max value
			}
		}
	}
	
	/**
	 * MEAN, opcode: uamean, sparse input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void s_uamean( SparseRow[] a, double[] c, int m, int n, KahanObject kbuff, Mean kmean )
	{
		int len = m * n;
		int count = 0;
		
		//correction remaining tuples (not sparse-safe)
		//note: before aggregate computation in order to
		//exploit 0 sum (noop) and better numerical stability
		for( int i=0; i<m; i++ )
			count += (a[i]==null)? n : n-a[i].size();
		
		//compute aggregate mean
		for( int i=0; i<m; i++ )
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				mean(avals, 0, alen, count, kbuff, kmean);
				count += alen;
			}
		}

		//OLD VERSION: correction remaining tuples (not sparse-safe)
		//mean(0, len-count, count, kbuff, kplus);
		
		c[0] = kbuff._sum;
		c[1] = len;
		c[2] = kbuff._correction;
	}
	
	/**
	 * ROWMEAN, opcode: uarmean, sparse input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void s_uarmean( SparseRow[] a, double[] c, int m, int n, KahanObject kbuff, Mean kmean ) 
	{
		for( int i=0, cix=0; i<m; i++, cix+=3 )
		{
			//correction remaining tuples (not sparse-safe)
			//note: before aggregate computation in order to
			//exploit 0 sum (noop) and better numerical stability
			int count = (a[i]==null)? n : n-a[i].size();
			
			kbuff.set(0, 0); //reset buffer
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				mean(avals, 0, alen, count, kbuff, kmean);
			}
			
			//OLD VERSION: correction remaining tuples (not sparse-safe)
			//int count = ((arow==null) ? 0 : arow.size());
			//mean(0, n-count, count, kbuff, kplus);
			
			c[cix+0] = kbuff._sum;
			c[cix+1] = n;
			c[cix+2] = kbuff._correction;
		}
	}
	
	/**
	 * COLMEAN, opcode: uacmean, sparse input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param init
	 * @param kbuff
	 * @param kplus
	 */
	private static void s_uacmean( SparseRow[] a, double[] c, int m, int n, KahanObject kbuff, Mean kmean ) 
	{
		//init output (base for incremental agg)
		Arrays.fill(c, 0);
		
		//correction remaining tuples (not sparse-safe)
		//note: before aggregate computation in order to
		//exploit 0 sum (noop) and better numerical stability
		Arrays.fill(c, n, n*2, m);
		for( int i=0; i<m; i++ ) 
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				int[] aix = arow.getIndexContainer();
				countDisAgg( avals, c, aix, n, alen );
			}
		} 
		
		//compute column aggregate means
		for( int i=0; i<m; i++ )
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				int[] aix = arow.getIndexContainer();
				meanAgg( avals, c, aix, alen, n, kbuff, kmean );
			}
		}
		
		//OLD VERSION: correction remaining tuples (not sparse-safe)
		/*for( int i=0; i<n; i++ ) //for all colmeans
		{
			int count = (int) c[n+i];
			kbuff.set(c[i], c[2*n+i]); //reset buffer
			mean(0, m-count, count, kbuff, kplus);
			
			c[i]     = kbuff._sum;
			c[i+n]   = m;
			c[i+2*n] = kbuff._correction;	
		}*/
	}
	
	/**
	 * PROD, opcode: ua*, sparse input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 */
	private static void s_uam( SparseRow[] a, double[] c, int m, int n )
	{
		double ret = 1;
		for( int i=0; i<m; i++ )
		{
			SparseRow arow = a[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				ret *= product(avals, alen);
				ret *= (alen<n) ? 0 : 1;
			}
			
			//early abort (note: in case of NaNs this is an invalid optimization)
			if( !NAN_AWARENESS && ret==0 ) break;
		}
		c[0] = ret;
	}
	
	
	////////////////////////////////////////////
	// performance-relevant utility functions //
	////////////////////////////////////////////
	
	/**
	 * 
	 * @param a
	 * @param ai
	 * @param len
	 * @param kbuff
	 * @param kplus
	 */
	private static void sum( double[] a, int ai, final int len, KahanObject kbuff, KahanPlus kplus )
	{
		for( int i=0; i<len; i++, ai++ )
			kplus.execute2(kbuff, a[ ai ]);
	}
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param ai
	 * @param ci
	 * @param len
	 * @param kbuff
	 * @param kplus
	 */
	private static void sumAgg( double[] a, double[] c, int ai, int ci, final int len, KahanObject kbuff, KahanPlus kplus )
	{
		for( int i=0; i<len; i++, ai++, ci++ )
		{
			kbuff._sum        = c[ci];
			kbuff._correction = c[ci+len];
			kplus.execute2(kbuff, a[ai]);
			c[ci]     = kbuff._sum;
			c[ci+len] = kbuff._correction;
		}
	}
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param ai
	 * @param len
	 * @param clen
	 * @param kbuff
	 * @param kplus
	 */
	private static void sumAgg( double[] a, double[] c, int[] ai, final int len, final int n, KahanObject kbuff, KahanPlus kplus )
	{
		for( int i=0; i<len; i++ )
		{
			kbuff._sum        = c[ ai[i] ];
			kbuff._correction = c[ ai[i]+n ];
			kplus.execute2( kbuff, a[i] );
			c[ ai[i] ]      = kbuff._sum;
			c[ ai[i]+n ] = kbuff._correction;
		}
	}
	
	/**
	 * 
	 * @param a
	 * @param len
	 * @return
	 */
	private static double product( double[] a, final int len )
	{
		double val = 1;
		
		if( NAN_AWARENESS )
		{
			//product without early abort
			//even if val is 0, it might turn into NaN.
			for( int i=0; i<len; i++ )
				val *= a[ i ];	
		}
		else
		{
			//product with early abort (if 0)
			//note: this will not work with NaNs (invalid optimization)
			for( int i=0; i<len && val!=0; i++ )
				val *= a[ i ];
		}
		
		return val;
	}
	
	/**
	 * 
	 * @param a
	 * @param ai
	 * @param len
	 * @param kbuff
	 * @param kplus
	 */
	private static void mean( double[] a, int ai, final int len, int count, KahanObject kbuff, Mean mean )
	{
		for( int i=0; i<len; i++, ai++, count++ )
		{
			//delta: (newvalue-buffer._sum)/count
			mean.execute2(kbuff, a[ai], count+1);
		}
	}
	
	/*
	private static void mean( final double aval, final int len, int count, KahanObject kbuff, KahanPlus kplus )
	{
		for( int i=0; i<len; i++, count++ )
		{
			//delta: (newvalue-buffer._sum)/count
			kplus.execute2(kbuff, (aval-kbuff._sum)/(count+1));
		}
	}
	*/
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param ai
	 * @param ci
	 * @param len
	 * @param kbuff
	 * @param kplus
	 */
	private static void meanAgg( double[] a, double[] c, int ai, int ci, final int len, KahanObject kbuff, Mean mean )
	{
		for( int i=0; i<len; i++, ai++, ci++ )
		{
			kbuff._sum        = c[ci];
			double count      = c[ci+len] + 1;
			kbuff._correction = c[ci+2*len];
			mean.execute2(kbuff, a[ai], count);
			c[ci]       = kbuff._sum;
			c[ci+len]   = count;
			c[ci+2*len] = kbuff._correction;
		}
	}
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param ai
	 * @param len
	 * @param kbuff
	 * @param kplus
	 */
	private static void meanAgg( double[] a, double[] c, int[] ai, final int len, final int n, KahanObject kbuff, Mean mean )
	{
		for( int i=0; i<len; i++ )
		{
			kbuff._sum        = c[ai[i]];
			double count      = c[ai[i]+n] + 1;
			kbuff._correction = c[ai[i]+2*n];
			mean.execute2(kbuff, a[ i ], count);
			c[ai[i]]     = kbuff._sum;
			c[ai[i]+n]   = count;
			c[ai[i]+2*n] = kbuff._correction;
		}
	}
	
	/**
	 * Meant for builtin function ops (min, max) 
	 * 
	 * @param a
	 * @param ai
	 * @param init
	 * @param len
	 * @param aggop
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static double builtin( double[] a, int ai, final double init, final int len, Builtin aggop ) 
	{
		double val = init;
		for( int i=0; i<len; i++, ai++ )
			val = aggop.execute2( val, a[ ai ] );
		
		return val;
	}
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param ai
	 * @param len
	 * @param aggop
	 */
	private static void builtinAgg( double[] a, double[] c, int ai, final int len, Builtin aggop ) 
	{
		for( int i=0; i<len; i++, ai++ )
			c[ i ] = aggop.execute2( c[ i ], a[ ai ] );
	}
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param ai
	 * @param len
	 * @param aggop
	 */
	private static void builtinAgg( double[] a, double[] c, int[] ai, final int len, Builtin aggop ) 
	{
		for( int i=0; i<len; i++ )
			c[ ai[i] ] = aggop.execute2( c[ ai[i] ], a[ i ] );
	}
	
	/**
	 * 
	 * @param a
	 * @param ai
	 * @param init
	 * @param len
	 * @param aggop
	 * @return
	 */
	private static int indexmax( double[] a, int ai, final double init, final int len, Builtin aggop ) 
	{
		double maxval = init;
		int maxindex = -1;
		for( int i=0; i<len; i++ )
		{
			double val = a[ai+i];
			if( aggop.execute2( val, maxval ) == 1 )
			{
				maxval = val;
				maxindex = i;
			}
		}
		
		return maxindex;
	}
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param ai
	 * @param len
	 */
	private static void countAgg( double[] a, int[] c, int[] ai, final int len ) 
	{
		final int bn = len%8;
		
		//compute rest, not aligned to 8-block
		for( int i=0; i<bn; i++ )
			c[ ai[i] ]++;
		
		//unrolled 8-block (for better instruction level parallelism)
		for( int i=bn; i<len; i+=8 )
		{
			c[ ai[ i+0 ] ] ++;
			c[ ai[ i+1 ] ] ++;
			c[ ai[ i+2 ] ] ++;
			c[ ai[ i+3 ] ] ++;
			c[ ai[ i+4 ] ] ++;
			c[ ai[ i+5 ] ] ++;
			c[ ai[ i+6 ] ] ++;
			c[ ai[ i+7 ] ] ++;
		}
	}
	
	private static void countDisAgg( double[] a, double[] c, int[] ai, final int ci, final int len ) 
	{
		final int bn = len%8;
		
		//compute rest, not aligned to 8-block
		for( int i=0; i<bn; i++ )
			c[ ci+ai[i] ]--;
		
		//unrolled 8-block (for better instruction level parallelism)
		for( int i=bn; i<len; i+=8 )
		{
			c[ ci+ai[ i+0 ] ] --;
			c[ ci+ai[ i+1 ] ] --;
			c[ ci+ai[ i+2 ] ] --;
			c[ ci+ai[ i+3 ] ] --;
			c[ ci+ai[ i+4 ] ] --;
			c[ ci+ai[ i+5 ] ] --;
			c[ ci+ai[ i+6 ] ] --;
			c[ ci+ai[ i+7 ] ] --;
		}
	}
	
	
	/*
	NOTE: those low level optimizations improved performance after JIT by 2x-3x.
	However, the runtime BEFORE JIT compile was significantly increased (e.g.
	8kx8k sum, 6s before JIT, 130ms after JIT). NON-CONCLUSIVE.
	
	private static KahanObject[] createBuffers( int num )
	{
		KahanObject[] buff = new KahanObject[ num ];
		
		for( int i=0; i<num; i++ )
			buff[i] = new KahanObject(0,0);
		
		return buff;
	}
	
	private static void resetBuffers( KahanObject[] buff)
	{
		int num = buff.length;
		for( int i=0; i<num; i++ )
			buff[i].set(0, 0);
	}
	
	private static void sum8( double[] a, int ai, int len, KahanObject kbuff, KahanObject[] lbuff8, KahanPlus kplus )
	{
		final int bn = len%8;
		
		for( int i=0; i<bn; i++, ai++ )
			kplus.execute2(kbuff, a[ai]);
		
		//unrolled 8-block (for better instruction level parallelism)
		for( int i=bn; i<len; i+=8, ai+=8 )
		{
			kplus.execute2( lbuff8[0], a[ ai+0 ] );
			kplus.execute2( lbuff8[1], a[ ai+1 ] );
			kplus.execute2( lbuff8[2], a[ ai+2 ] );
			kplus.execute2( lbuff8[3], a[ ai+3 ] );
			kplus.execute2( lbuff8[4], a[ ai+4 ] );
			kplus.execute2( lbuff8[5], a[ ai+5 ] );
			kplus.execute2( lbuff8[6], a[ ai+6 ] );
			kplus.execute2( lbuff8[7], a[ ai+7 ] );
		}
		
		//merge local buffers into return buffer
		for( int i=0; i<8; i++ )
			kplus.execute2(kbuff, lbuff8[i]._sum+lbuff8[i]._correction);
	}
	*/
}

