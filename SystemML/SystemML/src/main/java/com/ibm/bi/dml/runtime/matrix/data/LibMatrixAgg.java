/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.data;

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
import com.ibm.bi.dml.runtime.functionobjects.ReduceDiag;
import com.ibm.bi.dml.runtime.functionobjects.ReduceRow;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.instructions.cp.KahanObject;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;
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
 * ua*, uamean, uarmean, uacmean, uarimax, uaktrace.
 * 
 * TODO next opcode extensions: a+, colindexmax
 * TODO low level optimization (potential 3x, sum non-conclusive yet)
 */
public class LibMatrixAgg 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	//internal configuration parameters
	private static final boolean NAN_AWARENESS = false;

	////////////////////////////////
	// public matrix agg interface
	////////////////////////////////
	
	private enum AggType {
		KAHAN_SUM,
		CUM_KAHAN_SUM,
		MIN,
		MAX,
		MEAN,
		MAX_INDEX,
		MIN_INDEX,
		PROD,
		INVALID,
	}

	private LibMatrixAgg() {
		//prevent instantiation via private constructor
	}
	
	/**
	 * Core incremental matrix aggregate (ak+) as used in mapmult, tsmm, 
	 * cpmm, etc. Note that we try to keep the current 
	 * aggVal and aggCorr in dense format in order to allow efficient
	 * access according to the dense/sparse input. 
	 * 
	 * 
	 * @param in input matrix
	 * @param aggVal current aggregate values (in/out)
	 * @param aggCorr current aggregate correction (in/out)
	 * @throws DMLRuntimeException 
	 */
	public static void aggregateBinaryMatrix(MatrixBlock in, MatrixBlock aggVal, MatrixBlock aggCorr) 
		throws DMLRuntimeException
	{	
		//Timing time = new Timing(true);
		//boolean saggVal = aggVal.isInSparseFormat(), saggCorr = aggCorr.isInSparseFormat(); 
		//long naggVal = aggVal.getNonZeros(), naggCorr = aggCorr.getNonZeros();
		
		//core aggregation
		if(!in.sparse && !aggVal.sparse && !aggCorr.sparse)
			aggregateBinaryMatrixAllDense(in, aggVal, aggCorr);
		else if(in.sparse && !aggVal.sparse && !aggCorr.sparse)
			aggregateBinaryMatrixSparseDense(in, aggVal, aggCorr);
		else if(in.sparse ) //any aggVal, aggCorr
			aggregateBinaryMatrixSparseGeneric(in, aggVal, aggCorr);
		else //if( !in.sparse ) //any aggVal, aggCorr
			aggregateBinaryMatrixDenseGeneric(in, aggVal, aggCorr);
		
		//System.out.println("agg ("+in.rlen+","+in.clen+","+in.getNonZeros()+","+in.sparse+"), " +
		//		          "("+naggVal+","+saggVal+"), ("+naggCorr+","+saggCorr+") -> " +
		//		          "("+aggVal.getNonZeros()+","+aggVal.isInSparseFormat()+"), ("+aggCorr.getNonZeros()+","+aggCorr.isInSparseFormat()+") " +
		//		          "in "+time.stop()+"ms.");
	}
	
	/**
	 * Core incremental matrix aggregate (ak+) as used for uack+ and acrk+.
	 * Embedded correction values.
	 * 
	 * @param in
	 * @param aggVal
	 * @throws DMLRuntimeException
	 */
	public static void aggregateBinaryMatrix(MatrixBlock in, MatrixBlock aggVal, AggregateOperator aop) 
		throws DMLRuntimeException
	{	
		//Timing time = new Timing(true);
		//boolean saggVal = aggVal.isInSparseFormat(); 
		//long naggVal = aggVal.getNonZeros();
		
		//core aggregation
		boolean lastRowCorr = (aop.correctionLocation == CorrectionLocationType.LASTROW);
		boolean lastColCorr = (aop.correctionLocation == CorrectionLocationType.LASTCOLUMN);
		if( !in.sparse && lastRowCorr )
			aggregateBinaryMatrixLastRowDenseGeneric(in, aggVal);
		else if( in.sparse && lastRowCorr )
			aggregateBinaryMatrixLastRowSparseGeneric(in, aggVal);
		else if( !in.sparse && lastColCorr )
			aggregateBinaryMatrixLastColDenseGeneric(in, aggVal);
		else //if( in.sparse && lastColCorr )
			aggregateBinaryMatrixLastColSparseGeneric(in, aggVal);
		
		//System.out.println("agg ("+in.rlen+","+in.clen+","+in.getNonZeros()+","+in.sparse+"), ("+naggVal+","+saggVal+") -> " +
		//                   "("+aggVal.getNonZeros()+","+aggVal.isInSparseFormat()+") in "+time.stop()+"ms.");
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param vFn
	 * @param ixFn
	 * @throws DMLRuntimeException
	 */
	public static void aggregateUnaryMatrix(MatrixBlock in, MatrixBlock out, AggregateUnaryOperator uaop) 
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
	 * @param in
	 * @param out
	 * @param uop
	 * @throws DMLRuntimeException
	 */
	public static void aggregateUnaryMatrix(MatrixBlock in, MatrixBlock out, UnaryOperator uop) 
		throws DMLRuntimeException
	{
		//Timing time = new Timing(true);
		
		AggType aggtype = getAggType(uop);
		if( !in.sparse )
			aggregateUnaryMatrixDense(in, out, aggtype, null, null);
		else
			aggregateUnaryMatrixSparse(in, out, aggtype, null, null);
		
		//System.out.println("uop ("+in.rlen+","+in.clen+","+in.sparse+") in "+time.stop()+"ms.");
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
	
	public static boolean isSupportedUnaryOperator( UnaryOperator op )
	{
		AggType type = getAggType( op );
		return (type != AggType.INVALID);
	}
	
	/**
	 * Recompute outputs (e.g., maxindex or minindex) according to block indexes from MR.
	 * TODO: this should not be part of block operations but of the MR instruction.
	 * 
	 * @param out
	 * @param op
	 * @param brlen
	 * @param bclen
	 * @param ix
	 */
	public static void recomputeIndexes( MatrixBlock out, AggregateUnaryOperator op, int brlen, int bclen, MatrixIndexes ix )
	{
		AggType type = getAggType(op);
		if( (type == AggType.MAX_INDEX || type == AggType.MIN_INDEX) && ix.getColumnIndex()!=1 ) //MAXINDEX or MININDEX
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
		
		//(kahan) sum / trace (for ReduceDiag)
		if( vfn instanceof KahanPlus 
			&& (op.aggOp.correctionLocation == CorrectionLocationType.LASTCOLUMN ||op.aggOp.correctionLocation == CorrectionLocationType.LASTROW ) 
			&& (ifn instanceof ReduceAll || ifn instanceof ReduceCol || ifn instanceof ReduceRow || ifn instanceof ReduceDiag) )
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
				case MININDEX: return AggType.MIN_INDEX;
				default: //do nothing
			}
		}
		
		return AggType.INVALID;
	}
	
	/**
	 * 
	 * @param op
	 * @return
	 */
	private static AggType getAggType( UnaryOperator op )
	{
		ValueFunction vfn = op.fn;
	
		//cumsum
		if( vfn instanceof Builtin && ((Builtin) vfn).bFunc == BuiltinFunctionCode.CUMSUM )
		{
			return AggType.CUM_KAHAN_SUM;
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
	private static void aggregateBinaryMatrixAllDense(MatrixBlock in, MatrixBlock aggVal, MatrixBlock aggCorr) 
			throws DMLRuntimeException
	{
		if( in.denseBlock==null || in.isEmptyBlock(false) )
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
		//aggVal.examSparsity();
		//aggCorr.examSparsity(); 
	}

	/**
	 * 
	 * @param in
	 * @param aggVal
	 * @param aggCorr
	 * @throws DMLRuntimeException
	 */
	private static void aggregateBinaryMatrixSparseDense(MatrixBlock in, MatrixBlock aggVal, MatrixBlock aggCorr) 
			throws DMLRuntimeException
	{
		if( in.sparseRows==null || in.isEmptyBlock(false) )
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
		
		for( int i=0, cix=0; i<rlen; i++, cix+=n )
		{
			SparseRow arow = a[i];
			if( arow!=null && !arow.isEmpty() )
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
				}
			}
		}
		
		aggVal.recomputeNonZeros();
		aggCorr.recomputeNonZeros();
		//aggVal.examSparsity();
		//aggCorr.examSparsity(); 
	}
	
	/**
	 * 
	 * @param in
	 * @param aggVal
	 * @param aggCorr
	 * @throws DMLRuntimeException
	 */
	private static void aggregateBinaryMatrixSparseGeneric(MatrixBlock in, MatrixBlock aggVal, MatrixBlock aggCorr) 
			throws DMLRuntimeException
	{
		if( in.sparseRows==null || in.isEmptyBlock(false) )
			return;
		
		SparseRow[] a = in.getSparseRows();
		
		KahanObject buffer1 = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		final int m = in.rlen;
		final int rlen = Math.min(a.length, m);
		
		for( int i=0; i<rlen; i++ )
		{
			SparseRow arow = a[i];
			if( arow!=null && !arow.isEmpty() )
			{
				int alen = arow.size();
				int[] aix = arow.getIndexContainer();
				double[] avals = arow.getValueContainer();
				
				for( int j=0; j<alen; j++ )
				{
					int jix = aix[j];
					buffer1._sum        = aggVal.quickGetValue(i, jix);
					buffer1._correction = aggCorr.quickGetValue(i, jix);
					akplus.execute2(buffer1, avals[j]);
					aggVal.quickSetValue(i, jix, buffer1._sum);
					aggCorr.quickSetValue(i, jix, buffer1._correction);
				}
			}
		}
		
		//note: nnz of aggVal/aggCorr maintained internally 
		aggVal.examSparsity();
		aggCorr.examSparsity(); 
	}
	
	
	/**
	 * 
	 * @param in
	 * @param aggVal
	 * @param aggCorr
	 * @throws DMLRuntimeException
	 */
	private static void aggregateBinaryMatrixDenseGeneric(MatrixBlock in, MatrixBlock aggVal, MatrixBlock aggCorr) 
		throws DMLRuntimeException
	{	
		if( in.denseBlock==null || in.isEmptyBlock(false) )
			return;
		
		final int m = in.rlen;
		final int n = in.clen;
		
		double[] a = in.getDenseArray();
		
		KahanObject buffer = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		//incl implicit nnz maintenance
		for(int i=0, ix=0; i<m; i++)
			for(int j=0; j<n; j++, ix++)
			{
				buffer._sum = aggVal.quickGetValue(i, j);
				buffer._correction = aggCorr.quickGetValue(i, j);
				akplus.execute(buffer, a[ix]);
				aggVal.quickSetValue(i, j, buffer._sum);
				aggCorr.quickSetValue(i, j, buffer._correction);
			}
		
		//note: nnz of aggVal/aggCorr maintained internally 
		aggVal.examSparsity();
		aggCorr.examSparsity();
	}
	
	/**
	 * 
	 * @param in
	 * @param aggVal
	 * @throws DMLRuntimeException
	 */
	private static void aggregateBinaryMatrixLastRowDenseGeneric(MatrixBlock in, MatrixBlock aggVal) 
			throws DMLRuntimeException
	{
		if( in.denseBlock==null || in.isEmptyBlock(false) )
			return;
		
		final int m = in.rlen;
		final int n = in.clen;
		final int cix = (m-1)*n;
		
		double[] a = in.getDenseArray();
		
		KahanObject buffer = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		//incl implicit nnz maintenance
		for(int i=0, ix=0; i<m-1; i++)
			for(int j=0; j<n; j++, ix++)
			{
				buffer._sum = aggVal.quickGetValue(i, j);
				buffer._correction = aggVal.quickGetValue(m-1, j);
				akplus.execute(buffer, a[ix], a[cix+j]);
				aggVal.quickSetValue(i, j, buffer._sum);
				aggVal.quickSetValue(m-1, j, buffer._correction);
			}
		
		//note: nnz of aggVal maintained internally 
		aggVal.examSparsity();
	}
	
	/**
	 * 
	 * @param in
	 * @param aggVal
	 * @throws DMLRuntimeException
	 */
	private static void aggregateBinaryMatrixLastRowSparseGeneric(MatrixBlock in, MatrixBlock aggVal) 
			throws DMLRuntimeException
	{
		//sparse-safe operation
		if( in.sparseRows==null || in.isEmptyBlock(false) )
			return;
		
		SparseRow[] a = in.getSparseRows();
		
		KahanObject buffer1 = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		final int m = in.rlen;
		final int rlen = Math.min(a.length, m);
		
		for( int i=0; i<rlen-1; i++ )
		{
			SparseRow arow = a[i];
			if( arow!=null && !arow.isEmpty() )
			{
				int alen = arow.size();
				int[] aix = arow.getIndexContainer();
				double[] avals = arow.getValueContainer();
				
				for( int j=0; j<alen; j++ )
				{
					int jix = aix[j];
					double corr = in.quickGetValue(m-1, jix);
					buffer1._sum        = aggVal.quickGetValue(i, jix);
					buffer1._correction = aggVal.quickGetValue(m-1, jix);
					akplus.execute(buffer1, avals[j], corr);
					aggVal.quickSetValue(i, jix, buffer1._sum);
					aggVal.quickSetValue(m-1, jix, buffer1._correction);
				}
			}
		}
		
		//note: nnz of aggVal/aggCorr maintained internally 
		aggVal.examSparsity(); 
	}
	
	/**
	 * 
	 * @param in
	 * @param aggVal
	 * @throws DMLRuntimeException
	 */
	private static void aggregateBinaryMatrixLastColDenseGeneric(MatrixBlock in, MatrixBlock aggVal) 
			throws DMLRuntimeException
	{
		if( in.denseBlock==null || in.isEmptyBlock(false) )
			return;
		
		final int m = in.rlen;
		final int n = in.clen;
		
		double[] a = in.getDenseArray();
		
		KahanObject buffer = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		//incl implicit nnz maintenance
		for(int i=0, ix=0; i<m; i++, ix+=n)
			for(int j=0; j<n-1; j++)
			{
				buffer._sum = aggVal.quickGetValue(i, j);
				buffer._correction = aggVal.quickGetValue(i, n-1);
				akplus.execute(buffer, a[ix+j], a[ix+j+1]);
				aggVal.quickSetValue(i, j, buffer._sum);
				aggVal.quickSetValue(i, n-1, buffer._correction);
			}
		
		//note: nnz of aggVal maintained internally 
		aggVal.examSparsity();
	}
	
	/**
	 * 
	 * @param in
	 * @param aggVal
	 * @throws DMLRuntimeException
	 */
	private static void aggregateBinaryMatrixLastColSparseGeneric(MatrixBlock in, MatrixBlock aggVal) 
			throws DMLRuntimeException
	{
		//sparse-safe operation
		if( in.sparseRows==null || in.isEmptyBlock(false) )
			return;
		
		SparseRow[] a = in.getSparseRows();
		
		KahanObject buffer1 = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		final int m = in.rlen;
		final int n = in.clen;
		final int rlen = Math.min(a.length, m);
		
		for( int i=0; i<rlen; i++ )
		{
			SparseRow arow = a[i];
			if( arow!=null && !arow.isEmpty() )
			{
				int alen = arow.size();
				int[] aix = arow.getIndexContainer();
				double[] avals = arow.getValueContainer();
				
				for( int j=0; j<alen && aix[j]<n-1; j++ )
				{
					int jix = aix[j];
					double corr = in.quickGetValue(i, n-1);
					buffer1._sum        = aggVal.quickGetValue(i, jix);
					buffer1._correction = aggVal.quickGetValue(i, n-1);
					akplus.execute(buffer1, avals[j], corr);
					aggVal.quickSetValue(i, jix, buffer1._sum);
					aggVal.quickSetValue(i, n-1, buffer1._correction);
				}
			}
		}
		
		//note: nnz of aggVal/aggCorr maintained internally 
		aggVal.examSparsity(); 
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param vFn
	 * @param ixFn
	 * @throws DMLRuntimeException
	 */
	private static void aggregateUnaryMatrixDense(MatrixBlock in, MatrixBlock out, AggType optype, ValueFunction vFn, IndexFunction ixFn) 
			throws DMLRuntimeException
	{
		if( in.denseBlock==null || in.isEmptyBlock(false) ){
			aggregateUnaryMatrixEmpty(in, out, optype, ixFn);
			return;
		}	
		
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
			case KAHAN_SUM: //SUM/TRACE via k+, 
			{
				KahanObject kbuff = new KahanObject(0, 0);
				
				if( ixFn instanceof ReduceAll ) // SUM
					d_uakp(a, c, m, n, kbuff, (KahanPlus)vFn);
				else if( ixFn instanceof ReduceCol ) //ROWSUM
					d_uarkp(a, c, m, n, kbuff, (KahanPlus)vFn);
				else if( ixFn instanceof ReduceRow ) //COLSUM
					d_uackp(a, c, m, n, kbuff, (KahanPlus)vFn);
				else if( ixFn instanceof ReduceDiag ) //TRACE
					d_uakptrace(a, c, m, n, kbuff, (KahanPlus)vFn);
				break;
			}
			case CUM_KAHAN_SUM: //CUMSUM
			{
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				d_ucumkp(a, c, m, n, kbuff, kplus);
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
			case MIN_INDEX:
			{
				double init = Double.MAX_VALUE;
				
				if( ixFn instanceof ReduceCol ) //ROWINDEXMIN
					d_uarimin(a, c, m, n, init, (Builtin)vFn);
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
			
			default:
				throw new DMLRuntimeException("Unsupported aggregation type: "+optype);
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
	private static void aggregateUnaryMatrixSparse(MatrixBlock in, MatrixBlock out, AggType optype, ValueFunction vFn, IndexFunction ixFn) 
			throws DMLRuntimeException
	{
		//filter empty input blocks (incl special handling for sparse-unsafe operations)
		if( in.sparseRows==null || in.isEmptyBlock(false) ){
			aggregateUnaryMatrixEmpty(in, out, optype, ixFn);
			return;
		}
		
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
				else if( ixFn instanceof ReduceDiag ) //TRACE
					s_uakptrace(a, c, m, n, kbuff, (KahanPlus)vFn);
					
				break;
			}
			case CUM_KAHAN_SUM: //CUMSUM
			{
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				s_ucumkp(a, c, m, n, kbuff, kplus);
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
			case MIN_INDEX:
			{
				double init = Double.MAX_VALUE;
				
				if( ixFn instanceof ReduceCol ) //ROWINDEXMAX
					s_uarimin(a, c, m, n, init, (Builtin)vFn);
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

			default:
				throw new DMLRuntimeException("Unsupported aggregation type: "+optype);
		}
		
		//cleanup output and change representation (if necessary)
		out.recomputeNonZeros();
		out.examSparsity();
	}

	/**
	 * 
	 * @param in
	 * @param out
	 * @param optype
	 * @param ixFn
	 * @throws DMLRuntimeException 
	 */
	private static void aggregateUnaryMatrixEmpty(MatrixBlock in, MatrixBlock out, AggType optype, IndexFunction ixFn) 
		throws DMLRuntimeException
	{
		//do nothing for pseudo sparse-safe operations
		if(optype==AggType.KAHAN_SUM || optype==AggType.MIN || optype==AggType.MAX || optype==AggType.PROD 
			|| optype == AggType.CUM_KAHAN_SUM )
		{
			return;
		}
		
		//compute result based on meta data only
		switch( optype )
		{
			case MAX_INDEX:
			{
				if( ixFn instanceof ReduceCol ) { //ROWINDEXMAX
					for(int i=0; i<out.rlen; i++) {
						out.quickSetValue(i, 0, in.clen); //maxindex
					}
				}
				break;
			}
			case MIN_INDEX:
			{
				if( ixFn instanceof ReduceCol ) //ROWINDEXMIN
					for(int i=0; i<out.rlen; i++) {
						out.quickSetValue(i, 0, in.clen); //minindex
					}
				break;
			}
			case MEAN:
			{
				if( ixFn instanceof ReduceAll ) // MEAN
					out.quickSetValue(0, 1, in.rlen*in.clen); //count
				else if( ixFn instanceof ReduceCol ) //ROWMEAN
					for( int i=0; i<in.rlen; i++ ) //0-sum and 0-correction 
						out.quickSetValue(i, 1, in.clen); //count
				else if( ixFn instanceof ReduceRow ) //COLMEAN
					for( int j=0; j<in.clen; j++ ) //0-sum and 0-correction 
						out.quickSetValue(1, j, in.rlen); //count				
				break;
			}
			
			default:
				throw new DMLRuntimeException("Unsupported aggregation type: "+optype);
		}
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
	 * CUMSUM, opcode: ucumk+, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void d_ucumkp( double[] a, double[] c, int m, int n, KahanObject kbuff, KahanPlus kplus ) 
	{
		//init current sum/correction arrays 
		double[] csums = new double[ 2*n ]; 
		Arrays.fill(csums, 0);
		
		//scan once and compute prefix sums
		for( int i=0, aix=0; i<m; i++, aix+=n ) {
			sumAgg( a, csums, aix, 0, n, kbuff, kplus );
			System.arraycopy(csums, 0, c, aix, n);	
		}			
	}
	
	/**
	 * TRACE, opcode: uaktrace 
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void d_uakptrace( double[] a, double[] c, int m, int n, KahanObject kbuff, KahanPlus kplus ) 
	{
		//aggregate diag (via ix=n+1)
		for( int i=0, aix=0; i<m; i++, aix+=(n+1) )
			kplus.execute2(kbuff, a[ aix ]);			
		c[0] = kbuff._sum;
		c[1] = kbuff._correction;	
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
	 * ROWINDEXMIN, opcode: uarimin, dense input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param init
	 * @param builtin
	 */
	private static void d_uarimin( double[] a, double[] c, int m, int n, double init, Builtin builtin )
	{
		for( int i=0, aix=0, cix=0; i<m; i++, aix+=n, cix+=2 )
		{
			int minindex = indexmin(a, aix, init, n, builtin);
			c[cix+0] = (double)minindex + 1;
			c[cix+1] = a[aix+minindex]; //min value
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
			if( arow!=null && !arow.isEmpty() )
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
			if( arow!=null && !arow.isEmpty() )
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
			if( arow!=null && !arow.isEmpty() )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				int[] aix = arow.getIndexContainer();
				sumAgg( avals, c, aix, alen, n, kbuff, kplus );
			}
		}
	}
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void s_ucumkp( SparseRow[] a, double[] c, int m, int n, KahanObject kbuff, KahanPlus kplus )
	{
		//init current sum/correction arrays 
		double[] csums = new double[ 2*n ]; 
		Arrays.fill(csums, 0);
		
		//scan once and compute prefix sums
		for( int i=0, ix=0; i<m; i++, ix+=n )
		{
			SparseRow arow = a[i];
			if( arow!=null && !arow.isEmpty() )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				int[] aix = arow.getIndexContainer();
				sumAgg( avals, csums, aix, alen, n, kbuff, kplus );
			}
			//always copy current sum (not sparse-safe)
			System.arraycopy(csums, 0, c, ix, n);
		}
	}
	
	/**
	 * TRACE, opcode: uaktrace, sparse input.
	 * 
	 * @param a
	 * @param c
	 * @param m
	 * @param n
	 * @param kbuff
	 * @param kplus
	 */
	private static void s_uakptrace( SparseRow[] a, double[] c, int m, int n, KahanObject kbuff, KahanPlus kplus ) 
	{
		for( int i=0; i<m; i++ ) {
			SparseRow arow = a[i];
			if( arow!=null && !arow.isEmpty() ) 
			{
				double val = arow.get( i );
				kplus.execute2(kbuff, val);
			}
		}
		c[0] = kbuff._sum;
		c[1] = kbuff._correction;	
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
			if( arow!=null && !arow.isEmpty() )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				double lval = builtin(avals, 0, init, alen, builtin);
				ret = builtin.execute2(ret, lval);
			}
		
			//correction (not sparse-safe)
			if( arow==null || arow.size()<n )
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
			if( arow!=null && !arow.isEmpty() )
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
			if( arow!=null && !arow.isEmpty() )
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
			if( arow!=null && !arow.isEmpty() )
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
	
	private static void s_uarimin( SparseRow[] a, double[] c, int m, int n, double init, Builtin builtin ) 
	{
		for( int i=0, cix=0; i<m; i++, cix+=2 )
		{
			SparseRow arow = a[i];
			if( arow!=null && !arow.isEmpty() )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				int[] aix = arow.getIndexContainer();
				int minindex = indexmin(avals, 0, init, alen, builtin);
				c[cix+0] = (double)aix[minindex] + 1;
				c[cix+1] = avals[minindex]; //min value among non-zeros
				
				//correction (not sparse-safe)	
				if(alen < n && (builtin.execute2( 0, c[cix+1] ) == 1))
				{
					int ix = n-1; //find last 0 value
					for( int j=alen-1; j>=0; j--, ix-- )
						if( aix[j]!=ix )
							break;
					c[cix+0] = ix + 1; //min index (last)
					c[cix+1] = 0; //min value
				}
			}
			else //if( arow==null )
			{
				//correction (not sparse-safe)	
				c[cix+0] = n; //min index (last)
				c[cix+1] = 0; //min value
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
			if( arow!=null && !arow.isEmpty() )
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
			if( arow!=null && !arow.isEmpty() )
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
			if( arow!=null && !arow.isEmpty() )
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
			if( arow!=null && !arow.isEmpty() )
			{
				int alen = arow.size();
				double[] avals = arow.getValueContainer();
				int[] aix = arow.getIndexContainer();
				meanAgg( avals, c, aix, alen, n, kbuff, kmean );
			}
		}
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
			if( arow!=null && !arow.isEmpty() )
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
	 * @param ai
	 * @param init
	 * @param len
	 * @param aggop
	 * @return
	 */
	private static int indexmin( double[] a, int ai, final double init, final int len, Builtin aggop ) 
	{
		double minval = init;
		int minindex = -1;
		for( int i=0; i<len; i++ )
		{
			double val = a[ai+i];
			if( aggop.execute2( val, minval ) == 1 )
			{
				minval = val;
				minindex = i;
			}
		}
		
		return minindex;
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

