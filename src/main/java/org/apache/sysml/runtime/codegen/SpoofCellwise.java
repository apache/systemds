/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.codegen;

import java.io.Serializable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.BitmapEncoder;
import org.apache.sysml.runtime.compress.ColGroup;
import org.apache.sysml.runtime.compress.ColGroupValue;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysml.runtime.functionobjects.KahanFunction;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.functionobjects.KahanPlusSq;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.LibMatrixMult;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.util.UtilFunctions;

public abstract class SpoofCellwise extends SpoofOperator implements Serializable
{
	private static final long serialVersionUID = 3442528770573293590L;
	
	public enum CellType {
		NO_AGG,
		FULL_AGG,
		ROW_AGG,
		COL_AGG,
	}
	
	//redefinition of Hop.AggOp for cleaner imports in generate class
	public enum AggOp {
		SUM, 
		SUM_SQ,
		MIN,
		MAX,
	}
	
	private final CellType _type;
	private final AggOp _aggOp;
	private final boolean _sparseSafe;
	
	public SpoofCellwise(CellType type, boolean sparseSafe, AggOp aggOp) {
		_type = type;
		_aggOp = aggOp;
		_sparseSafe = sparseSafe;
	}
	
	public CellType getCellType() {
		return _type;
	}
	
	public AggOp getAggOp() {
		return _aggOp;
	}
	
	public boolean isSparseSafe() {
		return _sparseSafe;
	}
	
	@Override
	public String getSpoofType() {
		return "Cell" +  getClass().getName().split("\\.")[1];
	}
	
	private ValueFunction getAggFunction() {
		switch( _aggOp ) {
			case SUM: return KahanPlus.getKahanPlusFnObject();
			case SUM_SQ: return KahanPlusSq.getKahanPlusSqFnObject();
			case MIN: return Builtin.getBuiltinFnObject(BuiltinCode.MIN);
			case MAX: return Builtin.getBuiltinFnObject(BuiltinCode.MAX);
			default:
				throw new RuntimeException("Unsupported "
						+ "aggregation type: "+_aggOp.name());
		}
	}
	
	@Override
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, int k) 
		throws DMLRuntimeException 
	{
		//sanity check
		if( inputs==null || inputs.size() < 1  )
			throw new RuntimeException("Invalid input arguments.");
		
		//input preparation
		MatrixBlock a = inputs.get(0);
		SideInput[] b = prepInputMatrices(inputs);
		double[] scalars = prepInputScalars(scalarObjects);
		final int m = a.getNumRows();
		final int n = a.getNumColumns();
		
		//sparse safe check
		boolean sparseSafe = isSparseSafe() || (b.length == 0
				&& genexec( 0, b, scalars, m, n, 0, 0 ) == 0);
		
		long inputSize = sparseSafe ? 
			getTotalInputNnz(inputs) : getTotalInputSize(inputs);
		if( inputSize < PAR_NUMCELL_THRESHOLD ) {
			k = 1; //serial execution
		}
		
		double ret = 0;
		if( k <= 1 ) //SINGLE-THREADED
		{
			if( inputs.get(0) instanceof CompressedMatrixBlock )
				ret = executeCompressedAndAgg((CompressedMatrixBlock)a, b, scalars, m, n, sparseSafe, 0, m);
			else if( !inputs.get(0).isInSparseFormat() )
				ret = executeDenseAndAgg(a.getDenseBlock(), b, scalars, m, n, sparseSafe, 0, m);
			else
				ret = executeSparseAndAgg(a.getSparseBlock(), b, scalars, m, n, sparseSafe, 0, m);
		}
		else  //MULTI-THREADED
		{
			try {
				ExecutorService pool = Executors.newFixedThreadPool( k );
				ArrayList<ParAggTask> tasks = new ArrayList<>();
				int nk = (a instanceof CompressedMatrixBlock) ? k :
					UtilFunctions.roundToNext(Math.min(8*k,m/32), k);
				int blklen = (int)(Math.ceil((double)m/nk));
				if( a instanceof CompressedMatrixBlock )
					blklen = BitmapEncoder.getAlignedBlocksize(blklen);
				for( int i=0; i<nk & i*blklen<m; i++ )
					tasks.add(new ParAggTask(a, b, scalars, m, n, sparseSafe, i*blklen, Math.min((i+1)*blklen, m)));
				//execute tasks
				List<Future<Double>> taskret = pool.invokeAll(tasks);
				pool.shutdown();
				
				//aggregate partial results
				ValueFunction vfun = getAggFunction();
				if( vfun instanceof KahanFunction ) {
					KahanObject kbuff = new KahanObject(0, 0);
					KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
					for( Future<Double> task : taskret )
						kplus.execute2(kbuff, task.get());
					ret = kbuff._sum;
				}
				else {
					for( Future<Double> task : taskret )
						ret = vfun.execute(ret, task.get());
				}
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
		}
		
		//correction for min/max
		if( (_aggOp == AggOp.MIN || _aggOp == AggOp.MAX) && sparseSafe 
			&& a.getNonZeros()<a.getNumRows()*a.getNumColumns() )
			ret = getAggFunction().execute(ret, 0); //unseen 0 might be max or min value
		
		return new DoubleObject(ret);
	}

	@Override
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out)
		throws DMLRuntimeException
	{
		return execute(inputs, scalarObjects, out, 1);
	}
	
	@Override
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out, int k)
		throws DMLRuntimeException
	{
		//sanity check
		if( inputs==null || inputs.size() < 1 || out==null )
			throw new RuntimeException("Invalid input arguments.");
		
		//input preparation
		MatrixBlock a = inputs.get(0);
		SideInput[] b = prepInputMatrices(inputs);
		double[] scalars = prepInputScalars(scalarObjects);
		final int m = a.getNumRows();
		final int n = a.getNumColumns();
		
		//sparse safe check 
		boolean sparseSafe = isSparseSafe() || (b.length == 0
				&& genexec( 0, b, scalars, m, n, 0, 0 ) == 0);
		
		long inputSize = sparseSafe ? 
			getTotalInputNnz(inputs) : getTotalInputSize(inputs);
		if( inputSize < PAR_NUMCELL_THRESHOLD ) {
			k = 1; //serial execution
		}
		
		//result allocation and preparations
		boolean sparseOut = _type == CellType.NO_AGG
			&& sparseSafe && a.isInSparseFormat();
		switch( _type ) {
			case NO_AGG: out.reset(m, n, sparseOut); break;
			case ROW_AGG: out.reset(m, 1, false); break;
			case COL_AGG: out.reset(1, n, false); break;
			default: throw new DMLRuntimeException("Invalid cell type: "+_type);
		}
		out.allocateBlock();
		
		long lnnz = 0;
		if( k <= 1 ) //SINGLE-THREADED
		{
			if( inputs.get(0) instanceof CompressedMatrixBlock )
				lnnz = executeCompressed((CompressedMatrixBlock)a, b, scalars, out, m, n, sparseSafe, 0, m);
			else if( !inputs.get(0).isInSparseFormat() )
				lnnz = executeDense(a.getDenseBlock(), b, scalars, out, m, n, sparseSafe, 0, m);
			else
				lnnz = executeSparse(a.getSparseBlock(), b, scalars, out, m, n, sparseSafe, 0, m);
		}
		else  //MULTI-THREADED
		{
			try {
				ExecutorService pool = Executors.newFixedThreadPool( k );
				ArrayList<ParExecTask> tasks = new ArrayList<>();
				int nk = UtilFunctions.roundToNext(Math.min(8*k,m/32), k);
				int blklen = (int)(Math.ceil((double)m/nk));
				if( a instanceof CompressedMatrixBlock )
					blklen = BitmapEncoder.getAlignedBlocksize(blklen);
				for( int i=0; i<nk & i*blklen<m; i++ )
					tasks.add(new ParExecTask(a, b, scalars, out, m, n,
						sparseSafe, i*blklen, Math.min((i+1)*blklen, m)));
				//execute tasks
				List<Future<Long>> taskret = pool.invokeAll(tasks);
				pool.shutdown();
				
				//aggregate nnz and error handling
				for( Future<Long> task : taskret )
					lnnz += task.get();
				if( _type == CellType.COL_AGG ) {
					//aggregate partial results
					double[] c = out.getDenseBlock();
					ValueFunction vfun = getAggFunction();
					if( vfun instanceof KahanFunction ) {
						for( ParExecTask task : tasks )
							LibMatrixMult.vectAdd(task.getResult().getDenseBlock(), c, 0, 0, n);
					}
					else {
						for( ParExecTask task : tasks ) {
							double[] tmp = task.getResult().getDenseBlock();
							for(int j=0; j<n; j++)
								c[j] = vfun.execute(c[j], tmp[j]);
						}
					}
					lnnz = out.recomputeNonZeros();
				}
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
		}
		
		//post-processing
		out.setNonZeros(lnnz);
		out.examSparsity();
		return out;
	}
	
	/////////
	//function dispatch
	
	private long executeDense(double[] a, SideInput[] b, double[] scalars, 
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru) 
		throws DMLRuntimeException 
	{
		double[] c = out.getDenseBlock();
		SideInput[] lb = createSparseSideInputs(b);
		
		if( _type == CellType.NO_AGG ) {
			return executeDenseNoAgg(a, lb, scalars, c, m, n, sparseSafe, rl, ru);
		}
		else if( _type == CellType.ROW_AGG ) {
			if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
				return executeDenseRowAggSum(a, lb, scalars, c, m, n, sparseSafe, rl, ru);
			else
				return executeDenseRowAggMxx(a, lb, scalars, c, m, n, sparseSafe, rl, ru);
		}
		else if( _type == CellType.COL_AGG ) {
			if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
				return executeDenseColAggSum(a, lb, scalars, c, m, n, sparseSafe, rl, ru);
			else
				return executeDenseColAggMxx(a, lb, scalars, c, m, n, sparseSafe, rl, ru);
		}
		return -1;
	}
	
	private double executeDenseAndAgg(double[] a, SideInput[] b, double[] scalars, 
			int m, int n, boolean sparseSafe, int rl, int ru) throws DMLRuntimeException 
	{
		SideInput[] lb = createSparseSideInputs(b);
		
		//numerically stable aggregation for sum/sum_sq
		if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
			return executeDenseAggSum(a, lb, scalars, m, n, sparseSafe, rl, ru);
		else
			return executeDenseAggMxx(a, lb, scalars, m, n, sparseSafe, rl, ru);
	}
	
	private long executeSparse(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		if( sparseSafe && sblock == null )
			return 0;
		
		SideInput[] lb = createSparseSideInputs(b);
		if( _type == CellType.NO_AGG ) {
			if( out.isInSparseFormat() )
				return executeSparseNoAggSparse(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru);
			else
				return executeSparseNoAggDense(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru);
		}
		else if( _type == CellType.ROW_AGG ) {
			if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
				return executeSparseRowAggSum(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru);
			else
				return executeSparseRowAggMxx(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru);
		}
		else if( _type == CellType.COL_AGG ) {
			if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
				return executeSparseColAggSum(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru);
			else
				return executeSparseColAggMxx(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru);
		}
		
		return -1;
	}
	
	private double executeSparseAndAgg(SparseBlock sblock, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		if( sparseSafe && sblock == null )
			return 0;
		
		SideInput[] lb = createSparseSideInputs(b);
		if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
			return executeSparseAggSum(sblock, lb, scalars, m, n, sparseSafe, rl, ru);
		else
			return executeSparseAggMxx(sblock, lb, scalars, m, n, sparseSafe, rl, ru);
	}
	
	private long executeCompressed(CompressedMatrixBlock a, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		//NOTE: we don't create sparse side inputs w/ row-major cursors because 
		//compressed data is access in a column-major order 
		
		if( _type == CellType.NO_AGG ) {
			long lnnz = executeCompressedNoAgg(a, b, scalars, out, m, n, sparseSafe, rl, ru);
			if( out.isInSparseFormat() )
				out.sortSparseRows(rl, ru);
			return lnnz;
		}
		else if( _type == CellType.ROW_AGG ) {
			double[] c = out.getDenseBlock();
			if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
				return executeCompressedRowAggSum(a, b, scalars, c, m, n, sparseSafe, rl, ru);
			else
				return executeCompressedRowAggMxx(a, b, scalars, c, m, n, sparseSafe, rl, ru);
		}
		else if( _type == CellType.COL_AGG ) {
			double[] c = out.getDenseBlock();
			if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
				return executeCompressedColAggSum(a, b, scalars, c, m, n, sparseSafe, rl, ru);
			else
				return executeCompressedColAggMxx(a, b, scalars, c, m, n, sparseSafe, rl, ru);
		}
		return -1;
	}
	
	private double executeCompressedAndAgg(CompressedMatrixBlock a, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru) throws DMLRuntimeException 
	{
		//NOTE: we don't create sparse side inputs w/ row-major cursors because 
		//compressed data is access in a column-major order 
		
		//numerically stable aggregation for sum/sum_sq
		if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
			return executeCompressedAggSum(a, b, scalars, m, n, sparseSafe, rl, ru);
		else
			return executeCompressedAggMxx(a, b, scalars, m, n, sparseSafe, rl, ru);
	}
	
	/////////
	//core operator skeletons for dense, sparse, and compressed

	private long executeDenseNoAgg(double[] a, SideInput[] b, double[] scalars,
			double[] c, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		long lnnz = 0;
		for( int i=rl, ix=rl*n; i<ru; i++ )
			for( int j=0; j<n; j++, ix++ ) {
				double aval = (a != null) ? a[ix] : 0;
				if( aval != 0 || !sparseSafe) {
					c[ix] = genexec( aval, b, scalars, m, n, i, j);
					lnnz += (c[ix]!=0) ? 1 : 0;
				}
			}
		return lnnz;
	}
	
	private long executeDenseRowAggSum(double[] a, SideInput[] b, double[] scalars,
		double[] c, int m, int n, boolean sparseSafe, int rl, int ru)
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		long lnnz = 0;
		for( int i=rl, ix=rl*n; i<ru; i++ ) {
			kbuff.set(0, 0);
			for( int j=0; j<n; j++, ix++ ) {
				double aval = (a != null) ? a[ix] : 0;
				if( aval != 0 || !sparseSafe)
					kplus.execute2(kbuff, genexec(aval, b, scalars, m, n, i, j));
			}
			lnnz += ((c[i] = kbuff._sum)!=0) ? 1 : 0;
		}
		return lnnz;
	}
	
	private long executeDenseRowAggMxx(double[] a, SideInput[] b, double[] scalars,
			double[] c, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException 
	{
		double initialVal = (_aggOp==AggOp.MIN) ? Double.MAX_VALUE : -Double.MAX_VALUE;
		ValueFunction vfun = getAggFunction();
		long lnnz = 0;
		if( a == null && !sparseSafe ) { //empty
			for( int i=rl; i<ru; i++ ) {
				double tmp = initialVal;
				for( int j=0; j<n; j++ )
					tmp = vfun.execute(tmp, genexec(0, b, scalars, m, n, i, j));
				lnnz += ((c[i] = tmp)!=0) ? 1 : 0;
			}
		}
		else if( a != null ) { //general case
			for( int i=rl, ix=rl*n; i<ru; i++ ) {
				double tmp = initialVal;
				for( int j=0; j<n; j++, ix++ )
					if( a[ix] != 0 || !sparseSafe)
						tmp = vfun.execute(tmp, genexec(a[ix], b, scalars, m, n, i, j));
				if( sparseSafe && UtilFunctions.containsZero(a, ix-n, n) )
					tmp = vfun.execute(tmp, 0);
				lnnz += ((c[i] = tmp)!=0) ? 1 : 0;
			}
		}
		return lnnz;
	}
	
	private long executeDenseColAggSum(double[] a, SideInput[] b, double[] scalars,
		double[] c, int m, int n, boolean sparseSafe, int rl, int ru)
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		double[] corr = new double[n];
		
		for( int i=rl, ix=rl*n; i<ru; i++ )
			for( int j=0; j<n; j++, ix++ ) {
				double aval = (a != null) ? a[ix] : 0;
				if( aval != 0 || !sparseSafe) {
					kbuff.set(c[j], corr[j]);
					kplus.execute2(kbuff, genexec(aval, b, scalars, m, n, i, j));
					c[j] = kbuff._sum;
					corr[j] = kbuff._correction;
				}
			}
		return -1;
	}
	
	private long executeDenseColAggMxx(double[] a, SideInput[] b, double[] scalars,
			double[] c, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException 
	{
		double initialVal = (_aggOp==AggOp.MIN) ? Double.MAX_VALUE : -Double.MAX_VALUE;
		ValueFunction vfun = getAggFunction();
		Arrays.fill(c, initialVal);
		
		if( a == null && !sparseSafe ) { //empty
			for( int i=rl; i<ru; i++ )
				for( int j=0; j<n; j++ )
					c[j] = vfun.execute(c[j], genexec(0, b, scalars, m, n, i, j));
		}
		else if( a != null ) { //general case
			int[] counts = new int[n];
			for( int i=rl, ix=rl*n; i<ru; i++ )
				for( int j=0; j<n; j++, ix++ )
					if( a[ix] != 0 || !sparseSafe) {
						c[j] = vfun.execute(c[j], genexec(a[ix], b, scalars, m, n, i, j));
						counts[j] ++;
					}
			if( sparseSafe )
				for(int j=0; j<n; j++)
					if( counts[j] != ru-rl )
						c[j] = vfun.execute(c[j], 0);
		}
		return -1;
	}
	
	private double executeDenseAggSum(double[] a, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException 
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		
		for( int i=rl, ix=rl*n; i<ru; i++ ) 
			for( int j=0; j<n; j++, ix++ ) {
				double aval = (a != null) ? a[ix] : 0;
				if( aval != 0 || !sparseSafe)
					kplus.execute2(kbuff, genexec(aval, b, scalars, m, n, i, j));
			}
		return kbuff._sum;
	}
	
	private double executeDenseAggMxx(double[] a, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException 
	{
		//safe aggregation for min/max w/ handling of zero entries
		//note: sparse safe with zero value as min/max handled outside
		double ret = (_aggOp==AggOp.MIN) ? Double.MAX_VALUE : -Double.MAX_VALUE; 
		ValueFunction vfun = getAggFunction();
		
		for( int i=rl, ix=rl*n; i<ru; i++ ) 
			for( int j=0; j<n; j++, ix++ ) {
				double aval = (a != null) ? a[ix] : 0;
				if( aval != 0 || !sparseSafe)
					ret = vfun.execute(ret, genexec(aval, b, scalars, m, n, i, j));
			}
		return ret;
	}
	
	private long executeSparseNoAggSparse(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		//note: sequential scan algorithm for both sparse-safe and -unsafe
		//in order to avoid binary search for sparse-unsafe
		SparseBlock c = out.getSparseBlock();
		long lnnz = 0;
		for(int i=rl; i<ru; i++) {
			int lastj = -1;
			//handle non-empty rows
			if( sblock != null && !sblock.isEmpty(i) ) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				c.allocate(i, sparseSafe ? alen : n);
				for(int k=apos; k<apos+alen; k++) {
					//process zeros before current non-zero
					if( !sparseSafe )
						for(int j=lastj+1; j<aix[k]; j++)
							c.append(i, j, genexec(0, b, scalars, m, n, i, j));
					//process current non-zero
					lastj = aix[k];
					c.append(i, lastj, genexec(avals[k], b, scalars, m, n, i, lastj));
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					c.append(i, j, genexec(0, b, scalars, m, n, i, j));
			lnnz += c.size(i);
		}
		
		return lnnz;
	}
	
	private long executeSparseNoAggDense(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		//note: sequential scan algorithm for both sparse-safe and -unsafe
		//in order to avoid binary search for sparse-unsafe
		double[] c = out.getDenseBlock();
		long lnnz = 0;
		for(int i=rl, cix=rl*n; i<ru; i++, cix+=n) {
			int lastj = -1;
			//handle non-empty rows
			if( sblock != null && !sblock.isEmpty(i) ) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				for(int k=apos; k<apos+alen; k++) {
					//process zeros before current non-zero
					if( !sparseSafe )
						for(int j=lastj+1; j<aix[k]; j++)
							lnnz += ((c[cix+j]=genexec(0, b, scalars, m, n, i, j))!=0)?1:0;
					//process current non-zero
					lastj = aix[k];
					lnnz += ((c[cix+lastj]=genexec(avals[k], b, scalars, m, n, i, lastj))!=0)?1:0;
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					lnnz += ((c[cix+j]=genexec(0, b, scalars, m, n, i, j))!=0)?1:0;
		}
		return lnnz;
	}
	
	private long executeSparseRowAggSum(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException 
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);

		//note: sequential scan algorithm for both sparse-safe and -unsafe
		//in order to avoid binary search for sparse-unsafe
		double[] c = out.getDenseBlock();
		long lnnz = 0;
		for(int i=rl; i<ru; i++) {
			kbuff.set(0, 0);
			int lastj = -1;
			//handle non-empty rows
			if( sblock != null && !sblock.isEmpty(i) ) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				for(int k=apos; k<apos+alen; k++) {
					//process zeros before current non-zero
					if( !sparseSafe )
						for(int j=lastj+1; j<aix[k]; j++)
							kplus.execute2(kbuff, genexec(0, b, scalars, m, n, i, j));
					//process current non-zero
					lastj = aix[k];
					kplus.execute2(kbuff, genexec(avals[k], b, scalars, m, n, i, lastj));
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					kplus.execute2(kbuff, genexec(0, b, scalars, m, n, i, j));
			lnnz += ((c[i] = kbuff._sum)!=0) ? 1 : 0;
		}
		return lnnz;
	}
	
	private long executeSparseRowAggMxx(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		double initialVal = (_aggOp==AggOp.MIN) ? Double.MAX_VALUE : -Double.MAX_VALUE;
		ValueFunction vfun = getAggFunction();
		
		//note: sequential scan algorithm for both sparse-safe and -unsafe 
		//in order to avoid binary search for sparse-unsafe 
		double[] c = out.getDenseBlock();
		long lnnz = 0;
		for(int i=rl; i<ru; i++) {
			double tmp = (sparseSafe && sblock.size(i) < n) ? 0 : initialVal;
			int lastj = -1;
			//handle non-empty rows
			if( sblock != null && !sblock.isEmpty(i) ) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				for(int k=apos; k<apos+alen; k++) {
					//process zeros before current non-zero
					if( !sparseSafe )
						for(int j=lastj+1; j<aix[k]; j++)
							tmp = vfun.execute(tmp, genexec(0, b, scalars, m, n, i, j));
					//process current non-zero
					lastj = aix[k];
					tmp = vfun.execute( tmp, genexec(avals[k], b, scalars, m, n, i, lastj));
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					tmp = vfun.execute(tmp, genexec(0, b, scalars, m, n, i, j));
			lnnz += ((c[i] = tmp)!=0) ? 1 : 0;
		}
		return lnnz;
	}

	private long executeSparseColAggSum(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		double[] corr = new double[n];
		
		//note: sequential scan algorithm for both sparse-safe and -unsafe
		//in order to avoid binary search for sparse-unsafe
		double[] c = out.getDenseBlock();
		for(int i=rl; i<ru; i++) {
			kbuff.set(0, 0);
			int lastj = -1;
			//handle non-empty rows
			if( sblock != null && !sblock.isEmpty(i) ) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				for(int k=apos; k<apos+alen; k++) {
					//process zeros before current non-zero
					if( !sparseSafe )
						for(int j=lastj+1; j<aix[k]; j++) {
							kbuff.set(c[j], corr[j]);
							kplus.execute2(kbuff, genexec(0, b, scalars, m, n, i, j));
							c[j] = kbuff._sum;
							corr[j] = kbuff._correction;
						}
					//process current non-zero
					lastj = aix[k];
					kbuff.set(c[aix[k]], corr[aix[k]]);
					kplus.execute2(kbuff, genexec(avals[k], b, scalars, m, n, i, lastj));
					c[aix[k]] = kbuff._sum;
					corr[aix[k]] = kbuff._correction;
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++) {
					kbuff.set(c[j], corr[j]);
					kplus.execute2(kbuff, genexec(0, b, scalars, m, n, i, j));
					c[j] = kbuff._sum;
					corr[j] = kbuff._correction;
				}
		}
		return -1;
	}
	
	private long executeSparseColAggMxx(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		double initialVal = (_aggOp==AggOp.MIN) ? Double.MAX_VALUE : -Double.MAX_VALUE;
		ValueFunction vfun = getAggFunction();
		double[] c = out.getDenseBlock();
		Arrays.fill(c, initialVal);
		int[] count = new int[n];
		
		//note: sequential scan algorithm for both sparse-safe and -unsafe
		//in order to avoid binary search for sparse-unsafe
		for(int i=rl; i<ru; i++) {
			int lastj = -1;
			//handle non-empty rows
			if( sblock != null && !sblock.isEmpty(i) ) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				for(int k=apos; k<apos+alen; k++) {
					//process zeros before current non-zero
					if( !sparseSafe )
						for(int j=lastj+1; j<aix[k]; j++) {
							c[j] = vfun.execute(c[j], genexec(0, b, scalars, m, n, i, j));
							count[j] ++;
						}
					//process current non-zero
					lastj = aix[k];
					c[aix[k]] = vfun.execute(c[aix[k]], genexec(avals[k], b, scalars, m, n, i, lastj));
					count[aix[k]] ++;
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					c[j] = vfun.execute(c[j], genexec(0, b, scalars, m, n, i, j));
		}
		
		return -1;
	}
	
	private double executeSparseAggSum(SparseBlock sblock, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		
		//note: sequential scan algorithm for both sparse-safe and -unsafe 
		//in order to avoid binary search for sparse-unsafe 
		for(int i=rl; i<ru; i++) {
			int lastj = -1;
			//handle non-empty rows
			if( sblock != null && !sblock.isEmpty(i) ) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				for(int k=apos; k<apos+alen; k++) {
					//process zeros before current non-zero
					if( !sparseSafe )
						for(int j=lastj+1; j<aix[k]; j++)
							kplus.execute2(kbuff, genexec(0, b, scalars, m, n, i, j));
					//process current non-zero
					lastj = aix[k];
					kplus.execute2(kbuff, genexec(avals[k], b, scalars, m, n, i, lastj));
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					kplus.execute2(kbuff, genexec(0, b, scalars, m, n, i, j));
		}
		return kbuff._sum;
	}
	
	private double executeSparseAggMxx(SparseBlock sblock, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		double ret = (_aggOp==AggOp.MIN) ? Double.MAX_VALUE : -Double.MAX_VALUE;
		ret = (sparseSafe && sblock.size() < (long)m*n) ? 0 : ret;
		ValueFunction vfun = getAggFunction();
		
		//note: sequential scan algorithm for both sparse-safe and -unsafe
		//in order to avoid binary search for sparse-unsafe
		for(int i=rl; i<ru; i++) {
			int lastj = -1;
			//handle non-empty rows
			if( sblock != null && !sblock.isEmpty(i) ) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				for(int k=apos; k<apos+alen; k++) {
					//process zeros before current non-zero
					if( !sparseSafe )
						for(int j=lastj+1; j<aix[k]; j++)
							ret = vfun.execute(ret, genexec(0, b, scalars, m, n, i, j));
					//process current non-zero
					lastj = aix[k];
					ret = vfun.execute(ret, genexec(avals[k], b, scalars, m, n, i, lastj));
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					ret = vfun.execute(ret, genexec(0, b, scalars, m, n, i, j));
		}
		return ret;
	}
	
	private long executeCompressedNoAgg(CompressedMatrixBlock a, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		double[] c = out.getDenseBlock();
		SparseBlock csblock = out.getSparseBlock();
		
		//preallocate sparse rows to avoid reallocations
		//note: counting nnz requires segment-aligned boundaries, which is enforced
		//whenever k/2 * BITMAP_BLOCK_SZ > m (i.e., it does not limit parallelism)
		if( out.isInSparseFormat() && rl%BitmapEncoder.BITMAP_BLOCK_SZ==0
			&& ru%BitmapEncoder.BITMAP_BLOCK_SZ==0) {
			int[] rnnz = a.countNonZerosPerRow(rl, ru);
			for( int i=rl; i<ru; i++ )
				csblock.allocate(i, rnnz[i-rl]);
		}
		
		long lnnz = 0;
		Iterator<IJV> iter = a.getIterator(rl, ru, !sparseSafe);
		while( iter.hasNext() ) {
			IJV cell = iter.next();
			double val = genexec(cell.getV(), b, scalars, m, n, cell.getI(), cell.getJ());
			if( out.isInSparseFormat() ) {
				csblock.allocate(cell.getI());
				csblock.append(cell.getI(), cell.getJ(), val);
			}
			else
				c[cell.getI()*n+cell.getJ()] = val;
			lnnz += (val!=0) ? 1 : 0;
		}
		return lnnz;
	}
	
	private long executeCompressedRowAggSum(CompressedMatrixBlock a, SideInput[] b, double[] scalars,
			double[] c, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		long lnnz = 0;
		Iterator<IJV> iter = a.getIterator(rl, ru, !sparseSafe);
		while( iter.hasNext() ) {
			IJV cell = iter.next();
			double val = genexec(cell.getV(), b, scalars, m, n, cell.getI(), cell.getJ());
			kbuff.set(c[cell.getI()], 0);
			kplus.execute2(kbuff, val);
			c[cell.getI()] = kbuff._sum;
		}
		for( int i=rl; i<ru; i++ )
			lnnz += (c[i]!=0) ? 1 : 0;
		return lnnz;
	}
	
	private long executeCompressedRowAggMxx(CompressedMatrixBlock a, SideInput[] b, double[] scalars,
			double[] c, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		Arrays.fill(c, rl, ru, (_aggOp==AggOp.MIN) ? Double.MAX_VALUE : -Double.MAX_VALUE);
		ValueFunction vfun = getAggFunction();
		long lnnz = 0;
		Iterator<IJV> iter = a.getIterator(rl, ru, !sparseSafe);
		while( iter.hasNext() ) {
			IJV cell = iter.next();
			double val = genexec(cell.getV(), b, scalars, m, n, cell.getI(), cell.getJ());
			c[cell.getI()] = vfun.execute(c[cell.getI()], val);
		}
		for( int i=rl; i<ru; i++ )
			lnnz += (c[i]!=0) ? 1 : 0;
		return lnnz;
	}
	
	private long executeCompressedColAggSum(CompressedMatrixBlock a, SideInput[] b, double[] scalars,
		double[] c, int m, int n, boolean sparseSafe, int rl, int ru)
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		double[] corr = new double[n];
		
		Iterator<IJV> iter = a.getIterator(rl, ru, !sparseSafe);
		while( iter.hasNext() ) {
			IJV cell = iter.next();
			double val = genexec(cell.getV(), b, scalars, m, n, cell.getI(), cell.getJ());
			kbuff.set(c[cell.getJ()], corr[cell.getJ()]);
			kplus.execute2(kbuff, val);
			c[cell.getJ()] = kbuff._sum;
			corr[cell.getJ()] = kbuff._correction;
		}
		return -1;
	}
	
	private long executeCompressedColAggMxx(CompressedMatrixBlock a, SideInput[] b, double[] scalars,
			double[] c, int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		Arrays.fill(c, rl, ru, (_aggOp==AggOp.MIN) ? Double.MAX_VALUE : -Double.MAX_VALUE);
		ValueFunction vfun = getAggFunction();
		long lnnz = 0;
		Iterator<IJV> iter = a.getIterator(rl, ru, !sparseSafe);
		while( iter.hasNext() ) {
			IJV cell = iter.next();
			double val = genexec(cell.getV(), b, scalars, m, n, cell.getI(), cell.getJ());
			c[cell.getI()] = vfun.execute(c[cell.getI()], val);
		}
		for( int i=rl; i<ru; i++ )
			lnnz += (c[i]!=0) ? 1 : 0;
		return lnnz;
	}
	
	private double executeCompressedAggSum(CompressedMatrixBlock a, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		KahanObject kbuff2 = new KahanObject(0, 0);
		
		//special case: computation over value-tuples only
		if( sparseSafe && b.length==0 && !a.hasUncompressedColGroup() ) {
			//note: all remaining groups are guaranteed ColGroupValue
			boolean entireGrp = (rl==0 && ru==a.getNumRows());
			for( ColGroup grp : a.getColGroups() ) {
				ColGroupValue grpv = (ColGroupValue) grp;
				int[] counts = entireGrp ? 
					grpv.getCounts() : grpv.getCounts(rl, ru);
				for(int k=0; k<grpv.getNumValues(); k++) {
					kbuff2.set(0, 0);
					double in = grpv.sumValues(k, kplus, kbuff2);
					double out = genexec(in, b, scalars, m, n, -1, -1);
					kplus.execute3(kbuff, out, counts[k]);
				}
			}
		}
		//general case of arbitrary side inputs 
		else {
			Iterator<IJV> iter = a.getIterator(rl, ru, !sparseSafe);
			while( iter.hasNext() ) {
				IJV cell = iter.next();
				double val = genexec(cell.getV(), b, scalars, m, n, cell.getI(), cell.getJ());
				kplus.execute2(kbuff, val);
			}
		}
		return kbuff._sum;
	}
	
	private double executeCompressedAggMxx(CompressedMatrixBlock a, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru)
		throws DMLRuntimeException
	{
		//safe aggregation for min/max w/ handling of zero entries
		//note: sparse safe with zero value as min/max handled outside
		double ret = (_aggOp==AggOp.MIN) ? Double.MAX_VALUE : -Double.MAX_VALUE;
		ValueFunction vfun = getAggFunction();
		
		Iterator<IJV> iter = a.getIterator(rl, ru, !sparseSafe);
		while( iter.hasNext() ) {
			IJV cell = iter.next();
			double val = genexec(cell.getV(), b, scalars, m, n, cell.getI(), cell.getJ());
			ret = vfun.execute(ret, val);
		}
		return ret;
	}
	
	protected abstract double genexec( double a, SideInput[] b,
			double[] scalars, int m, int n, int rowIndex, int colIndex);
	
	private class ParAggTask implements Callable<Double> 
	{
		private final MatrixBlock _a;
		private final SideInput[] _b;
		private final double[] _scalars;
		private final int _rlen;
		private final int _clen;
		private final boolean _safe;
		private final int _rl;
		private final int _ru;

		protected ParAggTask( MatrixBlock a, SideInput[] b, double[] scalars, 
				int rlen, int clen, boolean sparseSafe, int rl, int ru ) {
			_a = a;
			_b = b;
			_scalars = scalars;
			_rlen = rlen;
			_clen = clen;
			_safe = sparseSafe;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Double call() throws DMLRuntimeException {
			if( _a instanceof CompressedMatrixBlock )
				return executeCompressedAndAgg((CompressedMatrixBlock)_a, _b, _scalars, _rlen, _clen, _safe, _rl, _ru);
			else if (!_a.isInSparseFormat())
				return executeDenseAndAgg(_a.getDenseBlock(), _b, _scalars, _rlen, _clen, _safe, _rl, _ru);
			else
				return executeSparseAndAgg(_a.getSparseBlock(), _b, _scalars, _rlen, _clen, _safe, _rl, _ru);
		}
	}

	private class ParExecTask implements Callable<Long>
	{
		private final MatrixBlock _a;
		private final SideInput[] _b;
		private final double[] _scalars;
		private MatrixBlock _c;
		private final int _rlen;
		private final int _clen;
		private final boolean _safe;
		private final int _rl;
		private final int _ru;

		protected ParExecTask( MatrixBlock a, SideInput[] b, double[] scalars, MatrixBlock c,
				int rlen, int clen, boolean sparseSafe, int rl, int ru ) {
			_a = a;
			_b = b;
			_scalars = scalars;
			_c = c;
			_rlen = rlen;
			_clen = clen;
			_safe = sparseSafe;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Long call() throws DMLRuntimeException {
			if( _type==CellType.COL_AGG ) {
				_c = new MatrixBlock(1,_clen, false);
				_c.allocateDenseBlock();
			}
			if( _a instanceof CompressedMatrixBlock )
				return executeCompressed((CompressedMatrixBlock)_a, _b, _scalars, _c, _rlen, _clen, _safe, _rl, _ru);
			else if( !_a.isInSparseFormat() )
				return executeDense(_a.getDenseBlock(), _b, _scalars, _c, _rlen, _clen, _safe, _rl, _ru);
			else
				return executeSparse(_a.getSparseBlock(), _b, _scalars, _c, _rlen, _clen, _safe, _rl, _ru);
		}
		
		public MatrixBlock getResult() {
			return _c;
		}
	}
}
