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

package org.apache.sysds.runtime.codegen;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

public abstract class SpoofCellwise extends SpoofOperator {

	private static final long serialVersionUID = 3442528770573293590L;
	
	// these values need to match with their native counterparts (spoof cuda ops)
	public enum CellType {
		NO_AGG(0),
		FULL_AGG(1),
		ROW_AGG(2),
		COL_AGG(3);
		
		private final int value;
		private final static HashMap<Integer, CellType> map = new HashMap<>();
		
		CellType(int value) {
			this.value = value;
		}
		
		static {
			for (CellType cellType : CellType.values()) {
				map.put(cellType.value, cellType);
			}
		}
		
		public static CellType valueOf(int cellType) {
			return map.get(cellType);
		}
		
		public int getValue() {
			return value;
		}			
	}
	
	//redefinition of Hop.AggOp for cleaner imports in generate class
	public enum AggOp {
		SUM, 
		SUM_SQ,
		MIN,
		MAX,
		PROD
	}
	
	protected final CellType _type;
	private final AggOp _aggOp;
	private final boolean _sparseSafe;
	private final boolean _containsSeq;
	
	public SpoofCellwise(CellType type, boolean sparseSafe, boolean containsSeq, AggOp aggOp) {
		_type = type;
		_aggOp = aggOp;
		_sparseSafe = sparseSafe;
		_containsSeq = containsSeq;
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
	
	public boolean containsSeq() {
		return _containsSeq;
	}
	
	@Override public SpoofCUDAOperator createCUDAInstrcution(Integer opID, SpoofCUDAOperator.PrecisionProxy ep) {
		return new SpoofCUDACellwise(_type, _sparseSafe, _containsSeq, _aggOp, opID, ep, this);
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
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, int k) {
		return execute(inputs, scalarObjects, k, 0);
	}
	
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, int k, long rix) {
		//sanity check
		if( inputs==null || inputs.size() < 1  )
			throw new RuntimeException("Invalid input arguments.");
		
		//input preparation
		MatrixBlock a = inputs.get(0);
		if(a instanceof CompressedMatrixBlock)
			a = CompressedMatrixBlock.getUncompressed(a);

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
			if( !a.isInSparseFormat() )
				ret = executeDenseAndAgg(a.getDenseBlock(), b, scalars, m, n, sparseSafe, 0, m, rix);
			else
				ret = executeSparseAndAgg(a.getSparseBlock(), b, scalars, m, n, sparseSafe, 0, m, rix);
		}
		else  //MULTI-THREADED
		{
			ExecutorService pool = CommonThreadPool.get(k);
			try {
				ArrayList<ParAggTask> tasks = new ArrayList<>();
				int nk = UtilFunctions.roundToNext(Math.min(8*k,m/32), k);
				int blklen = (int)(Math.ceil((double)m/nk));
				for( int i=0; i<nk & i*blklen<m; i++ )
					tasks.add(new ParAggTask(a, b, scalars, m, n, sparseSafe, i*blklen, Math.min((i+1)*blklen, m)));
				//execute tasks
				List<Future<Double>> taskret = pool.invokeAll(tasks);
				
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
			finally{
				pool.shutdown();
			}
		}
		
		//correction for min/max
		if( (_aggOp == AggOp.MIN || _aggOp == AggOp.MAX) && sparseSafe 
			&& a.getNonZeros()<a.getNumRows()*a.getNumColumns() )
			ret = getAggFunction().execute(ret, 0); //unseen 0 might be max or min value
		
		return new DoubleObject(ret);
	}

	@Override
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out) {
		return execute(inputs, scalarObjects, out, 1, 0);
	}
	
	@Override
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out, int k) {
		return execute(inputs, scalarObjects, out, k, 0);
	}
	
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out, int k, long rix) {
		//sanity check
		if( inputs==null || inputs.size() < 1 || out==null )
			throw new RuntimeException("Invalid input arguments.");
		
		//input preparation
		MatrixBlock a = inputs.get(0);
		if(a instanceof CompressedMatrixBlock)
			a = CompressedMatrixBlock.getUncompressed(a);
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
			if( !inputs.get(0).isInSparseFormat() )
				lnnz = executeDense(a.getDenseBlock(), b, scalars, out, m, n, sparseSafe, 0, m, rix);
			else
				lnnz = executeSparse(a.getSparseBlock(), b, scalars, out, m, n, sparseSafe, 0, m, rix);
			if(_type == CellType.COL_AGG)
				lnnz = out.recomputeNonZeros();
		}
		else  //MULTI-THREADED
		{
			ExecutorService pool = CommonThreadPool.get(k);
			try {
				ArrayList<ParExecTask> tasks = new ArrayList<>();
				int nk = UtilFunctions.roundToNext(Math.min(8*k,m/32), k);
				int blklen = (int)(Math.ceil((double)m/nk));
				for( int i=0; i<nk & i*blklen<m; i++ )
					tasks.add(new ParExecTask(a, b, scalars, out, m, n,
						sparseSafe, i*blklen, Math.min((i+1)*blklen, m)));
				//execute tasks
				List<Future<Long>> taskret = pool.invokeAll(tasks);
				
				//aggregate nnz and error handling
				for( Future<Long> task : taskret )
					lnnz += task.get();
				if( _type == CellType.COL_AGG ) {
					//aggregate partial results
					double[] c = out.getDenseBlockValues();
					ValueFunction vfun = getAggFunction();
					if( vfun instanceof KahanFunction ) {
						for( ParExecTask task : tasks )
							LibMatrixMult.vectAdd(task.getResult().getDenseBlockValues(), c, 0, 0, n);
					}
					else {
						for( ParExecTask task : tasks ) {
							double[] tmp = task.getResult().getDenseBlockValues();
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
			finally{
				pool.shutdown();
			}
		}
		
		//post-processing
		out.setNonZeros(lnnz);
		out.examSparsity();
		return out;
	}
	
	/////////
	//function dispatch
	
	private long executeDense(DenseBlock a, SideInput[] b, double[] scalars, 
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru, long rix) {
		DenseBlock c = out.getDenseBlock();
		SideInput[] lb = createSparseSideInputs(b);
		
		if( _type == CellType.NO_AGG ) {
			return executeDenseNoAgg(a, lb, scalars, c, m, n, sparseSafe, rl, ru, rix);
		}
		else if( _type == CellType.ROW_AGG ) {
			if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
				return executeDenseRowAggSum(a, lb, scalars, c, m, n, sparseSafe, rl, ru, rix);
			else if(_aggOp == AggOp.PROD)
				return executeDenseRowProd(a, lb, scalars, c, m, n, sparseSafe, rl, ru, rix);
			else
				return executeDenseRowAggMxx(a, lb, scalars, c, m, n, sparseSafe, rl, ru, rix);
		}
		else if( _type == CellType.COL_AGG ) {
			if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
				return executeDenseColAggSum(a, lb, scalars, c, m, n, sparseSafe, rl, ru, rix);
			else if(_aggOp == AggOp.PROD)
				return executeDenseColProd(a, lb, scalars, c, m, n, sparseSafe, rl, ru, rix);
			else
				return executeDenseColAggMxx(a, lb, scalars, c, m, n, sparseSafe, rl, ru, rix);
		}
		return -1;
	}
	
	private double executeDenseAndAgg(DenseBlock a, SideInput[] b, double[] scalars, 
			int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		SideInput[] lb = createSparseSideInputs(b);
		
		//numerically stable aggregation for sum/sum_sq
		if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
			return executeDenseAggSum(a, lb, scalars, m, n, sparseSafe, rl, ru, rix);
		else
			return executeDenseAggMxx(a, lb, scalars, m, n, sparseSafe, rl, ru, rix);
	}
	
	private long executeSparse(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		if( sparseSafe && sblock == null )
			return 0;
		
		SideInput[] lb = createSparseSideInputs(b);
		if( _type == CellType.NO_AGG ) {
			if( out.isInSparseFormat() )
				return executeSparseNoAggSparse(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru, rix);
			else
				return executeSparseNoAggDense(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru, rix);
		}
		else if( _type == CellType.ROW_AGG ) {
			if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
				return executeSparseRowAggSum(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru, rix);
			else if( _aggOp == AggOp.PROD)
				return executeSparseRowProd(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru, rix);
			else
				return executeSparseRowAggMxx(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru, rix);
		}
		else if( _type == CellType.COL_AGG ) {
			if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
				return executeSparseColAggSum(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru, rix);
			else if( _aggOp == AggOp.PROD)
				return executeSparseColProd(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru, rix);
			else
				return executeSparseColAggMxx(sblock, lb, scalars, out, m, n, sparseSafe, rl, ru, rix);
		}
		
		return -1;
	}
	
	private double executeSparseAndAgg(SparseBlock sblock, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		if( sparseSafe && sblock == null )
			return 0;
		
		SideInput[] lb = createSparseSideInputs(b);
		if( _aggOp == AggOp.SUM || _aggOp == AggOp.SUM_SQ )
			return executeSparseAggSum(sblock, lb, scalars, m, n, sparseSafe, rl, ru, rix);
		else
			return executeSparseAggMxx(sblock, lb, scalars, m, n, sparseSafe, rl, ru, rix);
	}
	
	/////////
	//core operator skeletons for dense, sparse, and compressed

	private long executeDenseNoAgg(DenseBlock a, SideInput[] b, double[] scalars,
			DenseBlock c, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		long lnnz = 0;
		if( a == null && !sparseSafe ) {
			for( int i=rl; i<ru; i++ ) {
				double[] cvals = c.values(i);
				int cix = c.pos(i);
				for( int j=0; j<n; j++ )
					lnnz += ((cvals[cix+j] = genexec(0, b, scalars, m, n, rix+i, i, j))!=0) ? 1 : 0;
			}
		}
		else if( a != null ) {
			for( int i=rl; i<ru; i++ ) {
				double[] avals = a.values(i);
				double[] cvals = c.values(i);
				int ix = a.pos(i);
				for( int j=0; j<n; j++ ) {
					double aval = avals[ix+j];
					if( aval != 0 || !sparseSafe)
						lnnz += ((cvals[ix+j] = genexec(aval, b, scalars, m, n, rix+i, i, j))!=0) ? 1 : 0;
				}
			}
		}
		
		return lnnz;
	}
	
	private long executeDenseRowAggSum(DenseBlock a, SideInput[] b, double[] scalars,
		DenseBlock c, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		//note: output always single block
		double[] lc = c.valuesAt(0);
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		
		long lnnz = 0;
		if( a == null && !sparseSafe ) {
			for( int i=rl; i<ru; i++ ) {
				kbuff.set(0, 0);
				for( int j=0; j<n; j++ )
					kplus.execute2(kbuff, genexec(0, b, scalars, m, n, rix+i, i, j));
				lnnz += ((lc[i] = kbuff._sum)!=0) ? 1 : 0;
			}
		}
		else if( a != null ) {
			for( int i=rl; i<ru; i++ ) {
				kbuff.set(0, 0);
				double[] avals = a.values(i);
				int aix = a.pos(i);
				for( int j=0; j<n; j++ ) {
					double aval = avals[aix+j];
					if( aval != 0 || !sparseSafe)
						kplus.execute2(kbuff, genexec(aval, b, scalars, m, n, rix+i, i, j));
				}
				lnnz += ((lc[i] = kbuff._sum)!=0) ? 1 : 0;
			}
		}
		
		return lnnz;
	}
	
	private long executeDenseRowAggMxx(DenseBlock a, SideInput[] b, double[] scalars,
			DenseBlock c, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		double[] lc = c.valuesAt(0); //single block
		
		double initialVal = (_aggOp==AggOp.MIN) ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
		ValueFunction vfun = getAggFunction();
		long lnnz = 0;
		if( a == null && !sparseSafe ) { //empty
			for( int i=rl; i<ru; i++ ) {
				double tmp = initialVal;
				for( int j=0; j<n; j++ )
					tmp = vfun.execute(tmp, genexec(0, b, scalars, m, n, rix+i, i, j));
				lnnz += ((lc[i] = tmp)!=0) ? 1 : 0;
			}
		}
		else if( a != null ) { //general case
			for( int i=rl; i<ru; i++ ) {
				double tmp = initialVal;
				double[] avals = a.values(i);
				int aix = a.pos(i);
				for( int j=0; j<n; j++ ) {
					double aval = avals[aix + j];
					if( aval != 0 || !sparseSafe)
						tmp = vfun.execute(tmp, genexec(aval, b, scalars, m, n, rix+i, i, j));
				}
				if( sparseSafe && UtilFunctions.containsZero(avals, aix, n) )
					tmp = vfun.execute(tmp, 0);
				lnnz += ((lc[i] = tmp)!=0) ? 1 : 0;
			}
		}
		return lnnz;
	}
	
	private long executeDenseColAggSum(DenseBlock a, SideInput[] b, double[] scalars,
		DenseBlock c, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		double[] lc = c.valuesAt(0); //single block
		
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		double[] corr = new double[n];
		
		if( a == null && !sparseSafe ) {
			for( int i=rl; i<ru; i++ )
				for( int j=0; j<n; j++ ) {
					kbuff.set(lc[j], corr[j]);
					kplus.execute2(kbuff, genexec(0, b, scalars, m, n, rix+i, i, j));
					lc[j] = kbuff._sum;
					corr[j] = kbuff._correction;
				}
		}
		else if( a != null ) {
			for( int i=rl; i<ru; i++ ) {
				double[] avals = a.values(i);
				int aix = a.pos(i);
				for( int j=0; j<n; j++ ) {
					double aval = avals[aix + j];
					if( aval != 0 || !sparseSafe ) {
						kbuff.set(lc[j], corr[j]);
						kplus.execute2(kbuff, genexec(aval, b, scalars, m, n, rix+i, i, j));
						lc[j] = kbuff._sum;
						corr[j] = kbuff._correction;
					}
				}
			}
		}
		
		return -1;
	}
	
	private long executeDenseColAggMxx(DenseBlock a, SideInput[] b, double[] scalars,
			DenseBlock c, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		double[] lc = c.valuesAt(0); //single block
		
		double initialVal = (_aggOp==AggOp.MIN) ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
		ValueFunction vfun = getAggFunction();
		Arrays.fill(lc, initialVal);
		
		if( a == null && !sparseSafe ) { //empty
			for( int i=rl; i<ru; i++ )
				for( int j=0; j<n; j++ )
					lc[j] = vfun.execute(lc[j], genexec(0, b, scalars, m, n, rix+i, i, j));
		}
		else if( a != null ) { //general case
			int[] counts = new int[n];
			for( int i=rl; i<ru; i++ ) {
				double[] avals = a.values(i);
				int aix = a.pos(i);
				for( int j=0; j<n; j++ ) {
					double aval = avals[aix + j];
					if( aval != 0 || !sparseSafe ) {
						lc[j] = vfun.execute(lc[j], genexec(aval, b, scalars, m, n, rix+i, i, j));
						counts[j] ++;
					}
				}
			}
			if( sparseSafe )
				for(int j=0; j<n; j++)
					if( counts[j] != ru-rl )
						lc[j] = vfun.execute(lc[j], 0);
		}
		return -1;
	}
	
	private double executeDenseAggSum(DenseBlock a, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		
		if( a == null && !sparseSafe ) {
			for( int i=rl; i<ru; i++ )
				for( int j=0; j<n; j++ )
					kplus.execute2(kbuff, genexec(0, b, scalars, m, n, rix+i, i, j));
		}
		else if( a != null ) {
			for( int i=rl; i<ru; i++ ) {
				double[] avals = a.values(i);
				int aix = a.pos(i);
				for( int j=0; j<n; j++ ) {
					double aval = avals[aix + j];
					if( aval != 0 || !sparseSafe)
						kplus.execute2(kbuff, genexec(aval, b, scalars, m, n, rix+i, i, j));
				}
			}
		}
		
		return kbuff._sum;
	}
	
	private double executeDenseAggMxx(DenseBlock a, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		//safe aggregation for min/max w/ handling of zero entries
		//note: sparse safe with zero value as min/max handled outside
		double ret = (_aggOp==AggOp.MIN) ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
		ValueFunction vfun = getAggFunction();
		
		if( a == null && !sparseSafe ) {
			for( int i=rl; i<ru; i++ )
				for( int j=0; j<n; j++ )
					ret = vfun.execute(ret, genexec(0, b, scalars, m, n, rix+i, i, j));
		}
		else if( a != null ) {
			for( int i=rl; i<ru; i++ ) {
				double[] avals = a.values(i);
				int aix = a.pos(i);
				for( int j=0; j<n; j++ ) {
					double aval = avals[aix + j];
					if( aval != 0 || !sparseSafe)
						ret = vfun.execute(ret, genexec(aval, b, scalars, m, n, rix+i, i, j));
				}
			}
		}
		
		return ret;
	}
	
	private long executeSparseNoAggSparse(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
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
							c.append(i, j, genexec(0, b, scalars, m, n, rix+i, i, j));
					//process current non-zero
					lastj = aix[k];
					c.append(i, lastj, genexec(avals[k], b, scalars, m, n, rix+i, i, lastj));
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					c.append(i, j, genexec(0, b, scalars, m, n, rix+i, i, j));
			lnnz += c.size(i);
		}
		
		return lnnz;
	}
	
	private long executeSparseNoAggDense(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		//note: sequential scan algorithm for both sparse-safe and -unsafe
		//in order to avoid binary search for sparse-unsafe
		DenseBlock c = out.getDenseBlock();
		long lnnz = 0;
		for(int i=rl; i<ru; i++) {
			int lastj = -1;
			//handle non-empty rows
			if( sblock != null && !sblock.isEmpty(i) ) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				double[] cvals = c.values(i);
				int cix = c.pos(i);
				for(int k=apos; k<apos+alen; k++) {
					//process zeros before current non-zero
					if( !sparseSafe )
						for(int j=lastj+1; j<aix[k]; j++)
							lnnz += ((cvals[cix+j]=genexec(0, b, scalars, m, n, rix+i, i, j))!=0)?1:0;
					//process current non-zero
					lastj = aix[k];
					lnnz += ((cvals[cix+lastj]=genexec(avals[k], b, scalars, m, n, rix+i, i, lastj))!=0)?1:0;
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++) {
					double[] cvals = c.values(i);
					int cix = c.pos(i);
					lnnz += ((cvals[cix+j]=genexec(0, b, scalars, m, n, rix+i, i, j))!=0)?1:0;
				}
		}
		return lnnz;
	}
	
	private long executeSparseRowAggSum(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);

		//note: sequential scan algorithm for both sparse-safe and -unsafe
		//in order to avoid binary search for sparse-unsafe
		double[] c = out.getDenseBlockValues();
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
							kplus.execute2(kbuff, genexec(0, b, scalars, m, n, rix+i, i, j));
					//process current non-zero
					lastj = aix[k];
					kplus.execute2(kbuff, genexec(avals[k], b, scalars, m, n, rix+i, i, lastj));
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					kplus.execute2(kbuff, genexec(0, b, scalars, m, n, rix+i, i, j));
			lnnz += ((c[i] = kbuff._sum)!=0) ? 1 : 0;
		}
		return lnnz;
	}
	
	private long executeSparseRowAggMxx(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		double initialVal = (_aggOp==AggOp.MIN) ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
		ValueFunction vfun = getAggFunction();
		
		//note: sequential scan algorithm for both sparse-safe and -unsafe 
		//in order to avoid binary search for sparse-unsafe 
		double[] c = out.getDenseBlockValues();
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
							tmp = vfun.execute(tmp, genexec(0, b, scalars, m, n, rix+i, i, j));
					//process current non-zero
					lastj = aix[k];
					tmp = vfun.execute( tmp, genexec(avals[k], b, scalars, m, n, rix+i, i, lastj));
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					tmp = vfun.execute(tmp, genexec(0, b, scalars, m, n, rix+i, i, j));
			lnnz += ((c[i] = tmp)!=0) ? 1 : 0;
		}
		return lnnz;
	}

	private long executeSparseColAggSum(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		KahanFunction kplus = (KahanFunction) getAggFunction();
		KahanObject kbuff = new KahanObject(0, 0);
		double[] corr = new double[n];
		
		//note: sequential scan algorithm for both sparse-safe and -unsafe
		//in order to avoid binary search for sparse-unsafe
		double[] c = out.getDenseBlockValues();
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
							kplus.execute2(kbuff, genexec(0, b, scalars, m, n, rix+i, i, j));
							c[j] = kbuff._sum;
							corr[j] = kbuff._correction;
						}
					//process current non-zero
					lastj = aix[k];
					kbuff.set(c[aix[k]], corr[aix[k]]);
					kplus.execute2(kbuff, genexec(avals[k], b, scalars, m, n, rix+i, i, lastj));
					c[aix[k]] = kbuff._sum;
					corr[aix[k]] = kbuff._correction;
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++) {
					kbuff.set(c[j], corr[j]);
					kplus.execute2(kbuff, genexec(0, b, scalars, m, n, rix+i, i, j));
					c[j] = kbuff._sum;
					corr[j] = kbuff._correction;
				}
		}
		return -1;
	}
	
	private long executeSparseColAggMxx(SparseBlock sblock, SideInput[] b, double[] scalars,
			MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		double initialVal = (_aggOp==AggOp.MIN) ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
		ValueFunction vfun = getAggFunction();
		double[] c = out.getDenseBlockValues();
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
							c[j] = vfun.execute(c[j], genexec(0, b, scalars, m, n, rix+i, i, j));
							count[j] ++;
						}
					//process current non-zero
					lastj = aix[k];
					c[aix[k]] = vfun.execute(c[aix[k]], genexec(avals[k], b, scalars, m, n, rix+i, i, lastj));
					count[aix[k]] ++;
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					c[j] = vfun.execute(c[j], genexec(0, b, scalars, m, n, rix+i, i, j));
		}
		
		return -1;
	}
	
	private double executeSparseAggSum(SparseBlock sblock, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru, long rix)
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
							kplus.execute2(kbuff, genexec(0, b, scalars, m, n, rix+i, i, j));
					//process current non-zero
					lastj = aix[k];
					kplus.execute2(kbuff, genexec(avals[k], b, scalars, m, n, rix+i, i, lastj));
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					kplus.execute2(kbuff, genexec(0, b, scalars, m, n, rix+i, i, j));
		}
		return kbuff._sum;
	}
	
	private double executeSparseAggMxx(SparseBlock sblock, SideInput[] b, double[] scalars,
			int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		double ret = (_aggOp==AggOp.MIN) ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
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
							ret = vfun.execute(ret, genexec(0, b, scalars, m, n, rix+i, i, j));
					//process current non-zero
					lastj = aix[k];
					ret = vfun.execute(ret, genexec(avals[k], b, scalars, m, n, rix+i, i, lastj));
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					ret = vfun.execute(ret, genexec(0, b, scalars, m, n, rix+i, i, j));
		}
		return ret;
	}

	private long executeDenseRowProd(DenseBlock a, SideInput[] b, double[] scalars,
		DenseBlock c, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		// single block output
		double[] lc = c.valuesAt(0);
		long lnnz = 0;
		if(a == null && !sparseSafe) {
			for(int i = rl; i < ru; i++) {
				for(int j = 0; j < n; j++) {
					lc[i] *= genexec(0, b, scalars, m, n, rix+i, i, j);
				}
				lnnz += (lc[i]!=0) ? 1 : 0;
			}
		}
		else if( a != null ) {
			for(int i = rl; i < ru; i++) {
				double[] avals = a.values(i);
				int aix = a.pos(i);
				for(int j = 0; j < n; j++) {
					double aval = avals[aix + j];
					if(j == 0) {
						lc[i] += genexec(aval, b, scalars, m, n, rix+i, i, j);
					} else if(aval != 0 || !sparseSafe)
						lc[i] *= genexec(aval, b, scalars, m, n, rix+i, i, j);
					else {
						lc[i] *= genexec(0, b, scalars, m, n, rix+i, i, j);
						break;
					}
					lnnz += (lc[i] != 0) ? 1 : 0;
				}
			}
		}
		return lnnz;
	}

	private long executeDenseColProd(DenseBlock a, SideInput[] b, double[] scalars,
		DenseBlock c, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		double[] lc = c.valuesAt(0);
		//track the cols that have a zero
		boolean[] zeroFlag = new boolean[n];
		if(a == null && !sparseSafe) {
			for(int i = rl; i < ru; i++) {
				for(int j = 0; j < n; j++) {
					if(!zeroFlag[j]) {
						lc[j] *= genexec(0, b, scalars, m, n, rix+i, i, j);
						zeroFlag[j] = true;
					}
				}
			}
		}
		else if(a != null) {
			for(int i = rl; i < ru; i++) {
				double[] avals = a.values(i);
				int aix = a.pos(i);
				for(int j = 0; j < n; j++) {
					if(!zeroFlag[j]) {
						double aval = avals[aix + j];
						if(i == 0)
							lc[j] += genexec(aval, b, scalars, m, n, rix + i, i, j);
						else if(aval != 0 || !sparseSafe)
							lc[j] *= genexec(aval, b, scalars, m, n, rix + i, i, j);
						else {
							lc[j] *= genexec(0, b, scalars, m, n, rix + i, i, j);
							zeroFlag[j] = true;
						}
					}
				}
			}
		}
		return -1;
	}

	private long executeSparseRowProd(SparseBlock sblock, SideInput[] b, double[] scalars,
		MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		double[] c = out.getDenseBlockValues();
		long lnnz = 0;
		for(int i = rl; i < ru; i++) {
			int lastj = -1;
			if(sblock != null && !sblock.isEmpty(i)) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				for(int j = apos; j < apos+alen; j++) {
					if(!sparseSafe) {
						// if there is a zero, no need to compute anymore
						if(aix[j] - (lastj+1) >= 1) {
							c[i] *= genexec(0, b, scalars, m, n, rix+i, i, j);
							break;
						}
					}
					lastj = aix[j];
					c[i] *= genexec(avals[j], b, scalars, m, n, rix+i, i, j);
				}
			}
			if(!sparseSafe)
				for(int j=lastj+1; j<n; j++)
					c[i] *= genexec(0, b, scalars, m, n, rix+i, i, j);
			lnnz += (c[i] != 0) ? 1 : 0;
		}
		return lnnz;
	}

	private long executeSparseColProd(SparseBlock sblock, SideInput[] b, double[] scalars,
		MatrixBlock out, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		double[] c = out.getDenseBlockValues();
		boolean[] zeroFlag = new boolean[n];

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
					if(!sparseSafe && (aix[k] - (lastj+1) >= 1)) {
						c[k] *= genexec(0, b, scalars, m, n, rix+i, i, k);
						zeroFlag[k] = true;
					}
					if(!zeroFlag[k]) {
						//process current non-zero
						lastj = aix[k];
						c[aix[k]] *= genexec(avals[aix[k]], b, scalars, m, n, rix+i, i, lastj);
					}
				}
			}
			//process empty rows or remaining zeros
			if(!sparseSafe)
				for(int j=lastj+1; j<n; j++) {
					if(i == 0) {
						c[j] += genexec(0, b, scalars, m, n, rix+i, i, j);
					} else {
						c[j] *= genexec(0, b, scalars, m, n, rix+i, i, j);
					}
				}
		}
		return -1;
	}

	//local execution where grix==rix
	protected final double genexec( double a, SideInput[] b,
		double[] scalars, int m, int n, int rix, int cix) {
		return genexec(a, b, scalars, m, n, rix, rix, cix);
	}
	
	//distributed execution with additional global row index
	protected abstract double genexec( double a, SideInput[] b,
		double[] scalars, int m, int n, long gix, int rix, int cix);
	
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
		public Double call() {
			if (!_a.isInSparseFormat())
				return executeDenseAndAgg(_a.getDenseBlock(), _b, _scalars, _rlen, _clen, _safe, _rl, _ru, 0);
			else
				return executeSparseAndAgg(_a.getSparseBlock(), _b, _scalars, _rlen, _clen, _safe, _rl, _ru, 0);
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
		public Long call() {
			if( _type==CellType.COL_AGG ) {
				_c = new MatrixBlock(1,_clen, false);
				_c.allocateDenseBlock();
			}
			if( !_a.isInSparseFormat() )
				return executeDense(_a.getDenseBlock(), _b, _scalars, _c, _rlen, _clen, _safe, _rl, _ru, 0);
			else
				return executeSparse(_a.getSparseBlock(), _b, _scalars, _c, _rlen, _clen, _safe, _rl, _ru, 0);
		}
		
		public MatrixBlock getResult() {
			return _c;
		}
	}
}
