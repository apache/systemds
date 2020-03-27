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
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

public abstract class SpoofOuterProduct extends SpoofOperator
{
	private static final long serialVersionUID = 2948612259863710279L;
	
	private static final int L2_CACHESIZE = 256 * 1024; //256KB (common size)
	
	public enum OutProdType {
		LEFT_OUTER_PRODUCT,
		RIGHT_OUTER_PRODUCT,
		CELLWISE_OUTER_PRODUCT, // (e.g., X*log(sigmoid(-(U%*%t(V)))))  )
		AGG_OUTER_PRODUCT       // (e.g.,sum(X*log(U%*%t(V)+eps)))   )
	}
	
	protected OutProdType _outerProductType;
	
	public SpoofOuterProduct(OutProdType type) {
		setOuterProdType(type);
	}
	
	public void setOuterProdType(OutProdType type) {
		_outerProductType = type;
	}
	
	public OutProdType getOuterProdType() {
		return _outerProductType;
	}
	
	@Override
	public String getSpoofType() {
		return "OP" +  getClass().getName().split("\\.")[1];
	}
	
	@Override
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects)	
	{
		//sanity check
		if( inputs==null || inputs.size() < 3 )
			throw new RuntimeException("Invalid input arguments.");
		if( inputs.get(0).isEmptyBlock(false) )
			return new DoubleObject(0);
		
		//input preparation
		DenseBlock[] ab = getDenseMatrices(prepInputMatrices(inputs, 1, 2, true, false));
		SideInput[] b = prepInputMatrices(inputs, 3, false);
		double[] scalars = prepInputScalars(scalarObjects);
		
		//core sequential execute
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();
		final int k = inputs.get(1).getNumColumns(); // rank
		
		MatrixBlock a = inputs.get(0);
		MatrixBlock out = new MatrixBlock(1, 1, false);
		out.allocateDenseBlock();
		
		if( !a.isInSparseFormat() )
			executeCellwiseDense(a.getDenseBlock(), ab[0], ab[1], b, scalars, out.getDenseBlock(), m, n, k, _outerProductType, 0, m, 0, n);
		else
			executeCellwiseSparse(a.getSparseBlock(), ab[0], ab[1], b, scalars, out, m, n, k, a.getNonZeros(), _outerProductType, 0, m, 0, n);
		return new DoubleObject(out.getDenseBlock().get(0, 0));
	}
	
	@Override
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, int numThreads)
	{
		//sanity check
		if( inputs==null || inputs.size() < 3 )
			throw new RuntimeException("Invalid input arguments.");
		if( inputs.get(0).isEmptyBlock(false) )
			return new DoubleObject(0);
		
		if( 2*inputs.get(0).getNonZeros()*inputs.get(1).getNumColumns() < PAR_MINFLOP_THRESHOLD )
			return execute(inputs, scalarObjects); //sequential
		
		//input preparation
		DenseBlock[] ab = getDenseMatrices(prepInputMatrices(inputs, 1, 2, true, false));
		SideInput[] b = prepInputMatrices(inputs, 3, false);
		double[] scalars = prepInputScalars(scalarObjects);
		
		//core sequential execute
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();
		final int k = inputs.get(1).getNumColumns(); // rank
		final long nnz = inputs.get(0).getNonZeros();
		double sum = 0;
		
		try 
		{
			ExecutorService pool = CommonThreadPool.get(k);
			ArrayList<ParOuterProdAggTask> tasks = new ArrayList<>();
			int numThreads2 = getPreferredNumberOfTasks(m, n, nnz, k, numThreads);
			int blklen = (int)(Math.ceil((double)m/numThreads2));
			for( int i=0; i<numThreads2 & i*blklen<m; i++ )
				tasks.add(new ParOuterProdAggTask(inputs.get(0), ab[0], ab[1], b, scalars, 
					m, n, k, _outerProductType, i*blklen, Math.min((i+1)*blklen,m), 0, n));
			//execute tasks
			List<Future<Double>> taskret = pool.invokeAll(tasks);
			pool.shutdown();
			for( Future<Double> task : taskret )
				sum += task.get();
		} 
		catch (Exception e) {
			throw new DMLRuntimeException(e);
		} 
		
		return new DoubleObject(sum);
	}
	
	@Override
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out)
	{
		//sanity check
		if( inputs==null || inputs.size() < 3 || out==null )
			throw new RuntimeException("Invalid input arguments.");
		
		//check empty result
		if( (_outerProductType == OutProdType.LEFT_OUTER_PRODUCT && inputs.get(1).isEmptyBlock(false)) //U is empty
			|| (_outerProductType == OutProdType.RIGHT_OUTER_PRODUCT &&  inputs.get(2).isEmptyBlock(false)) //V is empty
			|| inputs.get(0).isEmptyBlock(false) ) {  //X is empty
			out.examSparsity(); //turn empty dense into sparse
			return out;
		}
		
		//input preparation and result allocation (Allocate the output that is set by Sigma2CPInstruction) 
		if(_outerProductType == OutProdType.CELLWISE_OUTER_PRODUCT) {
			//assign it to the time and sparse representation of the major input matrix
			out.reset(inputs.get(0).getNumRows(), inputs.get(0).getNumColumns(), inputs.get(0).isInSparseFormat());
		}		
		else {	
			//if left outerproduct gives a value of k*n instead of n*k, change it back to n*k and then transpose the output
			if(_outerProductType == OutProdType.LEFT_OUTER_PRODUCT )
				out.reset(inputs.get(0).getNumColumns(), inputs.get(1).getNumColumns(), false); // n*k
			else if(_outerProductType == OutProdType.RIGHT_OUTER_PRODUCT )
				out.reset(inputs.get(0).getNumRows(), inputs.get(1).getNumColumns(), false); // m*k
		}
		
		//check for empty inputs; otherwise allocate result
		if( inputs.get(0).isEmptyBlock(false) )
			return out;
		out.allocateBlock();
		
		//input preparation
		DenseBlock[] ab = getDenseMatrices(prepInputMatrices(inputs, 1, 2, true, false));
		SideInput[] b = prepInputMatrices(inputs, 3, false);
		double[] scalars = prepInputScalars(scalarObjects);
		
		//core sequential execute
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();
		final int k = inputs.get(1).getNumColumns(); // rank
		
		MatrixBlock a = inputs.get(0);
		
		switch(_outerProductType) {
			case LEFT_OUTER_PRODUCT:
			case RIGHT_OUTER_PRODUCT:
				if( !a.isInSparseFormat() )
					executeDense(a.getDenseBlock(), ab[0], ab[1], b, scalars, out.getDenseBlock(), m, n, k, _outerProductType, 0, m, 0, n);
				else
					executeSparse(a.getSparseBlock(), ab[0], ab[1], b, scalars, out.getDenseBlock(), m, n, k, a.getNonZeros(), _outerProductType, 0, m, 0, n);
				break;
				
			case CELLWISE_OUTER_PRODUCT:
				if( !a.isInSparseFormat() )
					executeCellwiseDense(a.getDenseBlock(), ab[0], ab[1], b, scalars, out.getDenseBlock(), m, n, k, _outerProductType, 0, m, 0, n);
				else 
					executeCellwiseSparse(a.getSparseBlock(), ab[0], ab[1], b, scalars, out, m, n, k, a.getNonZeros(), _outerProductType, 0, m, 0, n);
				break;
	
			case AGG_OUTER_PRODUCT:
				throw new DMLRuntimeException("Wrong codepath for aggregate outer product.");	
		}
		
		//post-processing
		out.recomputeNonZeros();
		out.examSparsity();
		return out;
	}
	
	@Override
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out, int numThreads)	
	{
		//sanity check
		if( inputs==null || inputs.size() < 3 || out==null )
			throw new RuntimeException("Invalid input arguments.");
		
		//check empty result
		if( (_outerProductType == OutProdType.LEFT_OUTER_PRODUCT && inputs.get(1).isEmptyBlock(false)) //U is empty
			|| (_outerProductType == OutProdType.RIGHT_OUTER_PRODUCT && inputs.get(2).isEmptyBlock(false)) //V is empty
			|| inputs.get(0).isEmptyBlock(false) ) {  //X is empty
			out.examSparsity(); //turn empty dense into sparse
			return out; 
		}
		
		//input preparation and result allocation (Allocate the output that is set by Sigma2CPInstruction) 
		if(_outerProductType == OutProdType.CELLWISE_OUTER_PRODUCT)
		{
			//assign it to the time and sparse representation of the major input matrix
			out.reset(inputs.get(0).getNumRows(), inputs.get(0).getNumColumns(), inputs.get(0).isInSparseFormat());
			out.allocateBlock();
		}
		else
		{
			//if left outerproduct gives a value of k*n instead of n*k, change it back to n*k and then transpose the output
			if( _outerProductType == OutProdType.LEFT_OUTER_PRODUCT )
				out.reset(inputs.get(0).getNumColumns(),inputs.get(1).getNumColumns(), false); // n*k
			else if( _outerProductType == OutProdType.RIGHT_OUTER_PRODUCT )
				out.reset(inputs.get(0).getNumRows(),inputs.get(1).getNumColumns(), false); // m*k
			out.allocateDenseBlock();
		}	
		
		if( 2*inputs.get(0).getNonZeros()*inputs.get(1).getNumColumns() < PAR_MINFLOP_THRESHOLD )
			return execute(inputs, scalarObjects, out); //sequential
		
		//input preparation
		DenseBlock[] ab = getDenseMatrices(prepInputMatrices(inputs, 1, 2, true, false));
		SideInput[] b = prepInputMatrices(inputs, 3, false);
		double[] scalars = prepInputScalars(scalarObjects);
		
		//core sequential execute
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();
		final int k = inputs.get(1).getNumColumns(); // rank
		final long nnz = inputs.get(0).getNonZeros();
		
		MatrixBlock a = inputs.get(0);
		
		try 
		{
			ExecutorService pool = CommonThreadPool.get(numThreads);
			ArrayList<ParExecTask> tasks = new ArrayList<>();
			//create tasks (for wdivmm-left, parallelization over columns;
			//for wdivmm-right, parallelization over rows; both ensure disjoint results)
			
			if( _outerProductType == OutProdType.LEFT_OUTER_PRODUCT ) {
				//parallelize over column partitions
				int blklen = (int)(Math.ceil((double)n/numThreads));
				for( int j=0; j<numThreads & j*blklen<n; j++ )
					tasks.add(new ParExecTask(a, ab[0], ab[1], b, scalars, out, m, n, k,
						_outerProductType,  0, m, j*blklen, Math.min((j+1)*blklen, n)));
			}
			else { //right or cell-wise
				//parallelize over row partitions
				int numThreads2 = getPreferredNumberOfTasks(m, n, nnz, k, numThreads);
				int blklen = (int)(Math.ceil((double)m/numThreads2));
				for( int i=0; i<numThreads2 & i*blklen<m; i++ )
					tasks.add(new ParExecTask(a, ab[0], ab[1], b, scalars, out, m, n, k,
						_outerProductType, i*blklen, Math.min((i+1)*blklen,m), 0, n));
			}
			List<Future<Long>> taskret = pool.invokeAll(tasks);
			pool.shutdown();
			for( Future<Long> task : taskret )
				out.setNonZeros(out.getNonZeros() + task.get());
		} 
		catch (Exception e) {
			throw new DMLRuntimeException(e);
		}
		out.examSparsity();
		return out;
	}
	
	private static int getPreferredNumberOfTasks(int m, int n, long nnz, int rank, int k) {
		//compute number of tasks nk in range [k, 8k]
		int base = (int) Math.min(Math.min(8*k, m/32),
			Math.ceil((double)2*nnz*rank/PAR_MINFLOP_THRESHOLD));
		return UtilFunctions.roundToNext(base, k);
	}
	
	private void executeDense(DenseBlock a, DenseBlock u, DenseBlock v, SideInput[] b, double[] scalars,
		DenseBlock c, int m, int n, int k, OutProdType type, int rl, int ru, int cl, int cu )
	{
		//approach: iterate over non-zeros of w, selective mm computation
		//cache-conscious blocking: due to blocksize constraint (default 1000),
		//a blocksize of 16 allows to fit blocks of UV into L2 cache (256KB) 
		
		//NOTE: we don't create sparse side inputs w/ row-major cursors because 
		//cache blocking would lead to non-sequential access
		
		final int blocksizeIJ = 16; //u/v block (max at typical L2 size) 
		int cix = 0;
		//blocked execution
		for( int bi = rl; bi < ru; bi+=blocksizeIJ )
			for( int bj = cl, bimin = Math.min(ru, bi+blocksizeIJ); bj < cu; bj+=blocksizeIJ )
			{
				int bjmin = Math.min(cu, bj+blocksizeIJ);
				
				//core computation
				for( int i=bi; i<bimin; i++ ) {
					double[] avals = a.values(i);
					double[] uvals = u.values(i);
					int aix = a.pos(i), uix = u.pos(i);
					for( int j=bj; j<bjmin; j++)
						if( avals[aix+j] != 0 ) {
							int vix = v.pos(j);
							cix = (type == OutProdType.LEFT_OUTER_PRODUCT) ? vix : uix;
							genexecDense( avals[aix+j], uvals, uix, v.values(j), vix,
								b, scalars, c.values(j), cix, m, n, k, i, j); 
						}
				}
			}
	}
	
	private void executeCellwiseDense(DenseBlock a, DenseBlock u, DenseBlock v, SideInput[] b, double[] scalars,
		DenseBlock c, int m, int n, int k, OutProdType type, int rl, int ru, int cl, int cu )
	{
		//approach: iterate over non-zeros of w, selective mm computation
		//cache-conscious blocking: due to blocksize constraint (default 1000),
		//a blocksize of 16 allows to fit blocks of UV into L2 cache (256KB)
		
		//NOTE: we don't create sparse side inputs w/ row-major cursors because 
		//cache blocking would lead to non-sequential access
		
		final int blocksizeIJ = 16; //u/v block (max at typical L2 size)
		//blocked execution
		double sum = 0;
		for( int bi = rl; bi < ru; bi+=blocksizeIJ )
			for( int bj = cl, bimin = Math.min(ru, bi+blocksizeIJ); bj < cu; bj+=blocksizeIJ )
			{
				int bjmin = Math.min(cu, bj+blocksizeIJ);
				
				//core computation
				for( int i=bi; i<bimin; i++ ) {
					double[] avals = a.values(i);
					double[] uvals = u.values(i);
					int aix = a.pos(i), uix = u.pos(i);
					if(type == OutProdType.CELLWISE_OUTER_PRODUCT) {
						double[] cvals = c.values(i);
						for( int j=bj; j<bjmin; j++)
							if( avals[aix+j] != 0 )
								cvals[aix+j] = genexecCellwise( avals[aix+j], uvals, uix,
									v.values(j), v.pos(j), b, scalars, m, n, k, i, j );
					}
					else {
						for( int j=bj; j<bjmin; j++)
							if( avals[aix+j] != 0 )
								sum += genexecCellwise( avals[aix+j], uvals, uix,
									v.values(j), v.pos(j), b, scalars, m, n, k, i, j);
						
					}
				}
			}
		if( type != OutProdType.CELLWISE_OUTER_PRODUCT )
			c.set(0, 0, sum);
	}
	
	private void executeSparse(SparseBlock sblock, DenseBlock u, DenseBlock v, SideInput[] b, double[] scalars,
		DenseBlock c, int m, int n, int k, long nnz, OutProdType type, int rl, int ru, int cl, int cu) 
	{
		boolean left = (_outerProductType== OutProdType.LEFT_OUTER_PRODUCT);
		
		//approach: iterate over non-zeros of w, selective mm computation
		//blocked over ij, while maintaining front of column indexes, where the
		//blocksize is chosen such that we reuse each  Ui/Vj vector on average 8 times,
		//with custom blocksizeJ for wdivmm_left to avoid LLC misses on output.
		final int blocksizeI = (int) (8L*m*n/nnz);
		
		if( OptimizerUtils.getSparsity(m, n, nnz) < MatrixBlock.ULTRA_SPARSITY_TURN_POINT ) //ultra-sparse
		{
			//for ultra-sparse matrices, we do not allocate the index array because
			//its allocation and maintenance can dominate the total runtime.
			SideInput[] lb = createSparseSideInputs(b);
			
			//core wdivmm block matrix mult
			for( int i=rl; i<ru; i++ ) {
				if( sblock.isEmpty(i) ) continue;
				
				int wpos = sblock.pos(i);
				int wlen = sblock.size(i);
				int[] wix = sblock.indexes(i);
				double[] wvals = sblock.values(i);
				double[] uvals = u.values(i);
				int uix = u.pos(i);
				
				int index = (cl==0||sblock.isEmpty(i)) ? 0 : sblock.posFIndexGTE(i,cl);
				index = wpos + ((index>=0) ? index : n);
				for( ; index<wpos+wlen && wix[index]<cu; index++ ) {
					int jix = wix[index];
					genexecDense(wvals[index], uvals, uix, v.values(jix), v.pos(jix), lb, scalars,
						c.values(jix), (left ? v.pos(jix) : uix), m, n, k, i, wix[index]);
				}
			}
		}
		else //sparse
		{
			//NOTE: we don't create sparse side inputs w/ row-major cursors because 
			//cache blocking would lead to non-sequential access
			
			final int blocksizeJ = left ? Math.max(8,Math.min(L2_CACHESIZE/(k*8), blocksizeI)) : blocksizeI;
			int[] curk = new int[Math.min(blocksizeI,ru-rl)];
			
			for( int bi = rl; bi < ru; bi+=blocksizeI ) 
			{
				int bimin = Math.min(ru, bi+blocksizeI);
				//prepare starting indexes for block row
				for( int i=bi; i<bimin; i++ ) {
					int index = (cl==0||sblock.isEmpty(i)) ? 0 : sblock.posFIndexGTE(i,cl);
					curk[i-bi] = (index>=0) ? index : n;
				}
				
				//blocked execution over column blocks
				for( int bj = cl; bj < cu; bj+=blocksizeJ )
				{
					int bjmin = Math.min(cu, bj+blocksizeJ);
					//core wdivmm block matrix mult
					for( int i=bi; i<bimin; i++ ) {
						if( sblock.isEmpty(i) ) continue;
						
						int wpos = sblock.pos(i);
						int wlen = sblock.size(i);
						int[] wix = sblock.indexes(i);
						double[] wvals = sblock.values(i);
						double[] uvals = u.values(i);
						int uix = u.pos(i);
						
						int index = wpos + curk[i-bi];
						for( ; index<wpos+wlen && wix[index]<bjmin; index++ ) {
							int jix = wix[index];
							genexecDense(wvals[index], uvals, uix, v.values(jix), v.pos(jix), b, scalars,
								c.values(jix), (left ? wix[index]*k : uix), m, n, k, i, wix[index]);
						}
						curk[i-bi] = index - wpos;
					}
				}
			}
		}
	}
	
	private void executeCellwiseSparse(SparseBlock sblock, DenseBlock u, DenseBlock v, SideInput[] b, double[] scalars, 
		MatrixBlock out, int m, int n, int k, long nnz, OutProdType type, int rl, int ru, int cl, int cu ) 
	{
		//NOTE: we don't create sparse side inputs w/ row-major cursors because 
		//cache blocking would lead to non-sequential access
		
		final int blocksizeIJ = (int) (8L*m*n/nnz);
		int[] curk = new int[Math.min(blocksizeIJ, ru-rl)];
		
		if( !out.isInSparseFormat() ) //DENSE
		{
			DenseBlock c = out.getDenseBlock();
			double tmp = 0;
			for( int bi=rl; bi<ru; bi+=blocksizeIJ ) {
				int bimin = Math.min(ru, bi+blocksizeIJ);
				//prepare starting indexes for block row
				Arrays.fill(curk, 0); 
				//blocked execution over column blocks
				for( int bj=0; bj<n; bj+=blocksizeIJ ) {
					int bjmin = Math.min(n, bj+blocksizeIJ);
					for( int i=bi; i<bimin; i++ ) {
						if( sblock.isEmpty(i) ) continue;
						int wpos = sblock.pos(i);
						int wlen = sblock.size(i);
						int[] wix = sblock.indexes(i);
						double[] wvals = sblock.values(i);
						double[] cvals = c.values(i);
						double[] uvals = u.values(i);
						int uix = u.pos(i);
						int index = wpos + curk[i-bi];
						if( type == OutProdType.CELLWISE_OUTER_PRODUCT )
							for( ; index<wpos+wlen && wix[index]<bjmin; index++ ) {
								int jix = wix[index];
								cvals[jix] = genexecCellwise( wvals[index],
									uvals, uix, v.values(jix), v.pos(jix), b, scalars, m, n, k, i, wix[index] );
							}
						else
							for( ; index<wpos+wlen && wix[index]<bjmin; index++ ) {
								int jix = wix[index];
								tmp += genexecCellwise( wvals[index], 
									uvals, uix, v.values(jix), v.pos(jix), b, scalars, m, n, k, i, wix[index]);
							}
						curk[i-bi] = index - wpos;
					}
				}
			}
			if( type != OutProdType.CELLWISE_OUTER_PRODUCT )
				c.set(0, 0, tmp);
		}
		else //SPARSE
		{
			SparseBlock c = out.getSparseBlock();
			for( int bi=rl; bi<ru; bi+=blocksizeIJ ) {
				int bimin = Math.min(ru, bi+blocksizeIJ);
				//prepare starting indexes for block row
				Arrays.fill(curk, 0); 
				//blocked execution over column blocks
				for( int bj=0; bj<n; bj+=blocksizeIJ ) {
					int bjmin = Math.min(n, bj+blocksizeIJ);
					for( int i=bi; i<bimin; i++ ) {
						if( sblock.isEmpty(i) ) continue;
						int wpos = sblock.pos(i);
						int wlen = sblock.size(i);
						int[] wix = sblock.indexes(i);
						double[] wval = sblock.values(i);
						double[] uvals = u.values(i);
						int uix = u.pos(i);
						int index = wpos + curk[i-bi];
						for( ; index<wpos+wlen && wix[index]<bjmin; index++ ) {
							int jix = wix[index];
							c.append(i, wix[index], genexecCellwise( wval[index], uvals, uix,
								v.values(jix), v.pos(jix), b, scalars, m, n, k, i, wix[index] ));
						}
						curk[i-bi] = index - wpos;
					}
				}
			}
		}
	}
	
	protected abstract void genexecDense( double a, double[] u, int ui, double[] v, int vi, SideInput[] b,
		double[] scalars, double[] c, int ci, int m, int n, int k, int rowIndex, int colIndex);
	
	protected abstract double genexecCellwise( double a, double[] u, int ui, double[] v, int vi, SideInput[] b,
		double[] scalars, int m, int n, int k, int rowIndex, int colIndex);
	
	private class ParExecTask implements Callable<Long> 
	{
		private final MatrixBlock _a;
		private final DenseBlock _u;
		private final DenseBlock _v;
		private final SideInput[] _b;
		private final double[] _scalars;
		private final MatrixBlock _c;
		private final int _clen;
		private final int _rlen;
		private final int _k;
		private final OutProdType _type;
		private final int _rl;
		private final int _ru;
		private final int _cl;
		private final int _cu;
		
		protected ParExecTask( MatrixBlock a, DenseBlock u, DenseBlock v, SideInput[] b, double[] scalars , MatrixBlock c, int m, int n, int k, OutProdType type, int rl, int ru, int cl, int cu ) {
			_a = a;
			_u = u;
			_v = v;
			_b = b;
			_c = c;
			_scalars = scalars;
			_rlen = m;
			_clen = n;
			_k = k;
			_type = type;
			_rl = rl;
			_ru = ru;
			_cl = cl;
			_cu = cu;
		}
		
		@Override
		public Long call() {
			switch(_type)
			{
				case LEFT_OUTER_PRODUCT:
				case RIGHT_OUTER_PRODUCT:
					if( !_a.isInSparseFormat() )
						executeDense(_a.getDenseBlock(), _u, _v, _b, _scalars, _c.getDenseBlock(), _rlen, _clen, _k, _type, _rl, _ru, _cl, _cu);
					else
						executeSparse(_a.getSparseBlock(), _u, _v, _b, _scalars, _c.getDenseBlock(), _rlen, _clen, _k, _a.getNonZeros(), _type,  _rl, _ru, _cl, _cu);
					break;
				case CELLWISE_OUTER_PRODUCT:
					if( !_c.isInSparseFormat() )
						executeCellwiseDense(_a.getDenseBlock(), _u, _v, _b, _scalars, _c.getDenseBlock(), _rlen, _clen, _k, _type, _rl, _ru, _cl, _cu);
					else 
						executeCellwiseSparse(_a.getSparseBlock(), _u, _v, _b, _scalars, _c, _rlen, _clen, _k, _a.getNonZeros(), _type,  _rl, _ru, _cl, _cu);
					break;
				case AGG_OUTER_PRODUCT:
					throw new DMLRuntimeException("Wrong codepath for aggregate outer product.");
			}
			
			boolean left = (_outerProductType == OutProdType.LEFT_OUTER_PRODUCT);
			int rl = left ? _cl : _rl;
			int ru = left ? _cu : _ru;
			return _c.recomputeNonZeros(rl, ru-1, 0, _c.getNumColumns()-1);
		}
	}
	
	private class ParOuterProdAggTask implements Callable<Double> 
	{
		private final MatrixBlock _a;
		private final DenseBlock _u;
		private final DenseBlock _v;
		private final SideInput[] _b;
		private final double[] _scalars;
		private final int _rlen;
		private final int _clen;
		private final int _k;
		private final OutProdType _type;
		private final int _rl;
		private final int _ru;
		private final int _cl;
		private final int _cu;
		
		protected ParOuterProdAggTask( MatrixBlock a, DenseBlock u, DenseBlock v, SideInput[] b, double[] scalars, int m, int n, int k, OutProdType type, int rl, int ru, int cl, int cu ) {
			_a = a;
			_u = u;
			_v = v;
			_b = b;
			_scalars = scalars;
			_rlen = m;
			_clen = n;
			_k = k;
			_type = type;
			_rl = rl;
			_ru = ru;
			_cl = cl;
			_cu = cu;
		}
		
		@Override
		public Double call() {
			MatrixBlock out = new MatrixBlock(1, 1, false);
			out.allocateDenseBlock();
			if( !_a.isInSparseFormat() )
				executeCellwiseDense(_a.getDenseBlock(), _u, _v, _b, _scalars, out.getDenseBlock(), _rlen, _clen, _k, _type, _rl, _ru, _cl, _cu);
			else
				executeCellwiseSparse(_a.getSparseBlock(), _u, _v, _b, _scalars, out, _rlen, _clen, _k, _a.getNonZeros(), _type, _rl, _ru, _cl, _cu);
			return out.quickGetValue(0, 0);
		}
	}
}
