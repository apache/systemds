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
package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.math3.util.FastMath;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.util.ArrayList;
import java.util.concurrent.Callable;

import static org.apache.sysds.runtime.functionobjects.KahanPlus.getKahanPlusFnObject;
import static org.apache.sysds.runtime.instructions.InstructionUtils.*;

public class LibMatrixDNNLSTM {
	private static final int row_tile_size = 4;
	private static final boolean kahan = false;
	private static final boolean optimized = true;
	public static ArrayList<Callable<Long>> getLSTMWorkers(DnnParameters params) {
		ArrayList<Callable<Long>> ret = new ArrayList<>();
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int) (Math.ceil((double) params.N / k));

		//very small input => use less threads
		if(taskSize < row_tile_size && (params.D+params.M)*params.T < 256*25)
			taskSize = row_tile_size;
		for(int i = 0; i*taskSize < params.N; i++)
			ret.add(new LSTMExecutor(i*taskSize, Math.min((i+1)*taskSize, params.N),params));
		return ret;
	}

	public static void lstmTile(int n, int d, int T, int m, int start, int end, MatrixBlock x, MatrixBlock w,
							MatrixBlock bias, MatrixBlock out0, MatrixBlock c0, boolean return_sequences,
							MatrixBlock out, MatrixBlock cout, MatrixBlock cache_out, MatrixBlock cache_c, MatrixBlock cache_ifog){

		//inputs arrays
		double[] c_0_values = c0.getDenseBlockValues();
		double[] bias_values = bias.getDenseBlockValues();
		double[] out0_values = out0.getDenseBlockValues();
		double[] w_values = w.getDenseBlockValues();
		double[] x_values = x.getDenseBlockValues();

		double[] out_values = out.getDenseBlockValues();
		double[] cout_values = cout.getDenseBlockValues();
		double[] cache_out_values = cache_out.getDenseBlockValues();
		double[] cache_c_values = cache_c.getDenseBlockValues();
		double[] cache_ifog_values = cache_ifog.getDenseBlockValues();

		int c_prev_pointer;

		//constants
		final boolean biasAllocated = bias.isAllocated();
		final boolean xAllocated = x.isAllocated();
		final boolean wAllocated = w.isAllocated();
		final int tile_size_i = row_tile_size;
		final int tile_size_j = 32;
		final int tile_size_k = 1024;
		final int m_4 = 4*m;
		final int m_T = T*m;

		int[] pos_in_x = new int[tile_size_i];
		int pos_in_sequence;
		double[] ifog = new double[tile_size_i*4*m];

		KahanObject kbuff[] = kahan ? new KahanObject[tile_size_i*4*m] : null;
		if(kahan)
			for (int i = 0; i < tile_size_i*4*m; i++)
				kbuff[i] = new KahanObject(0,0);
		KahanPlus kplus = kahan ? getKahanPlusFnObject() : null;

		double[] out_prev_values = null;
		double[] c_prev_values = null;

		for( int bi = start; bi < end; bi+=tile_size_i ) {
			int bimin = Math.min(end, bi + tile_size_i);

			//init out_prev
			if (out0_values != null) {
				if (out_prev_values == null)
					out_prev_values = new double[m * tile_size_i];
				for (int i = bi, i_internal = 0; i < bimin; i++, i_internal++) {
					c_prev_pointer = i * m;
					for (int j = 0; j < m; j++)
						out_prev_values[j + i_internal * m] = out0_values[c_prev_pointer + j];
				}
			} else
				out_prev_values = new double[m * tile_size_i];

			//init c_prev
			if (c_0_values != null) {
				if (c_prev_values == null)
					c_prev_values = new double[m * tile_size_i];
				for (int i = bi, i_internal = 0; i < bimin; i++, i_internal++) {
					c_prev_pointer = i * m;
					for (int j = 0; j < m; j++)
						c_prev_values[j + i_internal * m] = c_0_values[c_prev_pointer + j];
				}
			} else
				c_prev_values = new double[m * tile_size_i];

			//calculate position of input token sequence for all rows in tile
			for (int i = bi, i_internal = 0; i < bimin; i++, i_internal++) {
				pos_in_x[i_internal] = i * x.getNumColumns();
			}
			//iterate timesteps
			for (int t = 0; t < T; t++) {
				pos_in_sequence = t * d;
				int offset_t_internal = t*m;
				int offset_t = offset_t_internal*n;
				int offset_t2 = offset_t*4;
				//init ifog with bias values
				for (int j = 0; j < 4 * m; j++) {
					//for all rows in the row tile
					for (int i = bi, i_internal = 0; i < bimin; i++, i_internal++) {
						if(kahan)
							kbuff[j + i_internal * m_4].set(biasAllocated ? bias_values[j] : 0.0, 0.0);
						else
							ifog[j + i_internal * m_4] = biasAllocated ? bias_values[j] : 0.0;
					}
				}

				//iterate input token tiles
				if(xAllocated)
					for (int bj = 0; bj < d; bj += tile_size_j)
						//iterate weight tiles
						if(wAllocated)
							for (int bk = 0, bjmin = Math.min(d, bj + tile_size_j); bk < m_4; bk += tile_size_k) {
								int bkmin = Math.min(m_4, bk + tile_size_k);

								//core loop: adds the input token to the ifog-gates
								for (int i = bi, i_internal = 0; i < bimin; i++, i_internal++) {
									int pos_internal_ifog_i = i_internal * m_4;
									int pos = pos_in_x[i_internal] + pos_in_sequence;
									for (int j = bj; j < bjmin; j++) {
										int offset_w = j * 4 * m;
										int offset_x = pos + j;
										for (int k = bk; k < bkmin; k++) {
											if (kahan)
												kplus.execute2(kbuff[pos_internal_ifog_i + k], x_values[offset_x] * w_values[k + offset_w]);
											else
												ifog[pos_internal_ifog_i + k] += x_values[offset_x] * w_values[k + offset_w];
										}
									}
								}
							}
				//iterate hidden state tiles
				for (int bj = 0; bj < m; bj += tile_size_j)
					//iterate weight tiles
					if(wAllocated)
						for (int bk = 0, bjmin = Math.min(m, bj + tile_size_j); bk < 4 * m; bk += tile_size_k) {
							int bkmin = Math.min(4 * m, bk + tile_size_k);

							//core loop: adds the hidden state to the ifog-gates
							for (int i = bi, i_internal = 0; i < bimin; i++, i_internal++) {
								int offset_out_prev = i_internal * m;
								int offset_internal = offset_out_prev*4;
								for (int j = bj; j < bjmin; j++){
									int offset_tmp = (j + d) * m_4;
									for (int k = bk; k < bkmin; k++){
										int offset_w = k + offset_tmp;
										if(kahan)
											kplus.execute2(kbuff[offset_internal + k], out_prev_values[offset_out_prev + j] * w_values[offset_w]);
										else
											ifog[offset_internal + k] += out_prev_values[offset_out_prev + j] * w_values[offset_w];
									}
								}
							}
						}

				//calculate new hidden state for the current tile
				for (int i = bi, i_internal = 0; i < bimin; i++, i_internal++) {
					//from now on only elementwise operations

					//calculate index offset for array operations
					int offset_internal_i = i_internal * 4 * m;
					int offset_internal_f = offset_internal_i + m;
					int offset_internal_o = offset_internal_f + m;
					int offset_internal_g = offset_internal_o + m;
					int offset_c_internal = i_internal * m;
					int offset_out = i*m_T + offset_t_internal;

					int offset_i = i*m;
					int offset_cache = offset_t + offset_i;
					int offset_cache_i = offset_t2 + offset_i*4;
					int offset_cache_f = offset_cache_i + m;
					int offset_cache_o = offset_cache_f + m;
					int offset_cache_g = offset_cache_o + m;

					for (int j = 0; j < m; j++) {
						double ig, fg, og,gg;
						if(kahan){
							ig = 1.0 / (FastMath.exp(-kbuff[offset_internal_i + j]._sum) + 1.0);
							fg = 1.0 / (FastMath.exp(-kbuff[offset_internal_f + j]._sum) + 1.0);
							og = 1.0 / (FastMath.exp(-kbuff[offset_internal_o + j]._sum) + 1.0);
							gg = FastMath.tanh(kbuff[offset_internal_g + j]._sum);
						} else{
							ig = 1.0 / (FastMath.exp(-ifog[offset_internal_i + j]) + 1.0);
							fg = 1.0 / (FastMath.exp(-ifog[offset_internal_f + j]) + 1.0);
							og = 1.0 / (FastMath.exp(-ifog[offset_internal_o + j]) + 1.0);
							gg = FastMath.tanh(ifog[offset_internal_g + j]);
						}
						//c_prev_values.shape = (N,M)
						double c = c_prev_values[offset_c_internal + j] * fg + ig * gg;
						double o = FastMath.tanh(c) * og;

						//out.shape = (N,T*M)
						if (return_sequences)
							out_values[offset_out + j] = o;
							//out.setValue(i, t * m + j, o);

						//set caches
						cache_out_values[offset_cache + j] = o;
						cache_c_values[offset_cache + j] = c;
						cache_ifog_values[offset_cache_i + j] = ig;
						cache_ifog_values[offset_cache_f + j] = fg;
						cache_ifog_values[offset_cache_o + j] = og;
						cache_ifog_values[offset_cache_g + j] = gg;

						c_prev_values[offset_c_internal + j] = c;
						out_prev_values[offset_c_internal + j] = o;

					}
				}
			}
			for (int i = bi, i_internal = 0; i < bimin; i++, i_internal++) {
				int offset_i = i*m;
				for (int j = 0; j < m; j++) {
					cout_values[offset_i + j] = c_prev_values[i_internal * m + j];
					if (!return_sequences)
						out_values[offset_i + j] = out_prev_values[i_internal * m + j];
				}
			}
		}
	}


	public static long lstmGeneric(DnnParameters params) {
		//applies the LSTM operation on the input matrices using the generic matrix block operations

		MatrixBlock x = params.input1, w = params.input2, bias = params.bias;
		MatrixBlock out = params.input3, c = params.input4;
		MatrixBlock cache_out = params.output3, cache_c = params.output4, cache_ifog = params.output5;

		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int M = params.M;

		//init Operators
		BinaryOperator plus = InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString(),k);
		BinaryOperator emult = InstructionUtils.parseBinaryOperator(Opcodes.MULT.toString(),k);
		UnaryOperator tanh =  InstructionUtils.parseUnaryOperator(Opcodes.TANH.toString(),k);
		UnaryOperator  sigmoid =  InstructionUtils.parseUnaryOperator(Opcodes.SIGMOID.toString(),k);
		AggregateBinaryOperator mmult = InstructionUtils.getMatMultOperator(k);

		//iterate time steps
		for (int t = 0; t < params.T; t++) {
			//Extract the current input vector
			MatrixBlock x_t = x.slice(0, x.rlen - 1, t*params.D , (t+1)*params.D - 1);

			// Compute input, forget, output, and g gates
			// ifog = input %*% W + b
			MatrixBlock ifog = x_t.append(out, true);
			ifog = ifog.aggregateBinaryOperations(ifog, w, mmult);
			ifog = ifog.binaryOperations(plus, bias);

			// Apply sigmoid to i, f, o gates and tanh to g gate
			MatrixBlock ifo = ifog.slice(0, ifog.rlen - 1, 0, 3*M - 1).unaryOperations(sigmoid);
			MatrixBlock i = ifo.slice(0, ifog.rlen - 1, 0, M - 1);
			MatrixBlock f = ifo.slice(0, ifog.rlen - 1, M, 2*M - 1);
			MatrixBlock o = ifo.slice(0, ifog.rlen - 1, 2*M, 3*M - 1);
			MatrixBlock g = ifog.slice(0, ifog.rlen - 1, 3*M, 4*M - 1).unaryOperations(tanh);

			// Update cell state
			// c = ifog[,M+1:2*M]*c_prev + ifog[,1:M]*ifog[,3*M+1:4*M]  # shape (N, M)
			MatrixBlock tmp = i.binaryOperations(emult, g);
			c = f.binaryOperations(emult, c).binaryOperations(plus, tmp, t == params.T-1 ? params.output2 : null);

			// Compute output
			// out_t = ifog[,2*M+1:3*M] * tanh::forward(c)  # shape (N, M)
			tmp = c.unaryOperations(tanh);
			if(params.return_sequences){
				out = o.binaryOperations(emult, tmp);
				params.output.leftIndexingOperations(out, 0, out.rlen - 1, t*M,(t + 1)*M - 1,
						null, MatrixObject.UpdateType.INPLACE );
			}
			else
				out = o.binaryOperations(emult, tmp, t == params.T-1 ? params.output : null);

			//store caches
			ifog = ifo.append(g, true);
			MatrixBlock cache_out_t = LibMatrixReorg.reshape(out, new MatrixBlock(), 1, cache_out.clen, true);
			cache_out.leftIndexingOperations(cache_out_t, t, t,0, cache_out.clen - 1, null, MatrixObject.UpdateType.INPLACE );

			MatrixBlock cache_c_t = LibMatrixReorg.reshape(c, new MatrixBlock(), 1, cache_c.clen, true);
			cache_c.leftIndexingOperations(cache_c_t, t, t,0, cache_c.clen - 1, null, MatrixObject.UpdateType.INPLACE );

			MatrixBlock cache_ifog_t = LibMatrixReorg.reshape(ifog, new MatrixBlock(), 1, cache_ifog.clen, true);
			cache_ifog.leftIndexingOperations(cache_ifog_t, t, t,0,cache_ifog.clen - 1, null, MatrixObject.UpdateType.INPLACE );
		}
		return params.output.recomputeNonZeros();
	}

	@SuppressWarnings("unused")
	public static long lstmBackwardGeneric(DnnParameters params) {
		//TODO elias: currently we apply operator each on the whole batch,
		// -> slice the batch into small parts -> each thread processes one part through all timesteps (maybe even slice
		// the batch into smaller section) -> this should help keep the data local -> cache friendly

		//inputs
		MatrixBlock x = params.input1, w = params.input2, bias = params.bias;
		MatrixBlock out0 = params.input3, c0 = params.input4, dout = params.input5, dc = params.input6;
		MatrixBlock cache_out = params.input7, cache_c = params.input8, cache_ifog = params.input9;

		//outputs
		MatrixBlock dX = params.output, dW = null, db = null;

		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int M = params.M;

		//init Operators
		BinaryOperator plus = parseBinaryOperator(Opcodes.PLUS.toString(),k);
		BinaryOperator emult = parseBinaryOperator(Opcodes.MULT.toString(),k);
		ScalarOperator exp2 = parseScalarBinaryOperator(Opcodes.POW2.toString(),false, 0.0, k);
		ScalarOperator minus = parseScalarBinaryOperator(Opcodes.MINUS.toString(),true, 1.0, k);
		UnaryOperator tanh = parseUnaryOperator(Opcodes.TANH.toString(), k);
		UnaryOperator sprop = parseUnaryOperator(Opcodes.SPROP.toString(), k);
		AggregateUnaryOperator colsum = parseBasicAggregateUnaryOperator(Opcodes.UACKP.toString(),k);
		ReorgOperator transpose =  new ReorgOperator(SwapIndex.getSwapIndexFnObject(), k);
		AggregateBinaryOperator mmult = InstructionUtils.getMatMultOperator(k);

		//if(!params.return_sequences): get the predecessing partial derivative
		//else:						 load the predecessing partial derivative for timestep t in the for loop
		MatrixBlock dout_prev = params.return_sequences ? null : dout;

		//precompute t(W)
		//Note elias: optionally calculated it multiple times in for loop
		w = w.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);

		//iterate time steps reversely (backpropagation)
		for(int t = params.T - 1; t >= 0; t--){
			//get the predecessing partial derivative
			if(params.return_sequences)
				if(t == params.T-1)
					dout_prev = dout.slice(0, dout.rlen-1, t*M, (t+1)*M - 1);
				else
					dout_prev = dout.slice(0, dout.rlen-1, t*M, (t+1)*M - 1).binaryOperations(plus, dout_prev);

			//load and reuse cached results from forward pass for the current time step
			MatrixBlock c_t = LibMatrixReorg.reshape(cache_c.slice(t, t, 0, cache_c.clen - 1), new MatrixBlock(), params.N, M, true);
			MatrixBlock c_prev = t==0 ? c0 : LibMatrixReorg.reshape(cache_c.slice(t - 1, t - 1, 0, cache_c.clen - 1), new MatrixBlock(), params.N, M, true);
			MatrixBlock ifog = LibMatrixReorg.reshape(cache_ifog.slice(t, t,0, cache_ifog.clen - 1), new MatrixBlock(), params.N, 4*M, true);
			MatrixBlock i = ifog.slice(0, ifog.rlen - 1, 0, M -1);
			MatrixBlock f = ifog.slice(0, ifog.rlen - 1, M, 2*M -1);
			MatrixBlock o = ifog.slice(0, ifog.rlen - 1, 2*M, 3*M -1);
			MatrixBlock g = ifog.slice(0, ifog.rlen - 1, 3*M, ifog.clen -1);

			//dct = dct + o*tanh::backward(dout_t, ct)  # shape (N, M)
			MatrixBlock tanh_forward = c_t.unaryOperations(tanh);
			MatrixBlock tanh_back = tanh_forward.scalarOperations(exp2, new MatrixBlock())
					.scalarOperations(minus, new MatrixBlock());
			tanh_back = tanh_back.binaryOperations(emult, dout_prev);
			MatrixBlock tmp = o.binaryOperations(emult, tanh_back);
			dc = dc.binaryOperations(plus, tmp);

			//do = tanh::forward(ct) * dout_t  # output gate, shape (N, M)
			MatrixBlock d_o = tanh_forward.binaryOperations(emult, dout_prev);

			//df = c_prev * dct  # forget gate, shape (N, M)
			MatrixBlock d_f = c_prev.binaryOperations(emult, dc);

			//di = g * dct  # input gate, shape (N, M)
			MatrixBlock d_i = g.binaryOperations(emult, dc);

			//dg = i * dct  # g gate, shape (N, M)
			MatrixBlock d_g = i.binaryOperations(emult, dc);

			//di_raw = i * (1-i) * di
			//df_raw = f * (1-f) * df
			//do_raw = o * (1-o) * do
			//dg_raw = (1-g^2) * dg
			//difog_raw = cbind(di_raw, df_raw, do_raw, dg_raw)  # shape (N, 4M)
			MatrixBlock difog_raw = new MatrixBlock(params.N, 4*M, false);
			MatrixBlock di_raw = i.unaryOperations(sprop, new MatrixBlock()).binaryOperations(emult, d_i);
			difog_raw.leftIndexingOperations(di_raw,0, difog_raw.rlen - 1, 0, M-1, null,
					MatrixObject.UpdateType.INPLACE);
			MatrixBlock df_raw = f.unaryOperations(sprop, new MatrixBlock()).binaryOperations(emult, d_f);
			difog_raw.leftIndexingOperations(df_raw,0, difog_raw.rlen - 1, M, 2*M-1, null,
					MatrixObject.UpdateType.INPLACE);
			MatrixBlock do_raw = o.unaryOperations(sprop, new MatrixBlock()).binaryOperations(emult, d_o);
			difog_raw.leftIndexingOperations(do_raw,0, difog_raw.rlen - 1, 2*M, 3*M-1, null,
					MatrixObject.UpdateType.INPLACE);
			MatrixBlock dg_raw = g.scalarOperations(exp2, new MatrixBlock()).scalarOperations(minus, new MatrixBlock()).binaryOperations(emult, d_g);
			difog_raw.leftIndexingOperations(dg_raw,0, difog_raw.rlen - 1, 3*M, 4*M-1, null,
					MatrixObject.UpdateType.INPLACE);

			//load the current input vector and in the cached previous hidden state
			MatrixBlock x_t = x.slice(0, x.rlen - 1, t*params.D , (t+1)*params.D - 1);
			MatrixBlock out_prev = t==0 ? out0 : LibMatrixReorg.reshape(cache_out.slice(t - 1, t - 1, 0, cache_out.clen - 1), new MatrixBlock(), params.N, M, true);

			//merge mm for dx and dout_prev: input = cbind(X_t, out_prev)  # shape (N, D+M)
			MatrixBlock in_t = x_t.append(out_prev, true).reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);

			//dW = dW + t(input) %*% difog_raw  # shape (D+M, 4M)
			tmp = in_t.aggregateBinaryOperations(in_t, difog_raw, params.T == 1 ? params.output2 : null,mmult);
			dW = (t==params.T-1) ? tmp : dW.binaryOperations(plus, tmp, (t == 0) ? params.output2 : null);

			//db = db + colSums(difog_raw)  # shape (1, 4M)
			tmp = difog_raw.aggregateUnaryOperations(colsum, params.T == 1 ? params.output3 : null, difog_raw.rlen, new MatrixIndexes(1,1), true);
			db = (t==params.T-1) ? tmp : db.binaryOperations(plus, tmp,(t == 0) ? params.output3 : null);

			//dinput = difog_raw %*% t(W)  # shape (N, D+M)
			MatrixBlock dinput = difog_raw.aggregateBinaryOperations(difog_raw, w, mmult);

			//dX[,(t-1)*D+1:t*D] = dinput[,1:D]
			dX.leftIndexingOperations(dinput.slice(0, dinput.rlen - 1, 0, params.D-1),0, dX.rlen - 1, t*params.D, (t+1)*params.D - 1, null, MatrixObject.UpdateType.INPLACE);

			//dout_prev = dinput[,D+1:D+M]  # shape (N, M)
			//if(t == 0) -> dout0 = dout_prev
			dout_prev = dinput.slice(0, dinput.rlen - 1, params.D, dinput.clen - 1, (t == 0) ? params.output4 : null);

			//dc_prev = f * dct  # shape (N, M)
			//if(t == 0) -> dc0 = dc_prev
			dc = f.binaryOperations(emult, dc, (t == 0) ? params.output5 : null);
		}

		return params.output.recomputeNonZeros();
	}

	public static boolean checkLSTMInputForOptimisation(DnnParameters params) {
		//optimised just for FP64 single block or Empty:
//		System.out.println(!params.input1.isAllocated() + " | " + !params.input1.sparse + " | " + (params.input1.denseBlock.numBlocks() == 1));
//		System.out.println(!params.input2.isAllocated() + " | " + !params.input2.sparse + " | " + (params.input2.denseBlock.numBlocks() == 1));
//		System.out.println(!params.bias.isAllocated() + " | " + !params.bias.sparse + " | " + (params.bias.denseBlock.numBlocks() == 1));
//		System.out.println(!params.input4.isAllocated() + " | " + !params.input4.sparse + " | " + (params.input4.denseBlock.numBlocks() == 1));
//		System.out.println(!params.input3.isAllocated() + " | " + !params.input3.sparse + " | " + (params.input3.denseBlock.numBlocks() == 1));
//		System.out.println(optimized);

		//largest output size if cache_ifog (T, N*M)
		boolean fits_FP64 = (UtilFunctions.prod(new int[]{params.T,params.N,params.M}) < Integer.MAX_VALUE);

		return  (!params.input1.isAllocated() || (!params.input1.sparse && params.input1.denseBlock.numBlocks() == 1))
				&& (!params.input2.isAllocated() || (!params.input2.sparse && params.input2.denseBlock.numBlocks() == 1))
				&& (!params.bias.isAllocated() || (!params.bias.sparse && params.bias.denseBlock.numBlocks() == 1))
				&& (!params.input4.isAllocated() || (!params.input4.sparse && params.input4.denseBlock.numBlocks() == 1))
				&& (!params.input3.isAllocated() || (!params.input3.sparse && params.input3.denseBlock.numBlocks() == 1))
				&& fits_FP64
				&& optimized;
	}

	public static boolean checkLSTMBackwardInputForOptimisation(DnnParameters params) {
		return false;
	}

	private static class LSTMExecutor implements Callable<Long> {
		protected final int _rl, _ru;
		protected final DnnParameters _params;

		public LSTMExecutor(int rl, int ru, DnnParameters params) {
			_rl = rl;
			_ru = ru;
			_params = params;
		}

		@Override
		public Long call() throws Exception {
			lstmTile(_params.N, _params.D, _params.T, _params.M, _rl, _ru, _params.input1, _params.input2, _params.bias, _params.input3, _params.input4, _params.return_sequences, _params.output, _params.output2, _params.output3, _params.output4, _params.output5);
			//multithreaded nnz maintenance of current working set
			return _params.output.recomputeNonZeros(_rl, _ru - 1);
		}
	}
}
