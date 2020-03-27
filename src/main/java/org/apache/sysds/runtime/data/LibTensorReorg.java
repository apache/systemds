/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tugraz.sysds.runtime.data;

import org.apache.commons.lang.NotImplementedException;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;

import java.util.Arrays;
import java.util.List;

public class LibTensorReorg {
	//allow shallow dense/sparse copy for unchanged data (which is
	//safe due to copy-on-write and safe update-in-place handling)
	private static final boolean SHALLOW_COPY_REORG = true;

	private LibTensorReorg() {
		//prevent instantiation via private constructor
	}

	/**
	 * CP reshape operation (single input, single output tensor)
	 *
	 * @param in input tensor
	 * @param out output tensor
	 * @param dims dimensions
	 * @return output tensor
	 */
	public static BasicTensorBlock reshape(BasicTensorBlock in, BasicTensorBlock out, int[] dims) {
		long length = 1;
		for (int dim : dims) {
			length *= dim;
		}
		int[] inDims = in.getDims();
		//check validity
		if(in.getLength() != length) {
			throw new DMLRuntimeException("Reshape tensor requires consistent numbers of input/output cells (" +
					Arrays.toString(inDims) + ", " + Arrays.toString(dims) + ").");
		}

		//check for same dimensions
		if( Arrays.equals(inDims, dims)) {
			//copy incl dims, nnz
			if( SHALLOW_COPY_REORG )
				out.copyShallow(in);
			else // TODO deep copy
				out.copy(in);
			return out;
		}

		// TODO eval sparse output
		out._sparse = false;

		//set output dimensions
		out._dims = dims;
		out._nnz = in._nnz;

		//core reshape (sparse or dense)
		if(!in.isSparse() && !out.isSparse())
			reshapeDense(in, out, dims);
		else if(in.isSparse() && out.isSparse())
			throw new NotImplementedException();
		else if(in.isSparse())
			throw new NotImplementedException();
		else
			throw new NotImplementedException();

		return out;
	}

	private static void reshapeDense(BasicTensorBlock in, BasicTensorBlock out, int[] dims) {
		//reshape empty block
		if( in._denseBlock == null )
			return;

		//shallow dense by-row reshape (w/o result allocation)
		if( SHALLOW_COPY_REORG && in._denseBlock.numBlocks()==1 ) {
			//since the physical representation of dense matrices is always the same,
			//we don't need to create a copy, given our copy on write semantics.
			//however, note that with update in-place this would be an invalid optimization
			DenseBlock denseBlock = in._denseBlock;
			if (denseBlock instanceof DenseBlockBool) {
				DenseBlockBool specificBlock = (DenseBlockBool) denseBlock;
				out._denseBlock = DenseBlockFactory.createDenseBlock(specificBlock.getData(), dims);
			} else if (denseBlock instanceof DenseBlockString) {
				DenseBlockString specificBlock = (DenseBlockString) denseBlock;
				out._denseBlock = DenseBlockFactory.createDenseBlock(specificBlock.getData(), dims);
			}else if (denseBlock instanceof DenseBlockFP64) {
				out._denseBlock = DenseBlockFactory.createDenseBlock(in._denseBlock.valuesAt(0), dims);
			} else if (denseBlock instanceof DenseBlockFP32) {
				DenseBlockFP32 specificBlock = (DenseBlockFP32) denseBlock;
				out._denseBlock = DenseBlockFactory.createDenseBlock(specificBlock.getData(), dims);
			} else if (denseBlock instanceof DenseBlockInt64) {
				DenseBlockInt64 specificBlock = (DenseBlockInt64) denseBlock;
				out._denseBlock = DenseBlockFactory.createDenseBlock(specificBlock.getData(), dims);
			} else if (denseBlock instanceof DenseBlockInt32) {
				DenseBlockInt32 specificBlock = (DenseBlockInt32) denseBlock;
				out._denseBlock = DenseBlockFactory.createDenseBlock(specificBlock.getData(), dims);
			}
			return;
		}
		out.set(in);
	}

	/**
	 * MR/SPARK reshape interface - for reshape we cannot view blocks independently, and hence,
	 * there are different CP and MR interfaces.
	 *
	 * @param in indexed tensor block
	 * @param mcIn input tensor characteristics
	 * @param mcOut output tensor characteristics
	 * @param rowwise if true, reshape by row
	 * @param outputEmptyBlocks output blocks with nnz=0
	 * @return list of indexed tensor block
	 */
	public static List<IndexedTensorBlock> reshape(IndexedTensorBlock in, DataCharacteristics mcIn,
	                                               DataCharacteristics mcOut, boolean rowwise, boolean outputEmptyBlocks ) {
		throw new DMLRuntimeException("Spark reshape not implemented for tensors.");
	}
}
