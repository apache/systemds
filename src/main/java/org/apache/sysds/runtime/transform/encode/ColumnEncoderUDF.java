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

package org.apache.sysds.runtime.transform.encode;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.EvalNaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.utils.stats.TransformStatistics;

public class ColumnEncoderUDF extends ColumnEncoder {

	//TODO pass execution context through encoder factory for arbitrary functions not just builtin
	//TODO integration into IPA to ensure existence of unoptimized functions
	
	private String _fName;
	public int _domainSize = 1;

	protected ColumnEncoderUDF(int ptCols, String name) {
		super(ptCols); // 1-based
		_fName = name;
	}

	public ColumnEncoderUDF() {
		this(-1, null);
	}

	@Override
	protected TransformType getTransformType() {
		return TransformType.UDF;
	}

	@Override
	public void build(CacheBlock<?> in) {
		// do nothing
	}

	@Override
	public List<DependencyTask<?>> getBuildTasks(CacheBlock<?> in) {
		return null;
	}
	
	@Override
	public void applyDense(CacheBlock<?> in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		//create execution context and input
		ExecutionContext ec = ExecutionContextFactory.createContext(new Program(new DMLProgram()));
		//MatrixBlock col = out.slice(0, in.getNumRows()-1, _colID-1, _colID-1, new MatrixBlock());
		MatrixBlock col = out.slice(0, in.getNumRows()-1, outputCol, outputCol+_domainSize-1, new MatrixBlock());
		ec.setVariable("I", new ListObject(new Data[] {ParamservUtils.newMatrixObject(col, true)}));
		ec.setVariable("O", ParamservUtils.newMatrixObject(col, true));
		
		//call UDF function via eval machinery
		var fun = new EvalNaryCPInstruction(null, "eval", "",
			new CPOperand("O", ValueType.FP64, DataType.MATRIX),
			new CPOperand[] {
				new CPOperand(_fName, ValueType.STRING, DataType.SCALAR, true),
				new CPOperand("I", ValueType.UNKNOWN, DataType.LIST)});
		fun.processInstruction(ec);

		//obtain result and in-place write back
		MatrixBlock ret = ((MatrixObject)ec.getCacheableData("O")).acquireReadAndRelease();
		//out.leftIndexingOperations(ret, 0, in.getNumRows()-1, _colID-1, _colID-1, ret, UpdateType.INPLACE);
		//out.leftIndexingOperations(ret, 0, in.getNumRows()-1, outputCol, outputCol+_domainSize-1, ret, UpdateType.INPLACE);
		//out.copy(0, in.getNumRows()-1, _colID-1, _colID-1, ret, true);
		out.copy(0, in.getNumRows()-1, outputCol, outputCol+_domainSize-1, ret, true);

		if (DMLScript.STATISTICS)
			TransformStatistics.incUDFApplyTime(System.nanoTime() - t0);
	}

	public void updateDomainSizes(List<ColumnEncoder> columnEncoders) {
		if(_colID == -1)
			return;
		for(ColumnEncoder columnEncoder : columnEncoders) {
			int distinct = -1;
			if(columnEncoder instanceof ColumnEncoderRecode) {
				ColumnEncoderRecode columnEncoderRecode = (ColumnEncoderRecode) columnEncoder;
				distinct = columnEncoderRecode.getNumDistinctValues();
			}
			else if(columnEncoder instanceof ColumnEncoderBin) {
				distinct = ((ColumnEncoderBin) columnEncoder)._numBin;
			}
			else if(columnEncoder instanceof ColumnEncoderFeatureHash){
				distinct = (int) ((ColumnEncoderFeatureHash) columnEncoder).getK();
			}

			if(distinct != -1) {
				_domainSize = distinct;
				LOG.debug("DummyCoder for column: " + _colID + " has domain size: " + _domainSize);
			}
		}
	}
	
	@Override
	protected ColumnApplyTask<ColumnEncoderUDF> getSparseTask(CacheBlock<?> in,
		MatrixBlock out, int outputCol, int startRow, int blk)
	{
		throw new DMLRuntimeException("UDF encoders do not support sparse tasks.");
	}
	
	@Override
	public void mergeAt(ColumnEncoder other) {
		if(other instanceof ColumnEncoderUDF)
			return;
		super.mergeAt(other);
	}

	@Override
	public void allocateMetaData(FrameBlock meta) {
		// do nothing
		return;
	}

	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		// do nothing
		return meta;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		// do nothing
	}

	@Override
	protected double getCode(CacheBlock<?> in, int row) {
		throw new DMLRuntimeException("UDF encoders only support full column access.");
	}

	@Override
	protected double[] getCodeCol(CacheBlock<?> in, int startInd, int endInd, double[] tmp) {
		throw new DMLRuntimeException("UDF encoders only support full column access.");
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		out.writeUTF(_fName != null ? _fName : "");
		out.writeInt(_domainSize);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		_fName = in.readUTF();
		if(_fName.isEmpty())
			_fName = null;
		_domainSize = in.readInt();
	}
}
