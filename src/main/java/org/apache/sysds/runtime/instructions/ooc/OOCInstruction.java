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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

import java.util.HashMap;

public abstract class OOCInstruction extends Instruction {
	protected static final Log LOG = LogFactory.getLog(OOCInstruction.class.getName());

	public enum OOCType {
		Reblock, Tee, Binary, Unary, AggregateUnary, AggregateBinary, MAPMM, MMTSJ, Reorg, CM
	}

	protected final OOCInstruction.OOCType _ooctype;
	protected final boolean _requiresLabelUpdate;

	protected OOCInstruction(OOCInstruction.OOCType type, String opcode, String istr) {
		this(type, null, opcode, istr);
	}

	protected OOCInstruction(OOCInstruction.OOCType type, Operator op, String opcode, String istr) {
		super(op);
		_ooctype = type;
		instString = istr;
		instOpcode = opcode;

		_requiresLabelUpdate = super.requiresLabelUpdate();
	}

	@Override
	public IType getType() {
		return IType.OUT_OF_CORE;
	}

	public OOCInstruction.OOCType getOOCInstructionType() {
		return _ooctype;
	}

	@Override
	public boolean requiresLabelUpdate() {
		return _requiresLabelUpdate;
	}

	@Override
	public String getGraphString() {
		return getOpcode();
	}

	@Override
	public Instruction preprocessInstruction(ExecutionContext ec) {
		// TODO
		return super.preprocessInstruction(ec);
	}

	@Override
	public abstract void processInstruction(ExecutionContext ec);

	@Override
	public void postprocessInstruction(ExecutionContext ec) {
		if(DMLScript.LINEAGE_DEBUGGER)
			ec.maintainLineageDebuggerInfo(this);
	}

	/**
	 * Tracks blocks and their counts to enable early emission
	 * once all blocks for a given index are processed.
	 */
	public static class OOCMatrixBlockTracker {
		private final long _emitThreshold;
		private final HashMap<Long, MatrixBlock> _blocks;
		private final HashMap<Long, Integer> _cnts;

		public OOCMatrixBlockTracker(long emitThreshold) {
			_emitThreshold = emitThreshold;
			_blocks = new HashMap<>();
			_cnts = new HashMap<>();
		}

		/**
		 * Adds or updates a block for the given index and updates its internal count.
		 * @param idx   block index
		 * @param block MatrixBlock
		 * @return true if the block count reached the threshold (ready to emit), false otherwise
		 */
		public boolean putAndIncrementCount(Long idx, MatrixBlock block) {
			_blocks.put(idx, block);
			int newCnt = _cnts.getOrDefault(idx, 0) + 1;
			if( newCnt < _emitThreshold )
				_cnts.put(idx, newCnt);
			return newCnt == _emitThreshold;
		}

		public boolean incrementCount(Long idx) {
			int newCnt = _cnts.get(idx) + 1;
			if( newCnt < _emitThreshold )
				_cnts.put(idx, newCnt);
			return newCnt == _emitThreshold;
		}

		public void putAndInitCount(Long idx, MatrixBlock block) {
			_blocks.put(idx, block);
			_cnts.put(idx, 0);
		}

		public MatrixBlock get(Long idx) {
			return _blocks.get(idx);
		}

		public void remove(Long idx) {
			_blocks.remove(idx);
			_cnts.remove(idx);
		}
	}
}
