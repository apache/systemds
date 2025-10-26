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
package org.apache.sysds.runtime.controlprogram;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.api.jmlc.JMLCUtils;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.ParseInfo;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.lineage.LineageCache;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MetaData;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.utils.stats.RecompileStatistics;
import org.apache.sysds.utils.Statistics;

public abstract class ProgramBlock implements ParseInfo {
	public static final String PRED_VAR = "__pred";

	protected static final Log LOG = LogFactory.getLog(ProgramBlock.class.getName());
	public static boolean CHECK_MATRIX_PROPERTIES = false;

	protected Program _prog; // pointer to Program this ProgramBlock is part of

	// optional exit instructions, necessary for proper cleanup in while/for/if
	// in case a variable needs to be removed (via rmvar) after the control block
	protected Instruction _exitInstruction = null; // single packed rmvar

	// additional attributes for recompile
	protected StatementBlock _sb = null;
	protected long _tid = 0; // by default _t0

	public ProgramBlock(Program prog) {
		_prog = prog;
	}

	////////////////////////////////////////////////
	// getters, setters and similar functionality
	////////////////////////////////////////////////

	public Program getProgram() {
		return _prog;
	}

	public StatementBlock getStatementBlock() {
		return _sb;
	}

	public void setStatementBlock(StatementBlock sb) {
		_sb = sb;
	}

	public void setThreadID(long id) {
		_tid = id;
	}

	public boolean hasThreadID() {
		return _tid != 0;
	}

	public static boolean isThreadID(long tid) {
		return tid != 0;
	}

	public long getThreadID() {
		return _tid;
	}

	public void setExitInstruction(Instruction rmVar) {
		_exitInstruction = rmVar;
	}

	public Instruction getExitInstruction() {
		return _exitInstruction;
	}

	/**
	 * Get the list of child program blocks if nested; otherwise this method returns null.
	 * 
	 * @return list of program blocks
	 */
	public abstract ArrayList<ProgramBlock> getChildBlocks();

	/**
	 * Indicates if the program block is nested, i.e., if it contains other program blocks (e.g., loops).
	 * 
	 * @return true if nested
	 */
	public abstract boolean isNested();

	//////////////////////////////////////////////////////////
	// core instruction execution (program block, predicate)
	//////////////////////////////////////////////////////////

	/**
	 * Executes this program block (incl recompilation if required).
	 *
	 * @param ec execution context
	 */
	public abstract void execute(ExecutionContext ec);

	/**
	 * Executes given predicate instructions (incl recompilation if required)
	 *
	 * @param inst              list of instructions
	 * @param hops              high-level operator
	 * @param requiresRecompile true if requires recompile
	 * @param retType           value type of the return type
	 * @param ec                execution context
	 * @return scalar object
	 */
	public ScalarObject executePredicate(ArrayList<Instruction> inst, Hop hops, boolean requiresRecompile,
		ValueType retType, ExecutionContext ec) {
		ArrayList<Instruction> tmp = inst;

		// dynamically recompile instructions if enabled and required
		try {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			if(ConfigurationManager.isDynamicRecompilation() && requiresRecompile) {
				tmp = Recompiler.recompileHopsDag(hops, ec.getVariables(), null, false, true, _tid);
				tmp = JMLCUtils.cleanupRuntimeInstructions(tmp, PRED_VAR);
			}
			if(DMLScript.STATISTICS) {
				long t1 = System.nanoTime();
				RecompileStatistics.incrementRecompileTime(t1 - t0);
				if(tmp != inst)
					RecompileStatistics.incrementRecompilePred();
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException("Unable to recompile predicate instructions.", ex);
		}

		// actual instruction execution
		return executePredicateInstructions(tmp, retType, ec);
	}

	protected void executeExitInstructions(String ctx, ExecutionContext ec) {
		try {
			if(_exitInstruction != null)
				executeSingleInstruction(_exitInstruction, ec);
		}
		catch(DMLScriptException e) {
			throw e;
		}
		catch(Exception e) {
			throw new DMLRuntimeException(printBlockErrorLocation() + "Error evaluating " + ctx + " exit instructions ",
				e);
		}
	}

	protected void executeInstructions(ArrayList<Instruction> inst, ExecutionContext ec) {
		for(int i = 0; i < inst.size(); i++) {
			// indexed access required due to dynamic add
			Instruction currInst = inst.get(i);
			// execute instruction
			executeSingleInstruction(currInst, ec);
		}
	}

	protected ScalarObject executePredicateInstructions(ArrayList<Instruction> inst, ValueType retType,
		ExecutionContext ec) {
		// execute all instructions (indexed access required due to debug mode)
		for(Instruction currInst : inst) {
			executeSingleInstruction(currInst, ec);
		}

		// get scalar return
		ScalarObject ret = ec.getScalarInput(PRED_VAR, retType, false);

		// check and correct scalar ret type (incl save double to int)
		if(retType != null && retType != ret.getValueType())
			ret = ScalarObjectFactory.createScalarObject(retType, ret);

		// remove predicate variable
		ec.removeVariable(PRED_VAR);
		return ret;
	}

	private void executeSingleInstruction(Instruction currInst, ExecutionContext ec) {
		try {
			// start time measurement for statistics
			long t0 = (DMLScript.STATISTICS || DMLScript.STATISTICS_NGRAMS || LOG.isTraceEnabled())
				? System.nanoTime() : 0;

			// pre-process instruction (inst patching, listeners, lineage)
			Instruction tmp = currInst.preprocessInstruction(ec);

			// try to reuse instruction result from lineage cache
			if(!LineageCache.reuse(tmp, ec)) {
				long et0 = (!ReuseCacheType.isNone() || DMLScript.LINEAGE_ESTIMATE) ? System.nanoTime() : 0;

				// record IO Access
				ec.recordIOAccess(tmp);

				// process actual instruction
				tmp.processInstruction(ec);

				// cache result
				LineageCache.putValue(tmp, ec, et0);

				// post-process instruction (debug)
				tmp.postprocessInstruction(ec);

				// maintain aggregate statistics
				if(DMLScript.STATISTICS) {
					Statistics.maintainCPHeavyHitters(tmp.getExtendedOpcode(), System.nanoTime() - t0);
				}

				if (DMLScript.STATISTICS_NGRAMS)
					Statistics.maintainNGramsFromLineage(tmp, ec, t0);
			}

			// optional trace information (instruction and runtime)
			if(LOG.isTraceEnabled()) {
				long t1 = System.nanoTime();
				String time = String.format("%.3f", ((double) t1 - t0) / 1000000000);
				LOG.trace("Instruction: " + tmp + " (executed in " + time + "s).");
			}

			// optional check for correct nnz and sparse/dense representation of all
			// variables in symbol table (for tracking source of wrong representation)
			if(CHECK_MATRIX_PROPERTIES) {
				checkSparsity(tmp, ec.getVariables(), ec);
				checkFederated(ec.getVariables());
			}
		}
		catch(DMLScriptException e) {
			throw e;
		}
		catch(Exception e) {
			throw new DMLRuntimeException(
				printBlockErrorLocation() + "Error evaluating instruction: " + currInst.toString(), e);
		}
	}

	protected UpdateType[] prepareUpdateInPlaceVariables(ExecutionContext ec, long tid) {
		if(_sb == null || _sb.getUpdateInPlaceVars().isEmpty())
			return null;

		ArrayList<String> varnames = _sb.getUpdateInPlaceVars();
		UpdateType[] flags = new UpdateType[varnames.size()];
		for(int i = 0; i < flags.length; i++) {
			String varname = varnames.get(i);
			if(!ec.isMatrixObject(varname))
				continue;
			MatrixObject mo = ec.getMatrixObject(varname);
			flags[i] = mo.getUpdateType();
			// create deep copy if required and if it fits in thread-local mem budget
			if(flags[i] == UpdateType.COPY && OptimizerUtils.getLocalMemBudget() / 2 > OptimizerUtils
				.estimateSizeExactSparsity(mo.getDataCharacteristics())) {
				MatrixObject moNew = new MatrixObject(mo);
				MatrixBlock mbVar = mo.acquireRead();
				moNew.acquireModify(mbVar instanceof CompressedMatrixBlock ? new CompressedMatrixBlock(
					(CompressedMatrixBlock) mbVar) : !mbVar.isInSparseFormat() ? new MatrixBlock(
						mbVar) : new MatrixBlock(mbVar, MatrixBlock.DEFAULT_INPLACE_SPARSEBLOCK, true));
				moNew.setFileName(mo.getFileName() + Lop.UPDATE_INPLACE_PREFIX + tid);
				mo.release();
				// cleanup old variable (e.g., remove from buffer pool)
				if(ec.removeVariable(varname) != null)
					ec.cleanupCacheableData(mo);
				moNew.release(); // after old removal to avoid unnecessary evictions
				moNew.setUpdateType(UpdateType.INPLACE);
				ec.setVariable(varname, moNew);
			}
		}

		return flags;
	}

	protected void resetUpdateInPlaceVariableFlags(ExecutionContext ec, UpdateType[] flags) {
		if(flags == null)
			return;
		// reset update-in-place flag to pre-loop status
		ArrayList<String> varnames = _sb.getUpdateInPlaceVars();
		for(int i = 0; i < varnames.size(); i++)
			if(ec.getVariable(varnames.get(i)) != null && flags[i] != null) {
				MatrixObject mo = ec.getMatrixObject(varnames.get(i));
				mo.setUpdateType(flags[i]);
			}
	}

	private static void checkSparsity(Instruction lastInst, LocalVariableMap vars, ExecutionContext ec) {
		for(String varname : vars.keySet()) {
			Data dat = vars.get(varname);
			if(dat instanceof MatrixObject) {
				MatrixObject mo = (MatrixObject) dat;
				if(mo.isDirty() && !mo.isPartitioned()) {
					MatrixBlock mb = mo.acquireRead();
					boolean sparse1 = mb.isInSparseFormat();
					long nnz1 = mb.getNonZeros();
					synchronized(mb) { // potential state change
						mb.recomputeNonZeros();
						mb.examSparsity();
					}
					if(mb.isInSparseFormat() && mb.isAllocated()) {
						mb.getSparseBlock().checkValidity(mb.getNumRows(), mb.getNumColumns(), mb.getNonZeros(), true);
					}
					boolean sparse2 = mb.isInSparseFormat();
					long nnz2 = mb.getNonZeros();
					mo.release();

					if(nnz1 != nnz2)
						throw new DMLRuntimeException("Matrix nnz meta data was incorrect: (" + varname + ", actual="
							+ nnz1 + ", expected=" + nnz2 + ", inst=" + lastInst + ")");

					if(sparse1 != sparse2 && mb.isAllocated())
						throw new DMLRuntimeException("Matrix was in wrong data representation: (" + varname
							+ ", actual=" + sparse1 + ", expected=" + sparse2 + ", nrow=" + mb.getNumRows() + ", ncol="
							+ mb.getNumColumns() + ", nnz=" + nnz1 + ", inst=" + lastInst + ")");
				}
				MetaData meta = mo.getMetaData();
				if( mo.getRDDHandle() != null && !(meta instanceof MetaDataFormat 
					&& ((MetaDataFormat)meta).getFileFormat() != FileFormat.BINARY) ) {
					SparkUtils.checkSparsity(varname, ec);
				}
			}
		}
	}

	private static void checkFederated(LocalVariableMap vars) {
		for(String varname : vars.keySet()) {
			Data dat = vars.get(varname);
			if(!(dat instanceof CacheableData))
				continue;

			CacheableData<?> mo = (CacheableData<?>) dat;
			if(mo.isFederated()) {
				if(mo.getFedMapping().getMap().isEmpty())
					throw new DMLRuntimeException("Invalid empty FederationMap for: " + mo);
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////
	// store position information for program blocks
	///////////////////////////////////////////////////////////////////////////

	public String _filename;
	public int _beginLine, _beginColumn;
	public int _endLine, _endColumn;
	public String _text;

	@Override
	public void setFilename(String passed) {
		_filename = passed;
	}

	@Override
	public void setBeginLine(int passed) {
		_beginLine = passed;
	}

	@Override
	public void setBeginColumn(int passed) {
		_beginColumn = passed;
	}

	@Override
	public void setEndLine(int passed) {
		_endLine = passed;
	}

	@Override
	public void setEndColumn(int passed) {
		_endColumn = passed;
	}

	@Override
	public void setText(String text) {
		_text = text;
	}

	@Override
	public String getFilename() {
		return _filename;
	}

	@Override
	public int getBeginLine() {
		return _beginLine;
	}

	@Override
	public int getBeginColumn() {
		return _beginColumn;
	}

	@Override
	public int getEndLine() {
		return _endLine;
	}

	@Override
	public int getEndColumn() {
		return _endColumn;
	}

	@Override
	public String getText() {
		return _text;
	}

	public String printBlockErrorLocation() {
		return "ERROR: Runtime error in program block generated from statement block between lines " + _beginLine
			+ " and " + _endLine + " -- ";
	}

	/**
	 * Set parse information.
	 *
	 * @param parseInfo parse information, such as beginning line position, beginning column position, ending line
	 *                  position, ending column position, text, and filename
	 */
	public void setParseInfo(ParseInfo parseInfo) {
		setBeginLine(parseInfo.getBeginLine());
		setBeginColumn(parseInfo.getBeginColumn());
		setEndLine(parseInfo.getEndLine());
		setEndColumn(parseInfo.getEndColumn());
		setText(parseInfo.getText());
		setFilename(parseInfo.getFilename());
	}
}
