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


package org.tugraz.sysds.runtime.lineage;

import org.apache.commons.lang.NotImplementedException;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.lineage.LineageItem.LineageItemType;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.DataGenOp;
import org.tugraz.sysds.hops.DataOp;
import org.tugraz.sysds.hops.Hop;
import org.tugraz.sysds.hops.LiteralOp;
import org.tugraz.sysds.hops.Hop.DataGenMethod;
import org.tugraz.sysds.hops.Hop.DataOpTypes;
import org.tugraz.sysds.hops.rewrite.HopRewriteUtils;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.lops.compile.Dag;
import org.tugraz.sysds.parser.DataExpression;
import org.tugraz.sysds.parser.DataIdentifier;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.BasicProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.Program;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.InstructionParser;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.instructions.cp.DataGenCPInstruction;
import org.tugraz.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.tugraz.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.tugraz.sysds.runtime.instructions.spark.SPInstruction.SPType;
import org.tugraz.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.tugraz.sysds.runtime.util.HDFSTool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.stream.Collectors;

public class LineageItemUtils {
	
	private static final String LVARPREFIX = "lvar";
	
	public static LineageItemType getType(String str) {
		if (str.length() == 1) {
			switch (str) {
				case "C":
					return LineageItemType.Creation;
				case "L":
					return LineageItemType.Literal;
				case "I":
					return LineageItemType.Instruction;
				case "D":
					return LineageItemType.Dedup;
				default:
					throw new DMLRuntimeException("Unknown LineageItemType given!");
			}
		} else
			throw new DMLRuntimeException("Unknown LineageItemType given!");
	}
	
	private static String getString(LineageItemType lit) {
		switch (lit) {
			case Creation:
				return "C";
			case Literal:
				return "L";
			case Instruction:
				return "I";
			case Dedup:
				return "D";
			default:
				throw new DMLRuntimeException("Unknown LineageItemType given!");
		}
	}
	
	private static String getString(LineageItem li) {
		return getString(li.getType());
	}
	
	public static String explainSingleLineageItem(LineageItem li) {
		StringBuilder sb = new StringBuilder();
		sb.append("(").append(li.getId()).append(") ");
		sb.append("(").append(getString(li)).append(") ");
		
		if (li.isLeaf()) {
			sb.append(li.getData()).append(" ");
		} else {
			if (li.getType() == LineageItemType.Dedup)
				sb.append(li.getOpcode()).append(li.getData()).append(" ");
			else
				sb.append(li.getOpcode()).append(" ");
			
			String ids = Arrays.stream(li.getInputs())
				.map(i -> String.format("(%d)", i.getId()))
				.collect(Collectors.joining(" "));
			sb.append(ids);
		}
		return sb.toString().trim();
	}
	
	public static Data computeByLineage(LineageItem root) {
		long rootId = root.getOpcode().equals("write") ?
			root.getInputs()[0].getId() : root.getId();
		String varname = LVARPREFIX + rootId;
		
		//recursively construct hops 
		root.resetVisitStatus();
		HashMap<Long, Hop> operands = new HashMap<>();
		rConstructHops(root, operands);
		Hop out = HopRewriteUtils.createTransientWrite(
			varname, operands.get(rootId));
		
		//generate instructions for temporary hops
		ExecutionContext ec = ExecutionContextFactory.createContext();
		BasicProgramBlock pb = new BasicProgramBlock(new Program());
		Dag<Lop> dag = new Dag<>();
		Lop lops = out.constructLops();
		lops.addToDag(dag);
		pb.setInstructions(dag.getJobs(null,
			ConfigurationManager.getDMLConfig()));
		
		// reset cache due to cleaned data objects
		LineageCache.resetCache();
		//execute instructions and get result
		pb.execute(ec);
		return ec.getVariable(varname);
	}
	
	public static LineageItem[] getLineage(ExecutionContext ec, CPOperand... operands) {
		return Arrays.stream(operands).filter(c -> c!=null)
			.map(c -> ec.getLineage().getOrCreate(c)).toArray(LineageItem[]::new);
	}
	
	private static void rConstructHops(LineageItem item, HashMap<Long, Hop> operands) {
		if (item.isVisited())
			return;
		
		//recursively process children (ordering by data dependencies)
		if (!item.isLeaf())
			for (LineageItem c : item.getInputs())
				rConstructHops(c, operands);
		
		//process current lineage item
		//NOTE: we generate instructions from hops (but without rewrites) to automatically
		//handle execution types, rmvar instructions, and rewiring of inputs/outputs
		switch (item.getType()) {
			case Creation: {
				Instruction inst = InstructionParser.parseSingleInstruction(item.getData());
				
				if (inst instanceof DataGenCPInstruction) {
					DataGenCPInstruction rand = (DataGenCPInstruction) inst;
					HashMap<String, Hop> params = new HashMap<>();
					params.put(DataExpression.RAND_ROWS, new LiteralOp(rand.getRows()));
					params.put(DataExpression.RAND_COLS, new LiteralOp(rand.getCols()));
					params.put(DataExpression.RAND_DIMS, new LiteralOp(rand.getDims()));
					params.put(DataExpression.RAND_MIN, new LiteralOp(rand.getMinValue()));
					params.put(DataExpression.RAND_MAX, new LiteralOp(rand.getMaxValue()));
					params.put(DataExpression.RAND_PDF, new LiteralOp(rand.getPdf()));
					params.put(DataExpression.RAND_LAMBDA, new LiteralOp(rand.getPdfParams()));
					params.put(DataExpression.RAND_SPARSITY, new LiteralOp(rand.getSparsity()));
					params.put(DataExpression.RAND_SEED, new LiteralOp(rand.getSeed()));
					Hop datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params);
					datagen.setBlocksize(rand.getBlocksize());
					operands.put(item.getId(), datagen);
				} else if (inst instanceof VariableCPInstruction
						&& ((VariableCPInstruction) inst).isCreateVariable()) {
					String parts[] = InstructionUtils.getInstructionPartsWithValueType(inst.toString());
					DataType dt = DataType.valueOf(parts[4]);
					ValueType vt = dt == DataType.MATRIX ? ValueType.FP64 : ValueType.STRING;
					HashMap<String, Hop> params = new HashMap<>();
					params.put(DataExpression.IO_FILENAME, new LiteralOp(parts[2]));
					params.put(DataExpression.READROWPARAM, new LiteralOp(Long.parseLong(parts[6])));
					params.put(DataExpression.READCOLPARAM, new LiteralOp(Long.parseLong(parts[7])));
					params.put(DataExpression.READNNZPARAM, new LiteralOp(Long.parseLong(parts[8])));
					params.put(DataExpression.FORMAT_TYPE, new LiteralOp(parts[5]));
					DataOp pread = new DataOp(parts[1].substring(5), dt, vt, DataOpTypes.PERSISTENTREAD, params);
					pread.setFileName(parts[2]);
					operands.put(item.getId(), pread);
				}
				break;
			}
			case Instruction: {
				CPType ctype = InstructionUtils.getCPTypeByOpcode(item.getOpcode());
				SPType stype = InstructionUtils.getSPTypeByOpcode(item.getOpcode());
				
				if (ctype != null) {
					switch (ctype) {
						case AggregateUnary: {
							Hop input = operands.get(item.getInputs()[0].getId());
							Hop aggunary = HopRewriteUtils.createAggUnaryOp(input, item.getOpcode());
							operands.put(item.getId(), aggunary);
							break;
						}
						case Unary: {
							Hop input = operands.get(item.getInputs()[0].getId());
							Hop unary = HopRewriteUtils.createUnary(input, item.getOpcode());
							operands.put(item.getId(), unary);
							break;
						}
						case Binary: {
							//handle special cases of binary operations 
							String opcode = ("^2".equals(item.getOpcode()) 
								|| "*2".equals(item.getOpcode())) ? 
								item.getOpcode().substring(0, 1) : item.getOpcode();
							Hop input1 = operands.get(item.getInputs()[0].getId());
							Hop input2 = operands.get(item.getInputs()[1].getId());
							Hop binary = HopRewriteUtils.createBinary(input1, input2, opcode);
							operands.put(item.getId(), binary);
							break;
						}
						case Ternary: {
							operands.put(item.getId(), HopRewriteUtils.createTernaryOp(
								operands.get(item.getInputs()[0].getId()), 
								operands.get(item.getInputs()[1].getId()), 
								operands.get(item.getInputs()[2].getId()), item.getOpcode()));
							break;
						}
						case MatrixIndexing: {
							if( "rightIndex".equals(item.getOpcode()) )
								operands.put(item.getId(), HopRewriteUtils.createIndexingOp(
									operands.get(item.getInputs()[0].getId()), //input
									operands.get(item.getInputs()[1].getId()), //rl
									operands.get(item.getInputs()[2].getId()), //ru
									operands.get(item.getInputs()[3].getId()), //cl
									operands.get(item.getInputs()[4].getId()))); //cu
							else if( "leftIndex".equals(item.getOpcode()) )
								operands.put(item.getId(), HopRewriteUtils.createLeftIndexingOp(
									operands.get(item.getInputs()[0].getId()), //input
									operands.get(item.getInputs()[1].getId()), //rhs
									operands.get(item.getInputs()[2].getId()), //rl
									operands.get(item.getInputs()[3].getId()), //ru
									operands.get(item.getInputs()[4].getId()), //cl
									operands.get(item.getInputs()[5].getId()))); //cu
							break;
						}
						case Variable: { //cpvar, write
							operands.put(item.getId(), operands.get(item.getInputs()[0].getId()));
							break;
						}
						default:
							throw new DMLRuntimeException("Unsupported instruction "
									+ "type: " + ctype.name() + " (" + item.getOpcode() + ").");
					}
				} else if (stype == SPType.Reblock) {
					Hop input = operands.get(item.getInputs()[0].getId());
					input.setBlocksize(ConfigurationManager.getBlocksize());
					input.setRequiresReblock(true);
					operands.put(item.getId(), input);
				} else
					throw new DMLRuntimeException("Unsupported instruction: " + item.getOpcode());
				break;
			}
			case Literal: {
				CPOperand op = new CPOperand(item.getData());
				operands.put(item.getId(), ScalarObjectFactory
					.createLiteralOp(op.getValueType(), op.getName()));
				break;
			}
			case Dedup: {
				throw new NotImplementedException();
			}
		}
		
		item.setVisited();
	}
	
	public static LineageItem rDecompress(LineageItem item) {
		if (item.getType() == LineageItemType.Dedup) {
			LineageItem dedupInput = rDecompress(item.getInputs()[0]);
			ArrayList<LineageItem> inputs = new ArrayList<>();
			
			for (LineageItem li : item.getInputs()[1].getInputs())
				inputs.add(rDecompress(li));
			
			LineageItem li = new LineageItem(item.getInputs()[1].getName(),
				item.getInputs()[1].getData(),
				item.getInputs()[1].getOpcode(), inputs.toArray(new LineageItem[0]));
			
			li.resetVisitStatus();
			rSetDedupInputOntoOutput(item.getName(), li, dedupInput);
			li.resetVisitStatus();
			return li;
		} else {
			ArrayList<LineageItem> inputs = new ArrayList<>();
			if (item.getInputs() != null) {
				for (LineageItem li : item.getInputs())
					inputs.add(rDecompress(li));
			}
			return new LineageItem(item.getName(), item.getData(),
				item.getOpcode(), inputs.toArray(new LineageItem[0]));
		}
	}
	
	public static void writeTraceToHDFS(String trace, String fname) {
		try {
			HDFSTool.writeStringToHDFS(trace, fname);
			FileSystem fs = IOUtilFunctions.getFileSystem(fname);
			if (fs instanceof LocalFileSystem) {
				Path path = new Path(fname);
				IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
			}
			
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}
	
	private static void rSetDedupInputOntoOutput(String name, LineageItem item, LineageItem dedupInput) {
		if (item.isVisited())
			return;
		
		if (item.getInputs() != null)
			for (int i = 0; i < item.getInputs().length; i++) {
				LineageItem li = item.getInputs()[i];
				
				if (li.getName().equals(name))
					item.getInputs()[i] = dedupInput;
				
				rSetDedupInputOntoOutput(name, li, dedupInput);
			}
		
		item.setVisited();
	}
	
	public static LineageItem replace(LineageItem root, LineageItem liOld, LineageItem liNew) {
		root.resetVisitStatus();
		rReplace(root, liOld, liNew);
		root.resetVisitStatus();
		return root;
	}
	
	private static void rReplace(LineageItem current, LineageItem liOld, LineageItem liNew) {
		if( current.isVisited() || current.getInputs() == null )
			return;
		//process children until old item found, then replace
		for(int i=0; i<current.getInputs().length; i++) {
			LineageItem tmp = current.getInputs()[i];
			if( tmp == liOld )
				current.getInputs()[i] = liNew;
			else
				rReplace(tmp, liOld, liNew);
		}
		current.setVisited();
	}
}
