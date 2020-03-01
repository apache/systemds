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
import org.tugraz.sysds.runtime.instructions.spark.RandSPInstruction;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.tugraz.sysds.runtime.lineage.LineageItem.LineageItemType;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.OpOpN;
import org.tugraz.sysds.common.Types.ReOrgOp;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.AggBinaryOp;
import org.tugraz.sysds.hops.BinaryOp;
import org.tugraz.sysds.hops.DataGenOp;
import org.tugraz.sysds.hops.DataOp;
import org.tugraz.sysds.hops.Hop;
import org.tugraz.sysds.hops.LiteralOp;
import org.tugraz.sysds.hops.ReorgOp;
import org.tugraz.sysds.hops.Hop.DataGenMethod;
import org.tugraz.sysds.hops.Hop.DataOpTypes;
import org.tugraz.sysds.hops.rewrite.HopRewriteUtils;
import org.tugraz.sysds.lops.Binary;
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
import org.tugraz.sysds.runtime.instructions.cp.ScalarObject;
import org.tugraz.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.tugraz.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.tugraz.sysds.runtime.instructions.spark.SPInstruction.SPType;
import org.tugraz.sysds.runtime.instructions.cp.CPInstruction;
import org.tugraz.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.tugraz.sysds.runtime.util.HDFSTool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
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
		Map<Long, Hop> operands = new HashMap<>();
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
	
	private static void rConstructHops(LineageItem item, Map<Long, Hop> operands) {
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
					if( rand.output.getDataType() == DataType.TENSOR)
						params.put(DataExpression.RAND_DIMS, new LiteralOp(rand.getDims()));
					else {
						params.put(DataExpression.RAND_ROWS, new LiteralOp(rand.getRows()));
						params.put(DataExpression.RAND_COLS, new LiteralOp(rand.getCols()));
					}
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
				else if  (inst instanceof RandSPInstruction) {
					RandSPInstruction rand = (RandSPInstruction) inst;
					HashMap<String, Hop> params = new HashMap<>();
					if (rand.output.getDataType() == DataType.TENSOR)
						params.put(DataExpression.RAND_DIMS, new LiteralOp(rand.getDims()));
					else {
						params.put(DataExpression.RAND_ROWS, new LiteralOp(rand.getRows()));
						params.put(DataExpression.RAND_COLS, new LiteralOp(rand.getCols()));
					}
					params.put(DataExpression.RAND_MIN, new LiteralOp(rand.getMinValue()));
					params.put(DataExpression.RAND_MAX, new LiteralOp(rand.getMaxValue()));
					params.put(DataExpression.RAND_PDF, new LiteralOp(rand.getPdf()));
					params.put(DataExpression.RAND_LAMBDA, new LiteralOp(rand.getPdfParams()));
					params.put(DataExpression.RAND_SPARSITY, new LiteralOp(rand.getSparsity()));
					params.put(DataExpression.RAND_SEED, new LiteralOp(rand.getSeed()));
					Hop datagen = new DataGenOp(DataGenMethod.RAND, new DataIdentifier("tmp"), params);
					datagen.setBlocksize(rand.getBlocksize());
					operands.put(item.getId(), datagen);
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
						case Reorg: {
							Hop input = operands.get(item.getInputs()[0].getId());
							Hop reorg = HopRewriteUtils.createReorg(input, ReOrgOp.TRANS);
							operands.put(item.getId(), reorg);
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
						case AggregateBinary: {
							Hop input1 = operands.get(item.getInputs()[0].getId());
							Hop input2 = operands.get(item.getInputs()[1].getId());
							Hop aggbinary = HopRewriteUtils.createMatrixMultiply(input1, input2);
							operands.put(item.getId(), aggbinary);
							break;
						}
						case Ternary: {
							operands.put(item.getId(), HopRewriteUtils.createTernaryOp(
								operands.get(item.getInputs()[0].getId()), 
								operands.get(item.getInputs()[1].getId()), 
								operands.get(item.getInputs()[2].getId()), item.getOpcode()));
							break;
						}
						case BuiltinNary: {
							operands.put(item.getId(), HopRewriteUtils.createNary(
								OpOpN.valueOf(item.getOpcode().toUpperCase()),
								createNaryInputs(item, operands)));
							break;
						}
						case MatrixIndexing: {
							operands.put(item.getId(), constructIndexingOp(item, operands));
							break;
						}
						case MMTSJ: {
							//TODO handling of tsmm type left and right -> placement transpose
							Hop input = operands.get(item.getInputs()[0].getId());
							Hop aggunary = HopRewriteUtils.createMatrixMultiply(
								HopRewriteUtils.createTranspose(input), input);
							operands.put(item.getId(), aggunary);
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
				}
				else if( stype != null ) {
					switch(stype) {
						case Reblock: {
							Hop input = operands.get(item.getInputs()[0].getId());
							input.setBlocksize(ConfigurationManager.getBlocksize());
							input.setRequiresReblock(true);
							operands.put(item.getId(), input);
							break;
						}
						case Checkpoint: {
							Hop input = operands.get(item.getInputs()[0].getId());
							operands.put(item.getId(), input);
							break;
						}
						case MatrixIndexing: {
							operands.put(item.getId(), constructIndexingOp(item, operands));
							break;
						}
						default:
							throw new DMLRuntimeException("Unsupported instruction "
								+ "type: " + stype.name() + " (" + item.getOpcode() + ").");
					}
				}
				else
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

	public static void constructLineageFromHops(Hop root, String claName) {
		//probe existence and only generate lineage if non-existing
		//(a fused operator might be used in multiple places of a program)
		if( LineageCodegenItem.getCodegenLTrace(claName) == null ) {
			//recursively construct lineage for fused operator
			Map<Long, LineageItem> operands = new HashMap<>();
			root.resetVisitStatus(); // ensure non-visited
			rConstructLineageFromHops(root, operands);
			
			//cache to avoid reconstruction
			LineageCodegenItem.setCodegenLTrace(claName, operands.get(root.getHopID()));
		}
	}

	public static void rConstructLineageFromHops(Hop root, Map<Long, LineageItem> operands) {
		if (root.isVisited())
			return;

		for (int i = 0; i < root.getInput().size(); i++) 
			rConstructLineageFromHops(root.getInput().get(i), operands);
	
		if ((root instanceof DataOp) && (((DataOp)root).getDataOpType() == DataOpTypes.TRANSIENTREAD)) {
			LineageItem li = new LineageItem(root.getName(), "InputPlaceholder", "Create"+String.valueOf(root.getHopID()));
			operands.put(root.getHopID(), li);
			return;
		}

		LineageItem li = null;
		ArrayList<LineageItem> LIinputs = new ArrayList<>();
		root.getInput().forEach(input->LIinputs.add(operands.get(input.getHopID())));
		String name = Dag.getNextUniqueVarname(root.getDataType());
		
		if (root instanceof ReorgOp)
			li = new LineageItem(name, "r'", LIinputs.toArray(new LineageItem[LIinputs.size()]));
		else if (root instanceof AggBinaryOp)
			li = new LineageItem(name, "ba+*", LIinputs.toArray(new LineageItem[LIinputs.size()]));
		else if (root instanceof BinaryOp)
			li = new LineageItem(name, Binary.getOpcode(Hop.HopsOpOp2LopsB.get(((BinaryOp)root).getOp())),
				LIinputs.toArray(new LineageItem[LIinputs.size()]));
		
		else if (root instanceof LiteralOp) {  //TODO: remove redundancy
			StringBuilder sb = new StringBuilder(root.getName());
			sb.append(Instruction.VALUETYPE_PREFIX);
			sb.append(root.getDataType().toString());
			sb.append(Instruction.VALUETYPE_PREFIX);
			sb.append(root.getValueType().toString());
			sb.append(Instruction.VALUETYPE_PREFIX);
			sb.append(true); //isLiteral = true
			li = new LineageItem(root.getName(), sb.toString());
		}
		//TODO: include all the other hops
		operands.put(root.getHopID(), li);
		
		root.setVisited();
	}
	
	private static Hop constructIndexingOp(LineageItem item, Map<Long, Hop> operands) {
		//TODO fix 
		if( "rightIndex".equals(item.getOpcode()) )
			return HopRewriteUtils.createIndexingOp(
				operands.get(item.getInputs()[0].getId()), //input
				operands.get(item.getInputs()[1].getId()), //rl
				operands.get(item.getInputs()[2].getId()), //ru
				operands.get(item.getInputs()[3].getId()), //cl
				operands.get(item.getInputs()[4].getId())); //cu
		else if( "leftIndex".equals(item.getOpcode()) 
				|| "mapLeftIndex".equals(item.getOpcode()) )
			return HopRewriteUtils.createLeftIndexingOp(
				operands.get(item.getInputs()[0].getId()), //input
				operands.get(item.getInputs()[1].getId()), //rhs
				operands.get(item.getInputs()[2].getId()), //rl
				operands.get(item.getInputs()[3].getId()), //ru
				operands.get(item.getInputs()[4].getId()), //cl
				operands.get(item.getInputs()[5].getId())); //cu
		throw new DMLRuntimeException("Unsupported opcode: "+item.getOpcode());
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
			if( tmp.equals(liOld) )
				current.getInputs()[i] = liNew;
			else
				rReplace(tmp, liOld, liNew);
		}
		current.setVisited();
	}
	
	public static void replaceDagLeaves(ExecutionContext ec, LineageItem root, CPOperand[] newLeaves) {
		LineageItem[] newLIleaves = LineageItemUtils.getLineage(ec, newLeaves);
		HashMap<String, LineageItem> newLImap = new HashMap<>();
		for (int i=0; i<newLIleaves.length; i++)
			newLImap.put(newLeaves[i].getName(), newLIleaves[i]);

		//find and replace the old leaves
		HashSet<LineageItem> oldLeaves = new HashSet<>();
		root.resetVisitStatus();
		rGetDagLeaves(oldLeaves, root);
		for (LineageItem leaf : oldLeaves) {
			if (leaf.getType() != LineageItemType.Literal)
				replace(root, leaf, newLImap.get(leaf.getName()));
		}
	}

	public static void rGetDagLeaves(HashSet<LineageItem> leaves, LineageItem root) {
		if (root.isVisited())
			return;
		
		if (root.isLeaf())
			leaves.add(root);
		else {
			for (LineageItem li : root.getInputs())
				rGetDagLeaves(leaves, li);
		}
		root.setVisited();
	}
	
	private static Hop[] createNaryInputs(LineageItem item, Map<Long, Hop> operands) {
		int len = item.getInputs().length;
		Hop[] ret = new Hop[len];
		for( int i=0; i<len; i++ )
			ret[i] = operands.get(item.getInputs()[i].getId());
		return ret;
	}
	
	public static boolean containsRandDataGen(HashSet<LineageItem> entries, LineageItem root) {
		boolean isRand = false;
		if (entries.contains(root))
			return false;
		if (isNonDeterministic(root))
			isRand |= true;
		if (!root.isLeaf()) 
			for (LineageItem input : root.getInputs())
				isRand = isRand ? true : containsRandDataGen(entries, input);
		return isRand;
		//TODO: unmark for caching in compile time
	}
	
	private static boolean isNonDeterministic(LineageItem li) {
		if (li.getType() != LineageItemType.Creation)
			return false;

		boolean isND = false;
		CPInstruction CPins = (CPInstruction) InstructionParser.parseSingleInstruction(li.getData());
		if (!(CPins instanceof DataGenCPInstruction))
			return false;

		DataGenCPInstruction ins = (DataGenCPInstruction)CPins;
		switch(li.getOpcode().toUpperCase())
		{
			case "RAND":
				if ((ins.getMinValue() != ins.getMaxValue()) || (ins.getSparsity() != 1))
					isND = true;
					//NOTE:It is hard to detect in runtime if rand was called with unspecified seed
					//as -1 is already replaced by computed seed. Solution is to unmark for caching in 
					//compile time. That way we can differentiate between given and unspecified seed.
				break;
			case "SAMPLE":
				isND = true;
				break;
			default:
				isND = false;
				break;
		}
		//TODO: add 'read' in this list
		return isND;
	}
	
	public static LineageItem[] getLineageItemInputstoSB(ArrayList<String> inputs, ExecutionContext ec) {
		if (ReuseCacheType.isNone())
			return null;
		
		ArrayList<CPOperand> CPOpInputs = inputs.size() > 0 ? new ArrayList<>() : null;
		for (int i=0; i<inputs.size(); i++) {
			Data value = ec.getVariable(inputs.get(i));
			if (value != null) {
				CPOpInputs.add(new CPOperand(value instanceof ScalarObject ? value.toString() : inputs.get(i),
					value.getValueType(), value.getDataType()));
			}
		}
		return(CPOpInputs != null ? LineageItemUtils.getLineage(ec, 
			CPOpInputs.toArray(new CPOperand[CPOpInputs.size()])) : null);
	}
	
}
