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

package org.apache.sysds.runtime.lineage;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.runtime.instructions.spark.RandSPInstruction;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.lineage.LineageItem.LineageItemType;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.codegen.SpoofFusedOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.PartialAggregate;
import org.apache.sysds.lops.UnaryCP;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionParser;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction.SPType;
import org.apache.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.apache.sysds.runtime.util.HDFSTool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Stack;
import java.util.stream.Collectors;

public class LineageItemUtils {
	
	private static final String LVARPREFIX = "lvar";
	private static final String LPLACEHOLDER = "IN#";
	
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
		root.resetVisitStatusNR();
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
					if( rand.getOpcode().equals("rand") ) {
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
					}
					else if( rand.getOpcode().equals("seq") ) {
						params.put(Statement.SEQ_FROM, new LiteralOp(rand.getFrom()));
						params.put(Statement.SEQ_TO, new LiteralOp(rand.getTo()));
						params.put(Statement.SEQ_INCR, new LiteralOp(rand.getIncr()));
					}
					Hop datagen = new DataGenOp(OpOpDG.valueOf(rand.getOpcode().toUpperCase()),
						new DataIdentifier("tmp"), params);
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
					DataOp pread = new DataOp(parts[1].substring(5), dt, vt, OpOpData.PERSISTENTREAD, params);
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
					Hop datagen = new DataGenOp(OpOpDG.RAND, new DataIdentifier("tmp"), params);
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
							Hop aggunary = InstructionUtils.isUnaryMetadata(item.getOpcode()) ?
								HopRewriteUtils.createUnary(input, OpOp1.valueOfByOpcode(item.getOpcode())) :
								HopRewriteUtils.createAggUnaryOp(input, item.getOpcode());
							operands.put(item.getId(), aggunary);
							break;
						}
						case AggregateBinary: {
							Hop input1 = operands.get(item.getInputs()[0].getId());
							Hop input2 = operands.get(item.getInputs()[1].getId());
							Hop aggbinary = HopRewriteUtils.createMatrixMultiply(input1, input2);
							operands.put(item.getId(), aggbinary);
							break;
						}
						case AggregateTernary: {
							Hop input1 = operands.get(item.getInputs()[0].getId());
							Hop input2 = operands.get(item.getInputs()[1].getId());
							Hop input3 = operands.get(item.getInputs()[2].getId());
							Hop aggternary = HopRewriteUtils.createSum(
								HopRewriteUtils.createBinary(
								HopRewriteUtils.createBinary(input1, input2, OpOp2.MULT),
								input3, OpOp2.MULT));
							operands.put(item.getId(), aggternary);
							break;
						}
						case Unary:
						case Builtin: {
							Hop input = operands.get(item.getInputs()[0].getId());
							Hop unary = HopRewriteUtils.createUnary(input, item.getOpcode());
							operands.put(item.getId(), unary);
							break;
						}
						case Reorg: {
							operands.put(item.getId(), HopRewriteUtils.createReorg(
								operands.get(item.getInputs()[0].getId()), item.getOpcode()));
							break;
						}
						case Reshape: {
							ArrayList<Hop> inputs = new ArrayList<>();
							for(int i=0; i<5; i++)
								inputs.add(operands.get(item.getInputs()[i].getId()));
							operands.put(item.getId(), HopRewriteUtils.createReorg(inputs, ReOrgOp.RESHAPE));
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
							operands.put(item.getId(), HopRewriteUtils.createTernary(
								operands.get(item.getInputs()[0].getId()), 
								operands.get(item.getInputs()[1].getId()), 
								operands.get(item.getInputs()[2].getId()), item.getOpcode()));
							break;
						}
						case Ctable: { //e.g., ctable 
							if( item.getInputs().length==3 )
								operands.put(item.getId(), HopRewriteUtils.createTernary(
									operands.get(item.getInputs()[0].getId()),
									operands.get(item.getInputs()[1].getId()),
									operands.get(item.getInputs()[2].getId()), OpOp3.CTABLE));
							else if( item.getInputs().length==5 )
								operands.put(item.getId(), HopRewriteUtils.createTernary(
									operands.get(item.getInputs()[0].getId()),
									operands.get(item.getInputs()[1].getId()),
									operands.get(item.getInputs()[2].getId()),
									operands.get(item.getInputs()[3].getId()),
									operands.get(item.getInputs()[4].getId()), OpOp3.CTABLE));
							break;
						}
						case BuiltinNary: {
							String opcode = item.getOpcode().equals("n+") ? "plus" : item.getOpcode();
							operands.put(item.getId(), HopRewriteUtils.createNary(
								OpOpN.valueOf(opcode.toUpperCase()), createNaryInputs(item, operands)));
							break;
						}
						case ParameterizedBuiltin: {
							operands.put(item.getId(), constructParameterizedBuiltinOp(item, operands));
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
						case Variable: {
							if( item.getOpcode().startsWith("cast") )
								operands.put(item.getId(), HopRewriteUtils.createUnary(
									operands.get(item.getInputs()[0].getId()),
									OpOp1.valueOfByOpcode(item.getOpcode())));
							else //cpvar, write
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
						case GAppend: {
							operands.put(item.getId(), HopRewriteUtils.createBinary(
								operands.get(item.getInputs()[0].getId()),
								operands.get(item.getInputs()[1].getId()), OpOp2.CBIND));
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

	public static void constructLineageFromHops(Hop[] roots, String claName, Hop[] inputs, HashMap<Long, Hop> spoofmap) {
		//probe existence and only generate lineage if non-existing
		//(a fused operator might be used in multiple places of a program)
		if( LineageCodegenItem.getCodegenLTrace(claName) == null ) {
			//recursively construct lineage for fused operator
			Map<Long, LineageItem> operands = new HashMap<>();
			//reset visit status once for entire sub DAG
			for( Hop root : roots )
				root.resetVisitStatus();

			//construct lineage dags for roots (potentially overlapping)
			for( Hop root : roots )
				rConstructLineageFromHops(root, inputs, operands, spoofmap);
			
			//create single lineage item (single root or multiagg)
			LineageItem out = operands.get(roots[0].getHopID());
			if( roots.length > 1 ) { //multi-agg
				LineageItem[] outputs = Arrays.stream(roots)
					.map(h -> new LineageItem("", UnaryCP.CAST_AS_MATRIX_OPCODE,
						new LineageItem[]{operands.get(h.getHopID())}))
					.toArray(LineageItem[]::new);
				out = new LineageItem("", "cbind", outputs);
			}
			
			//cache to avoid reconstruction
			LineageCodegenItem.setCodegenLTrace(claName, out);
			
			for (Hop root : roots)
				root.resetVisitStatus();
		}
	}

	public static void rConstructLineageFromHops(Hop root, Hop[] inputs, Map<Long, LineageItem> operands, HashMap<Long, Hop> spoofmap) {
		if (root.isVisited())
			return;
		
		boolean spoof = root instanceof SpoofFusedOp && ArrayUtils.contains(inputs, spoofmap.get(root.getHopID())); 
		if (ArrayUtils.contains(inputs, root) || spoof) {
			Hop tmp = spoof ? spoofmap.get(root.getHopID()) : root;
			int pos = ArrayUtils.indexOf(inputs, tmp);
			LineageItem li = new LineageItem(LPLACEHOLDER+pos,
				"Create"+String.valueOf(root.getHopID()));
			operands.put(tmp.getHopID(), li);
			return;
		}

		for (int i = 0; i < root.getInput().size(); i++) 
			rConstructLineageFromHops(root.getInput().get(i), inputs, operands, spoofmap);
	
		LineageItem li = null;
		LineageItem[] LIinputs = root.getInput().stream()
			.map(h->ArrayUtils.contains(inputs, spoofmap.get(h.getHopID())) ? spoofmap.get(h.getHopID()) : h)
			.map(h->operands.get(h.getHopID()))
			.toArray(LineageItem[]::new);

		String name = Dag.getNextUniqueVarname(root.getDataType());
		
		if (root instanceof ReorgOp)
			li = new LineageItem(name, "r'", LIinputs);
		else if (root instanceof UnaryOp) {
			String opcode = ((UnaryOp) root).getOp().toString();
			li = new LineageItem(name, opcode, LIinputs);
		}
		else if (root instanceof AggBinaryOp)
			li = new LineageItem(name, "ba+*", LIinputs);
		else if (root instanceof BinaryOp)
			li = new LineageItem(name, ((BinaryOp)root).getOp().toString(), LIinputs);
		else if (root instanceof TernaryOp) {
			String opcode = ((TernaryOp) root).getOp().toString();
			li = new LineageItem(name, opcode, LIinputs);
		}
		else if (root instanceof AggUnaryOp) {
			AggOp op = ((AggUnaryOp) root).getOp();
			Direction dir = ((AggUnaryOp) root).getDirection();
			String opcode = PartialAggregate.getOpcode(op, dir);
			li = new LineageItem(name, opcode, LIinputs);
		}
		else if (root instanceof IndexingOp)
			li = new LineageItem(name, "rightIndex", LIinputs);
		else if (root instanceof SpoofFusedOp)
			li = LineageCodegenItem.getCodegenLTrace(((SpoofFusedOp) root).getClassName());
		
		else if (root instanceof LiteralOp) {  //TODO: remove redundancy
			StringBuilder sb = new StringBuilder(root.getName());
			sb.append(Instruction.VALUETYPE_PREFIX);
			sb.append(root.getDataType().toString());
			sb.append(Instruction.VALUETYPE_PREFIX);
			sb.append(root.getValueType().toString());
			sb.append(Instruction.VALUETYPE_PREFIX);
			sb.append(true); //isLiteral = true
			li = new LineageItem(sb.toString());
		}
		else
			throw new DMLRuntimeException("Unsupported hop: "+root.getOpString());

		//TODO: include all the other hops
		operands.put(root.getHopID(), li);
		root.setVisited();
	}
	
	private static Hop constructIndexingOp(LineageItem item, Map<Long, Hop> operands) {
		Hop input = operands.get(item.getInputs()[0].getId());
		if( "rightIndex".equals(item.getOpcode()) )
			return HopRewriteUtils.createIndexingOp(input,
				operands.get(item.getInputs()[1].getId()), //rl
				operands.get(item.getInputs()[2].getId()), //ru
				operands.get(item.getInputs()[3].getId()), //cl
				operands.get(item.getInputs()[4].getId())); //cu
		else if( "leftIndex".equals(item.getOpcode()) 
				|| "mapLeftIndex".equals(item.getOpcode()) )
			return HopRewriteUtils.createLeftIndexingOp(input,
				operands.get(item.getInputs()[1].getId()), //rhs
				operands.get(item.getInputs()[2].getId()), //rl
				operands.get(item.getInputs()[3].getId()), //ru
				operands.get(item.getInputs()[4].getId()), //cl
				operands.get(item.getInputs()[5].getId())); //cu
		throw new DMLRuntimeException("Unsupported opcode: "+item.getOpcode());
	}
	
	private static Hop constructParameterizedBuiltinOp(LineageItem item, Map<Long, Hop> operands) {
		String opcode = item.getOpcode();
		Hop target = operands.get(item.getInputs()[0].getId());
		LinkedHashMap<String,Hop> args = new LinkedHashMap<>();
		if( opcode.equals("groupedagg") ) {
			args.put("target", target);
			args.put(Statement.GAGG_GROUPS, operands.get(item.getInputs()[1].getId()));
			args.put(Statement.GAGG_WEIGHTS, operands.get(item.getInputs()[2].getId()));
			args.put(Statement.GAGG_FN, operands.get(item.getInputs()[3].getId()));
			args.put(Statement.GAGG_NUM_GROUPS, operands.get(item.getInputs()[4].getId()));
		}
		else if (opcode.equalsIgnoreCase("rmempty")) {
			args.put("target", target);
			args.put("margin", operands.get(item.getInputs()[1].getId()));
			args.put("select", operands.get(item.getInputs()[2].getId()));
		}
		else if(opcode.equalsIgnoreCase("replace")) {
			args.put("target", target);
			args.put("pattern", operands.get(item.getInputs()[1].getId()));
			args.put("replacement", operands.get(item.getInputs()[2].getId()));
		}
		else if(opcode.equalsIgnoreCase("rexpand")) {
			args.put("target", target);
			args.put("max", operands.get(item.getInputs()[1].getId()));
			args.put("dir", operands.get(item.getInputs()[2].getId()));
			args.put("cast", operands.get(item.getInputs()[3].getId()));
			args.put("ignore", operands.get(item.getInputs()[4].getId()));
		}
		
		return HopRewriteUtils.createParameterizedBuiltinOp(
			target, args, ParamBuiltinOp.valueOf(opcode.toUpperCase()));
	}
	
	public static LineageItem rDecompress(LineageItem item) {
		if (item.getType() == LineageItemType.Dedup) {
			LineageItem dedupInput = rDecompress(item.getInputs()[0]);
			ArrayList<LineageItem> inputs = new ArrayList<>();
			
			for (LineageItem li : item.getInputs()[1].getInputs())
				inputs.add(rDecompress(li));
			
			LineageItem li = new LineageItem(item.getInputs()[1].getData(),
				item.getInputs()[1].getOpcode(), inputs.toArray(new LineageItem[0]));
			
			li.resetVisitStatusNR();
			rSetDedupInputOntoOutput(item.getData(), li, dedupInput);
			li.resetVisitStatusNR();
			return li;
		}
		else {
			ArrayList<LineageItem> inputs = new ArrayList<>();
			if (item.getInputs() != null) {
				for (LineageItem li : item.getInputs())
					inputs.add(rDecompress(li));
			}
			return new LineageItem(
				item.getData(), item.getOpcode(), inputs.toArray(new LineageItem[0]));
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
				//replace CPOperand literals (placeholders)
				//TODO should use the same placeholder meta data as codegen
				if( li.getType() == LineageItemType.Literal ) {
					CPOperand tmp = new CPOperand(li.getData());
					if( !tmp.isLiteral() && tmp.getName().equals(name) )
						item.getInputs()[i] = dedupInput;
				}
				
				rSetDedupInputOntoOutput(name, li, dedupInput);
			}
		
		item.setVisited();
	}
	
	public static LineageItem replace(LineageItem root, LineageItem liOld, LineageItem liNew) {
		if( liNew == null )
			throw new DMLRuntimeException("Invalid null lineage item for "+liOld.getId());
		root.resetVisitStatusNR();
		rReplaceNR(root, liOld, liNew);
		root.resetVisitStatusNR();
		return root;
	}
	
	/**
	 * Non-recursive equivalent of {@link #rReplace(LineageItem, LineageItem, LineageItem)} 
	 * for robustness with regard to stack overflow errors.
	 */
	public static void rReplaceNR(LineageItem current, LineageItem liOld, LineageItem liNew) {
		Stack<LineageItem> q = new Stack<>();
		q.push(current);
		while( !q.empty() ) {
			LineageItem tmp = q.pop();
			if( tmp.isVisited() || tmp.getInputs() == null )
				continue;
			//process children until old item found, then replace
			for(int i=0; i<tmp.getInputs().length; i++) {
				LineageItem ctmp = tmp.getInputs()[i];
				if (liOld.getId() == ctmp.getId() && liOld.equals(ctmp))
					tmp.setInput(i, liNew);
				else
					q.push(ctmp);
			}
			tmp.setVisited(true);
		}
	}
	
	@Deprecated
	@SuppressWarnings("unused")
	private static void rReplace(LineageItem current, LineageItem liOld, LineageItem liNew) {
		if( current.isVisited() || current.getInputs() == null )
			return;
		if( liNew == null )
			throw new DMLRuntimeException("Invalid null lineage item for "+liOld.getId());
		//process children until old item found, then replace
		for(int i=0; i<current.getInputs().length; i++) {
			LineageItem tmp = current.getInputs()[i];
			if (liOld.equals(tmp))
				current.setInput(i, liNew);
			else
				rReplace(tmp, liOld, liNew);
		}
		current.setVisited();
	}
	
	public static void replaceDagLeaves(ExecutionContext ec, LineageItem root, CPOperand[] newLeaves) {
		//find and replace the placeholder leaves
		root.resetVisitStatusNR();
		rReplaceDagLeaves(root, LineageItemUtils.getLineage(ec, newLeaves));
		root.resetVisitStatusNR();
	}
	
	public static void rReplaceDagLeaves(LineageItem root, LineageItem[] newleaves) {
		if (root.isVisited() || root.isLeaf())
			return;
		
		for (int i=0; i<root.getInputs().length; i++) {
			LineageItem li = root.getInputs()[i];
			if (li.isLeaf() && li.getType() != LineageItemType.Literal
				&& li.getData().startsWith(LPLACEHOLDER))
				//order-preserving replacement. IN#<xxx> represents relative position xxx
				root.setInput(i, newleaves[Integer.parseInt(li.getData().substring(3))]);
			else
				rReplaceDagLeaves(li, newleaves);
		}

		root.setVisited();
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
	
	public static void checkCycles(LineageItem current) {
		current.resetVisitStatusNR();
		rCheckCycles(current, new HashSet<Long>(), true);
		current.resetVisitStatusNR();
	}
	
	public static void rCheckCycles(LineageItem current, Set<Long> probe, boolean useObjIdent) {
		if( current.isVisited() )
			return;
		long id = useObjIdent ? System.identityHashCode(current) : current.getId();
		if( probe.contains(id) )
			throw new DMLRuntimeException("Cycle detected for "+current.toString());
		probe.add(id);
		if( current.getInputs() != null )
			for(LineageItem li : current.getInputs())
				rCheckCycles(li, probe, useObjIdent);
		current.setVisited();
	}
	
	private static Hop[] createNaryInputs(LineageItem item, Map<Long, Hop> operands) {
		int len = item.getInputs().length;
		Hop[] ret = new Hop[len];
		for( int i=0; i<len; i++ )
			ret[i] = operands.get(item.getInputs()[i].getId());
		return ret;
	}
	
	public static boolean containsRandDataGen(HashSet<LineageItem> entries, LineageItem root) {
		if (entries.contains(root) || root.isVisited())
			return false;
		boolean isRand = isNonDeterministic(root);
		if (!root.isLeaf() && !isRand) 
			for (LineageItem input : root.getInputs())
				isRand |= containsRandDataGen(entries, input);
		root.setVisited();
		return isRand;
	}
	
	private static boolean isNonDeterministic(LineageItem li) {
		if (li.getType() != LineageItemType.Creation)
			return false;

		boolean isND = false;
		DataGenCPInstruction cprand = null;
		RandSPInstruction sprand = null;
		Instruction ins = InstructionParser.parseSingleInstruction(li.getData());

		if (ins instanceof DataGenCPInstruction)
			cprand = (DataGenCPInstruction)ins;
		else if (ins instanceof RandSPInstruction)
			sprand = (RandSPInstruction)ins;
		else 
			return false;

		switch(li.getOpcode().toUpperCase())
		{
			case "RAND":
				if (cprand != null)
					if ((cprand.getMinValue() != cprand.getMaxValue()) || (cprand.getSparsity() != 1))
						isND = true;
				if (sprand!= null)
					if ((sprand.getMinValue() != sprand.getMaxValue()) || (sprand.getSparsity() != 1))
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
	
	public static void addAllDataLineage(ExecutionContext ec) {
		for( Entry<String, Data> e : ec.getVariables().entrySet() ) {
			if( e.getValue() instanceof CacheableData<?> ) {
				CacheableData<?> cdata = (CacheableData<?>) e.getValue();
				//only createvar instruction with pREAD prefix added to lineage
				String fromVar = org.apache.sysds.lops.Data.PREAD_PREFIX+e.getKey();
				ec.traceLineage(VariableCPInstruction.prepCreatevarInstruction(
					fromVar, "CacheableData::"+cdata.getUniqueID(), false, "binary"));
				//move from pREADx to x
				ec.traceLineage(VariableCPInstruction.prepMoveInstruction(fromVar, e.getKey()));
			}
		}
	}
}
