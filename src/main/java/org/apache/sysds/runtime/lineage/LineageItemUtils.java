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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.runtime.instructions.spark.RandSPInstruction;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.lineage.LineageItem.LineageItemType;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.codegen.SpoofFusedOp;
import org.apache.sysds.lops.PartialAggregate;
import org.apache.sysds.lops.UnaryCP;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionParser;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.util.HDFSTool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Stack;
import java.util.stream.Collectors;

public class LineageItemUtils {
	
	public static final String LPLACEHOLDER = "IN#";
	
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
	
	public static LineageItem[] getLineage(ExecutionContext ec, CPOperand... operands) {
		return Arrays.stream(operands).filter(c -> c!=null)
			.map(c -> ec.getLineage().getOrCreate(c)).toArray(LineageItem[]::new);
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
	
	@Deprecated
	@SuppressWarnings("unused")
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
				if (li.getType() == LineageItemType.Creation) {
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
	 * 
	 * @param current Current lineage item
	 * @param liOld Old lineage item
	 * @param liNew New Lineage item.
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
		if (ReuseCacheType.isNone() && !DMLScript.LINEAGE_DEDUP)
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
