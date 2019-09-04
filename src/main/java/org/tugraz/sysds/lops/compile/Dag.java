/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.lops.compile;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.conf.DMLConfig;
import org.tugraz.sysds.hops.HopsException;
import org.tugraz.sysds.lops.Data;
import org.tugraz.sysds.lops.Data.OperationTypes;
import org.tugraz.sysds.lops.FunctionCallCP;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.lops.Lop.Type;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.lops.LopsException;
import org.tugraz.sysds.lops.OutputParameters;
import org.tugraz.sysds.lops.OutputParameters.Format;
import org.tugraz.sysds.parser.DataExpression;
import org.tugraz.sysds.parser.StatementBlock;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.tugraz.sysds.runtime.instructions.CPInstructionParser;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.Instruction.IType;
import org.tugraz.sysds.runtime.instructions.InstructionParser;
import org.tugraz.sysds.runtime.instructions.SPInstructionParser;
import org.tugraz.sysds.runtime.instructions.cp.CPInstruction;
import org.tugraz.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.tugraz.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;


/**
 * 
 * Class to maintain a DAG of lops and compile it into 
 * runtime instructions, incl piggybacking into jobs.
 * 
 * @param <N> the class parameter has no affect and is
 * only kept for documentation purposes.
 */
public class Dag<N extends Lop>
{
	private static final Log LOG = LogFactory.getLog(Dag.class.getName());

	private static IDSequence job_id = null;
	private static IDSequence var_index = null;
	
	private String scratch = "";
	private String scratchFilePath = null;

	// list of all nodes in the dag
	private ArrayList<Lop> nodes = null;

	static {
		job_id = new IDSequence();
		var_index = new IDSequence();
	}

	private static class NodeOutput {
		OutputInfo outInfo;
		ArrayList<Instruction> preInstructions; // instructions added before a MR instruction
		ArrayList<Instruction> lastInstructions;
		
		NodeOutput() {
			outInfo = null;
			preInstructions = new ArrayList<>(); 
			lastInstructions = new ArrayList<>();
		}
		
		public OutputInfo getOutInfo() {
			return outInfo;
		}
		public void setOutInfo(OutputInfo outInfo) {
			this.outInfo = outInfo;
		}
		public ArrayList<Instruction> getPreInstructions() {
			return preInstructions;
		}
		public void addPreInstruction(Instruction inst) {
			preInstructions.add(inst);
		}
		public ArrayList<Instruction> getLastInstructions() {
			return lastInstructions;
		}
		public void addLastInstruction(Instruction inst) {
			lastInstructions.add(inst);
		}
	}
	
	public Dag() {
		//allocate internal data structures
		nodes = new ArrayList<>();
	}
	
	///////
	// filename handling
	
	private String getFilePath() {
		if ( scratchFilePath == null ) {
			scratchFilePath = scratch + Lop.FILE_SEPARATOR
				+ Lop.PROCESS_PREFIX + DMLScript.getUUID()
				+ Lop.FILE_SEPARATOR + Lop.FILE_SEPARATOR
				+ Lop.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR;
		}
		return scratchFilePath;
	}

	public static String getNextUniqueFilenameSuffix() {
		return "temp" + job_id.getNextID();
	}
	
	public String getNextUniqueFilename() {
		return getFilePath() + getNextUniqueFilenameSuffix();
	}
	
	public static String getNextUniqueVarname(DataType dt) {
		return (dt.isMatrix() ? Lop.MATRIX_VAR_NAME_PREFIX :
			dt.isFrame() ? Lop.FRAME_VAR_NAME_PREFIX :
			Lop.SCALAR_VAR_NAME_PREFIX) + var_index.getNextID();
	}
	
	///////
	// Dag modifications
	
	/**
	 * Method to add a node to the DAG.
	 * 
	 * @param node low-level operator
	 * @return true if node was not already present, false if not.
	 */
	public boolean addNode(Lop node) {
		if (nodes.contains(node))
			return false;
		nodes.add(node);
		return true;
	}

	/**
	 * Method to compile a dag generically
	 * 
	 * @param sb statement block
	 * @param config dml configuration
	 * @return list of instructions
	 */
	public ArrayList<Instruction> getJobs(StatementBlock sb, DMLConfig config) {
		if (config != null) {
			scratch = config.getTextValue(DMLConfig.SCRATCH_SPACE) + "/";
		}
		
		// create ordering of lops (for MR, we sort by level, while for all
		// other exec types we use a two-level sorting of )
		List<Lop> node_v = 
			//doTopologicalSortStrictOrder(nodes) :
			doTopologicalSortTwoLevelOrder(nodes);
		
		// do greedy grouping of operations
		ArrayList<Instruction> inst =
			//doGreedyGrouping(sb, node_v) :
			doPlainInstructionGen(sb, node_v);
		
		// cleanup instruction (e.g., create packed rmvar instructions)
		return cleanupInstructions(inst);
	}
	
	private List<Lop> doTopologicalSortTwoLevelOrder(List<Lop> v) {
		//partition nodes into leaf/inner nodes and dag root nodes,
		//+ sort leaf/inner nodes by ID to force depth-first scheduling
		//+ append root nodes in order of their original definition 
		//  (which also preserves the original order of prints)
		List<Lop> nodes = Stream.concat(
			v.stream().filter(l -> !l.getOutputs().isEmpty()).sorted(Comparator.comparing(l -> l.getID())),
			v.stream().filter(l -> l.getOutputs().isEmpty())).collect(Collectors.toList());

		//NOTE: in contrast to hadoop execution modes, we avoid computing the transitive
		//closure here to ensure linear time complexity because its unnecessary for CP and Spark
		return nodes;
	}
	
	private ArrayList<Instruction> doPlainInstructionGen(StatementBlock sb, List<Lop> nodes)
	{
		//prepare basic instruction sets
		List<Instruction> deleteInst = new ArrayList<>();
		List<Instruction> writeInst = deleteUpdatedTransientReadVariables(sb, nodes);
		List<Instruction> endOfBlockInst = generateRemoveInstructions(sb);
		ArrayList<Instruction> inst = generateInstructionsForInputVariables(nodes);
		
		// filter out non-executable nodes
		List<Lop> execNodes = nodes.stream()
			.filter(l -> (!l.isDataExecLocation() 
				|| (((Data)l).getOperationType()==OperationTypes.WRITE && !isTransientWriteRead((Data)l))
				|| (((Data)l).isPersistentRead() && l.getDataType().isScalar())))
			.collect(Collectors.toList());
		
		// generate executable instruction
		generateControlProgramJobs(execNodes, inst, writeInst, deleteInst);

		// add write and delete inst at the very end.
		inst.addAll(writeInst);
		inst.addAll(deleteInst);
		inst.addAll(endOfBlockInst);
		return inst;
	}
	
	private boolean isTransientWriteRead(Data dnode) {
		Lop input = dnode.getInputs().get(0);
		return dnode.isTransient() 
			&& input.isDataExecLocation() && ((Data)input).isTransient() 
			&& dnode.getOutputParameters().getLabel().equals(input.getOutputParameters().getLabel());
	}
	
	private static List<Instruction> deleteUpdatedTransientReadVariables(StatementBlock sb, List<Lop> nodeV) {
		List<Instruction> insts = new ArrayList<>();
		if ( sb == null ) //return modifiable list
			return insts;
		
		if( LOG.isTraceEnabled() )
			LOG.trace("In delete updated variables");

		// CANDIDATE list of variables which could have been updated in this statement block 
		HashMap<String, Lop> labelNodeMapping = new HashMap<>();
		
		// ACTUAL list of variables whose value is updated, AND the old value of the variable 
		// is no longer accessible/used.
		HashSet<String> updatedLabels = new HashSet<>();
		HashMap<String, Lop> updatedLabelsLineNum =  new HashMap<>();
		
		// first capture all transient read variables
		for ( Lop node : nodeV ) {
			if (node.isDataExecLocation()
					&& ((Data) node).isTransient()
					&& ((Data) node).getOperationType() == OperationTypes.READ
					&& ((Data) node).getDataType() == DataType.MATRIX) {
				// "node" is considered as updated ONLY IF the old value is not used any more
				// So, make sure that this READ node does not feed into any (transient/persistent) WRITE
				boolean hasWriteParent=false;
				for(Lop p : node.getOutputs()) {
					if(p.isDataExecLocation()) {
						// if the "p" is of type Data, then it has to be a WRITE
						hasWriteParent = true;
						break;
					}
				}
				if ( !hasWriteParent ) {
					// node has no parent of type WRITE, so this is a CANDIDATE variable 
					// add it to labelNodeMapping so that it is considered in further processing
					labelNodeMapping.put(node.getOutputParameters().getLabel(), node);
				}
			}
		}

		// capture updated transient write variables
		for ( Lop node : nodeV ) {
			if (node.isDataExecLocation()
					&& ((Data) node).isTransient()
					&& ((Data) node).getOperationType() == OperationTypes.WRITE
					&& ((Data) node).getDataType() == DataType.MATRIX
					&& labelNodeMapping.containsKey(node.getOutputParameters().getLabel()) // check to make sure corresponding (i.e., with the same label/name) transient read is present
					&& !labelNodeMapping.containsValue(node.getInputs().get(0)) ){ // check to avoid cases where transient read feeds into a transient write
				updatedLabels.add(node.getOutputParameters().getLabel());
				updatedLabelsLineNum.put(node.getOutputParameters().getLabel(), node);
			}
		}
		
		// generate RM instructions
		Instruction rm_inst = null;
		for ( String label : updatedLabels ) {
			rm_inst = VariableCPInstruction.prepareRemoveInstruction(label);
			rm_inst.setLocation(updatedLabelsLineNum.get(label));
			if( LOG.isTraceEnabled() )
				LOG.trace(rm_inst.toString());
			insts.add(rm_inst);
		}
		return insts;
	}
	
	private static List<Instruction> generateRemoveInstructions(StatementBlock sb) {
		if ( sb == null ) 
			return Collections.emptyList();
		ArrayList<Instruction> insts = new ArrayList<>();
		
		if( LOG.isTraceEnabled() )
			LOG.trace("In generateRemoveInstructions()");
		
		// RULE 1: if in IN and not in OUT, then there should be an rmvar or rmfilevar inst
		// (currently required for specific cases of external functions)
		for (String varName : sb.liveIn().getVariableNames()) {
			if (!sb.liveOut().containsVariable(varName)) {
				Instruction inst = VariableCPInstruction.prepareRemoveInstruction(varName);
				inst.setLocation(sb.getFilename(), sb.getEndLine(), sb.getEndLine(), -1, -1);
				insts.add(inst);
				if( LOG.isTraceEnabled() )
					LOG.trace("  Adding " + inst.toString());
			}
		}
		return insts;
	}

	/**
	 * Method to generate createvar instructions, which creates a new entry
	 * in the symbol table. One instruction is generated for every LOP that is 
	 * 1) type Data and 
	 * 2) persistent and 
	 * 3) matrix and 
	 * 4) read
	 * 
	 * Transient reads needn't be considered here since the previous program 
	 * block would already create appropriate entries in the symbol table.
	 * 
	 * @param nodes_v list of nodes
	 * @return list of instructions
	 */
	private static ArrayList<Instruction> generateInstructionsForInputVariables(List<Lop> nodes_v) {
		ArrayList<Instruction> insts = new ArrayList<>();
		for(Lop n : nodes_v) {
			if (n.isDataExecLocation() && !((Data) n).isTransient()
					&& ((Data) n).getOperationType() == OperationTypes.READ
					&& (n.getDataType() == DataType.MATRIX || n.getDataType() == DataType.FRAME) ) {
				if ( !((Data)n).isLiteral() ) {
					try {
						String inst_string = n.getInstructions();
						CPInstruction currInstr = CPInstructionParser.parseSingleInstruction(inst_string);
						currInstr.setLocation(n);
						insts.add(currInstr);
					} catch (DMLRuntimeException e) {
						throw new LopsException(n.printErrorLocation() + "error generating instructions from input variables in Dag -- \n", e);
					}
				}
			}
		}
		return insts;
	}
	
	/**
	 * Exclude rmvar instruction for varname from deleteInst, if exists
	 * 
	 * @param varName variable name
	 * @param deleteInst list of instructions
	 */
	private static void excludeRemoveInstruction(String varName, List<Instruction> deleteInst) {
		for(int i=0; i < deleteInst.size(); i++) {
			Instruction inst = deleteInst.get(i);
			if ((inst.getType() == IType.CONTROL_PROGRAM  || inst.getType() == IType.SPARK)
					&& ((CPInstruction)inst).getCPInstructionType() == CPType.Variable 
					&& ((VariableCPInstruction)inst).isRemoveVariable(varName) ) {
				deleteInst.remove(i);
			}
		}
	}
	
	/**
	 * Generate rmvar instructions for the inputs, if their consumer count becomes zero.
	 * 
	 * @param node low-level operator
	 * @param inst list of instructions
	 * @param delteInst list of instructions
	 */
	private static void processConsumersForInputs(Lop node, List<Instruction> inst, List<Instruction> delteInst) {
		// reduce the consumer count for all input lops
		// if the count becomes zero, then then variable associated w/ input can be removed
		for(Lop in : node.getInputs() )
			processConsumers(in, inst, delteInst, null);
	}
	
	private static void processConsumers(Lop node, List<Instruction> inst, List<Instruction> deleteInst, Lop locationInfo) {
		// reduce the consumer count for all input lops
		// if the count becomes zero, then then variable associated w/ input can be removed
		if ( node.removeConsumer() == 0 ) {
			if ( node.isDataExecLocation() && ((Data)node).isLiteral() ) {
				return;
			}
			
			String label = node.getOutputParameters().getLabel();
			Instruction currInstr = VariableCPInstruction.prepareRemoveInstruction(label);
			if (locationInfo != null)
				currInstr.setLocation(locationInfo);
			else
				currInstr.setLocation(node);
			
			inst.add(currInstr);
			excludeRemoveInstruction(label, deleteInst);
		}
	}
	
	/**
	 * Method to generate instructions that are executed in Control Program. At
	 * this point, this DAG has no dependencies on the MR dag. ie. none of the
	 * inputs are outputs of MR jobs
	 * 
	 * @param execNodes list of low-level operators
	 * @param inst list of instructions
	 * @param writeInst list of write instructions
	 * @param deleteInst list of delete instructions
	 */
	private void generateControlProgramJobs(List<Lop> execNodes,
		List<Instruction> inst, List<Instruction> writeInst, List<Instruction> deleteInst) {

		// nodes to be deleted from execnodes
		ArrayList<Lop> markedNodes = new ArrayList<>();

		// variable names to be deleted
		ArrayList<String> var_deletions = new ArrayList<>();
		HashMap<String, Lop> var_deletionsLineNum =  new HashMap<>();
		
		boolean doRmVar = false;

		for (int i = 0; i < execNodes.size(); i++) {
			Lop node = execNodes.get(i);
			doRmVar = false;

			// mark input scalar read nodes for deletion
			if (node.isDataExecLocation()
					&& ((Data) node).getOperationType() == Data.OperationTypes.READ
					&& ((Data) node).getDataType() == DataType.SCALAR 
					&& node.getOutputParameters().getFile_name() == null ) {
				markedNodes.add(node);
				continue;
			}
			
			// output scalar instructions and mark nodes for deletion
			if (!node.isDataExecLocation()) {

				if (node.getDataType() == DataType.SCALAR) {
					// Output from lops with SCALAR data type must
					// go into Temporary Variables (Var0, Var1, etc.)
					NodeOutput out = setupNodeOutputs(node, ExecType.CP, false, false);
					inst.addAll(out.getPreInstructions()); // dummy
					deleteInst.addAll(out.getLastInstructions());
				} else {
					// Output from lops with non-SCALAR data type must
					// go into Temporary Files (temp0, temp1, etc.)
					
					NodeOutput out = setupNodeOutputs(node, ExecType.CP, false, false);
					inst.addAll(out.getPreInstructions());
					
					boolean hasTransientWriteParent = false;
					for ( Lop parent : node.getOutputs() ) {
						if ( parent.isDataExecLocation() 
								&& ((Data)parent).getOperationType() == Data.OperationTypes.WRITE 
								&& ((Data)parent).isTransient() ) {
							hasTransientWriteParent = true;
							break;
						}
					}
					
					if ( !hasTransientWriteParent ) {
						deleteInst.addAll(out.getLastInstructions());
					} 
					else {
						var_deletions.add(node.getOutputParameters().getLabel());
						var_deletionsLineNum.put(node.getOutputParameters().getLabel(), node);
					}
				}

				String inst_string = "";

				// Lops with arbitrary number of inputs (ParameterizedBuiltin, GroupedAggregate, DataGen)
				// are handled separately, by simply passing ONLY the output variable to getInstructions()
				if (node.getType() == Lop.Type.ParameterizedBuiltin
						|| node.getType() == Lop.Type.GroupedAgg 
						|| node.getType() == Lop.Type.DataGen){
					inst_string = node.getInstructions(node.getOutputParameters().getLabel());
				} 
				
				// Lops with arbitrary number of inputs and outputs are handled
				// separately as well by passing arrays of inputs and outputs
				else if ( node.getType() == Lop.Type.FunctionCallCP )
				{
					String[] inputs = new String[node.getInputs().size()];
					String[] outputs = new String[node.getOutputs().size()];
					int count = 0;
					for( Lop in : node.getInputs() )
						inputs[count++] = in.getOutputParameters().getLabel();
					count = 0;
					for( Lop out : node.getOutputs() )
						outputs[count++] = out.getOutputParameters().getLabel();
					inst_string = node.getInstructions(inputs, outputs);
				}
				else if (node.getType() == Lop.Type.Nary) {
					String[] inputs = new String[node.getInputs().size()];
					int count = 0;
					for( Lop in : node.getInputs() )
						inputs[count++] = in.getOutputParameters().getLabel();
					inst_string = node.getInstructions(inputs, 
						node.getOutputParameters().getLabel());
				}
				else {
					if ( node.getInputs().isEmpty() ) {
						// currently, such a case exists only for Rand lop
						inst_string = node.getInstructions(node.getOutputParameters().getLabel());
					}
					else if (node.getInputs().size() == 1) {
						inst_string = node.getInstructions(node.getInputs()
								.get(0).getOutputParameters().getLabel(),
								node.getOutputParameters().getLabel());
					} 
					else if (node.getInputs().size() == 2) {
						inst_string = node.getInstructions(
								node.getInputs().get(0).getOutputParameters().getLabel(),
								node.getInputs().get(1).getOutputParameters().getLabel(),
								node.getOutputParameters().getLabel());
					} 
					else if (node.getInputs().size() == 3 || node.getType() == Type.Ctable) {
						inst_string = node.getInstructions(
								node.getInputs().get(0).getOutputParameters().getLabel(),
								node.getInputs().get(1).getOutputParameters().getLabel(),
								node.getInputs().get(2).getOutputParameters().getLabel(),
								node.getOutputParameters().getLabel());
					}
					else if (node.getInputs().size() == 4) {
						inst_string = node.getInstructions(
								node.getInputs().get(0).getOutputParameters().getLabel(),
								node.getInputs().get(1).getOutputParameters().getLabel(),
								node.getInputs().get(2).getOutputParameters().getLabel(),
								node.getInputs().get(3).getOutputParameters().getLabel(),
								node.getOutputParameters().getLabel());
					}
					else if (node.getInputs().size() == 5) {
						inst_string = node.getInstructions(
								node.getInputs().get(0).getOutputParameters().getLabel(),
								node.getInputs().get(1).getOutputParameters().getLabel(),
								node.getInputs().get(2).getOutputParameters().getLabel(),
								node.getInputs().get(3).getOutputParameters().getLabel(),
								node.getInputs().get(4).getOutputParameters().getLabel(),
								node.getOutputParameters().getLabel());
					}
					else if (node.getInputs().size() == 6) {
						inst_string = node.getInstructions(
								node.getInputs().get(0).getOutputParameters().getLabel(),
								node.getInputs().get(1).getOutputParameters().getLabel(),
								node.getInputs().get(2).getOutputParameters().getLabel(),
								node.getInputs().get(3).getOutputParameters().getLabel(),
								node.getInputs().get(4).getOutputParameters().getLabel(),
								node.getInputs().get(5).getOutputParameters().getLabel(),
								node.getOutputParameters().getLabel());
					}
					else if (node.getInputs().size() == 7) {
						inst_string = node.getInstructions(
								node.getInputs().get(0).getOutputParameters().getLabel(),
								node.getInputs().get(1).getOutputParameters().getLabel(),
								node.getInputs().get(2).getOutputParameters().getLabel(),
								node.getInputs().get(3).getOutputParameters().getLabel(),
								node.getInputs().get(4).getOutputParameters().getLabel(),
								node.getInputs().get(5).getOutputParameters().getLabel(),
								node.getInputs().get(6).getOutputParameters().getLabel(),
								node.getOutputParameters().getLabel());
					}
					else {
						String[] inputs = new String[node.getInputs().size()];
						for( int j=0; j<node.getInputs().size(); j++ )
							inputs[j] = node.getInputs().get(j).getOutputParameters().getLabel();
						inst_string = node.getInstructions(inputs,
								node.getOutputParameters().getLabel());
					}
				}
				
				try {
					if( LOG.isTraceEnabled() )
						LOG.trace("Generating instruction - "+ inst_string);
					Instruction currInstr = InstructionParser.parseSingleInstruction(inst_string);
					if(currInstr == null) {
						 throw new LopsException("Error parsing the instruction:" + inst_string);
					}
					if (node._beginLine != 0)
						currInstr.setLocation(node);
					else if ( !node.getOutputs().isEmpty() )
						currInstr.setLocation(node.getOutputs().get(0));
					else if ( !node.getInputs().isEmpty() )
						currInstr.setLocation(node.getInputs().get(0));
						
					inst.add(currInstr);
				} catch (Exception e) {
					throw new LopsException(node.printErrorLocation() + "Problem generating simple inst - "
							+ inst_string, e);
				}

				markedNodes.add(node);
				doRmVar = true;
			}
			else if (node.isDataExecLocation() ) {
				Data dnode = (Data)node;
				Data.OperationTypes op = dnode.getOperationType();
				
				if ( op == Data.OperationTypes.WRITE ) {
					NodeOutput out = null;
						out = setupNodeOutputs(node, ExecType.CP, false, false);
						if ( dnode.getDataType() == DataType.SCALAR ) {
							// processing is same for both transient and persistent scalar writes 
							writeInst.addAll(out.getLastInstructions());
							doRmVar = false;
						}
						else {
							// setupNodeOutputs() handles both transient and persistent matrix writes 
							if ( dnode.isTransient() ) {
								deleteInst.addAll(out.getLastInstructions());
								doRmVar = false;
							}
							else {
								// In case of persistent write lop, write instruction will be generated 
								// and that instruction must be added to <code>inst</code> so that it gets
								// executed immediately. If it is added to <code>deleteInst</code> then it
								// gets executed at the end of program block's execution
								inst.addAll(out.getLastInstructions());
								doRmVar = true;
							}
						}
						markedNodes.add(node);
					
				}
				else {
					// generate a temp label to hold the value that is read from HDFS
					if ( node.getDataType() == DataType.SCALAR ) {
						node.getOutputParameters().setLabel(Lop.SCALAR_VAR_NAME_PREFIX + var_index.getNextID());
						String io_inst = node.getInstructions(node.getOutputParameters().getLabel(), 
								node.getOutputParameters().getFile_name());
						CPInstruction currInstr = CPInstructionParser.parseSingleInstruction(io_inst);
						currInstr.setLocation(node);
						
						inst.add(currInstr);
						
						Instruction tempInstr = VariableCPInstruction.prepareRemoveInstruction(node.getOutputParameters().getLabel());
						tempInstr.setLocation(node);
						deleteInst.add(tempInstr);
					}
					else {
						throw new LopsException("Matrix READs are not handled in CP yet!");
					}
					markedNodes.add(node);
					doRmVar = true;
				}
			}
			
			// see if rmvar instructions can be generated for node's inputs
			if(doRmVar)
				processConsumersForInputs(node, inst, deleteInst);
			doRmVar = false;
		}
		
		for ( String var : var_deletions ) {
			Instruction rmInst = VariableCPInstruction.prepareRemoveInstruction(var);
			if( LOG.isTraceEnabled() )
				LOG.trace("  Adding var_deletions: " + rmInst.toString());
			
			rmInst.setLocation(var_deletionsLineNum.get(var));
			
			deleteInst.add(rmInst);
		}

		// delete all marked nodes
		for ( Lop node : markedNodes ) {
			execNodes.remove(node);
		}
	}
	
	/**
	 * Method that determines the output format for a given node.
	 * 
	 * @param node low-level operator
	 * @param cellModeOverride override mode
	 * @return output info
	 */
	private static OutputInfo getOutputInfo(Lop node, boolean cellModeOverride)
	{
		if ( (node.getDataType() == DataType.SCALAR && node.getExecType() == ExecType.CP) 
				|| node instanceof FunctionCallCP )
			return null;
	
		OutputInfo oinfo = null;
		OutputParameters oparams = node.getOutputParameters();
		
		if (oparams.isBlocked()) {
			if ( !cellModeOverride )
				oinfo = OutputInfo.BinaryBlockOutputInfo;
			else {
				// output format is overridden, for example, due to recordReaderInstructions in the job
				oinfo = OutputInfo.BinaryCellOutputInfo;
				
				// record decision of overriding in lop's outputParameters so that 
				// subsequent jobs that use this lop's output know the correct format.
				// TODO: ideally, this should be done by having a member variable in Lop 
				//       which stores the outputInfo.   
				try {
					oparams.setDimensions(oparams.getNumRows(), oparams.getNumCols(), -1, oparams.getNnz(), oparams.getUpdateType());
				} catch(HopsException e) {
					throw new LopsException(node.printErrorLocation() + "error in getOutputInfo in Dag ", e);
				}
			}
		} else {
			if (oparams.getFormat() == Format.TEXT || oparams.getFormat() == Format.MM)
				oinfo = OutputInfo.TextCellOutputInfo;
			else if ( oparams.getFormat() == Format.CSV ) {
				oinfo = OutputInfo.CSVOutputInfo;
			}
			else {
				oinfo = OutputInfo.BinaryCellOutputInfo;
			}
		}

		if (node.getType() == Type.CentralMoment
				|| node.getType() == Type.CoVariance )  
		{
			// CMMR always operate in "cell mode",
			// and the output is always in cell format
			oinfo = OutputInfo.BinaryCellOutputInfo;
		}

		return oinfo;
	}
	
	private static String prepareAssignVarInstruction(Lop input, Lop node) {
		StringBuilder sb = new StringBuilder();
		
		sb.append(ExecType.CP);
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append("assignvar");
		sb.append(Lop.OPERAND_DELIMITOR);

		sb.append( input.prepScalarInputOperand(ExecType.CP) );
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append(node.prepOutputOperand());

		return sb.toString();
	}

	/**
	 * Method to setup output filenames and outputInfos, and to generate related instructions
	 * 
	 * @param node low-level operator
	 * @param et exec type
	 * @param cellModeOverride override mode
	 * @param copyTWrite ?
	 * @return node output
	 */
	private NodeOutput setupNodeOutputs(Lop node, ExecType et, boolean cellModeOverride, boolean copyTWrite) {
		
		OutputParameters oparams = node.getOutputParameters();
		NodeOutput out = new NodeOutput();
		
		node.setConsumerCount(node.getOutputs().size());
		
		// Compute the output format for this node
		out.setOutInfo(getOutputInfo(node, cellModeOverride));
		
		// If node is NOT of type Data then we must generate
		// a variable to hold the value produced by this node
		// note: functioncallcp requires no createvar, rmvar since
		// since outputs are explicitly specified
		if( !node.isDataExecLocation() ) 
		{
			if (node.getDataType() == DataType.SCALAR || node.getDataType() == DataType.LIST) {
				oparams.setLabel(Lop.SCALAR_VAR_NAME_PREFIX + var_index.getNextID());
				Instruction currInstr = VariableCPInstruction.prepareRemoveInstruction(oparams.getLabel());
				
				currInstr.setLocation(node);
				
				out.addLastInstruction(currInstr);
			}
			else if(!(node instanceof FunctionCallCP)) //general case
			{
				// generate temporary filename and a variable name to hold the
				// output produced by "rootNode"
				oparams.setFile_name(getNextUniqueFilename());
				oparams.setLabel(getNextUniqueVarname(node.getDataType()));

				// generate an instruction that creates a symbol table entry for the new variable
				//String createInst = prepareVariableInstruction("createvar", node);
				//out.addPreInstruction(CPInstructionParser.parseSingleInstruction(createInst));
				int blen = (int) oparams.getBlocksize();
				Instruction createvarInst = VariableCPInstruction.prepareCreateVariableInstruction(
					oparams.getLabel(), oparams.getFile_name(), true, node.getDataType(),
					OutputInfo.outputInfoToString(getOutputInfo(node, false)),
					new MatrixCharacteristics(oparams.getNumRows(), oparams.getNumCols(), blen, oparams.getNnz()),
					oparams.getUpdateType());
				
				createvarInst.setLocation(node);
				
				out.addPreInstruction(createvarInst);

				// temp file as well as the variable has to be deleted at the end
				Instruction currInstr = VariableCPInstruction.prepareRemoveInstruction(oparams.getLabel());
				
				currInstr.setLocation(node);
				
				out.addLastInstruction(currInstr);
			}
			else {
				// If the function call is set with output lops (e.g., multi return builtin),
				// generate a createvar instruction for each function output
				FunctionCallCP fcall = (FunctionCallCP) node;
				if ( fcall.getFunctionOutputs() != null ) {
					for( Lop fnOut: fcall.getFunctionOutputs()) {
						OutputParameters fnOutParams = fnOut.getOutputParameters();
						//OutputInfo oinfo = getOutputInfo((N)fnOut, false);
						Instruction createvarInst = VariableCPInstruction.prepareCreateVariableInstruction(
							fnOutParams.getLabel(), getFilePath() + fnOutParams.getLabel(), 
							true, fnOut.getDataType(),
							OutputInfo.outputInfoToString(getOutputInfo(fnOut, false)),
							new MatrixCharacteristics(fnOutParams.getNumRows(), fnOutParams.getNumCols(), (int)fnOutParams.getBlocksize(), fnOutParams.getNnz()),
							oparams.getUpdateType());
						
						if (node._beginLine != 0)
							createvarInst.setLocation(node);
						else
							createvarInst.setLocation(fnOut);
						
						out.addPreInstruction(createvarInst);
					}
				}
			}
		}
		// rootNode is of type Data
		else {
			
			if ( node.getDataType() == DataType.SCALAR ) {
				// generate assignment operations for final and transient writes
				if ( oparams.getFile_name() == null && !(node instanceof Data && ((Data)node).isPersistentWrite()) ) {
					String io_inst = prepareAssignVarInstruction(node.getInputs().get(0), node);
					CPInstruction currInstr = CPInstructionParser.parseSingleInstruction(io_inst);
					
					if (node._beginLine != 0)
						currInstr.setLocation(node);
					else if ( !node.getInputs().isEmpty() )
						currInstr.setLocation(node.getInputs().get(0));
					
					out.addLastInstruction(currInstr);
				}
				else {
					//CP PERSISTENT WRITE SCALARS
					Lop fname = ((Data)node).getNamedInputLop(DataExpression.IO_FILENAME);
					String io_inst = node.getInstructions(node.getInputs().get(0).getOutputParameters().getLabel(), fname.getOutputParameters().getLabel());
					CPInstruction currInstr = CPInstructionParser.parseSingleInstruction(io_inst);
					
					if (node._beginLine != 0)
						currInstr.setLocation(node);
					else if ( !node.getInputs().isEmpty() )
						currInstr.setLocation(node.getInputs().get(0));
					
					out.addLastInstruction(currInstr);
				}
			}
			else {
				if ( ((Data)node).isTransient() ) {
					
					if ( et == ExecType.CP ) {
						// If transient matrix write is in CP then its input MUST be executed in CP as well.
						
						// get variable and filename associated with the input
						String inputVarName = node.getInputs().get(0).getOutputParameters().getLabel();
						
						String constVarName = oparams.getLabel();
						
						/*
						 * Symbol Table state must change as follows:
						 * 
						 * FROM:
						 *     mvar1 -> temp21
						 *  
						 * TO:
						 *     mVar1 -> temp21
						 *     tVarH -> temp21
						 */
						Instruction currInstr = VariableCPInstruction.prepareCopyInstruction(inputVarName, constVarName);
						
						currInstr.setLocation(node);
						
						out.addLastInstruction(currInstr);
					}
					else {
						if(copyTWrite) {
							Instruction currInstr = VariableCPInstruction.prepareCopyInstruction(node.getInputs().get(0).getOutputParameters().getLabel(), oparams.getLabel());
							
							currInstr.setLocation(node);
							
							out.addLastInstruction(currInstr);
							return out;
						}
						
						/*
						 * Since the "rootNode" is a transient data node, we first need to generate a 
						 * temporary filename as well as a variable name to hold the <i>immediate</i> 
						 * output produced by "rootNode". These generated HDFS filename and the 
						 * variable name must be changed at the end of an iteration/program block 
						 * so that the subsequent iteration/program block can correctly access the 
						 * generated data. Therefore, we need to distinguish between the following:
						 * 
						 *   1) Temporary file name & variable name: They hold the immediate output 
						 *   produced by "rootNode". Both names are generated below.
						 *   
						 *   2) Constant file name & variable name: They are constant across iterations. 
						 *   Variable name is given by rootNode's label that is created in the upper layers.  
						 *   File name is generated by concatenating "temporary file name" and "constant variable name".
						 *   
						 * Temporary files must be moved to constant files at the end of the iteration/program block.
						 */
						
						// generate temporary filename & var name
						String tempVarName = oparams.getLabel() + "temp";
						String tempFileName = getNextUniqueFilename();
						
						int blen = (int) oparams.getBlocksize();
						Instruction createvarInst = VariableCPInstruction.prepareCreateVariableInstruction(
							tempVarName, tempFileName, true, node.getDataType(),
							OutputInfo.outputInfoToString(out.getOutInfo()), 
							new MatrixCharacteristics(oparams.getNumRows(), oparams.getNumCols(), blen, oparams.getNnz()),
							oparams.getUpdateType());
						
						createvarInst.setLocation(node);
						
						out.addPreInstruction(createvarInst);

						
						String constVarName = oparams.getLabel();
						String constFileName = tempFileName + constVarName;
						
						oparams.setFile_name(getFilePath() + constFileName);
						
						/*
						 * Since this is a node that denotes a transient read/write, we need to make sure 
						 * that the data computed for a given variable in a given iteration is passed on 
						 * to the next iteration. This is done by generating miscellaneous instructions 
						 * that gets executed at the end of the program block.
						 * 
						 * The state of the symbol table must change 
						 * 
						 * FROM: 
						 *     tVarA -> temp21tVarA (old copy of temp21)
						 *     tVarAtemp -> temp21  (new copy that should override the old copy) 
						 *
						 * TO:
						 *     tVarA -> temp21tVarA
						 */
						
						// Generate a single mvvar instruction (e.g., mvvar tempA A) 
						//    instead of two instructions "cpvar tempA A" and "rmvar tempA"
						Instruction currInstr = VariableCPInstruction.prepareMoveInstruction(tempVarName, constVarName);
						
						currInstr.setLocation(node);
						
						out.addLastInstruction(currInstr);
					}
				} 
				// rootNode is not a transient write. It is a persistent write.
				else {
					{ //CP PERSISTENT WRITE
						// generate a write instruction that writes matrix to HDFS
						Lop fname = ((Data)node).getNamedInputLop(DataExpression.IO_FILENAME);
						
						String io_inst = node.getInstructions(
							node.getInputs().get(0).getOutputParameters().getLabel(), 
							fname.getOutputParameters().getLabel());
						Instruction currInstr = (node.getExecType() == ExecType.SPARK) ?
							SPInstructionParser.parseSingleInstruction(io_inst) :
							CPInstructionParser.parseSingleInstruction(io_inst);
						currInstr.setLocation((!node.getInputs().isEmpty() 
							&& node.getInputs().get(0)._beginLine != 0) ? node.getInputs().get(0) : node);
						
						out.addLastInstruction(currInstr);
					}
				}
			}
		}
		
		return out;
	}

	/**
	 * Performs various cleanups on the list of instructions in order to reduce the
	 * number of instructions to simply debugging and reduce interpretation overhead. 
	 * 
	 * @param insts list of instructions
	 * @return new list of potentially modified instructions
	 */
	private static ArrayList<Instruction> cleanupInstructions(List<Instruction> insts) {
		//step 1: create mvvar instructions: assignvar s1 s2, rmvar s1 -> mvvar s1 s2
		List<Instruction> tmp1 = collapseAssignvarAndRmvarInstructions(insts);
		
		//step 2: create packed rmvar instructions: rmvar m1, rmvar m2 -> rmvar m1 m2
		ArrayList<Instruction> tmp2 = createPackedRmvarInstructions(tmp1);
		
		return tmp2;
	}
	
	private static List<Instruction> collapseAssignvarAndRmvarInstructions(List<Instruction> insts) {
		ArrayList<Instruction> ret = new ArrayList<>();
		Iterator<Instruction> iter = insts.iterator();
		while( iter.hasNext() ) {
			Instruction inst = iter.next();
			if( iter.hasNext() && inst instanceof VariableCPInstruction
				&& ((VariableCPInstruction)inst).isAssignVariable() ) {
				VariableCPInstruction inst1 = (VariableCPInstruction) inst;
				Instruction inst2 = iter.next();
				if( inst2 instanceof VariableCPInstruction
					&& ((VariableCPInstruction)inst2).isRemoveVariableNoFile()
					&& inst1.getInput1().getName().equals(
						((VariableCPInstruction)inst2).getInput1().getName()) ) {
					ret.add(VariableCPInstruction.prepareMoveInstruction(
						inst1.getInput1().getName(), inst1.getInput2().getName()));
				}
				else {
					ret.add(inst1);
					ret.add(inst2);
				}
			}
			else {
				ret.add(inst);
			}
		}
		return ret;
	}
	
	private static ArrayList<Instruction> createPackedRmvarInstructions(List<Instruction> insts) {
		ArrayList<Instruction> ret = new ArrayList<>();
		ArrayList<String> currRmVar = new ArrayList<>();
		for( Instruction inst : insts ) {
			if( inst instanceof VariableCPInstruction 
				&& ((VariableCPInstruction)inst).isRemoveVariableNoFile() ) {
				//collect all subsequent rmvar instructions
				currRmVar.add(((VariableCPInstruction)inst).getInput1().getName());
			}
			else {
				//construct packed rmvar instruction
				if( !currRmVar.isEmpty() ) {
					ret.add(VariableCPInstruction.prepareRemoveInstruction(
						currRmVar.toArray(new String[0])));
					currRmVar.clear();
				}
				//add other instruction
				ret.add(inst);
			}
		}
		//construct last packed rmvar instruction
		if( !currRmVar.isEmpty() ) {
			ret.add(VariableCPInstruction.prepareRemoveInstruction(
				currRmVar.toArray(new String[0])));
		}
		return ret;
	}
}
