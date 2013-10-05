/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.utils.visualize;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ArithmeticBinaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BuiltinBinaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ComputationCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FileCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.VariableCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.Instruction.INSTRUCTION_TYPE;


public class InstructionGraph 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private static String trimHashes(String s) {
		return (s.split("##"))[1];
	}

	private static String getInput(String inst, int index) {
		String operand = (((inst.split(Lop.OPERAND_DELIMITOR))[index]).split(Lop.VALUETYPE_PREFIX))[0];
		if (operand.startsWith("##")) {
			String s = trimHashes(operand);
			if (s != null)
				return s;
			else
				throw new RuntimeException("Unexpected error in getInput() while plotting InstructionsDAG.");
		} else
			return operand;
	}

	private static String getOperator(String inst) {
		return ((inst.split(Lop.OPERAND_DELIMITOR))[0]);
	}

	private static ArrayList<String> getVariableNames(String str, ArrayList<String> varsIn) {
		ArrayList<String> varsOut = new ArrayList<String>(varsIn);

		for (String inst : str.split(Lop.INSTRUCTION_DELIMITOR)) {
			if (inst.contains("Var")) {
				for (String part : inst.split(Lop.OPERAND_DELIMITOR)) {
					if (part.contains("Var")) {
						String s = (part.split(Lop.VALUETYPE_PREFIX))[0];
						if (s.startsWith("##")) {
							varsOut.add(trimHashes(s));
						} else
							throw new RuntimeException(
									"Unexpected error in getVariableNames() while plotting InstructionsDAG -- expected ## for variable: " + s);
					}
				}
			}
		}
		return varsOut;
	}

	private static String prepareInstructionsNodeList(ProgramBlock pb, HashMap<String, String> outputMap, int blockid) throws DMLUnsupportedOperationException {
		String lastFile="nodeFile", lastVar="nodeVar", lastScalar="nodeScalar";
		
		String s = new String("");
		for (int i = 0; i < pb.getNumInstructions(); i++) {
			Instruction inst = pb.getInstruction(i);
			String idStr = "node" + inst.hashCode();

			if (inst.getType() == INSTRUCTION_TYPE.MAPREDUCE_JOB) {
				MRJobInstruction mrInst = (MRJobInstruction) inst;
				if ( mrInst.getJobType() == JobType.MMCJ )
					s += "    " + idStr + " [label=\"" + inst.getGraphString() + "\", color=bisque ]; \n";
				else if ( mrInst.getJobType() == JobType.MMRJ )
					s += "    " + idStr + " [label=\"" + inst.getGraphString() + "\", color=palegreen ]; \n";
				else
					s += "    " + idStr + " [label=\"" + inst.getGraphString() + "\", color=white ]; \n";
				String[] outputs = mrInst.getOutputVars();
				for (String out : outputs) {
					outputMap.put(out, idStr);
				}

				String[] inputs = mrInst.getInputVars();
				if (inputs != null) {
					for (String in : inputs) {
						if (in != null && in.contains("##")) {
							String tempnode = "tnodeB" + blockid + trimHashes(in);
							s += "    " + tempnode + " [label=\"" + in.replaceAll("##", "") + "\", color=grey ]; \n";
							String edge = tempnode + " -> " + idStr + " [label=\"" + in.replaceAll("##", "") + "\" ] " + "; \n";
							s += edge;
						} else {
							String producer = outputMap.get(in);
							String edge = producer + " -> " + idStr + " [label=\"" + in.replaceAll("##", "") + "\" ] " + "; \n";
							s += edge;
						}
					}
				}

				// Get all variables (Var) used in miscellaneous instructions
				// All Var's are used only as input operands
				ArrayList<String> vars = new ArrayList<String>();
				if (mrInst.getIv_shuffleInstructions().contains("Var")) {
					vars = getVariableNames(mrInst.getIv_shuffleInstructions(), vars);
				}
				if (mrInst.getIv_aggInstructions().contains("Var")) {
					vars = getVariableNames(mrInst.getIv_aggInstructions(), vars);
				}
				if (mrInst.getIv_instructionsInMapper().contains("Var")) {
					vars = getVariableNames(mrInst.getIv_instructionsInMapper(), vars);
				}
				if (mrInst.getIv_otherInstructions().contains("Var")) {
					vars = getVariableNames(mrInst.getIv_instructionsInMapper(), vars);
				}

				String producer = null;
				for (String var : vars) {
					producer = outputMap.get(var);
					if (producer != null) {
						String edge = producer + " -> " + idStr + " ; \n";
						// String edge = producer + " -> " + idStr + " [label=\"castToScalar\" ] " + "; \n";
						s += edge;
					}
				}
			} else {
				s += "    " + idStr + " [label=\"" + inst.getGraphString() + "\", color=white ]; \n";
				CPInstruction simpleInst = (CPInstruction) inst;
				String instStr = simpleInst.toString();
				String opStr = getOperator(instStr);
				//OpCode opcode = OpCode.getOpcode(opStr);
				
				if ( simpleInst instanceof VariableCPInstruction ) {
					
					// VAR_COPY produces outputs that are consumed by other MR jobs
					// So, update outputMap
					if ( simpleInst.getCPInstructionType() == CPINSTRUCTION_TYPE.Variable && opStr.equalsIgnoreCase("assignvar") ) {
					// if (opcode.getType() == OpCode.Type.Var && ((VarOpcode) opcode).getVarOp() == VarOperation.VAR_COPY) {
						String in1 = getInput(instStr, 1);
						String out = getInput(instStr, 2);
						String producer = outputMap.get(in1);
						if (producer != null) {
							outputMap.put(out, producer);
						}
					}
					
					if ( opStr.equalsIgnoreCase("assignvarwithfile") ) {
						// handle cast_to_scalar instruction specially!
						String in = getInput(instStr, 1);
						String out = getInput(instStr, 2);
						outputMap.put(out, idStr);

						String producer = outputMap.get(in);
						if (producer != null) {
							String edge = producer + " -> " + idStr + " [label=\"castToScalar\" ] " + "; \n";
							s += edge;
						}
					}
					else {
						// create an edge under the default list of all VariablSimpleInstructions
						String edge = lastVar + " -> " + idStr + " [label=\".\" ] " + "; \n";
						lastVar = idStr;
						s += edge;
					}
				} else if (simpleInst instanceof ComputationCPInstruction) {
					if (simpleInst instanceof ArithmeticBinaryCPInstruction) {

						String in1 = getInput(instStr, 1);
						String in2 = getInput(instStr, 2);
						String out = getInput(instStr, 3);
						outputMap.put(out, idStr);
						String producer1 = outputMap.get(in1);
						String producer2 = outputMap.get(in2);
						if (producer1 != null) {
							String edge = producer1 + " -> " + idStr + " [label=\".\" ] " + "; \n";
							s += edge;
						}
						if (producer2 != null) {
							String edge = producer2 + " -> " + idStr + " [label=\".\" ] " + "; \n";
							s += edge;
						}
					}
					else if (simpleInst instanceof BuiltinBinaryCPInstruction ) {
						BuiltinBinaryCPInstruction binst = (BuiltinBinaryCPInstruction) simpleInst;
						int arity = binst.getArity();
						if ( arity == 1 ) {
							String in1 = getInput(instStr, 1);
							String out = getInput(instStr, 2);
							outputMap.put(out, idStr);
							String producer1 = outputMap.get(in1);
							if (producer1 != null) {
								String edge = producer1 + " -> " + idStr + " [label=\".\" ] " + "; \n";
								s += edge;
							}
						}
						else if ( arity == 2 ) {
							String in1 = getInput(instStr, 1);
							String in2 = getInput(instStr, 2);
							String out = getInput(instStr, 3);
							outputMap.put(out, idStr);
							String producer1 = outputMap.get(in1);
							String producer2 = outputMap.get(in2);
							if (producer1 != null) {
								String edge = producer1 + " -> " + idStr + " [label=\".\" ] " + "; \n";
								s += edge;
							}
							if (producer2 != null) {
								String edge = producer2 + " -> " + idStr + " [label=\".\" ] " + "; \n";
								s += edge;
							}
						}
						else {
							// create an edge under the default list of all ScalarSimpleInstructions
							String edge = lastScalar + " -> " + idStr + " [label=\".\" ] " + "; \n";
							lastScalar = idStr;
							s += edge;
						}
					}
					else {
						// create an edge under the default list of all ScalarSimpleInstructions
						String edge = lastScalar + " -> " + idStr + " [label=\".\" ] " + "; \n";
						lastScalar = idStr;
						s += edge;
					}
				} else if (simpleInst instanceof FileCPInstruction) {
					// create an edge under the default list of all FileSimpleInstructions
					String edge = lastFile + " -> " + idStr + " [label=\".\" ] " + "; \n";
					lastFile = idStr;
					s += edge;
				}
			}
		}
		return s;
	}

	// TODO: FIX THIS TO MAKE IT NESTED -----
	public static String getInstructionGraphString(Program dmlp, String title, int x, int y, String basePath) throws DMLUnsupportedOperationException {
		String graphString = new String("digraph G { \n node [ shape = box, label = \"\\N\", style=filled ]; \n ");
		HashMap<String, String> outputMap = new HashMap<String, String>();

		graphString += "    nodeVar  [label=\"VarSimpleInst\", color=green ]; \n";
		graphString += "    nodeFile  [label=\"FileSimpleInst\", color=green ]; \n";
		graphString += "    nodeScalar  [label=\"ScalarSimpleInst\", color=green ]; \n";
		
		int blockid = 0;
		
		for (ProgramBlock pb : dmlp.getProgramBlocks()) {
			if ( pb instanceof WhileProgramBlock ){
				// System.out.println(pb.toString());
				for ( ProgramBlock cpb: ((WhileProgramBlock)pb).getChildBlocks()) {
					String pbStr = prepareInstructionsNodeList(cpb, outputMap, blockid );
					blockid++;
					graphString += pbStr;
				}
			}
			else { 
				// System.out.println(pb.toString());
				String pbStr = prepareInstructionsNodeList(pb, outputMap, blockid );
				blockid++;
				graphString += pbStr;
			}
		}

		graphString += "\n } \n";
		return graphString;
	}
}
