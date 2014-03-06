/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops.compile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Vector;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.SequenceFileInputFormat;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.Hop.FileFormatTypes;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.lops.CombineBinary;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Data.OperationTypes;
import com.ibm.bi.dml.lops.FunctionCallCP;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.Lop.Type;
import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.OutputParameters;
import com.ibm.bi.dml.lops.OutputParameters.Format;
import com.ibm.bi.dml.lops.PartitionLop;
import com.ibm.bi.dml.lops.PickByCount;
import com.ibm.bi.dml.lops.Unary;
import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.parser.Expression;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.CPInstructionParser;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.Instruction.INSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.VariableCPInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.sort.PickFromCompactInputFormat;



/**
 * 
 *  Class to maintain a DAG and compile it into jobs
 * @param <N>
 */
public class Dag<N extends Lop> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(Dag.class.getName());

	private static final int CHILD_BREAKS_ALIGNMENT = 2;
	private static final int CHILD_DOES_NOT_BREAK_ALIGNMENT = 1;
	private static final int MRCHILD_NOT_FOUND = 0;
	private static final int MR_CHILD_FOUND_BREAKS_ALIGNMENT = 4;
	private static final int MR_CHILD_FOUND_DOES_NOT_BREAK_ALIGNMENT = 5;

	private static int total_reducers = -1;
	private static String scratch = "";
	
	private static IDSequence job_id = null;
	private static IDSequence var_index = null;
	
	static {
		job_id = new IDSequence();
		var_index = new IDSequence();
	}
	
	// hash set for all nodes in dag
	private ArrayList<N> nodes = null;
	
	/* 
	 * Hashmap to translates the nodes in the DAG to a sequence of numbers
	 *     key:   Lop ID
	 *     value: Sequence Number (0 ... |DAG|)
	 *     
	 * This map is primarily used in performing DFS on the DAG, and subsequently in performing ancestor-descendant checks.
	 */
	private HashMap<Long, Integer> IDMap = null;


	public static class PiggyProfiler {
		public static long setupTime=0;
		public static long topoTime=0, groupingTime=0;
		public static long topoSetup=0, topoFindSrcs=0, topoSetLevels=0, topoSort=0, topoCopy=0, topoSanity=0;
		
		public static void print() {
			System.out.print("Piggybacking: setup = " + setupTime + ", topo = " + topoTime + ", grouping = " + groupingTime);
			System.out.println(" --> Total = " + (setupTime + topoTime + groupingTime) + " msec.");
			System.out.println("TOPO: " + topoSetup + "   " +  topoFindSrcs + "   " +  topoSetLevels + "   " +  topoSort + "   " +  topoCopy + "   " +  topoSanity);
		}
	}
	
	private class NodeOutput {
		String fileName;
		String varName;
		OutputInfo outInfo;
		ArrayList<Instruction> preInstructions; // instructions added before a MR instruction
		ArrayList<Instruction> postInstructions; // instructions added after a MR instruction
		ArrayList<Instruction> lastInstructions;
		
		NodeOutput() {
			fileName = null;
			varName = null;
			outInfo = null;
			preInstructions = new ArrayList<Instruction>(); 
			postInstructions = new ArrayList<Instruction>(); 
			lastInstructions = new ArrayList<Instruction>();
		}
		
		public String getFileName() {
			return fileName;
		}
		public void setFileName(String fileName) {
			this.fileName = fileName;
		}
		public String getVarName() {
			return varName;
		}
		public void setVarName(String varName) {
			this.varName = varName;
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
		public ArrayList<Instruction> getPostInstructions() {
			return postInstructions;
		}
		public void addPostInstruction(Instruction inst) {
			postInstructions.add(inst);
		}
		public ArrayList<Instruction> getLastInstructions() {
			return lastInstructions;
		}
		public void addLastInstruction(Instruction inst) {
			lastInstructions.add(inst);
		}
		
	}
	
	/**
	 * Constructor
	 */
	public Dag() 
	{
		//allocate internal data structures
		nodes = new ArrayList<N>();
		IDMap = new HashMap<Long, Integer>();

		// get number of reducers from dml config
		DMLConfig conf = ConfigurationManager.getConfig();
		total_reducers = conf.getIntValue(DMLConfig.NUM_REDUCERS);
	}

	/**
	 * 
	 * @param config
	 * @return
	 * @throws LopsException
	 * @throws IOException
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public ArrayList<Instruction> getJobs(DMLConfig config)
			throws LopsException, IOException, DMLRuntimeException, DMLUnsupportedOperationException 
	{
		return getJobs(null, config);
	}
	
	/**
	 * Method to compile a dag generically
	 * 
	 * @param config
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */

	public ArrayList<Instruction> getJobs(StatementBlock sb, DMLConfig config)
			throws LopsException, IOException, DMLRuntimeException,
			DMLUnsupportedOperationException {
		
		if (config != null) 
		{
			total_reducers = config.getIntValue(DMLConfig.NUM_REDUCERS);
			scratch = config.getTextValue(DMLConfig.SCRATCH_SPACE) + "/";
		}
		
		// hold all nodes in a vector (needed for ordering)
		Vector<N> node_v = new Vector<N>();
		node_v.addAll(nodes);
		
		/*
		 * Sort the nodes by topological order.
		 * 
		 * 1) All nodes with level i appear prior to the nodes in level i+1.
		 * 2) All nodes within a level are ordered by their ID i.e., in the order
		 * they are created
		 */
		doTopologicalSort_strict_order(node_v);
		
		// do greedy grouping of operations
		ArrayList<Instruction> inst = doGreedyGrouping(sb, node_v);
		
		return inst;

	}

	/**
	 * Method to remove transient reads that do not have a transient write
	 * 
	 * @param nodeV
	 * @param inst
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private void deleteUnwantedTransientReadVariables(Vector<N> nodeV,
			ArrayList<Instruction> inst) throws DMLRuntimeException,
			DMLUnsupportedOperationException {
		HashMap<String, N> labelNodeMapping = new HashMap<String, N>();
		
		LOG.trace("In delete unwanted variables");

		// first capture all transient read variables
		for (int i = 0; i < nodeV.size(); i++) {
			N node = nodeV.get(i);

			if (node.getExecLocation() == ExecLocation.Data
					&& ((Data) node).isTransient()
					&& ((Data) node).getOperationType() == OperationTypes.READ) {
				labelNodeMapping.put(node.getOutputParameters().getLabel(),
						node);
			}
		}

		// generate delete instructions for all transient read variables without
		// a transient write
		// first capture all transient write variables
		for (int i = 0; i < nodeV.size(); i++) {
			N node = nodeV.get(i);

			if (node.getExecLocation() == ExecLocation.Data
					&& ((Data) node).isTransient()
					&& ((Data) node).getOperationType() == OperationTypes.WRITE ) {
				if (node.getInputs().get(0).getExecLocation() == ExecLocation.Data) {
					// this transient write is NOT a result of actual computation, but is simple copy.
					// in such a case, preserve the input variable so that appropriate copy instruction (cpvar or GMR) is generated
					// therefore, remove the input label from labelNodeMapping
					labelNodeMapping.remove(node.getInputs().get(0).getOutputParameters().getLabel());
				}
				else {
					if(labelNodeMapping.containsKey(node.getOutputParameters().getLabel())) 
						// corresponding transient read exists (i.e., the variable is updated)
						// in such a case, generate rmvar instruction so that OLD data gets deleted
						labelNodeMapping.remove(node.getOutputParameters().getLabel());
				}
			}
		}

		// generate RM instructions

		Iterator<String> it = labelNodeMapping.keySet().iterator();
		Instruction rm_inst = null;
		while (it.hasNext()) {
			String label = it.next();
			N node = labelNodeMapping.get(label);

			if (((Data) node).get_dataType() == DataType.SCALAR) {
				// if(DEBUG)
				// System.out.println("rmvar" + Lops.OPERAND_DELIMITOR + label);
				// inst.add(new VariableSimpleInstructions("rmvar" +
				// Lops.OPERAND_DELIMITOR + label));

			} else {
				rm_inst = VariableCPInstruction.prepareRemoveInstruction(label);
				
				LOG.trace(rm_inst.toString());
				inst.add(rm_inst);
				
			}
		}

	}

	private void deleteUpdatedTransientReadVariables(StatementBlock sb, Vector<N> nodeV,
			ArrayList<Instruction> inst) throws DMLRuntimeException,
			DMLUnsupportedOperationException {

		if ( sb == null ) 
			return;
		
		LOG.trace("In delete updated variables");

		/*
		Set<String> in = sb.liveIn().getVariables().keySet();
		Set<String> out = sb.liveOut().getVariables().keySet();
		Set<String> updated = sb.variablesUpdated().getVariables().keySet();
		
		Set<String> intersection = in;
		intersection.retainAll(out);
		intersection.retainAll(updated);
		
		for (String var : intersection) {
			inst.add(VariableCPInstruction.prepareRemoveInstruction(var));
		}
		*/

		// CANDIDATE list of variables which could have been updated in this statement block 
		HashMap<String, N> labelNodeMapping = new HashMap<String, N>();
		
		// ACTUAL list of variables whose value is updated, AND the old value of the variable 
		// is no longer accessible/used.
		HashSet<String> updatedLabels = new HashSet<String>();
		
		// first capture all transient read variables
		for (int i = 0; i < nodeV.size(); i++) {
			N node = nodeV.get(i);

			if (node.getExecLocation() == ExecLocation.Data
					&& ((Data) node).isTransient()
					&& ((Data) node).getOperationType() == OperationTypes.READ
					&& ((Data) node).get_dataType() == DataType.MATRIX) {
				
				// "node" is considered as updated ONLY IF the old value is not used any more
				// So, make sure that this READ node does not feed into any (transient/persistent) WRITE
				boolean hasWriteParent=false;
				for(Lop p : node.getOutputs()) {
					if(p.getExecLocation() == ExecLocation.Data) {
						// if the "p" is of type Data, then it has to be a WRITE
						hasWriteParent = true;
						break;
					}
				}
				
				if ( hasWriteParent == false ) {
					// node has no parent of type WRITE, so this is a CANDIDATE variable 
					// add it to labelNodeMapping so that it is considered in further processing  
					labelNodeMapping.put(node.getOutputParameters().getLabel(), node);
				}
			}
		}

		// capture updated transient write variables
		for (int i = 0; i < nodeV.size(); i++) {
			N node = nodeV.get(i);

			if (node.getExecLocation() == ExecLocation.Data
					&& ((Data) node).isTransient()
					&& ((Data) node).getOperationType() == OperationTypes.WRITE
					&& ((Data) node).get_dataType() == DataType.MATRIX
					&& labelNodeMapping.containsKey(node.getOutputParameters().getLabel()) // check to make sure corresponding (i.e., with the same label/name) transient read is present
					&& !labelNodeMapping.containsValue(node.getInputs().get(0)) // check to avoid cases where transient read feeds into a transient write 
				) {
				updatedLabels.add(node.getOutputParameters().getLabel());
			}
		}
		
		// generate RM instructions
		Iterator<String> it = updatedLabels.iterator();
		Instruction rm_inst = null;
		while (it.hasNext()) {
			String label = it.next();

			rm_inst = VariableCPInstruction.prepareRemoveInstruction(label);
			
			LOG.trace(rm_inst.toString());
			inst.add(rm_inst);
		}

	}
	  
	private void generateRemoveInstructions(StatementBlock sb,
			ArrayList<Instruction> deleteInst)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		if ( sb == null ) 
			return;
		
		LOG.trace("In generateRemoveInstructions()");
		

		Instruction inst = null;
		// RULE 1: if in IN and not in OUT, then there should be an rmvar or rmfilevar inst
		// (currently required for specific cases of external functions)
		for (String varName : sb.liveIn().getVariableNames()) {
			if (!sb.liveOut().containsVariable(varName)) {
				// DataType dt = in.getVariable(varName).getDataType();
				// if( !(dt==DataType.MATRIX || dt==DataType.UNKNOWN) )
				// continue; //skip rm instructions for non-matrix objects

				inst = VariableCPInstruction.prepareRemoveInstruction(varName);
				deleteInst.add(inst);

				LOG.trace("  Adding " + inst.toString());
			}
		}

		// RULE 2: if in KILL and not in IN and not in OUT, then there should be an rmvar or rmfilevar inst
		// (currently required for specific cases of nested loops)
		// i.e., local variables which are created within the block, and used entirely within the block 
		/*for (String varName : sb.getKill().getVariableNames()) {
			if ((!sb.liveIn().containsVariable(varName))
					&& (!sb.liveOut().containsVariable(varName))) {
				// DataType dt =
				// sb.getKill().getVariable(varName).getDataType();
				// if( !(dt==DataType.MATRIX || dt==DataType.UNKNOWN) )
				// continue; //skip rm instructions for non-matrix objects

				inst = createCleanupInstruction(varName);
				deleteInst.add(inst);

				if (DMLScript.DEBUG)
					System.out.println("Adding instruction (r2) "
							+ inst.toString());
			}
		}*/
	}

	/**
	 * Method to add a node to the DAG.
	 * 
	 * @param node
	 * @return true if node was not already present, false if not.
	 */

	public boolean addNode(N node) {
		if (nodes.contains(node))
			return false;
		else {
			nodes.add(node);
			return true;
		}
		
	}

	private ArrayList<Vector<N>> createNodeVectors(int size) {
		ArrayList<Vector<N>> arr = new ArrayList<Vector<N>>();

		// for each job type, we need to create a vector.
		// additionally, create another vector for execNodes
		for (int i = 0; i < size; i++) {
			arr.add(new Vector<N>());
		}
		return arr;
	}

	private void clearNodeVectors(ArrayList<Vector<N>> arr) {
		for (int i = 0; i < arr.size(); i++) {
			arr.get(i).clear();
		}
	}

	private boolean isCompatible(Vector<N> nodes, JobType jt, int from,
			int to) throws LopsException {
		
		int base = jt.getBase();

		for (int i = from; i < to; i++) {
			if ((nodes.get(i).getCompatibleJobs() & base) == 0) {
				LOG.trace("Not compatible "
							+ nodes.get(i).toString());
				return false;
			}
		}
		return true;
	}

	private boolean isCompatible(N node, JobType jt) {
		return ((node.getCompatibleJobs() & jt.getBase()) > 0);
	}

	/*
	 * Add node, and its relevant children to job-specific node vectors.
	 */
	private void addNodeByJobType(N node, ArrayList<Vector<N>> arr,
			Vector<N> execNodes, boolean eliminate) throws LopsException {
		
		if (eliminate == false) {
			// Check if this lop defines a MR job.
			if ( node.definesMRJob() ) {
				
				// find the corresponding JobType
				JobType jt = JobType.findJobTypeFromLop(node);
				
				if ( jt == null ) {
					throw new LopsException(node.printErrorLocation() + "No matching JobType is found for a the lop type: " + node.getType() + " \n");
				}
				
				// Add "node" to corresponding job vector
				
				if ( jt == JobType.GMR ) {
					// if "node" (in this case, a group lop) has any inputs from RAND 
					// then add it to RAND job. Otherwise, create a GMR job
					if (hasChildNode(node, arr.get(JobType.DATAGEN.getId()) )) {
						arr.get(JobType.DATAGEN.getId()).add(node);
						// we should NOT call 'addChildren' because appropriate
						// child nodes would have got added to RAND job already
					} else {
						int gmr_index = JobType.GMR.getId();
						arr.get(gmr_index).add(node);
						int from = arr.get(gmr_index).size();
						addChildren(node, arr.get(gmr_index), execNodes);
						int to = arr.get(gmr_index).size();
						if (!isCompatible(arr.get(gmr_index),JobType.GMR, from, to)) {
							throw new LopsException(node.printErrorLocation() + "Error during compatibility check \n");
						}
					}
				}
				else {
					int index = jt.getId();
					arr.get(index).add(node);
					int from = arr.get(index).size();
					addChildren(node, arr.get(index), execNodes);
					int to = arr.get(index).size();
					// check if all added nodes are compatible with current job
					if (!isCompatible(arr.get(index), jt, from, to)) {
						throw new LopsException( 
								"Unexpected error in addNodeByType.");
					}
				}
				return;
			}
		}

		/*
		 * If this lop does not define a job, check if it uses the output of any
		 * specialized job. i.e., if this lop has a child node in any of the
		 * job-specific vector, then add it to the vector. Note: This lop must
		 * be added to ONLY ONE of the job-specific vectors.
		 */

		int numAdded = 0;
		for ( JobType j : JobType.values() ) {
			if ( j.getId() > 0 && hasChildNode(node, arr.get(j.getId()))) {
				if (eliminate || isCompatible(node, j)) {
					arr.get(j.getId()).add(node);
					numAdded += 1;
				}
			}
		}
		if (numAdded > 1) {
			throw new LopsException("Unexpected error in addNodeByJobType(): A given lop can ONLY be added to a single job vector (numAdded = " + numAdded + ")." );
		}
	}

	/*
	 * Remove the node from all job-specific node vectors. This method is
	 * invoked from removeNodesForNextIteration().
	 */
	private void removeNodeByJobType(N node, ArrayList<Vector<N>> arr) {
		for ( JobType jt : JobType.values())
			if ( jt.getId() > 0 ) 
				arr.get(jt.getId()).remove(node);
	}

	/**
	 * As some jobs only write one output, all operations in the mapper need to
	 * be redone and cannot be marked as finished.
	 * 
	 * @param execNodes
	 * @param jobNodes
	 * @throws LopsException
	 */

	private void handleSingleOutputJobs(Vector<N> execNodes,
			ArrayList<Vector<N>> jobNodes, Vector<N> finishedNodes)
			throws LopsException {
		/*
		 * If the input of a MMCJ/MMRJ job (must have executed in a Mapper) is used
		 * by multiple lops then we should mark it as not-finished.
		 */
		Vector<N> nodesWithUnfinishedOutputs = new Vector<N>();
		int[] jobIndices = {JobType.MMCJ.getId()};
		Lop.Type[] lopTypes = { Lop.Type.MMCJ};
		
		// TODO: SortByValue should be treated similar to MMCJ, since it can
		// only sort one file now
		
		for ( int jobi=0; jobi < jobIndices.length; jobi++ ) {
			int jindex = jobIndices[jobi];
			if (jobNodes.get(jindex).size() > 0) {
				Vector<N> vec = jobNodes.get(jindex);

				// first find all nodes with more than one parent that is not
				// finished.

				for (int i = 0; i < vec.size(); i++) {
					N node = vec.get(i);
					if (node.getExecLocation() == ExecLocation.MapOrReduce
							|| node.getExecLocation() == ExecLocation.Map) {
						N MRparent = getParentNode(node, execNodes, ExecLocation.MapAndReduce);
						if ( MRparent != null && MRparent.getType() == lopTypes[jobi]) {
							int numParents = node.getOutputs().size();
							if (numParents > 1) {
								for (int j = 0; j < numParents; j++) {
									if (!finishedNodes.contains(node.getOutputs()
											.get(j)))
										nodesWithUnfinishedOutputs.add(node);
								}
	
							}
						}
					} 
					/*
					// Following condition will not work for MMRJ because execNodes may contain non-MapAndReduce 
					// lops that are compatible with MMRJ (e.g., Data-WRITE)
					else if (!(node.getExecLocation() == ExecLocation.MapAndReduce 
							     && node.getType() == lopTypes[jobi])) {
						throw new LopsException(
								"Error in handleSingleOutputJobs(): encountered incompatible execLocation='"
										+ node.getExecLocation() + "' for lop (ID="
										+ node.getID() + ").");
					}
					*/
				}

				// need to redo all nodes in nodesWithOutput as well as their
				// children

				for (int i = 0; i < vec.size(); i++) {
					N node = vec.get(i);
					if (node.getExecLocation() == ExecLocation.MapOrReduce
							|| node.getExecLocation() == ExecLocation.Map) {
						if (nodesWithUnfinishedOutputs.contains(node))
							finishedNodes.remove(node);

						if (hasParentNode(node, nodesWithUnfinishedOutputs))
							finishedNodes.remove(node);

					}
				}
			}			
		}
		
	}

	/** Method to check if a lop can be eliminated from checking **/
	private boolean canEliminateLop(N node, Vector<N> execNodes) {
		// this function can only eliminate "aligner" lops such a group
		if (!node.isAligner())
			return false;

		// find the child whose execLoc = 'MapAndReduce'
		int ret = getChildAlignment(node, execNodes, ExecLocation.MapAndReduce);

		if (ret == CHILD_BREAKS_ALIGNMENT)
			return false;
		else if (ret == CHILD_DOES_NOT_BREAK_ALIGNMENT)
			return true;
		else if (ret == MRCHILD_NOT_FOUND)
			return false;
		else if (ret == MR_CHILD_FOUND_BREAKS_ALIGNMENT)
			return false;
		else if (ret == MR_CHILD_FOUND_DOES_NOT_BREAK_ALIGNMENT)
			return true;
		else
			throw new RuntimeException("Should not happen. \n");
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
	 * @param nodes
	 * @throws LopsException 
	 */
	private void generateInstructionsForInputVariables(Vector<N> nodes_v, ArrayList<Instruction> inst) throws LopsException, IOException {
		for(N n : nodes_v) {
			if (n.getExecLocation() == ExecLocation.Data && !((Data) n).isTransient() 
					&& ((Data) n).getOperationType() == OperationTypes.READ 
					&& n.get_dataType() == DataType.MATRIX) {
				
				if ( !((Data)n).isLiteral() ) {
					try {
						String inst_string = n.getInstructions();
						inst.add(CPInstructionParser.parseSingleInstruction(inst_string));
					} catch (DMLUnsupportedOperationException e) {
						throw new LopsException(n.printErrorLocation() + "error generating instructions from input variables in Dag -- \n", e);
					} catch (DMLRuntimeException e) {
						throw new LopsException(n.printErrorLocation() + "error generating instructions from input variables in Dag -- \n", e);
					}
				}
			}
		}
	}
	
	/*private String getOutputFormat(Data node) { 
		if(node.getFileFormatType() == FileFormatTypes.TEXT)
			return "textcell";
		else if ( node.getOutputParameters().isBlocked_representation() )
			return "binaryblock";
		else
			return "binarycell";
	}*/
	
	/**
	 * Determine whether to send <code>node</code> to MR or to process it in the control program.
	 * It is sent to MR in the following cases:
	 * 
	 * 1) if input lop gets processed in MR then <code>node</code> can be piggybacked
	 * 
	 * 2) if the exectype of write lop itself is marked MR i.e., memory estimate > memory budget.
	 * 
	 * @param node
	 * @return
	 */
	@SuppressWarnings("unchecked")
	private boolean sendWriteLopToMR(N node) {
		if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE )
			return false;
		N in = (N) node.getInputs().get(0);
		if( in.getExecType() == ExecType.MR  || node.getExecType() == ExecType.MR )
		{
			return true;
		}
		else
			return false;
	}
	
	/**
	 * Method to group a vector of sorted lops.
	 * 
	 * @param node_v
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */

	@SuppressWarnings("unchecked")
	private ArrayList<Instruction> doGreedyGrouping(StatementBlock sb, Vector<N> node_v)
			throws LopsException, IOException, DMLRuntimeException,
			DMLUnsupportedOperationException

	{
		LOG.trace("Grouping DAG ============");

		// nodes to be executed in current iteration
		Vector<N> execNodes = new Vector<N>();
		// nodes that have already been processed
		Vector<N> finishedNodes = new Vector<N>();
		// nodes that are queued for the following iteration
		Vector<N> queuedNodes = new Vector<N>();

		ArrayList<Vector<N>> jobNodes = createNodeVectors(JobType.getNumJobTypes());

		// list of instructions
		ArrayList<Instruction> inst = new ArrayList<Instruction>();

		//ArrayList<Instruction> preWriteDeleteInst = new ArrayList<Instruction>();
		ArrayList<Instruction> writeInst = new ArrayList<Instruction>();
		ArrayList<Instruction> deleteInst = new ArrayList<Instruction>();
		ArrayList<Instruction> endOfBlockInst = new ArrayList<Instruction>();

		// delete transient variables that are no longer needed
		//deleteUnwantedTransientReadVariables(node_v, deleteInst);

		// remove files for transient reads that are updated.
		deleteUpdatedTransientReadVariables(sb, node_v, deleteInst);
		
		generateRemoveInstructions(sb, endOfBlockInst);

		generateInstructionsForInputVariables(node_v, inst);
		
		
		boolean done = false;
		String indent = "    ";

		while (!done) {
			LOG.trace("Grouping nodes in DAG");

			execNodes.clear();
			queuedNodes.clear();
			clearNodeVectors(jobNodes);

			for (int i = 0; i < node_v.size(); i++) {
				N node = node_v.elementAt(i);

				// finished nodes don't need to be processed

				if (finishedNodes.contains(node))
					continue;

				LOG.trace("Processing node (" + node.getID()
							+ ") " + node.toString() + " exec nodes size is " + execNodes.size());
				
				
		        //if node defines MR job, make sure it is compatible with all 
		        //its children nodes in execNodes 
		        if(node.definesMRJob() && !compatibleWithChildrenInExecNodes(execNodes, node))
		        {
		          LOG.trace(indent + "Queueing node "
		                + node.toString() + " (code 1)");
		
		          queuedNodes.add(node);
		          removeNodesForNextIteration(node, finishedNodes, execNodes, queuedNodes, jobNodes);
		          continue;
		        }

				// if child is queued, this node will be processed in the later
				// iteration
				if (hasChildNode(node,queuedNodes)) {

					LOG.trace(indent + "Queueing node "
								+ node.toString() + " (code 2)");

					queuedNodes.add(node);

					// if node has more than two inputs,
					// remove children that will be needed in a future
					// iterations
					// may also have to remove parent nodes of these children
					removeNodesForNextIteration(node, finishedNodes, execNodes,
							queuedNodes, jobNodes);

					continue;
				}
				
				// if inputs come from different jobs, then queue

				if (node.getInputs().size() == 2) {
					int j1 = jobType(node.getInputs().get(0), jobNodes);
					int j2 = jobType(node.getInputs().get(1), jobNodes);
					if (j1 != -1 && j2 != -1 && j1 != j2) {
						LOG.trace(indent + "Queueing node "
									+ node.toString() + " (code 3)");

						queuedNodes.add(node);

						removeNodesForNextIteration(node, finishedNodes,
								execNodes, queuedNodes, jobNodes);

						continue;
					}
				}

				// See if this lop can be eliminated
				// This check is for "aligner" lops (e.g., group)
				boolean eliminate = false;
				eliminate = canEliminateLop(node, execNodes);
				if (eliminate) {
					LOG.trace(indent + "Adding -"
								+ node.toString());
					execNodes.add(node);
					finishedNodes.add(node);
					addNodeByJobType(node, jobNodes, execNodes, eliminate);
					continue;
				}

				// If the node defines a MR Job then make sure none of its
				// children that defines a MR Job are present in execNodes
				if (node.definesMRJob()) {
					if (hasMRJobChildNode(node, execNodes)) {
						// "node" must NOT be queued when node=group and the child that defines job is Rand
						// this is because "group" can be pushed into the "Rand" job.
						if (! (node.getType() == Lop.Type.Grouping && getMRJobChildNode(node,execNodes).getType() == Lop.Type.DataGen) ) {
							LOG.trace(indent + "Queueing node "
										+ node.toString() + " (code 4)");

							queuedNodes.add(node);

							removeNodesForNextIteration(node, finishedNodes,
									execNodes, queuedNodes, jobNodes);

							continue;
						}
					}
				}

				// if "node" has more than one input, and has a descendant lop
				// in execNodes that is of type RecordReader
				// then all its inputs must be ancestors of RecordReader. If
				// not, queue "node"
				if (node.getInputs().size() > 1
						&& hasChildNode(node, execNodes, ExecLocation.RecordReader)) {
					// get the actual RecordReader lop
					N rr_node = getChildNode(node, execNodes, ExecLocation.RecordReader);

					// all inputs of "node" must be ancestors of rr_node
					boolean queue_it = false;
					for (int in = 0; in < node.getInputs().size(); in++) {
						// each input should be ancestor of RecordReader lop
						N n = (N) node.getInputs().get(in);
						if (!n.equals(rr_node) && !isChild(rr_node, n, IDMap)) {
							queue_it = true; // i.e., "node" must be queued
							break;
						}
					}

					if (queue_it) {
						// queue node
						LOG.trace(indent + "Queueing -"
									+ node.toString() + " (code 5)");
						queuedNodes.add(node);
						// TODO: does this have to be modified to handle
						// recordreader lops?
						removeNodesForNextIteration(node, finishedNodes,
								execNodes, queuedNodes, jobNodes);
						continue;
					} else {
						// nothing here.. subsequent checks have to be performed
						// on "node"
						;
					}
				}

				// data node, always add if child not queued
				// only write nodes are kept in execnodes
				if (node.getExecLocation() == ExecLocation.Data) {
					LOG.trace(indent + "Adding Data -"
								+ node.toString());

					finishedNodes.add(node);

					Data dnode = (Data) node;
					if ( dnode.getOperationType() == OperationTypes.READ ) {
						// TODO: avoid readScalar instruction, and read it on-demand just like the way Matrices are read in control program
						if ( node.get_dataType() == DataType.SCALAR 
								//TODO: LEO check the following condition is still needed
								&& node.getOutputParameters().getFile_name() != null ) {
							// this lop corresponds to reading a scalar from HDFS file
							// add it to execNodes so that "readScalar" instruction gets generated
							execNodes.add(node);
							// note: no need to add it to any job vector
						}
					}
					else if (dnode.getOperationType() == OperationTypes.WRITE) {
						// Skip the transient write <code>node</code> if the input is a 
						// transient read with the same variable name. i.e., a dummy copy. 
						// Hence, <code>node</code> can be avoided.
						// TODO: this case should ideally be handled in the language layer 
						//       prior to the construction of Hops Dag 
						if ( dnode.isTransient() 
								&& dnode.getInputs().get(0).getExecLocation() == ExecLocation.Data 
								&& ((Data)dnode.getInputs().get(0)).isTransient() 
								&& dnode.getOutputParameters().getLabel().compareTo(
										dnode.getInputs().get(0).getOutputParameters().getLabel()) == 0 ) {
							// do nothing, <code>node</code> must not processed any further.
							;
						}
						else {
							execNodes.add(node);
							if ( sendWriteLopToMR(node) ) {
								addNodeByJobType(node, jobNodes, execNodes, false);
							}
						}
					}
					continue;
				}

				// map or reduce node, can always be piggybacked with parent
				if (node.getExecLocation() == ExecLocation.MapOrReduce) {
					LOG.trace(indent + "Adding -"
								+ node.toString());
					execNodes.add(node);
					finishedNodes.add(node);
					addNodeByJobType(node, jobNodes, execNodes, false);

					continue;
				}

				// RecordReader node, add, if no parent needs reduce, else queue
				if (node.getExecLocation() == ExecLocation.RecordReader) {
					// "node" should not have any children in
					// execNodes .. it has to be the first one in the job!
					if (!hasChildNode(node, execNodes, ExecLocation.Map)
							&& !hasChildNode(node, execNodes,
									ExecLocation.MapAndReduce)) {
						LOG.trace(indent + "Adding -"
									+ node.toString());
						execNodes.add(node);
						finishedNodes.add(node);
						addNodeByJobType(node, jobNodes, execNodes, false);
					} else {
						LOG.trace(indent + "Queueing -"
									+ node.toString() + " (code 6)");
						queuedNodes.add(node);
						removeNodesForNextIteration(node, finishedNodes,
								execNodes, queuedNodes, jobNodes);

					}
					continue;
				}

				// map node, add, if no parent needs reduce, else queue
				if (node.getExecLocation() == ExecLocation.Map) {
					boolean queueThisNode = false;
					if ( node.usesDistributedCache() ) {
						// if an input to <code>node</code> must come from distributed cache
						// then that input must get executed in one of previous jobs.
						int dcInputIndex = node.distributedCacheInputIndex();
						N dcInput = (N) node.getInputs().get(dcInputIndex-1);
						if ( dcInput.getType() != Lop.Type.Data &&  execNodes.contains(dcInput) )
							queueThisNode = true;
					}
					if (!queueThisNode && !hasChildNode(node, execNodes,ExecLocation.MapAndReduce)&& !hasMRJobChildNode(node, execNodes)) {
						LOG.trace(indent + "Adding -"
									+ node.toString());
						execNodes.add(node);
						finishedNodes.add(node);
						addNodeByJobType(node, jobNodes, execNodes, false);
					} else {
						LOG.trace(indent + "Queueing -"
									+ node.toString() + " (code 7)");
						queuedNodes.add(node);
						removeNodesForNextIteration(node, finishedNodes,
								execNodes, queuedNodes, jobNodes);

					}
					continue;
				}

				// reduce node, make sure no parent needs reduce, else queue
				if (node.getExecLocation() == ExecLocation.MapAndReduce) {

					// boolean eliminate = false;
					// eliminate = canEliminateLop(node, execNodes);
					// if (eliminate || (!hasChildNode(node, execNodes,
					// ExecLocation.MapAndReduce)) &&
					// !hasMRJobChildNode(node,execNodes)) {

					// TODO: statiko -- keep the middle condition
					// discuss about having a lop that is MapAndReduce but does
					// not define a job
					LOG.trace(indent + "Adding -"
								+ node.toString());
					execNodes.add(node);
					finishedNodes.add(node);
					addNodeByJobType(node, jobNodes, execNodes, eliminate);

					// } else {
					// if (DEBUG)
					// System.out.println("Queueing -" + node.toString());
					// queuedNodes.add(node);
					// removeNodesForNextIteration(node, finishedNodes,
					// execNodes, queuedNodes, jobNodes);
					// }
					continue;
				}

				// aligned reduce, make sure a parent that is reduce exists
				if (node.getExecLocation() == ExecLocation.Reduce) {
					if (   hasChildNode(node, execNodes, ExecLocation.MapAndReduce)
						|| hasChildNode(node, execNodes, ExecLocation.Map)) 
					{ 
						LOG.trace(indent + "Adding -"
									+ node.toString());
						execNodes.add(node);
						finishedNodes.add(node);
						addNodeByJobType(node, jobNodes, execNodes, false);
					} else {
						LOG.trace(indent + "Queueing -"
									+ node.toString() + " (code 8)");
						queuedNodes.add(node);
						removeNodesForNextIteration(node, finishedNodes,
								execNodes, queuedNodes, jobNodes);
					}

					continue;

				}

				// add Scalar to execNodes if it has no child in exec nodes
				// that will be executed in a MR job.
				if (node.getExecLocation() == ExecLocation.ControlProgram) {
					for (int j = 0; j < node.getInputs().size(); j++) {
						if (execNodes.contains(node.getInputs().get(j))
								&& !(node.getInputs().get(j).getExecLocation() == ExecLocation.Data)
								&& !(node.getInputs().get(j).getExecLocation() == ExecLocation.ControlProgram)) {
							LOG.trace(indent + "Queueing -"
										+ node.toString() + " (code 9)");

							queuedNodes.add(node);
							removeNodesForNextIteration(node, finishedNodes,
									execNodes, queuedNodes, jobNodes);
							break;
						}
					}

					if (queuedNodes.contains(node))
						continue;

					LOG.trace(indent + "Adding - scalar"
								+ node.toString());
					execNodes.add(node);
					addNodeByJobType(node, jobNodes, execNodes, false);
					finishedNodes.add(node);
					continue;
				}

			}

			// no work to do
			if (execNodes.size() == 0) {
			  
			  if(queuedNodes.size() != 0)
			  {
			    if (LOG.isDebugEnabled())
			    {
			      //System.err.println("Queued nodes should be 0");
			      throw new LopsException("Queued nodes should not be 0 at this point \n");
			    }
			  }
			  
				LOG.trace("All done! queuedNodes = "
							+ queuedNodes.size());
				done = true;
			} else {
				// work to do

				LOG.trace("Generating jobs for group -- Node count="
									+ execNodes.size());

				// first process scalar instructions
				generateControlProgramJobs(execNodes, inst, writeInst, deleteInst);

				// next generate MR instructions
				if (execNodes.size() > 0)
					generateMRJobs(execNodes, inst, deleteInst,
							jobNodes);

				handleSingleOutputJobs(execNodes, jobNodes, finishedNodes);

			}

		}

		// add write and delete inst at the very end.

		//inst.addAll(preWriteDeleteInst);
		inst.addAll(writeInst);
		inst.addAll(deleteInst);
		inst.addAll(endOfBlockInst);

		return inst;

	}

	private boolean compatibleWithChildrenInExecNodes(Vector<N> execNodes, N node) {
	  for(int i=0; i < execNodes.size(); i++)
	  {
	    N tmpNode = execNodes.elementAt(i);
	    // for lops that execute in control program, compatibleJobs property is set to LopProperties.INVALID
	    // we should not consider such lops in this check
	    if (isChild(tmpNode, node, IDMap) 
	    		&& tmpNode.getExecLocation() != ExecLocation.ControlProgram
	    		//&& tmpNode.getCompatibleJobs() != LopProperties.INVALID 
	    		&& (tmpNode.getCompatibleJobs() & node.getCompatibleJobs()) == 0)
	      return false;
	  }
	  return true;
	}

	/**
	 * Exclude rmvar instruction for <varname> from deleteInst, if exists
	 * 
	 * @param varName
	 * @param deleteInst
	 */
	private void excludeRemoveInstruction(String varName, ArrayList<Instruction> deleteInst) {
		//for(Instruction inst : deleteInst) {
		for(int i=0; i < deleteInst.size(); i++) {
			Instruction inst = deleteInst.get(i);
			if (inst.getType() == INSTRUCTION_TYPE.CONTROL_PROGRAM 
					&& ((CPInstruction)inst).getCPInstructionType() == CPINSTRUCTION_TYPE.Variable 
					&& ((VariableCPInstruction)inst).isRemoveVariable(varName) ) {
				deleteInst.remove(i);
			}
		}
	}
	
	/**
	 * Generate rmvar instructions for the inputs, if their consumer count becomes zero.
	 * 
	 * @param node
	 * @param inst
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private void processConsumersForInputs(N node, ArrayList<Instruction> inst, ArrayList<Instruction> delteInst) throws DMLRuntimeException, DMLUnsupportedOperationException {
		// reduce the consumer count for all input lops
		// if the count becomes zero, then then variable associated w/ input can be removed
		for(Lop in : node.getInputs() ) {
			processConsumers((N)in, inst, delteInst);
			/*if ( in.removeConsumer() == 0 ) {
				String label = in.getOutputParameters().getLabel();
				inst.add(VariableCPInstruction.prepareRemoveInstruction(label));
			}*/
		}
	}
	
	private void processConsumers(N node, ArrayList<Instruction> inst, ArrayList<Instruction> deleteInst) throws DMLRuntimeException, DMLUnsupportedOperationException {
		// reduce the consumer count for all input lops
		// if the count becomes zero, then then variable associated w/ input can be removed
		if ( node.removeConsumer() == 0 ) {
			if ( node.getExecLocation() == ExecLocation.Data && ((Data)node).isLiteral() ) {
				return;
			}
			
			String label = node.getOutputParameters().getLabel();
			inst.add(VariableCPInstruction.prepareRemoveInstruction(label));
			excludeRemoveInstruction(label, deleteInst);
		}
	}
	
	/**
	 * Method to generate instructions that are executed in Control Program. At
	 * this point, this DAG has no dependencies on the MR dag. ie. none of the
	 * inputs are outputs of MR jobs
	 * 
	 * @param execNodes
	 * @param inst
	 * @param deleteInst
	 * @throws LopsException
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */

	@SuppressWarnings("unchecked")
	private void generateControlProgramJobs(Vector<N> execNodes,
			ArrayList<Instruction> inst, ArrayList<Instruction> writeInst, ArrayList<Instruction> deleteInst) throws LopsException,
			DMLUnsupportedOperationException, DMLRuntimeException {

		// nodes to be deleted from execnodes
		Vector<N> markedNodes = new Vector<N>();

		// variable names to be deleted
		Vector<String> var_deletions = new Vector<String>();
		
		boolean doRmVar = false;

		for (int i = 0; i < execNodes.size(); i++) {
			N node = execNodes.get(i);
			doRmVar = false;

			// mark input scalar read nodes for deletion
			// TODO: statiko -- check if this condition ever evaluated to TRUE
			if (node.getExecLocation() == ExecLocation.Data
					&& ((Data) node).getOperationType() == Data.OperationTypes.READ
					&& ((Data) node).get_dataType() == DataType.SCALAR 
					&& node.getOutputParameters().getFile_name() == null ) {
				markedNodes.add(node);
				continue;
			}
			
			// output scalar instructions and mark nodes for deletion
			if (node.getExecLocation() == ExecLocation.ControlProgram) {

				if (node.get_dataType() == DataType.SCALAR) {
					// Output from lops with SCALAR data type must
					// go into Temporary Variables (Var0, Var1, etc.)
					NodeOutput out = setupNodeOutputs(node, ExecType.CP, false);
					inst.addAll(out.getPreInstructions()); // dummy
					deleteInst.addAll(out.getLastInstructions());
				} else {
					// Output from lops with non-SCALAR data type must
					// go into Temporary Files (temp0, temp1, etc.)
					
					NodeOutput out = setupNodeOutputs(node, ExecType.CP, false);
					inst.addAll(out.getPreInstructions());
					
					boolean hasTransientWriteParent = false;
					for ( int pid=0; pid < node.getOutputs().size(); pid++ ) {
						N parent = (N)node.getOutputs().get(pid); 
						if ( parent.getExecLocation() == ExecLocation.Data 
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
						//System.out.println("    --> skipping " + out.getLastInstructions() + " while processing node " + node.getID());
					}
				}

				String inst_string = "";

				// Lops with arbitrary number of inputs (ParameterizedBuiltin, GroupedAggregate, DataGen)
				// are handled separately, by simply passing ONLY the output variable to getInstructions()
				if (node.getType() == Lop.Type.ParameterizedBuiltin
						|| node.getType() == Lop.Type.GroupedAgg 
						|| node.getType() == Lop.Type.DataGen ){ 
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
					{
						outputs[count++] = out.getOutputParameters().getLabel();
					}
					
					inst_string = node.getInstructions(inputs, outputs);
				}
				else {
					if ( node.getInputs().size() == 0 ) {
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
					else if (node.getInputs().size() == 3) {
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
						throw new LopsException(node.printErrorLocation() + "Node with " + node.getInputs().size() + " inputs is not supported in CP yet! \n");
					}
				}
				
				try {
					LOG.trace("Generating simple instruction - "+ inst_string);
					inst.add(CPInstructionParser.parseSingleInstruction(inst_string));
				} catch (Exception e) {
					throw new LopsException(node.printErrorLocation() + "Problem generating simple inst - "
							+ inst_string, e);
				}

				markedNodes.add(node);
				doRmVar = true;
				//continue;
			}
			else if (node.getExecLocation() == ExecLocation.Data ) {
				Data dnode = (Data)node;
				Data.OperationTypes op = dnode.getOperationType();
				
				if ( op == Data.OperationTypes.WRITE ) {
					NodeOutput out = null;
					if ( sendWriteLopToMR(node) ) {
						// In this case, Data WRITE lop goes into MR, and 
						// we don't have to do anything here
						doRmVar = false;
					}
					else {
						out = setupNodeOutputs(node, ExecType.CP, false);
						if ( dnode.get_dataType() == DataType.SCALAR ) {
							// processing is same for both transient and persistent scalar writes 
							writeInst.addAll(out.getLastInstructions());
							//inst.addAll(out.getLastInstructions());
							doRmVar = false;
						}
						else {
							// setupNodeOutputs() handles both transient and persistent matrix writes 
							if ( dnode.isTransient() ) {
								//inst.addAll(out.getPreInstructions()); // dummy ?
								deleteInst.addAll(out.getLastInstructions());
								doRmVar = false;
							}
							else {
								// In case of persistent write lop, write() instruction will be generated 
								// and that instruction must be added to <code>inst</code> so that it gets
								// executed immediately. If it is added to <code>deleteInst</code> then it
								// gets executed at the end of program block's execution
								inst.addAll(out.getLastInstructions());
								doRmVar = true;
							}
						}
						markedNodes.add(node);
						//continue;
					}
				}
				else {
					// generate a temp label to hold the value that is read from HDFS
					if ( node.get_dataType() == DataType.SCALAR ) {
						node.getOutputParameters().setLabel(Lop.SCALAR_VAR_NAME_PREFIX + var_index.getNextID());
						String io_inst = node.getInstructions(node.getOutputParameters().getLabel(), 
								node.getOutputParameters().getFile_name());
						inst.add(CPInstructionParser.parseSingleInstruction(io_inst));
						
						deleteInst.add(VariableCPInstruction.prepareRemoveInstruction(node.getOutputParameters().getLabel()));
					}
					else {
						throw new LopsException("Matrix READs are not handled in CP yet!");
					}
					markedNodes.add(node);
					doRmVar = true;
					//continue;
				}
			}
			
			// see if rmvar instructions can be generated for node's inputs
			if(doRmVar)
				processConsumersForInputs(node, inst, deleteInst);
			doRmVar = false;
		}
		
		for ( String var : var_deletions ) {
			Instruction rmInst = VariableCPInstruction.prepareRemoveInstruction(var);
			LOG.trace("  Adding var_deletions: " + rmInst.toString());
			deleteInst.add(rmInst);
		}

		// delete all marked nodes
		for (int i = 0; i < markedNodes.size(); i++) {
			execNodes.remove(markedNodes.elementAt(i));
		}

	}

	/**
	 * Method to remove all child nodes of a queued node that should be executed
	 * in a following iteration.
	 * 
	 * @param node
	 * @param finishedNodes
	 * @param execNodes
	 * @param queuedNodes
	 * @throws LopsException
	 */

	private void removeNodesForNextIteration(N node, Vector<N> finishedNodes,
			Vector<N> execNodes, Vector<N> queuedNodes,
			ArrayList<Vector<N>> jobvec) throws LopsException {
		// only queued nodes with two inputs need to be handled.

		// TODO: statiko -- this should be made == 1
		if (node.getInputs().size() != 2)
			return;
		
		
		//if both children are queued, nothing to do. 
	    if (queuedNodes.contains(node.getInputs().get(0))
	        && queuedNodes.contains(node.getInputs().get(1)))
	      return;
	    
	    LOG.trace("Before remove nodes for next iteration -- size of execNodes " + execNodes.size());
 
	    

		boolean inputs_in_same_job = false;
		// TODO: handle tertiary
		if (jobType(node.getInputs().get(0), jobvec) != -1
				&& jobType(node.getInputs().get(0), jobvec) == jobType(node
						.getInputs().get(1), jobvec)) {
			inputs_in_same_job = true;
		}

		boolean unassigned_inputs = false;
		// here.. scalars shd be ignored
		if ((node.getInputs().get(0).getExecLocation() != ExecLocation.ControlProgram && jobType(
				node.getInputs().get(0), jobvec) == -1)
				|| (node.getInputs().get(1).getExecLocation() != ExecLocation.ControlProgram && jobType(
						node.getInputs().get(1), jobvec) == -1)) {
			unassigned_inputs = true;
		}

		boolean child_queued = false;

		// check if atleast one child was queued.
		if (queuedNodes.contains(node.getInputs().get(0))
				|| queuedNodes.contains(node.getInputs().get(1)))
			child_queued = true;

		// nodes to be dropped
		Vector<N> markedNodes = new Vector<N>();

		for (int i = 0; i < execNodes.size(); i++) {

			N tmpNode = execNodes.elementAt(i);

			if (LOG.isTraceEnabled()) {
				LOG.trace("Checking for removal (" + tmpNode.getID()
						+ ") " + tmpNode.toString());
				LOG.trace("Inputs in same job " + inputs_in_same_job);
				LOG.trace("Unassigned inputs " + unassigned_inputs);
				LOG.trace("Child queued " + child_queued);

			}
			
			//if(node.definesMRJob() && isChild(tmpNode,node) && (tmpNode.getCompatibleJobs() & node.getCompatibleJobs()) == 0)
			//  continue;

			// TODO: statiko -- check if this is too conservative?
			if (child_queued) {
  	     // if one of the children are queued, 
	       // remove some child nodes on other leg that may be needed later on. 
	       // For e.g. Group lop. 
 
			  if((tmpNode == node.getInputs().get(0) || tmpNode == node.getInputs().get(1)) && 
			      tmpNode.isAligner())
			  {
			    markedNodes.add(tmpNode);
			    LOG.trace("Removing for next iteration: ("
                + tmpNode.getID() + ") " + tmpNode.toString());
			  }
			  else {
				if (!hasOtherQueuedParentNode(tmpNode, queuedNodes, node) 
						&& isChild(tmpNode, node, IDMap)  && branchHasNoOtherUnExecutedParents(tmpNode, node, execNodes, finishedNodes)) {

	
				    if( 
				        //e.g. MMCJ
				        (node.getExecLocation() == ExecLocation.MapAndReduce &&
						branchCanBePiggyBackedMapAndReduce(tmpNode, node, execNodes, finishedNodes) && !tmpNode.definesMRJob() )
						||
						//e.g. Binary
						(node.getExecLocation() == ExecLocation.Reduce && branchCanBePiggyBackedReduce(tmpNode, node, execNodes, finishedNodes))  )
						{
					
				      LOG.trace("Removing for next iteration: ("
								    + tmpNode.getID() + ") " + tmpNode.toString());
					
					
	        				markedNodes.add(tmpNode);
						
						}
				 }
			  }
			} else {
				/*
				 * "node" has no other queued children.
				 * 
				 * If inputs are in the same job and "node" is of type
				 * MapAndReduce, then remove nodes of all types other than
				 * Reduce, MapAndReduce, and the ones that define a MR job as
				 * they can be piggybacked later.
				 * 
				 * e.g: A=Rand, B=Rand, C=A%*%B Here, both inputs of MMCJ lop
				 * come from Rand job, and they should not be removed.
				 * 
				 * Other examples: -- MMCJ whose children are of type
				 * MapAndReduce (say GMR) -- Inputs coming from two different
				 * jobs .. GMR & REBLOCK
				 */
				if ((inputs_in_same_job || unassigned_inputs)
						&& node.getExecLocation() == ExecLocation.MapAndReduce
						&& !hasOtherMapAndReduceParentNode(tmpNode, execNodes,
								node)
						&& isChild(tmpNode, node, IDMap) &&
						branchCanBePiggyBackedMapAndReduce(tmpNode, node, execNodes, finishedNodes)
						&& tmpNode.definesMRJob() != true) {
					LOG.trace("Removing for next iteration:: ("
								+ tmpNode.getID() + ") " + tmpNode.toString());

					markedNodes.add(tmpNode);
				}

				// as this node has inputs coming from different jobs, need to
				// free up everything
				// below and include the closest MapAndReduce lop if this is of
				// type Reduce.
				// if it is of type MapAndReduce, don't need to free any nodes

				if (!inputs_in_same_job && !unassigned_inputs
						&& isChild(tmpNode, node, IDMap) && 
						(tmpNode == node.getInputs().get(0) || tmpNode == node.getInputs().get(1)) && 
		            tmpNode.isAligner()) 
				{
					LOG.trace("Removing for next iteration ("
										+ tmpNode.getID()
										+ ") "
										+ tmpNode.toString());

					markedNodes.add(tmpNode);

				}

			}
		}

		// we also need to delete all parent nodes of marked nodes
		for (int i = 0; i < execNodes.size(); i++) {
			LOG.trace("Checking for removal - ("
						+ execNodes.elementAt(i).getID() + ") "
						+ execNodes.elementAt(i).toString());

			if (hasChildNode(execNodes.elementAt(i), markedNodes)) {
				markedNodes.add(execNodes.elementAt(i));
				LOG.trace("Removing for next iteration - ("
							+ execNodes.elementAt(i).getID() + ") "
							+ execNodes.elementAt(i).toString());
			}
		}

		// delete marked nodes from finishedNodes and execNodes
		// add to queued nodes
		for (int i = 0; i < markedNodes.size(); i++) {
			finishedNodes.remove(markedNodes.elementAt(i));
			execNodes.remove(markedNodes.elementAt(i));
			removeNodeByJobType(markedNodes.elementAt(i), jobvec);
			queuedNodes.add(markedNodes.elementAt(i));
		}
	}

	private boolean branchCanBePiggyBackedReduce(N tmpNode, N node,
      Vector<N> execNodes, Vector<N> finishedNodes) {
	   for(int i=0; i < execNodes.size(); i++)
     {
       N n = execNodes.elementAt(i);
       
       if(n.equals(node))
         continue;
       
       
       
       if(n.equals(tmpNode) && n.getExecLocation() != ExecLocation.Map && n.getExecLocation() != ExecLocation.MapOrReduce)
         return false;
       
       if(isChild(n, node, IDMap) && isChild(tmpNode, n, IDMap) && !node.getInputs().contains(tmpNode) && n.getExecLocation() != ExecLocation.Map && n.getExecLocation() != ExecLocation.MapOrReduce)
         return false;
         
      }
     
     
     return true;

  }

	/**
	 * Function that checks if <code>tmpNode</code> can be piggybacked with MapAndReduce 
	 * lop <code>node</code>. 
	 * 
	 * Decision depends on the exec location of <code>tmpNode</code>. If the exec location is: 
	 * MapAndReduce: CAN NOT be piggybacked since it defines its own MR job
	 * Reduce: CAN NOT be piggybacked since it must execute before <code>node</code>
	 * Map or MapOrReduce: CAN be piggybacked ONLY IF it is comatible w/ <code>tmpNode</code> 
	 * 
	 * @param tmpNode
	 * @param node
	 * @param execNodes
	 * @param finishedNodes
	 * @return
	 */
  private boolean branchCanBePiggyBackedMapAndReduce(N tmpNode, N node,
      Vector<N> execNodes, Vector<N> finishedNodes) {
	   
	  if ( node.getExecLocation() != ExecLocation.MapAndReduce ) 
		  return false;
	  JobType jt = JobType.findJobTypeFromLop(node);
	  
	  for(int i=0; i < execNodes.size(); i++)
	    {
	      N n = execNodes.elementAt(i);
	      
	      if(n.equals(node))
	        continue;
	      
	      boolean eval = false;
	      if ( n.equals(tmpNode) || (isChild(n, node, IDMap) && isChild(tmpNode, n, IDMap)) )
	    	  eval = true;
	      
	      if ( eval ) {
	    	  ExecLocation el = n.getExecLocation();
	    	  if ( el != ExecLocation.Map && el != ExecLocation.MapOrReduce )
	    		  return false;
	    	  else if ( !isCompatible(n, jt) )
	    		  return false;
	      }
	      
	      /*if(n.equals(tmpNode) && n.getExecLocation() != ExecLocation.Map && n.getExecLocation() != ExecLocation.MapOrReduce)
	        return false;
	      
	      if(isChild(n, node, IDMap) && isChild(tmpNode, n, IDMap) && n.getExecLocation() != ExecLocation.Map && n.getExecLocation() != ExecLocation.MapOrReduce)
	        return false;*/
	        
	     }
	    return true;
  }

  private boolean branchHasNoOtherUnExecutedParents(N tmpNode, N node,
      Vector<N> execNodes, Vector <N> finishedNodes) {

	  //if tmpNode has more than one unfinished output, return false 
	  if(tmpNode.getOutputs().size() > 1)
	  {
	    int cnt = 0;
	    for (int j = 0; j < tmpNode.getOutputs().size(); j++) {
        if (!finishedNodes.contains(tmpNode.getOutputs().get(j)))
          cnt++;
      } 
	    
	    if(cnt != 1)
	      return false;
	  }
	  
	  //check to see if any node between node and tmpNode has more than one unfinished output 
	  for(int i=0; i < execNodes.size(); i++)
	  {
	    N n = execNodes.elementAt(i);
	    
	    if(n.equals(node) || n.equals(tmpNode))
	      continue;
	    
	    if(isChild(n, node, IDMap) && isChild(tmpNode, n, IDMap))
	    {      
	      int cnt = 0;
	      for (int j = 0; j < n.getOutputs().size(); j++) {
	        if (!finishedNodes.contains(n.getOutputs().get(j)))    
	          cnt++;
	      } 
	      
	      if(cnt != 1)
	        return false;
	    }
	  }
	  
	  return true;
  }

  /**
	 * Method to check of there is a node of type MapAndReduce between a child
	 * (tmpNode) and its parent (node)
	 * 
	 * @param tmpNode
	 * @param execNodes
	 * @param node
	 * @return
	 */

	private boolean hasMapAndReduceParentNode(N tmpNode, Vector<N> execNodes,
			N node) {
		for (int i = 0; i < tmpNode.getOutputs().size(); i++) {
			N n = (N) tmpNode.getOutputs().get(i);

			if (execNodes.contains(n)
					&& n.getExecLocation() == ExecLocation.MapAndReduce
					&& isChild(n, node, IDMap)) {
				return true;
			} else {
				if (hasMapAndReduceParentNode(n, execNodes, node))
					return true;
			}

		}

		return false;
	}

	/**
	 * Method to return the job index for a lop.
	 * 
	 * @param lops
	 * @param jobvec
	 * @return
	 * @throws LopsException
	 */

	private int jobType(Lop lops, ArrayList<Vector<N>> jobvec) throws LopsException {
		for ( JobType jt : JobType.values()) {
			int i = jt.getId();
			if (i > 0 && jobvec.get(i) != null && jobvec.get(i).contains(lops)) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * Method to see if there is a node of type Reduce between tmpNode and node
	 * in execNodes
	 * 
	 * @param tmpNode
	 * @param execNodes
	 * @param node
	 * @return
	 */

	@SuppressWarnings("unchecked")
	private boolean hasOtherMapAndReduceParentNode(N tmpNode,
			Vector<N> execNodes, N node) {
		for (int i = 0; i < tmpNode.getOutputs().size(); i++) {
			N n = (N) tmpNode.getOutputs().get(i);

			if (execNodes.contains(n) && !n.equals(node)
					&& n.getExecLocation() == ExecLocation.MapAndReduce
					&& isChild(n, node, IDMap)) {
				return true;
			} else {
				if (hasOtherMapAndReduceParentNode(n, execNodes, node))
					return true;
			}

		}

		return false;
	}

	/**
	 * Method to check if there is a queued node that is a parent of both tmpNode and node
	 * 
	 * @param tmpNode
	 * @param queuedNodes
	 * @param node
	 * @return
	 */

	@SuppressWarnings("unchecked")
	private boolean hasOtherQueuedParentNode(N tmpNode, Vector<N> queuedNodes, N node) {
		if ( queuedNodes.size() == 0 )
			return false;
		
		boolean[] nodeMarked = node.get_reachable();
		boolean[] tmpMarked  = tmpNode.get_reachable();
		
		for ( int i=0; i < queuedNodes.size(); i++ ) {
			int id = IDMap.get(queuedNodes.get(i).getID());
			if (nodeMarked[id] && tmpMarked[id])
				return true;
		}
		
		return false;
	}

	@SuppressWarnings("unchecked")
	private boolean hasOtherQueuedParentNodeOLD(N tmpNode, Vector<N> queuedNodes,
			N node) {
		for (int i = 0; i < tmpNode.getOutputs().size(); i++) {
			N n = (N) tmpNode.getOutputs().get(i);
			
			// if node is a child of n then n's level must be greater than node
			if(n.getLevel() <= node.getLevel() )
				continue;
			
			if (queuedNodes.contains(n) && !n.equals(node) && isChild(n, node, IDMap)) {
				return true;
			} else {
				if (hasOtherQueuedParentNode(n, queuedNodes, node))
					return true;
			}
		}

		return false;
	}

	/**
	 * Method to print the lops grouped by job type
	 * 
	 * @param jobNodes
	 * @throws DMLRuntimeException
	 */
	void printJobNodes(ArrayList<Vector<N>> jobNodes)
			throws DMLRuntimeException {
		if (LOG.isTraceEnabled()){
			for ( JobType jt : JobType.values() ) {
				int i = jt.getId();
				if (i > 0 && jobNodes.get(i) != null && jobNodes.get(i).size() > 0) {
					LOG.trace(jt.getName() + " Job Nodes:");
					for (int j = 0; j < jobNodes.get(i).size(); j++) {
						LOG.trace("    "
								+ jobNodes.get(i).elementAt(j).getID() + ") "
								+ jobNodes.get(i).elementAt(j).toString());
					}
				}
			}
			
		}

		
	}

	/**
	 * Method to check if there exists any lops with ExecLocation=RecordReader
	 * 
	 * @param nodes
	 * @param loc
	 * @return
	 */
	boolean hasANode(Vector<N> nodes, ExecLocation loc) {
		for (int i = 0; i < nodes.size(); i++) {
			if (nodes.get(i).getExecLocation() == ExecLocation.RecordReader)
				return true;
		}
		return false;
	}

	ArrayList<Vector<N>> splitGMRNodesByRecordReader(Vector<N> gmrnodes) {

		// obtain the list of record reader nodes
		ArrayList<N> rrnodes = new ArrayList<N>();
		for (int i = 0; i < gmrnodes.size(); i++) {
			if (gmrnodes.get(i).getExecLocation() == ExecLocation.RecordReader)
				rrnodes.add(gmrnodes.get(i));
		}

		/*
		 * We allocate one extra vector to hold lops that do not depend on any
		 * recordreader lops
		 */
		ArrayList<Vector<N>> splitGMR = createNodeVectors(rrnodes.size() + 1);

		// flags to indicate whether a lop has been added to one of the node
		// vectors
		boolean[] flags = new boolean[gmrnodes.size()];
		for (int i = 0; i < gmrnodes.size(); i++) {
			flags[i] = false;
		}

		// first, obtain all ancestors of recordreader lops
		for (int rrid = 0; rrid < rrnodes.size(); rrid++) {
			// prepare node list for i^th record reader lop

			// add record reader lop
			splitGMR.get(rrid).add(rrnodes.get(rrid));

			for (int j = 0; j < gmrnodes.size(); j++) {
				if (rrnodes.get(rrid).equals(gmrnodes.get(j)))
					flags[j] = true;
				else if (isChild(rrnodes.get(rrid), gmrnodes.get(j), IDMap)) {
					splitGMR.get(rrid).add(gmrnodes.get(j));
					flags[j] = true;
				}
			}
		}

		// add all remaining lops to a separate job
		int jobindex = rrnodes.size(); // the last node vector
		for (int i = 0; i < gmrnodes.size(); i++) {
			if (flags[i] == false) {
				splitGMR.get(jobindex).add(gmrnodes.get(i));
				flags[i] = true;
			}
		}

		return splitGMR;
	}

	/**
	 * Method to generate hadoop jobs. Exec nodes can contains a mixture of node
	 * types requiring different mr jobs. This method breaks the job into
	 * sub-types and then invokes the appropriate method to generate
	 * instructions.
	 * 
	 * @param execNodes
	 * @param inst
	 * @param deleteinst
	 * @param jobNodes
	 * @throws LopsException
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */

	public void generateMRJobs(Vector<N> execNodes,
			ArrayList<Instruction> inst,
			ArrayList<Instruction> deleteinst, ArrayList<Vector<N>> jobNodes)
			throws LopsException, DMLUnsupportedOperationException,
			DMLRuntimeException

	{

		// copy unassigned lops in execnodes to gmrnodes
		for (int i = 0; i < execNodes.size(); i++) {
			N node = execNodes.elementAt(i);
			if (jobType(node, jobNodes) == -1) {
				jobNodes.get(JobType.GMR.getId()).add(node);
				addChildren(node, jobNodes.get(JobType.GMR.getId()),
						execNodes);
			}
		}

		printJobNodes(jobNodes);
		
		ArrayList<Instruction> rmvarinst = new ArrayList<Instruction>();
		for (JobType jt : JobType.values()) { 
			
			// do nothing, if jt = INVALID or ANY
			if ( jt == JobType.INVALID || jt == JobType.ANY )
				continue;
			
			// TODO: Following hardcoding of JobType.PARTITION must be removed
			if ( jt == JobType.PARTITION ) {
				if ( jobNodes.get(jt.getId()).size() > 0 )
					generatePartitionJob(jobNodes.get(jt.getId()), inst, deleteinst);
				continue;
			}
			
			int index = jt.getId(); // job id is used as an index into jobNodes
			Vector<N> currNodes = jobNodes.get(index);
			
			// generate MR job
			if (currNodes != null && currNodes.size() > 0) {

				LOG.trace("Generating " + jt.getName() + " job");

				if (jt.allowsRecordReaderInstructions() && hasANode(jobNodes.get(index), ExecLocation.RecordReader)) {
					// split the nodes by recordReader lops
					ArrayList<Vector<N>> rrlist = splitGMRNodesByRecordReader(jobNodes.get(index));
					for (int i = 0; i < rrlist.size(); i++) {
						generateMapReduceInstructions(rrlist.get(i), inst, deleteinst, rmvarinst, jt);
					}
				}
				else if ( jt.allowsSingleShuffleInstruction() ) {
					// These jobs allow a single shuffle instruction. 
					// We should split the nodes so that a separate job is produced for each shuffle instruction.
					Lop.Type splittingLopType = jt.getShuffleLopType();
					
					Vector<N> nodesForASingleJob = new Vector<N>();
					for (int i = 0; i < jobNodes.get(index).size(); i++) {
						if (jobNodes.get(index).elementAt(i).getType() == splittingLopType) {
							nodesForASingleJob.clear();
							
							// Add the lop that defines the split 
							nodesForASingleJob.add(jobNodes.get(index).elementAt(i));
							
							/*
							 * Add the splitting lop's children. This call is redundant when jt=SORT
							 * because a sort job ALWAYS has a SINGLE lop in the entire job
							 * i.e., there are no children to add when jt=SORT. 
							 */
							addChildren(jobNodes.get(index).elementAt(i), nodesForASingleJob, jobNodes.get(index));
							
							if ( jt.isCompatibleWithParentNodes() ) {
								/*
								 * If the splitting lop is compatible with parent nodes 
								 * then they must be added to the job. For example, MMRJ lop 
								 * may have a Data(Write) lop as its parent, which can be 
								 * executed along with MMRJ.
								 */
								addParents(jobNodes.get(index).elementAt(i), nodesForASingleJob, jobNodes.get(index));
							}
							
							generateMapReduceInstructions(nodesForASingleJob, inst, deleteinst, rmvarinst, jt);
						}
					}
				}
				else {
					// the default case
					generateMapReduceInstructions(jobNodes.get(index), inst, deleteinst, rmvarinst, jt);
				}
			}
		}
		inst.addAll(rmvarinst);

	}

	/**
	 * Method to get the input format for a lop
	 * 
	 * @param elementAt
	 * @return
	 * @throws LopsException
	 */

	// This code is replicated in ReBlock.java
	private Format getChildFormat(N node) throws LopsException {

		if (node.getOutputParameters().getFile_name() != null
				|| node.getOutputParameters().getLabel() != null) {
			return node.getOutputParameters().getFormat();
		} else {
			if (node.getInputs().size() > 1)
				throw new LopsException(node.printErrorLocation() + "Should only have one child! \n");
			/*
			 * Return the format of the child node (i.e., input lop) No need of
			 * recursion here.. because 1) Reblock lop's input can either be
			 * DataLop or some intermediate computation If it is Data then we
			 * just take its format (TEXT or BINARY) If it is intermediate lop
			 * then it is always BINARY since we assume that all intermediate
			 * computations will be in Binary format 2) Note that Reblock job
			 * will never have any instructions in the mapper => the input lop
			 * (if it is other than Data) is always executed in a different job
			 */
			return node.getInputs().get(0).getOutputParameters().getFormat();
			// return getChildFormat((N) node.getInputs().get(0));
		}

	}

	/**
	 * Method to add all parents of "node" in exec_n to node_v.
	 * 
	 * @param node
	 * @param node_v
	 * @param exec_n
	 */

	private void addParents(N node, Vector<N> node_v, Vector<N> exec_n) {
		for (int i = 0; i < exec_n.size(); i++) {
			if (isChild(node, exec_n.elementAt(i), IDMap)) {
				if (!node_v.contains(exec_n.elementAt(i))) {
					LOG.trace("Adding parent - "
								+ exec_n.elementAt(i).toString());
					node_v.add(exec_n.elementAt(i));
				}
			}

		}

	}

	/**
	 * Method to add all relevant data nodes for set of exec nodes.
	 * 
	 * @param node
	 * @param node_v
	 * @param exec_n
	 */

	@SuppressWarnings("unchecked")
	private void addChildren(N node, Vector<N> node_v, Vector<N> exec_n) {

		/** add child in exec nodes that is not of type scalar **/
		if (exec_n.contains(node)
				&& node.getExecLocation() != ExecLocation.ControlProgram) {
			if (!node_v.contains(node)) {
				node_v.add(node);
				if (node != null && LOG.isTraceEnabled())
					LOG.trace("      Added child " + node.toString());
			}

		}

		if (!exec_n.contains(node))
			return;

		/**
		 * recurse
		 */
		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);
			addChildren(n, node_v, exec_n);
		}
	}

	private void generatePartitionJob(Vector<N> execNodes,
			ArrayList<Instruction> inst, ArrayList<Instruction> deleteinst)
			throws LopsException, DMLRuntimeException {
		ArrayList<Byte> resultIndices = new ArrayList<Byte>();
		ArrayList<String> inputs = new ArrayList<String>();
		ArrayList<String> outputs = new ArrayList<String>();
		ArrayList<InputInfo> inputInfos = new ArrayList<InputInfo>();
		ArrayList<OutputInfo> outputInfos = new ArrayList<OutputInfo>();
		ArrayList<Long> numRows = new ArrayList<Long>();
		ArrayList<Long> numCols = new ArrayList<Long>();
		ArrayList<Long> numRowsPerBlock = new ArrayList<Long>();
		ArrayList<Long> numColsPerBlock = new ArrayList<Long>();
		ArrayList<String> inputLabels = new ArrayList<String>();
		ArrayList<String> outputLabels = new ArrayList<String>();
		LocalVariableMap outputLabelValueMapping = new LocalVariableMap ();

		ArrayList<Lop> inputLops = new ArrayList<Lop>();
		
		HashMap<N, Integer> nodeIndexMapping = new HashMap<N, Integer>();

		int numReducers = total_reducers;
		int replication = 1;

		Vector<N> rootNodes = new Vector<N>();
		// find all root nodes
		getOutputNodes(execNodes, rootNodes, false);

		if (LOG.isTraceEnabled()) {
			LOG.trace("rootNodes.size() = " + rootNodes.size()
					+ ", execNodes.size() = " + execNodes.size());
			rootNodes.get(0).printMe();
		}

		for (int i = 0; i < rootNodes.size(); i++)
			getInputPathsAndParameters(rootNodes.elementAt(i), execNodes,
					inputs, inputInfos, numRows, numCols, numRowsPerBlock,
					numColsPerBlock, nodeIndexMapping, inputLabels, inputLops);

		Vector<N> partitionNodes = new Vector<N>();
		getPartitionNodes(execNodes, partitionNodes);
		PartitionParams pp = ((PartitionLop) partitionNodes.get(0)).getPartitionParams();

		InputInfo[] inputInfo = new InputInfo[inputs.size()];
		for (int i = 0; i < inputs.size(); i++)
			inputInfo[i] = inputInfos.get(i);

		MRJobInstruction mr = new MRJobInstruction(JobType.PARTITION);

		// update so folds are only reset for submatrix
		if (pp.isEL == false && pp.pt.name().equals("submatrix"))
			pp.numFoldsForSubMatrix(getLongArray(numRows)[0],getLongArray(numCols)[0]);

		outputLabelValueMapping = pp.getOutputLabelValueMapping();
		String[] outputStrings = pp.getOutputStrings();
		
		/*mr.setPartitionInstructions(getStringArray(inputs), inputInfo,
				outputStrings, numReducers, replication, getLongArray(numRows),
				getLongArray(numCols), getIntArray(numRowsPerBlock),
				getIntArray(numColsPerBlock), pp.getResultIndexes(), pp
						.getResultDimsUnknown(), pp, inputLabels, outputLabels,
				outputLabelValueMapping);*/
		inst.add(mr);
	}
	
	/**
	 * Method that determines the output format for a given node.
	 * 
	 * @param node
	 * @return
	 * @throws LopsException 
	 */
	OutputInfo getOutputInfo(N node, boolean cellModeOverride) 
		throws LopsException 
	{
		if ( (node.get_dataType() == DataType.SCALAR && node.getExecType() == ExecType.CP) 
				|| node instanceof FunctionCallCP )
			return null;
	
		OutputInfo oinfo = null;
		OutputParameters oparams = node.getOutputParameters();
		
		if (oparams.isBlocked_representation()) {
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
					oparams.setDimensions(oparams.getNum_rows(), oparams.getNum_cols(), -1, -1, oparams.getNnz());
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

		/* Instead of following hardcoding, one must get this information from Lops */
		if (node.getType() == Type.SortKeys && node.getExecType() == ExecType.MR) {
			oinfo = OutputInfo.OutputInfoForSortOutput; //new OutputInfo(CompactOutputFormat.class,
					//DoubleWritable.class, IntWritable.class);
		} else if (node.getType() == Type.CombineBinary) {
			// Output format of CombineBinary (CB) depends on how the output is consumed
			CombineBinary combine = (CombineBinary) node;
			if ( combine.getOperation() == com.ibm.bi.dml.lops.CombineBinary.OperationTypes.PreSort ) {
				oinfo = OutputInfo.OutputInfoForSortInput; //new OutputInfo(SequenceFileOutputFormat.class,
						//DoubleWritable.class, IntWritable.class);
			}
			else if ( combine.getOperation() == com.ibm.bi.dml.lops.CombineBinary.OperationTypes.PreCentralMoment 
					  || combine.getOperation() == com.ibm.bi.dml.lops.CombineBinary.OperationTypes.PreCovUnweighted 
					  || combine.getOperation() == com.ibm.bi.dml.lops.CombineBinary.OperationTypes.PreGroupedAggUnweighted ) {
				oinfo = OutputInfo.WeightedPairOutputInfo;
			}
		} else if ( node.getType() == Type.CombineTertiary) {
			oinfo = OutputInfo.WeightedPairOutputInfo;
		} else if (node.getType() == Type.CentralMoment
				|| node.getType() == Type.CoVariance )  
		{
			// CMMR always operate in "cell mode",
			// and the output is always in cell format
			oinfo = OutputInfo.BinaryCellOutputInfo;
		}

		return oinfo;
	}

	/** 
	 * Method that determines the output format for a given node, 
	 * and returns a string representation of OutputInfo. This 
	 * method is primarily used while generating instructions that
	 * execute in the control program.
	 * 
	 * @param node
	 * @return
	 */
/*  	String getOutputFormat(N node) {
  		return OutputInfo.outputInfoToString(getOutputInfo(node, false));
  	}
*/  	
	
	public String prepareAssignVarInstruction(Lop input, Lop node) {
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
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 * @throws LopsException 
	 */
	private NodeOutput setupNodeOutputs(N node, ExecType et, boolean cellModeOverride) 
	throws DMLUnsupportedOperationException, DMLRuntimeException, LopsException {
		
		OutputParameters oparams = node.getOutputParameters();
		NodeOutput out = new NodeOutput();
		
		node.setConsumerCount(node.getOutputs().size());
		
		// Compute the output format for this node
		out.setOutInfo(getOutputInfo(node, cellModeOverride));
		
		// If node is NOT of type Data then we must generate
		// a variable to hold the value produced by this node
		// note: functioncallcp requires no createvar, rmvar since
		// since outputs are explicitly specified
		if (node.getExecLocation() != ExecLocation.Data ) 
		{
			if (node.get_dataType() == DataType.SCALAR) {
				oparams.setLabel(Lop.SCALAR_VAR_NAME_PREFIX + var_index.getNextID());
				out.setVarName(oparams.getLabel());
				out.addLastInstruction(VariableCPInstruction.prepareRemoveInstruction(oparams.getLabel()));
			} 
			else if(!(node instanceof FunctionCallCP)) //general case
			{
				// generate temporary filename and a variable name to hold the
				// output produced by "rootNode"
				oparams.setFile_name(scratch + Lop.FILE_SEPARATOR
						+ Lop.PROCESS_PREFIX + DMLScript.getUUID()
						+ Lop.FILE_SEPARATOR + Lop.FILE_SEPARATOR
						+ ProgramConverter.CP_ROOT_THREAD_ID
						+ Lop.FILE_SEPARATOR + "temp" + job_id.getNextID());
				oparams.setLabel(Lop.MATRIX_VAR_NAME_PREFIX + var_index.getNextID());

				// generate an instruction that creates a symbol table entry for the new variable
				//String createInst = prepareVariableInstruction("createvar", node);
				//out.addPreInstruction(CPInstructionParser.parseSingleInstruction(createInst));
				int rpb = Integer.parseInt("" + oparams.get_rows_in_block());
				int cpb = Integer.parseInt("" + oparams.get_cols_in_block());
				Instruction createvarInst = VariableCPInstruction.prepareCreateVariableInstruction(
									        oparams.getLabel(),
											oparams.getFile_name(), 
											true, 
											OutputInfo.outputInfoToString(getOutputInfo(node, false)),
											new MatrixCharacteristics(oparams.getNum_rows(), oparams.getNum_cols(), rpb, cpb, oparams.getNnz())
										);
				out.addPreInstruction(createvarInst);

				// temp file as well as the variable has to be deleted at the end
				out.addLastInstruction(VariableCPInstruction.prepareRemoveInstruction(oparams.getLabel()));

				// finally, add the generated filename and variable name to the list of outputs
				out.setFileName(oparams.getFile_name());
				out.setVarName(oparams.getLabel());
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
								fnOutParams.getLabel(),
								fnOutParams.getLabel(), 
								true, 
								OutputInfo.outputInfoToString(getOutputInfo((N)fnOut, false)),
								new MatrixCharacteristics(fnOutParams.getNum_rows(), fnOutParams.getNum_cols(), fnOutParams.get_rows_in_block().intValue(), fnOutParams.get_cols_in_block().intValue(), fnOutParams.getNnz())
							);
						out.addPreInstruction(createvarInst);
					}
				}
			}
		}
		// rootNode is of type Data
		else {
			
			if ( node.get_dataType() == DataType.SCALAR ) {
				// generate assignment operations for final and transient writes
				if ( oparams.getFile_name() == null ) {
					String io_inst = prepareAssignVarInstruction(node.getInputs().get(0), node);
					out.addLastInstruction(CPInstructionParser.parseSingleInstruction(io_inst));
				}
				else {
					String io_inst = node.getInstructions(node.getInputs().get(0).getOutputParameters().getLabel(), oparams.getFile_name());
					out.addLastInstruction(CPInstructionParser.parseSingleInstruction(io_inst));
				}
			}
			else {
				if ( ((Data)node).isTransient() ) {
					
					if ( et == ExecType.CP ) {
						// If transient matrix write is in CP then its input MUST be executed in CP as well.
						
						// get variable and filename associated with the input
						String inputFileName = node.getInputs().get(0).getOutputParameters().getFile_name();
						String inputVarName = node.getInputs().get(0).getOutputParameters().getLabel();
						
						String constVarName = oparams.getLabel();
						String constFileName = inputFileName + constVarName;
						
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
						out.addLastInstruction(VariableCPInstruction.prepareCopyInstruction(inputVarName, constVarName));
						out.setFileName(constFileName);
					}
					else {
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
						String tempFileName = scratch + Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR + 
								   					    Lop.FILE_SEPARATOR + ProgramConverter.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR +  
								   					    "temp" + job_id.getNextID();
						
						//String createInst = prepareVariableInstruction("createvar", tempVarName, node.get_dataType(), node.get_valueType(), tempFileName, oparams, out.getOutInfo());
						//out.addPreInstruction(CPInstructionParser.parseSingleInstruction(createInst));
						
						int rpb = Integer.parseInt(""+oparams.get_rows_in_block());
						int cpb = Integer.parseInt(""+oparams.get_cols_in_block());
						Instruction createvarInst = VariableCPInstruction.prepareCreateVariableInstruction(
													tempVarName, 
													tempFileName, 
													true, 
													OutputInfo.outputInfoToString(out.getOutInfo()), 
													new MatrixCharacteristics(oparams.getNum_rows(), oparams.getNum_cols(), rpb, cpb, oparams.getNnz())
												);
						out.addPreInstruction(createvarInst);

						
						String constVarName = oparams.getLabel();
						String constFileName = tempFileName + constVarName;
						
						oparams.setFile_name(scratch +  Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR + 
								   					    Lop.FILE_SEPARATOR + ProgramConverter.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR + 
								   					    constFileName);
						
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
						
						// rename the temp variable to constant variable (e.g., cpvar tVarAtemp tVarA)
						out.addLastInstruction(VariableCPInstruction.prepareCopyInstruction(tempVarName, constVarName));
						out.addLastInstruction(VariableCPInstruction.prepareRemoveInstruction(tempVarName));
						
						// finally, add the temporary filename and variable name to the list of outputs 
						out.setFileName(tempFileName);
						out.setVarName(tempVarName);
					}
				} 
				// rootNode is not a transient write. It is a persistent write.
				else {
					if(et == ExecType.MR) {
						// create a variable to hold the result produced by this "rootNode"
						oparams.setLabel("pVar" + var_index.getNextID() );
						
						//String createInst = prepareVariableInstruction("createvar", node);
						//out.addPreInstruction(CPInstructionParser.parseSingleInstruction(createInst));

						int rpb = Integer.parseInt(""+oparams.get_rows_in_block());
						int cpb = Integer.parseInt(""+oparams.get_cols_in_block());
						Instruction createvarInst;
						
						// for MatrixMarket format, the creatvar will output the result to a temporary file in textcell format 
						// the CP write instruction (post instruction) after the MR instruction will merge the result into a single
						// part MM format file on hdfs.
						if (oparams.getFormat() == Format.CSV)  {
							
							String tempFileName = scratch + Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR + 
			   					    Lop.FILE_SEPARATOR + ProgramConverter.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR +  
			   					    "temp" + job_id.getNextID();
							
							String createInst = node.getInstructions(tempFileName);
							createvarInst= CPInstructionParser.parseSingleInstruction(createInst);
							
							String writeInst = node.getInstructions(oparams.getLabel(), oparams.getFile_name());
							out.addPostInstruction(CPInstructionParser.parseSingleInstruction(writeInst));
							
							// remove the variable
							out.addLastInstruction(CPInstructionParser.parseSingleInstruction(
									"CP" + Lop.OPERAND_DELIMITOR + "rmfilevar" + Lop.OPERAND_DELIMITOR 
									+ oparams.getLabel() + Lop.VALUETYPE_PREFIX + Expression.ValueType.UNKNOWN + Lop.OPERAND_DELIMITOR 
									+ "true" + Lop.VALUETYPE_PREFIX + "BOOLEAN"));
						} 
						else if (oparams.getFormat() == Format.MM )  {
							
							String tempFileName = scratch + Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR + 
			   					    Lop.FILE_SEPARATOR + ProgramConverter.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR +  
			   					    "temp" + job_id.getNextID();
							
							createvarInst= VariableCPInstruction.prepareCreateVariableInstruction(
													oparams.getLabel(), 
													tempFileName, 
													false, 
													OutputInfo.outputInfoToString(getOutputInfo(node, false)), 
													new MatrixCharacteristics(oparams.getNum_rows(), oparams.getNum_cols(), rpb, cpb, oparams.getNnz())
												);
							String writeInst = node.getInstructions(oparams.getLabel(), oparams.getFile_name());
							out.addPostInstruction(CPInstructionParser.parseSingleInstruction(writeInst));
							
							// remove the variable
							out.addLastInstruction(CPInstructionParser.parseSingleInstruction(
									"CP" + Lop.OPERAND_DELIMITOR + "rmfilevar" + Lop.OPERAND_DELIMITOR 
									+ oparams.getLabel() + Lop.VALUETYPE_PREFIX + Expression.ValueType.UNKNOWN + Lop.OPERAND_DELIMITOR 
									+ "true" + Lop.VALUETYPE_PREFIX + "BOOLEAN"));
						} 
						else {
							createvarInst= VariableCPInstruction.prepareCreateVariableInstruction(
									                oparams.getLabel(), 
									                oparams.getFile_name(), 
									                false, 
									                OutputInfo.outputInfoToString(getOutputInfo(node, false)), 
									                new MatrixCharacteristics(oparams.getNum_rows(), oparams.getNum_cols(), rpb, cpb, oparams.getNnz())
								                 );
							// remove the variable
							out.addLastInstruction(CPInstructionParser.parseSingleInstruction(
									"CP" + Lop.OPERAND_DELIMITOR + "rmfilevar" + Lop.OPERAND_DELIMITOR 
									+ oparams.getLabel() + Lop.VALUETYPE_PREFIX + Expression.ValueType.UNKNOWN + Lop.OPERAND_DELIMITOR 
									+ "false" + Lop.VALUETYPE_PREFIX + "BOOLEAN"));
							
						}
						out.addPreInstruction(createvarInst);
						

						// finally, add the filename and variable name to the list of outputs 
						out.setFileName(oparams.getFile_name());
						out.setVarName(oparams.getLabel());
					}
					else {
						// generate a write instruction that writes matrix to HDFS
						String io_inst = node.getInstructions(
								node.getInputs().get(0).getOutputParameters().getLabel(), 
								node.getOutputParameters().getFile_name());
						out.addLastInstruction(CPInstructionParser.parseSingleInstruction(io_inst));
					}
				}
			}
		}
		
		return out;
	}
	
	/**
	 * Method to generate MapReduce job instructions from a given set of nodes.
	 * 
	 * @param execNodes
	 * @param inst
	 * @param deleteinst
	 * @param jobType
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public void generateMapReduceInstructions(Vector<N> execNodes,
			ArrayList<Instruction> inst, ArrayList<Instruction> deleteinst, ArrayList<Instruction> rmvarinst, 
			JobType jt) throws LopsException,
			DMLUnsupportedOperationException, DMLRuntimeException
	{
		ArrayList<Byte> resultIndices = new ArrayList<Byte>();
		ArrayList<String> inputs = new ArrayList<String>();
		ArrayList<String> outputs = new ArrayList<String>();
		ArrayList<InputInfo> inputInfos = new ArrayList<InputInfo>();
		ArrayList<OutputInfo> outputInfos = new ArrayList<OutputInfo>();
		ArrayList<Long> numRows = new ArrayList<Long>();
		ArrayList<Long> numCols = new ArrayList<Long>();
		ArrayList<Long> numRowsPerBlock = new ArrayList<Long>();
		ArrayList<Long> numColsPerBlock = new ArrayList<Long>();
		ArrayList<String> mapperInstructions = new ArrayList<String>();
		ArrayList<String> randInstructions = new ArrayList<String>();
		ArrayList<String> recordReaderInstructions = new ArrayList<String>();
		int numReducers = 0;
		int replication = 1;
		ArrayList<String> inputLabels = new ArrayList<String>();
		ArrayList<String> outputLabels = new ArrayList<String>();
		ArrayList<Instruction> renameInstructions = new ArrayList<Instruction>();
		ArrayList<Instruction> variableInstructions = new ArrayList<Instruction>();
		ArrayList<Instruction> writeInstructions = new ArrayList<Instruction>();
		
		ArrayList<Lop> inputLops = new ArrayList<Lop>();
		
		boolean cellModeOverride = false;
		
		/* Find the nodes that produce an output */
		Vector<N> rootNodes = new Vector<N>();
		getOutputNodes(execNodes, rootNodes, !jt.producesIntermediateOutput());
		LOG.trace("# of root nodes = " + rootNodes.size());
		
		
		/* Remove transient writes that are simple copy of transient reads */
		if (jt == JobType.GMR) {
			Vector<N> markedNodes = new Vector<N>();
			// only keep data nodes that are results of some computation.
			for (int i = 0; i < rootNodes.size(); i++) {
				N node = rootNodes.elementAt(i);
				if (node.getExecLocation() == ExecLocation.Data
						&& ((Data) node).isTransient()
						&& ((Data) node).getOperationType() == OperationTypes.WRITE
						&& ((Data) node).get_dataType() == DataType.MATRIX) {
					// no computation, just a copy
					if (node.getInputs().get(0).getExecLocation() == ExecLocation.Data
							&& ((Data) node.getInputs().get(0)).isTransient()
							&& node.getOutputParameters().getLabel().compareTo(
									node.getInputs().get(0)
											.getOutputParameters().getLabel()) == 0) {
						markedNodes.add(node);
					}
				}
			}
			// delete marked nodes
			rootNodes.removeAll(markedNodes);
			markedNodes.clear();
			if (rootNodes.size() == 0)
				return;
		}
		
		// structure that maps node to their indices that will be used in the instructions
		HashMap<N, Integer> nodeIndexMapping = new HashMap<N, Integer>();
		
		/* Determine all input data files */
		
		for (int i = 0; i < rootNodes.size(); i++) {
			getInputPathsAndParameters(rootNodes.elementAt(i), execNodes,
					inputs, inputInfos, numRows, numCols, numRowsPerBlock,
					numColsPerBlock, nodeIndexMapping, inputLabels, inputLops);
		}
		
		// In case of RAND job, instructions are defined in the input file
		if (jt == JobType.DATAGEN)
			randInstructions = inputs;
		
		int[] start_index = new int[1];
		start_index[0] = inputs.size();
		
		/* Get RecordReader Instructions */
		
		// currently, recordreader instructions are allowed only in GMR jobs
		if (jt == JobType.GMR) {
			for (int i = 0; i < rootNodes.size(); i++) {
				getRecordReaderInstructions(rootNodes.elementAt(i), execNodes,
						inputs, recordReaderInstructions, nodeIndexMapping,
						start_index, inputLabels, inputLops);
				if ( recordReaderInstructions.size() > 1 )
					throw new LopsException("MapReduce job can only have a single recordreader instruction: " + recordReaderInstructions.toString());
			}
		}
		
		/*
		 * Handle cases when job's output is FORCED to be cell format.
		 * - If there exist a cell input, then output can not be blocked. 
		 *   Only exception is when jobType = REBLOCK/CSVREBLOCK (for obvisous reason)
		 *   or when jobType = RAND since RandJob takes a special input file, 
		 *   whose format should not be used to dictate the output format.
		 * - If there exists a recordReader instruction
		 * - If jobtype = GroupedAgg. This job can only run in cell mode.
		 */
		
		// 
		if ( jt != JobType.REBLOCK && jt != JobType.CSV_REBLOCK && jt != JobType.DATAGEN ) {
			for (int i=0; i < inputInfos.size(); i++)
				if ( inputInfos.get(i) == InputInfo.BinaryCellInputInfo || inputInfos.get(i) == InputInfo.TextCellInputInfo )
					cellModeOverride = true;
		}
		
		if ( recordReaderInstructions.size() > 0 || jt == JobType.GROUPED_AGG )
			cellModeOverride = true;
		
		
		/* Get Mapper Instructions */
	
		for (int i = 0; i < rootNodes.size(); i++) {
			getMapperInstructions(rootNodes.elementAt(i), execNodes, inputs,
					mapperInstructions, nodeIndexMapping, start_index,
					inputLabels, inputLops);
		}
		
		if (LOG.isTraceEnabled()) {
			LOG.trace("    Input strings: " + inputs.toString());
			if (jt == JobType.DATAGEN)
				LOG.trace("    Rand instructions: " + getCSVString(randInstructions));
			if (jt == JobType.GMR)
				LOG.trace("    RecordReader instructions: " + getCSVString(recordReaderInstructions));
			LOG.trace("    Mapper instructions: " + getCSVString(mapperInstructions));
		}

		/* Get Shuffle and Reducer Instructions */
		
		ArrayList<String> shuffleInstructions = new ArrayList<String>();
		ArrayList<String> aggInstructionsReducer = new ArrayList<String>();
		ArrayList<String> otherInstructionsReducer = new ArrayList<String>();

		for (int i = 0; i < rootNodes.size(); i++) {
			int resultIndex = getAggAndOtherInstructions(
					rootNodes.elementAt(i), execNodes, shuffleInstructions, aggInstructionsReducer,
					otherInstructionsReducer, nodeIndexMapping, start_index,
					inputLabels, inputLops);
			if ( resultIndex == -1)
				throw new LopsException("Unexpected error in piggybacking!");
			resultIndices.add(new Byte((byte) resultIndex));
			
			// setup output filenames and outputInfos and generate related instructions
			NodeOutput out = setupNodeOutputs(rootNodes.elementAt(i), ExecType.MR, cellModeOverride);
			outputLabels.add(out.getVarName());
			outputs.add(out.getFileName());
			//if ( out.getFileName() == null )
			//	System.err.println("error here .. ");
			outputInfos.add(out.getOutInfo());
			renameInstructions.addAll(out.getLastInstructions());
			variableInstructions.addAll(out.getPreInstructions());
			writeInstructions.addAll(out.getPostInstructions());
			
		}
		
		// a sanity check
		if (resultIndices.size() != rootNodes.size()) {
			throw new LopsException("Unexected error in piggybacking: "
					+ resultIndices.size() + " != " + rootNodes.size());
		}
		
		/* Determine if the output dimensions are known */
		
		byte[] resultIndicesByte = new byte[resultIndices.size()];
		for (int i = 0; i < resultIndicesByte.length; i++) {
			resultIndicesByte[i] = resultIndices.get(i).byteValue();
		}
		
		if (LOG.isTraceEnabled()) {
			LOG.trace("    Shuffle/Aggregate Instructions: " + getCSVString(aggInstructionsReducer));
			LOG.trace("    Other instructions =" + getCSVString(otherInstructionsReducer));
			LOG.trace("    Output strings: " + outputs.toString());
			LOG.trace("    ResultIndices = " + resultIndices.toString());
		}
		
		/* Prepare the MapReduce job instruction */
		
		MRJobInstruction mr = new MRJobInstruction(jt);
		
		// check if this is a map-only job. If not, set the number of reducers
		if ( shuffleInstructions.size() > 0 || aggInstructionsReducer.size() > 0 || otherInstructionsReducer.size() > 0 )
			numReducers = total_reducers;
		
		// set inputs, outputs, and other other properties for the job 
		mr.setInputOutputLabels(getStringArray(inputLabels), getStringArray(outputLabels));
		mr.setOutputs(resultIndicesByte);
		mr.setDimsUnknownFilePrefix(scratch +  
				                      Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR + 
				                      Lop.FILE_SEPARATOR + ProgramConverter.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR );
		
		mr.setNumberOfReducers(numReducers);
		mr.setReplication(replication);
		
		// set instructions for recordReader and mapper
		mr.setRecordReaderInstructions(getCSVString(recordReaderInstructions));
		mr.setMapperInstructions(getCSVString(mapperInstructions));
		
		if ( jt == JobType.DATAGEN )
			mr.setRandInstructions(getCSVString(randInstructions));
		
		// set shuffle instructions
		mr.setShuffleInstructions(getCSVString(shuffleInstructions));
		
		// set reducer instruction
		mr.setAggregateInstructionsInReducer(getCSVString(aggInstructionsReducer));
		mr.setOtherInstructionsInReducer(getCSVString(otherInstructionsReducer));

		/* Add the prepared instructions to output set */
		inst.addAll(variableInstructions);
		inst.add(mr);
		inst.addAll(writeInstructions);
		deleteinst.addAll(renameInstructions);
		
		for (Lop l : inputLops) {
			processConsumers((N)l, rmvarinst, deleteinst);
		}

	}

	/**
	 * Convert a byte array into string
	 * 
	 * @param arr
	 * @return none
	 */
	public String getString(byte[] arr) {
		String s = "";
		for (int i = 0; i < arr.length; i++)
			s = s + "," + Byte.toString(arr[i]);

		return s;
	}

	/**
	 * converts an array list into a comma separated string
	 * 
	 * @param inputStrings
	 * @return
	 */
	private String getCSVString(ArrayList<String> inputStrings) {
		String s = "";
		for (int i = 0; i < inputStrings.size(); i++) {
			if (inputStrings.get(i) != null) {
				if (s.compareTo("") == 0)
					s = inputStrings.get(i);
				else
					s += Lop.INSTRUCTION_DELIMITOR + inputStrings.get(i);
			}
		}
		return s;

	}

	/**
	 * converts an array list of strings into an array of string
	 * 
	 * @param list
	 * @return
	 */

	private String[] getStringArray(ArrayList<String> list) {
		String[] arr = new String[list.size()];

		for (int i = 0; i < arr.length; i++) {
			arr[i] = list.get(i);
		}

		return arr;
	}

	/**
	 * converts an array list of OutputInfo into an array of OutputInfo
	 * 
	 * @param infos
	 * @return
	 */

	private OutputInfo[] getOutputInfoArray(ArrayList<OutputInfo> infos) {
		OutputInfo[] arr = new OutputInfo[infos.size()];

		for (int i = 0; i < arr.length; i++) {
			arr[i] = infos.get(i);
		}

		return arr;
	}

	/**
	 * converts an array list of InputInfo into an array of InputInfo
	 * 
	 * @param infos
	 * @return
	 */

	private InputInfo[] getInputInfoArray(ArrayList<InputInfo> infos) {
		InputInfo[] arr = new InputInfo[infos.size()];

		for (int i = 0; i < arr.length; i++) {
			arr[i] = infos.get(i);
		}

		return arr;
	}

	/**
	 * converts an array list of long into an array of long
	 * 
	 * @param list
	 * @return
	 */
	private long[] getLongArray(ArrayList<Long> list) {
		long[] arr = new long[list.size()];

		for (int i = 0; i < arr.length; i++) {
			arr[i] = list.get(i);
		}

		return arr;

	}

	/**
	 * converts an array of longs into an array of int
	 * 
	 * @param list
	 * @return
	 */
	private int[] getIntArray(ArrayList<Long> list) {
		int[] arr = new int[list.size()];

		for (int i = 0; i < arr.length; i++) {
			arr[i] = list.get(i).intValue();
		}

		return arr;

	}

	/**
	 * Method to populate aggregate and other instructions in reducer.
	 * 
	 * @param node
	 * @param execNodes
	 * @param aggInstructionsReducer
	 * @param otherInstructionsReducer
	 * @param nodeIndexMapping
	 * @param start_index
	 * @param inputLabels
	 * @return
	 * @throws LopsException
	 */

	@SuppressWarnings("unchecked")
	private int getAggAndOtherInstructions(N node, Vector<N> execNodes,
			ArrayList<String> shuffleInstructions,
			ArrayList<String> aggInstructionsReducer,
			ArrayList<String> otherInstructionsReducer,
			HashMap<N, Integer> nodeIndexMapping, int[] start_index,
			ArrayList<String> inputLabels, ArrayList<Lop> inputLops) throws LopsException

	{

		int ret_val = -1;

		if (nodeIndexMapping.containsKey(node))
			return nodeIndexMapping.get(node);

		// if not an input source and not in exec nodes, return.

		if (!execNodes.contains(node))
			return ret_val;

		ArrayList<Integer> inputIndices = new ArrayList<Integer>();

		// recurse
		// Leo: For WRITE, since the first element from input is the real input (the other elements
		// are parameters for the WRITE operation), so we only need to take care of the
		// first element.
		if (node.getType() == Lop.Type.Data && ((Data)node).getOperationType() == Data.OperationTypes.WRITE) {
			ret_val = getAggAndOtherInstructions((N) node.getInputs().get(0),
					execNodes, shuffleInstructions, aggInstructionsReducer,
					otherInstructionsReducer, nodeIndexMapping, start_index,
					inputLabels, inputLops);
			inputIndices.add(ret_val);
		}
		else {
			for (int i = 0; i < node.getInputs().size(); i++) {
				ret_val = getAggAndOtherInstructions((N) node.getInputs().get(i),
						execNodes, shuffleInstructions, aggInstructionsReducer,
						otherInstructionsReducer, nodeIndexMapping, start_index,
						inputLabels, inputLops);
				inputIndices.add(ret_val);
			}
		}

		// have to verify if this is needed
		if (node.getExecLocation() == ExecLocation.Data ) {
			if ( ((Data)node).getFileFormatType() == FileFormatTypes.CSV ) {
				// Generate write instruction, which goes into CSV_WRITE Job
				int output_index = start_index[0];
				shuffleInstructions.add(node.getInstructions(inputIndices.get(0), output_index));
				nodeIndexMapping.put(node, output_index);
				start_index[0]++; 
				return output_index;
			}
			else {
				return ret_val;
			}
		}

		if (node.getExecLocation() == ExecLocation.MapAndReduce) {
			
			/* Generate Shuffle Instruction for "node", and return the index associated with produced output */
			
			boolean instGenerated = true;
			int output_index = start_index[0];
			switch(node.getType()) {
			
			/* Lop types that take a single input */
			case ReBlock:
			case CSVReBlock:
			case SortKeys:
			case CentralMoment:
			case CoVariance:
			case GroupedAgg:
			case DataPartition:
				shuffleInstructions.add(node.getInstructions(inputIndices.get(0), output_index));
				break;
				
			/* Lop types that take two inputs */
			case MMCJ:
			case MMRJ:
			case CombineBinary:
				shuffleInstructions.add(node.getInstructions(inputIndices.get(0), inputIndices.get(1), output_index));
				break;

			/* Lop types that take three inputs */
			case CombineTertiary:
				shuffleInstructions.add(node.getInstructions(inputIndices
						.get(0), inputIndices.get(1), inputIndices.get(2), output_index));
				break;
			
			default:
				instGenerated = false;
				break;
			}
			
			if ( instGenerated ) { 
				nodeIndexMapping.put(node, output_index);
				start_index[0]++;
				return output_index;
			}
			else {
				return inputIndices.get(0);
			}
		}

		/* Get instructions for aligned reduce and other lops below the reduce. */
		if (node.getExecLocation() == ExecLocation.Reduce
				|| node.getExecLocation() == ExecLocation.MapOrReduce
				|| hasChildNode(node, execNodes, ExecLocation.MapAndReduce)) {

			if (inputIndices.size() == 1) {
				int output_index = start_index[0];
				start_index[0]++;

				if (node.getType() == Type.Aggregate)
					aggInstructionsReducer.add(node.getInstructions(
							inputIndices.get(0), output_index));
				else {
					otherInstructionsReducer.add(node.getInstructions(
							inputIndices.get(0), output_index));
				}
				nodeIndexMapping.put(node, output_index);

				return output_index;
			} else if (inputIndices.size() == 2) {
				int output_index = start_index[0];
				start_index[0]++;

				otherInstructionsReducer.add(node.getInstructions(inputIndices
						.get(0), inputIndices.get(1), output_index));
				nodeIndexMapping.put(node, output_index);

				// populate list of input labels.
				// only Unary lops can contribute to labels

				if (node instanceof Unary && node.getInputs().size() > 1) {
					int index = 0;
					for (int i = 0; i < node.getInputs().size(); i++) {
						if (node.getInputs().get(i).get_dataType() == DataType.SCALAR) {
							index = i;
							break;
						}
					}

					if (node.getInputs().get(index).getExecLocation() == ExecLocation.Data
							&& !((Data) (node.getInputs().get(index))).isLiteral()) {
						inputLabels.add(node.getInputs().get(index).getOutputParameters().getLabel());
						inputLops.add(node.getInputs().get(index));
					}

					if (node.getInputs().get(index).getExecLocation() != ExecLocation.Data) {
						inputLabels.add(node.getInputs().get(index).getOutputParameters().getLabel());
						inputLops.add(node.getInputs().get(index));
					}

				}

				return output_index;
			} else if (inputIndices.size() == 3) {
				int output_index = start_index[0];
				start_index[0]++;

				if (node.getType() == Type.Tertiary ) {
					//Tertiary.OperationTypes op = ((Tertiary<?, ?, ?, ?>) node).getOperationType();
					//if ( op == Tertiary.OperationTypes.CTABLE_TRANSFORM ) {
					
					// in case of CTABLE_TRANSFORM_SCALAR_WEIGHT: inputIndices.get(2) would be -1
					otherInstructionsReducer.add(node.getInstructions(
							inputIndices.get(0), inputIndices.get(1),
							inputIndices.get(2), output_index));
					nodeIndexMapping.put(node, output_index);
					//}
				}
				else if( node.getType() == Type.ParameterizedBuiltin ){
					otherInstructionsReducer.add(node.getInstructions(
							inputIndices.get(0), inputIndices.get(1),
							inputIndices.get(2), output_index));
					nodeIndexMapping.put(node, output_index);
				}

				return output_index;

			}
			else if (inputIndices.size() == 4) {
				int output_index = start_index[0];
				start_index[0]++;
				otherInstructionsReducer.add(node.getInstructions(
						inputIndices.get(0), inputIndices.get(1),
						inputIndices.get(2), inputIndices.get(3), output_index));
				nodeIndexMapping.put(node, output_index);
				return output_index;
			}
			else
				throw new LopsException("Invalid number of inputs to a lop: "
						+ inputIndices.size());
		}

		return -1;

	}

	/**
	 * Method to get record reader instructions for a MR job.
	 * 
	 * @param node
	 * @param execNodes
	 * @param inputStrings
	 * @param recordReaderInstructions
	 * @param nodeIndexMapping
	 * @param start_index
	 * @param inputLabels
	 * @return
	 * @throws LopsException
	 */

	@SuppressWarnings("unchecked")
	private int getRecordReaderInstructions(N node, Vector<N> execNodes,
			ArrayList<String> inputStrings,
			ArrayList<String> recordReaderInstructions,
			HashMap<N, Integer> nodeIndexMapping, int[] start_index,
			ArrayList<String> inputLabels, ArrayList<Lop> inputLops) throws LopsException

	{

		// if input source, return index
		if (nodeIndexMapping.containsKey(node))
			return nodeIndexMapping.get(node);

		// not input source and not in exec nodes, then return.
		if (!execNodes.contains(node))
			return -1;

		ArrayList<Integer> inputIndices = new ArrayList<Integer>();
		int max_input_index = -1;
		N child_for_max_input_index = null;

		// get mapper instructions
		for (int i = 0; i < node.getInputs().size(); i++) {

			// recurse
			N childNode = (N) node.getInputs().get(i);
			int ret_val = getRecordReaderInstructions(childNode, execNodes,
					inputStrings, recordReaderInstructions, nodeIndexMapping,
					start_index, inputLabels, inputLops);

			inputIndices.add(ret_val);

			if (ret_val > max_input_index) {
				max_input_index = ret_val;
				child_for_max_input_index = childNode;
			}
		}

		// only lops with execLocation as RecordReader can contribute
		// instructions
		if ((node.getExecLocation() == ExecLocation.RecordReader)) {
			int output_index = max_input_index;

			// cannot reuse index if this is true
			// need to add better indexing schemes
			// if (child_for_max_input_index.getOutputs().size() > 1) {
			output_index = start_index[0];
			start_index[0]++;
			// }

			nodeIndexMapping.put(node, output_index);

			// populate list of input labels.
			// only Ranagepick lop can contribute to labels
			if (node.getType() == Type.PickValues) {

				PickByCount pbc = (PickByCount) node;
				if (pbc.getOperationType() == PickByCount.OperationTypes.RANGEPICK) {
					int scalarIndex = 1; // always the second input is a scalar

					// if data lop not a literal -- add label
					if (node.getInputs().get(scalarIndex).getExecLocation() == ExecLocation.Data
							&& !((Data) (node.getInputs().get(scalarIndex))).isLiteral()) {
						inputLabels.add(node.getInputs().get(scalarIndex).getOutputParameters().getLabel());
						inputLops.add(node.getInputs().get(scalarIndex));
					}

					// if not data lop, then this is an intermediate variable.
					if (node.getInputs().get(scalarIndex).getExecLocation() != ExecLocation.Data) {
						inputLabels.add(node.getInputs().get(scalarIndex).getOutputParameters().getLabel());
						inputLops.add(node.getInputs().get(scalarIndex));
					}
				}
			}

			// get recordreader instruction.
			if (node.getInputs().size() == 2)
				recordReaderInstructions.add(node.getInstructions(inputIndices
						.get(0), inputIndices.get(1), output_index));
			else
				throw new LopsException(
						"Unexpected number of inputs while generating a RecordReader Instruction");

			return output_index;

		}

		return -1;

	}

	/**
	 * Method to get mapper instructions for a MR job.
	 * 
	 * @param node
	 * @param execNodes
	 * @param inputStrings
	 * @param instructionsInMapper
	 * @param nodeIndexMapping
	 * @param start_index
	 * @param inputLabels
	 * @return
	 * @throws LopsException
	 */

	@SuppressWarnings("unchecked")
	private int getMapperInstructions(N node, Vector<N> execNodes,
			ArrayList<String> inputStrings,
			ArrayList<String> instructionsInMapper,
			HashMap<N, Integer> nodeIndexMapping, int[] start_index,
			ArrayList<String> inputLabels, ArrayList<Lop> inputLops) throws LopsException

	{

		// if input source, return index
		if (nodeIndexMapping.containsKey(node))
			return nodeIndexMapping.get(node);

		// not input source and not in exec nodes, then return.
		if (!execNodes.contains(node))
			return -1;

		ArrayList<Integer> inputIndices = new ArrayList<Integer>();
		int max_input_index = -1;
		// get mapper instructions
		for (int i = 0; i < node.getInputs().size(); i++) {

			// recurse
			N childNode = (N) node.getInputs().get(i);
			int ret_val = getMapperInstructions(childNode, execNodes,
					inputStrings, instructionsInMapper, nodeIndexMapping,
					start_index, inputLabels, inputLops);

			inputIndices.add(ret_val);

			if (ret_val > max_input_index) {
				max_input_index = ret_val;
			}
		}

		// only map and map-or-reduce without a reduce child node can contribute
		// to mapper instructions.
		if ((node.getExecLocation() == ExecLocation.Map || node
				.getExecLocation() == ExecLocation.MapOrReduce)
				&& !hasChildNode(node, execNodes, ExecLocation.MapAndReduce)
				&& !hasChildNode(node, execNodes, ExecLocation.Reduce)
				) {
			int output_index = max_input_index;

			// cannot reuse index if this is true
			// need to add better indexing schemes
			// if (child_for_max_input_index.getOutputs().size() > 1) {
			output_index = start_index[0];
			start_index[0]++;
			// }

			nodeIndexMapping.put(node, output_index);

			// populate list of input labels.
			// only Unary lops can contribute to labels

			if (node instanceof Unary && node.getInputs().size() > 1) {
				// Following code must be executed only for those Unary
				// operators that have more than one input
				// It should not be executed for "true" unary operators like
				// cos(A).

				int index = 0;
				for (int i1 = 0; i1 < node.getInputs().size(); i1++) {
					if (node.getInputs().get(i1).get_dataType() == DataType.SCALAR) {
						index = i1;
						break;
					}
				}

				// if data lop not a literal -- add label
				if (node.getInputs().get(index).getExecLocation() == ExecLocation.Data
						&& !((Data) (node.getInputs().get(index))).isLiteral()) {
					inputLabels.add(node.getInputs().get(index).getOutputParameters().getLabel());
					inputLops.add(node.getInputs().get(index));
				}

				// if not data lop, then this is an intermediate variable.
				if (node.getInputs().get(index).getExecLocation() != ExecLocation.Data) {
					inputLabels.add(node.getInputs().get(index).getOutputParameters().getLabel());
					inputLops.add(node.getInputs().get(index));
				}

			}

			// get mapper instruction.
			if (node.getInputs().size() == 1)
				instructionsInMapper.add(node.getInstructions(inputIndices
						.get(0), output_index));
			else if (node.getInputs().size() == 2) {
				instructionsInMapper.add(node.getInstructions(inputIndices
						.get(0), inputIndices.get(1), output_index));
			}
			else if (node.getInputs().size() == 3)
				instructionsInMapper.add(node.getInstructions(inputIndices.get(0), 
															  inputIndices.get(1), 
															  inputIndices.get(2), 
															  output_index));
			else if ( node.getInputs().size() == 4) {
				// Example: Reshape
				instructionsInMapper.add(node.getInstructions(
						inputIndices.get(0),
						inputIndices.get(1),
						inputIndices.get(2),
						inputIndices.get(3),
						output_index ));
			}			
			else if ( node.getInputs().size() == 5) {
				// Example: RangeBasedReIndex A[row_l:row_u, col_l:col_u]
				instructionsInMapper.add(node.getInstructions(
						inputIndices.get(0),
						inputIndices.get(1),
						inputIndices.get(2),
						inputIndices.get(3),
						inputIndices.get(4),
						output_index ));
			}
			else if ( node.getInputs().size() == 7 ) {
				// Example: RangeBasedReIndex A[row_l:row_u, col_l:col_u] = B
				instructionsInMapper.add(node.getInstructions(
						inputIndices.get(0),
						inputIndices.get(1),
						inputIndices.get(2),
						inputIndices.get(3),
						inputIndices.get(4),
						inputIndices.get(5),
						inputIndices.get(6),
						output_index ));
			}
			else
				throw new LopsException("Node with " + node.getInputs().size() + " inputs is not supported in dag.java.");
			
			
			return output_index;

		}

		return -1;

	}

	// Method to populate inputs and also populates node index mapping.
	@SuppressWarnings("unchecked")
	private void getInputPathsAndParameters(N node, Vector<N> execNodes,
			ArrayList<String> inputStrings, ArrayList<InputInfo> inputInfos,
			ArrayList<Long> numRows, ArrayList<Long> numCols,
			ArrayList<Long> numRowsPerBlock, ArrayList<Long> numColsPerBlock,
			HashMap<N, Integer> nodeIndexMapping, ArrayList<String> inputLabels, ArrayList<Lop> inputLops)
			throws LopsException {
		// treat rand as an input.
		if (node.getType() == Type.DataGen && execNodes.contains(node)
				&& !nodeIndexMapping.containsKey(node)) {
			numRows.add(node.getOutputParameters().getNum_rows());
			numCols.add(node.getOutputParameters().getNum_cols());
			numRowsPerBlock.add(node.getOutputParameters()
					.get_rows_in_block());
			numColsPerBlock.add(node.getOutputParameters()
					.get_cols_in_block());
			inputStrings.add(node.getInstructions(inputStrings.size(),
					inputStrings.size()));
			inputInfos.add(InputInfo.TextCellInputInfo);
			nodeIndexMapping.put(node, inputStrings.size() - 1);

			return;
		}

		// && ( !(node.getExecLocation() == ExecLocation.ControlProgram)
		// || (node.getExecLocation() == ExecLocation.ControlProgram &&
		// node.get_dataType() != DataType.SCALAR )
		// )
		// get input file names
		if (!execNodes.contains(node)
				&& !nodeIndexMapping.containsKey(node)
				&& !(node.getExecLocation() == ExecLocation.Data)
				&& (!(node.getExecLocation() == ExecLocation.ControlProgram && node
						.get_dataType() == DataType.SCALAR))
				|| (!execNodes.contains(node)
						&& node.getExecLocation() == ExecLocation.Data
						&& ((Data) node).getOperationType() == Data.OperationTypes.READ
						&& ((Data) node).get_dataType() == DataType.MATRIX && !nodeIndexMapping
						.containsKey(node))) {
			if (node.getOutputParameters().getFile_name() != null) {
				inputStrings.add(node.getOutputParameters().getFile_name());
			} else {
				// use label name
				inputStrings.add(Lop.VARIABLE_NAME_PLACEHOLDER + node.getOutputParameters().getLabel()
						               + Lop.VARIABLE_NAME_PLACEHOLDER);
			}
			//if ( node.getType() == Lops.Type.Data && ((Data)node).isTransient())
			//	inputStrings.add("##" + node.getOutputParameters().getLabel() + "##");
			//else
			//	inputStrings.add(node.getOutputParameters().getLabel());
			inputLabels.add(node.getOutputParameters().getLabel());
			inputLops.add(node);

			numRows.add(node.getOutputParameters().getNum_rows());
			numCols.add(node.getOutputParameters().getNum_cols());
			numRowsPerBlock.add(node.getOutputParameters()
					.get_rows_in_block());
			numColsPerBlock.add(node.getOutputParameters()
					.get_cols_in_block());

			InputInfo nodeInputInfo = null;
			// Check if file format type is binary or text and update infos
			if (node.getOutputParameters().isBlocked_representation()) {
				if (node.getOutputParameters().getFormat() == Format.BINARY)
					nodeInputInfo = InputInfo.BinaryBlockInputInfo;
				else 
					throw new LopsException("Invalid format (" + node.getOutputParameters().getFormat() + ") encountered for a node/lop with blocked output.");
				// inputInfos.add(InputInfo.BinaryBlockInputInfo);
			} else {
				if (node.getOutputParameters().getFormat() == Format.TEXT)
					nodeInputInfo = InputInfo.TextCellInputInfo;
				// inputInfos.add(InputInfo.TextCellInputInfo);
				else {
					nodeInputInfo = InputInfo.BinaryCellInputInfo;
					// inputInfos.add(InputInfo.BinaryCellInputInfo);
				}
			}

			/*
			 * Hardcode output Key and Value Classes for SortKeys
			 */
			// TODO: statiko -- remove this hardcoding -- i.e., lops must encode
			// the information on key/value classes
			if (node.getType() == Type.SortKeys) {
				// SortKeys is the input to some other lop (say, L)
				// InputInfo of L is the ouputInfo of SortKeys, which is
				// (compactformat, doubleWriteable, IntWritable)
				nodeInputInfo = new InputInfo(PickFromCompactInputFormat.class,
						DoubleWritable.class, IntWritable.class);
			} else if (node.getType() == Type.CombineBinary) {
				
				// CombineBinary is the input to some other lop (say, L)
				// InputInfo of L is the ouputInfo of CombineBinary
				// And, the outputInfo of CombineBinary depends on the operation!
				CombineBinary combine = (CombineBinary) node;
				if ( combine.getOperation() == com.ibm.bi.dml.lops.CombineBinary.OperationTypes.PreSort ) {
					nodeInputInfo = new InputInfo(SequenceFileInputFormat.class,
							DoubleWritable.class, IntWritable.class);
				}
				else if ( combine.getOperation() == com.ibm.bi.dml.lops.CombineBinary.OperationTypes.PreCentralMoment 
						  || combine.getOperation() == com.ibm.bi.dml.lops.CombineBinary.OperationTypes.PreCovUnweighted
						  || combine.getOperation() == com.ibm.bi.dml.lops.CombineBinary.OperationTypes.PreGroupedAggUnweighted ) {
					nodeInputInfo = InputInfo.WeightedPairInputInfo;
				}
			} else if ( node.getType() == Type.CombineTertiary ) {
				nodeInputInfo = InputInfo.WeightedPairInputInfo;
			}

			inputInfos.add(nodeInputInfo);

			nodeIndexMapping.put(node, inputStrings.size() - 1);
			return;

		}

		// if exec nodes does not contain node at this point, return.

		if (!execNodes.contains(node))
			return;

		// process children recursively

		for (int i = 0; i < node.getInputs().size(); i++) {
			N childNode = (N) node.getInputs().get(i);
			getInputPathsAndParameters(childNode, execNodes, inputStrings,
					inputInfos, numRows, numCols, numRowsPerBlock,
					numColsPerBlock, nodeIndexMapping, inputLabels, inputLops);
		}

	}

	/**
	 * Method to find all terminal nodes.
	 * 
	 * @param execNodes
	 * @param rootNodes
	 */

	private void getOutputNodes(Vector<N> execNodes, Vector<N> rootNodes,
			boolean include_intermediate) {
		for (int i = 0; i < execNodes.size(); i++) {
			N node = execNodes.elementAt(i);

			// terminal node
			if (node.getOutputs().size() == 0 && !rootNodes.contains(node)) {
				rootNodes.add(node);
			} else {
				// check for nodes with at least one child outside execnodes
				int cnt = 0;
				for (int j = 0; j < node.getOutputs().size(); j++) {
					if (!execNodes.contains(node.getOutputs().get(j))) {
						cnt++;
					}
				}

				if (cnt != 0
						&& cnt <= node.getOutputs().size()
						&& !rootNodes.contains(node)
						&& !(node.getExecLocation() == ExecLocation.Data
								&& ((Data) node).getOperationType() == OperationTypes.READ && ((Data) node)
								.get_dataType() == DataType.MATRIX)) {

					if (cnt < node.getOutputs().size()) {
						if (include_intermediate)
							rootNodes.add(node);
					} else
						rootNodes.add(node);
				}
			}
		}
	}

	private void getPartitionNodes(Vector<N> execNodes, Vector<N> partitionNodes) {
		for (int i = 0; i < execNodes.size(); i++) {
			N node = execNodes.elementAt(i);
			if (node.getType() == Lop.Type.PartitionLop)
				partitionNodes.add(node);
		}
	}

	/**
	 * check to see if a is the child of b (i.e., there is a directed path from a to b)
	 * 
	 * @param a
	 * @param b
	 */

	public static boolean isChild(Lop a, Lop b, HashMap<Long, Integer> IDMap) {
		//int aID = IDMap.get(a.getID());
		int bID = IDMap.get(b.getID());
		return a.get_reachable()[bID];
	}
	
	public static boolean isChildOLD(Lop a, Lop b) {
		for (int i = 0; i < b.getInputs().size(); i++) {
			if (b.getInputs().get(i).equals(a))
				return true;

			/**
			 * dfs search
			 */
			boolean return_val = isChildOLD(a, b.getInputs().get(i));

			/**
			 * return true, if matching parent found, else continue
			 */
			if (return_val)
				return true;
		}

		return false;
	}

	@SuppressWarnings("unchecked")
	/**
	 * Method to topologically sort lops
	 * 
	 * @param v
	 */
	private void doTopologicalSort_strict_order(Vector<N> v) {
		//int numNodes = v.size();

		/*
		 * Step 1: compute the level for each node in the DAG. Level for each node is 
		 *   computed as lops are created. So, this step is need not be performed here.
		 * Step 2: sort the nodes by level, and within a level by node ID.
		 */
		
		// Step1: Performed at the time of creating Lops
		
		// Step2: sort nodes by level, and then by node ID
		Object[] nodearray = v.toArray();
		Arrays.sort(nodearray, new LopComparator());

		// Copy sorted nodes into "v" and construct a mapping between Lop IDs and sequence of numbers
		v.clear();
		IDMap.clear();
		
		for (int i = 0; i < nodearray.length; i++) {
			v.add((N) nodearray[i]);
			IDMap.put(v.get(i).getID(), i);
		}
		
		/*
		 * Compute of All-pair reachability graph (Transitive Closure) of the DAG.
		 * - Perform a depth-first search (DFS) from every node $u$ in the DAG 
		 * - and construct the list of reachable nodes from the node $u$
		 * - store the constructed reachability information in $u$.reachable[] boolean array
		 */
		// 
		//  
		for (int i = 0; i < nodearray.length; i++) {
			boolean[] arr = v.get(i).create_reachable(nodearray.length);
			Arrays.fill(arr, false);
			dagDFS(v.get(i), arr);
		}

		// Sanity check -- can be removed
/*		for (int i = 1; i < v.size(); i++) {
			if (v.get(i).getLevel() < v.get(i - 1).getLevel()
					|| (v.get(i).getLevel() == v.get(i - 1).getLevel() && v
							.get(i).getID() <= v.get(i - 1).getID())) {
				try {
					throw new DMLRuntimeException(
							"Unexpected error in topological sort.");
				} catch (DMLRuntimeException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
*/
		// print the nodes in sorted order
		if (LOG.isTraceEnabled()) {
			for (int i = 0; i < v.size(); i++) {
				// System.out.print(sortedNodes.get(i).getID() + "("
				// + levelmap.get(sortedNodes.get(i).getID()) + "), ");
				LOG.trace(v.get(i).getID() + "(" + v.get(i).getLevel()
						+ "), ");
				//System.out.print(v.get(i).getID() + "(" + v.get(i).getLevel()+ "), ");
			}
			
			//System.out.println("");
			LOG.trace("topological sort -- done");
		}

	}

	/**
	 * Method to perform depth-first traversal from a given node in the DAG.
	 * Store the reachability information in marked[] boolean array.
	 * 
	 * @param root
	 * @param marked
	 */
	void dagDFS(N root, boolean[] marked) {
		//contains check currently required for globalopt, will be removed when cleaned up
		if( !IDMap.containsKey(root.getID()) )
			return;
		
		int mapID = IDMap.get(root.getID());
		if ( marked[mapID] )
			return;
		marked[mapID] = true;
		for(int i=0; i < root.getOutputs().size(); i++) {
			//System.out.println("CALLING DFS "+root);
			dagDFS((N)root.getOutputs().get(i), marked);
		}
	}
	
	@SuppressWarnings("unchecked")
	private boolean hasChildNode(N node, Vector<N> childNodes, ExecLocation type) {
		if ( childNodes.size() == 0 ) 
			return false;
		
		int index = IDMap.get(node.getID());
		for(int i=0; i < childNodes.size(); i++) {
			N cn = childNodes.get(i);
			if ( (type == ExecLocation.INVALID || cn.getExecLocation() == type) && cn.get_reachable()[index]) {
				return true;
			}
		}
		return false;
	}

	@SuppressWarnings("unchecked")
	private boolean hasChildNodeOLD(N node, Vector<N> childNodes, ExecLocation type) {
		if(childNodes.size() == 0)
			return false;
		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);
			if (childNodes.contains(n) && n.getExecLocation() == type) {
				return true;
			} else {
				if (hasChildNodeOLD(n, childNodes, type))
					return true;
			}

		}

		return false;

	}

	@SuppressWarnings("unchecked")
	private N getChildNode(N node, Vector<N> childNodes, ExecLocation type) {
		if ( childNodes.size() == 0 )
			return null;
		
		int index = IDMap.get(node.getID());
		for(int i=0; i < childNodes.size(); i++) {
			N cn = childNodes.get(i);
			if ( cn.getExecLocation() == type && cn.get_reachable()[index]) {
				return cn;
			}
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	private N getChildNodeOLD(N node, Vector<N> childNodes, ExecLocation type) {
		if(childNodes.size() == 0)
			return null;
		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);
			if (childNodes.contains(n) && n.getExecLocation() == type) {
				return n;
			} else {
				return getChildNodeOLD(n, childNodes, type);
			}

		}

		return null;

	}

	/*
	 * Returns a node "n" such that 
	 * 1) n \in parentNodes
	 * 2) n is an ancestor of "node"
	 * 3) n.ExecLocation = type
	 * 
	 * Returns null if no such "n" exists
	 * 
	 */
	@SuppressWarnings("unchecked")
	private N getParentNode(N node, Vector<N> parentNodes, ExecLocation type) {
		if ( parentNodes.size() == 0 )
			return null;
		for(int i=0; i < parentNodes.size(); i++ ) {
			N pn = parentNodes.get(i);
			int index = IDMap.get( pn.getID() );
			if ( pn.getExecLocation() == type && node.get_reachable()[index])
				return pn;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	private N getParentNodeOLD(N node, Vector<N> parentNodes, ExecLocation type) {
		if(parentNodes.size() ==0)
			return null;
		
		for (int i = 0; i < node.getOutputs().size(); i++) {
			N n = (N) node.getOutputs().get(i);
			if (parentNodes.contains(n) && n.getExecLocation() == type) {
				return n;
			} else {
				return getParentNodeOLD(n, parentNodes, type);
			}

		}

		return null;

	}

	// Checks if "node" has any descendants in nodesVec with definedMRJob flag
	// set to true
	@SuppressWarnings("unchecked")
	private boolean hasMRJobChildNode(N node, Vector<N> nodesVec) {
		if ( nodesVec.size() == 0 )
			return false;
		
		int index = IDMap.get(node.getID());
		
		for (int i = 0; i < nodesVec.size(); i++) {
			N n = nodesVec.get(i);
			if ( n.definesMRJob() && n.get_reachable()[index]) 
				return true;
		}
		return false;
	}

	@SuppressWarnings("unchecked")
	private boolean hasMRJobChildNodeOLD(N node, Vector<N> nodesVec) {
		if(nodesVec.size() ==0)
			return false;
		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);
			if (nodesVec.contains(n) && n.definesMRJob()) {
				return true;
			} else {
				if (hasMRJobChildNodeOLD(n, nodesVec))
					return true;
			}

		}

		return false;

	}

	// Find the descendant of "node" in "nodeVec" that has its definedMRJob flag set to true
	// returns null if no such descendant exists in nodeVec
	@SuppressWarnings("unchecked")
	private N getMRJobChildNode(N node, Vector<N> nodesVec) {
		if(nodesVec.size() == 0)
			return null;
		
		int index = IDMap.get(node.getID());
		for (int i = 0; i < nodesVec.size(); i++) {
			N n = nodesVec.get(i);
			if ( n.definesMRJob() && n.get_reachable()[index]) 
				return n;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	private N getMRJobChildNodeOLD(N node, Vector<N> nodesVec) {
		if(nodesVec.size() == 0)
			return null;
		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);
			if (nodesVec.contains(n) && n.definesMRJob()) {
				return n;
			} else {
				N ret = getMRJobChildNodeOLD(n, nodesVec);
				if ( ret != null ) 
					return ret;
			}

		}
		return null;
	}

	private int getChildAlignment(N node, Vector<N> execNodes, ExecLocation type) {

		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);

			if (!execNodes.contains(n))
				continue;

			if (execNodes.contains(n) && n.getExecLocation() == type) {
				if (n.getBreaksAlignment())
					return MR_CHILD_FOUND_BREAKS_ALIGNMENT;
				else
					return MR_CHILD_FOUND_DOES_NOT_BREAK_ALIGNMENT;
			} else {
				int ret = getChildAlignment(n, execNodes, type);
				if (ret == MR_CHILD_FOUND_DOES_NOT_BREAK_ALIGNMENT
						|| ret == CHILD_DOES_NOT_BREAK_ALIGNMENT) {
					if (n.getBreaksAlignment())
						return CHILD_BREAKS_ALIGNMENT;
					else
						return CHILD_DOES_NOT_BREAK_ALIGNMENT;
				}

				else if (ret == MRCHILD_NOT_FOUND
						|| ret == CHILD_BREAKS_ALIGNMENT
						|| ret == MR_CHILD_FOUND_BREAKS_ALIGNMENT)
					return ret;
				else
					throw new RuntimeException(
							"Something wrong in getChildAlignment().");
			}
		}

		return MRCHILD_NOT_FOUND;

	}

	@SuppressWarnings("unchecked")
	private boolean hasChildNode(N node, Vector<N> nodes) {
		return hasChildNode(node, nodes, ExecLocation.INVALID);
	}

	@SuppressWarnings("unchecked")
	private boolean hasChildNodeOLD(N node, Vector<N> nodes) {
		if(nodes.size() == 0)
			return false;
		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);
			if (nodes.contains(n)) {
				return true;
			} else {
				if (hasChildNodeOLD(n, nodes))
					return true;
			}

		}

		return false;

	}

	@SuppressWarnings("unchecked")
	private boolean hasParentNode(N node, Vector<N> parentNodes) {
		if ( parentNodes.size() == 0 )
			return false;
		
		for( int i=0; i < parentNodes.size(); i++ ) {
			int index = IDMap.get( parentNodes.get(i).getID() );
			if ( node.get_reachable()[index])
				return true;
		}
		return false;
	}
	
	@SuppressWarnings("unchecked")
	private boolean hasParentNodeOLD(N node, Vector<N> childNodes) {
		if(childNodes.size() == 0)
			return false;
		for (int i = 0; i < node.getOutputs().size(); i++) {
			N n = (N) node.getOutputs().get(i);
			if (childNodes.contains(n)) {
				return true;
			} else {
				if (hasParentNodeOLD(n, childNodes))
					return true;
			}

		}

		return false;
	}
	
}
