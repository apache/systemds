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

package org.apache.sysml.lops.compile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.Hop.FileFormatTypes;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.AppendM;
import org.apache.sysml.lops.BinaryM;
import org.apache.sysml.lops.CombineBinary;
import org.apache.sysml.lops.Data;
import org.apache.sysml.lops.PMMJ;
import org.apache.sysml.lops.ParameterizedBuiltin;
import org.apache.sysml.lops.SortKeys;
import org.apache.sysml.lops.Data.OperationTypes;
import org.apache.sysml.lops.FunctionCallCP;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.Lop.Type;
import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.MapMult;
import org.apache.sysml.lops.OutputParameters;
import org.apache.sysml.lops.OutputParameters.Format;
import org.apache.sysml.lops.PickByCount;
import org.apache.sysml.lops.Unary;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.ParameterizedBuiltinFunctionExpression;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.instructions.CPInstructionParser;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.Instruction.INSTRUCTION_TYPE;
import org.apache.sysml.runtime.instructions.InstructionParser;
import org.apache.sysml.runtime.instructions.SPInstructionParser;
import org.apache.sysml.runtime.instructions.cp.CPInstruction;
import org.apache.sysml.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysml.runtime.instructions.cp.CPInstruction.CPINSTRUCTION_TYPE;
import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.sort.PickFromCompactInputFormat;



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

	private static final int CHILD_BREAKS_ALIGNMENT = 2;
	private static final int CHILD_DOES_NOT_BREAK_ALIGNMENT = 1;
	private static final int MRCHILD_NOT_FOUND = 0;
	private static final int MR_CHILD_FOUND_BREAKS_ALIGNMENT = 4;
	private static final int MR_CHILD_FOUND_DOES_NOT_BREAK_ALIGNMENT = 5;

	private static IDSequence job_id = null;
	private static IDSequence var_index = null;
	
	private int total_reducers = -1;
	private String scratch = "";
	private String scratchFilePath = null;
	
	private double gmrMapperFootprint = 0;
	
	static {
		job_id = new IDSequence();
		var_index = new IDSequence();
	}
	
	// hash set for all nodes in dag
	private ArrayList<Lop> nodes = null;
	
	/* 
	 * Hashmap to translates the nodes in the DAG to a sequence of numbers
	 *     key:   Lop ID
	 *     value: Sequence Number (0 ... |DAG|)
	 *     
	 * This map is primarily used in performing DFS on the DAG, and subsequently in performing ancestor-descendant checks.
	 */
	private HashMap<Long, Integer> IDMap = null;


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
	
	private String getFilePath() {
		if ( scratchFilePath == null ) {
			scratchFilePath = scratch + Lop.FILE_SEPARATOR
								+ Lop.PROCESS_PREFIX + DMLScript.getUUID()
								+ Lop.FILE_SEPARATOR + Lop.FILE_SEPARATOR
								+ ProgramConverter.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR;
		}
		return scratchFilePath;
	}
	
	/**
	 * Constructor
	 */
	public Dag() 
	{
		//allocate internal data structures
		nodes = new ArrayList<Lop>();
		IDMap = new HashMap<Long, Integer>();

		// get number of reducers from dml config
		total_reducers = ConfigurationManager.getNumReducers();
	}

	/**
	 * Method to add a node to the DAG.
	 * 
	 * @param node
	 * @return true if node was not already present, false if not.
	 */

	public boolean addNode(Lop node) {
		if (nodes.contains(node))
			return false;		
		nodes.add(node);
		return true;
	}
	
	/**
	 * 
	 * @param config
	 * @return
	 * @throws LopsException
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	public ArrayList<Instruction> getJobs(DMLConfig config)
			throws LopsException, IOException, DMLRuntimeException 
	{
		return getJobs(null, config);
	}
	
	/**
	 * Method to compile a dag generically
	 * 
	 * @param config
	 * @throws LopsException
	 * @throws DMLRuntimeException
	 */

	public ArrayList<Instruction> getJobs(StatementBlock sb, DMLConfig config)
			throws LopsException, IOException, DMLRuntimeException {
		
		if (config != null) 
		{
			total_reducers = config.getIntValue(DMLConfig.NUM_REDUCERS);
			scratch = config.getTextValue(DMLConfig.SCRATCH_SPACE) + "/";
		}
		
		// hold all nodes in a vector (needed for ordering)
		ArrayList<Lop> node_v = new ArrayList<Lop>();
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

	private static void deleteUpdatedTransientReadVariables(StatementBlock sb, ArrayList<Lop> nodeV,
			ArrayList<Instruction> inst) throws DMLRuntimeException {

		if ( sb == null ) 
			return;
		
		if( LOG.isTraceEnabled() )
			LOG.trace("In delete updated variables");

		// CANDIDATE list of variables which could have been updated in this statement block 
		HashMap<String, Lop> labelNodeMapping = new HashMap<String, Lop>();
		
		// ACTUAL list of variables whose value is updated, AND the old value of the variable 
		// is no longer accessible/used.
		HashSet<String> updatedLabels = new HashSet<String>();
		HashMap<String, Lop> updatedLabelsLineNum =  new HashMap<String, Lop>();
		
		// first capture all transient read variables
		for ( Lop node : nodeV ) {

			if (node.getExecLocation() == ExecLocation.Data
					&& ((Data) node).isTransient()
					&& ((Data) node).getOperationType() == OperationTypes.READ
					&& ((Data) node).getDataType() == DataType.MATRIX) {
				
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
				
				if ( !hasWriteParent ) {
					// node has no parent of type WRITE, so this is a CANDIDATE variable 
					// add it to labelNodeMapping so that it is considered in further processing  
					labelNodeMapping.put(node.getOutputParameters().getLabel(), node);
				}
			}
		}

		// capture updated transient write variables
		for ( Lop node : nodeV ) {

			if (node.getExecLocation() == ExecLocation.Data
					&& ((Data) node).isTransient()
					&& ((Data) node).getOperationType() == OperationTypes.WRITE
					&& ((Data) node).getDataType() == DataType.MATRIX
					&& labelNodeMapping.containsKey(node.getOutputParameters().getLabel()) // check to make sure corresponding (i.e., with the same label/name) transient read is present
					&& !labelNodeMapping.containsValue(node.getInputs().get(0)) // check to avoid cases where transient read feeds into a transient write 
				) {
				updatedLabels.add(node.getOutputParameters().getLabel());
				updatedLabelsLineNum.put(node.getOutputParameters().getLabel(), node);
				
			}
		}
		
		// generate RM instructions
		Instruction rm_inst = null;
		for ( String label : updatedLabels ) 
		{
			rm_inst = VariableCPInstruction.prepareRemoveInstruction(label);
			rm_inst.setLocation(updatedLabelsLineNum.get(label));
			
			if( LOG.isTraceEnabled() )
				LOG.trace(rm_inst.toString());
			inst.add(rm_inst);
		}

	}
	  
	private static void generateRemoveInstructions(StatementBlock sb, ArrayList<Instruction> deleteInst)
		throws DMLRuntimeException {
		
		if ( sb == null ) 
			return;

		if( LOG.isTraceEnabled() )
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
				inst.setLocation(sb.getEndLine(), sb.getEndLine(), -1, -1);
				
				deleteInst.add(inst);

				if( LOG.isTraceEnabled() )
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

	private static ArrayList<ArrayList<Lop>> createNodeVectors(int size) {
		ArrayList<ArrayList<Lop>> arr = new ArrayList<ArrayList<Lop>>();

		// for each job type, we need to create a vector.
		// additionally, create another vector for execNodes
		for (int i = 0; i < size; i++) {
			arr.add(new ArrayList<Lop>());
		}
		return arr;
	}

	private static void clearNodeVectors(ArrayList<ArrayList<Lop>> arr) {
		for (ArrayList<Lop> tmp : arr) {
			tmp.clear();
		}
	}

	private static boolean isCompatible(ArrayList<Lop> nodes, JobType jt, int from, int to) 
		throws LopsException 
	{
		int base = jt.getBase();
		for ( Lop node : nodes ) {
			if ((node.getCompatibleJobs() & base) == 0) {
				if( LOG.isTraceEnabled() )
					LOG.trace("Not compatible "+ node.toString());
				return false;
			}
		}
		return true;
	}

	/**
	 * Function that determines if the two input nodes can be executed together 
	 * in at least on job.
	 * 
	 * @param node1
	 * @param node2
	 * @return
	 */
	private static boolean isCompatible(Lop node1, Lop node2) {
		return( (node1.getCompatibleJobs() & node2.getCompatibleJobs()) > 0);
	}
	
	/**
	 * Function that checks if the given node executes in the job specified by jt.
	 * 
	 * @param node
	 * @param jt
	 * @return
	 */
	private static boolean isCompatible(Lop node, JobType jt) {
		if ( jt == JobType.GMRCELL )
			jt = JobType.GMR;
		return ((node.getCompatibleJobs() & jt.getBase()) > 0);
	}

	/*
	 * Add node, and its relevant children to job-specific node vectors.
	 */
	private void addNodeByJobType(Lop node, ArrayList<ArrayList<Lop>> arr,
			ArrayList<Lop> execNodes, boolean eliminate) throws LopsException {
		
		if (!eliminate) {
			// Check if this lop defines a MR job.
			if ( node.definesMRJob() ) {
				
				// find the corresponding JobType
				JobType jt = JobType.findJobTypeFromLop(node);
				
				if ( jt == null ) {
					throw new LopsException(node.printErrorLocation() + "No matching JobType is found for a the lop type: " + node.getType() + " \n");
				}
				
				// Add "node" to corresponding job vector
				
				if ( jt == JobType.GMR ) {
					if ( node.hasNonBlockedInputs() ) {
						int gmrcell_index = JobType.GMRCELL.getId();
						arr.get(gmrcell_index).add(node);
						int from = arr.get(gmrcell_index).size();
						addChildren(node, arr.get(gmrcell_index), execNodes);
						int to = arr.get(gmrcell_index).size();
						if (!isCompatible(arr.get(gmrcell_index),JobType.GMR, from, to))  // check against GMR only, not against GMRCELL
							throw new LopsException(node.printErrorLocation() + "Error during compatibility check \n");
					}
					else {
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
							if (!isCompatible(arr.get(gmr_index),JobType.GMR, from, to)) 
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

		if ( eliminate ) {
			// Eliminated lops are directly added to GMR queue. 
			// Note that eliminate flag is set only for 'group' lops
			if ( node.hasNonBlockedInputs() )
				arr.get(JobType.GMRCELL.getId()).add(node);
			else
				arr.get(JobType.GMR.getId()).add(node);
			return;
		}
		
		/*
		 * If this lop does not define a job, check if it uses the output of any
		 * specialized job. i.e., if this lop has a child node in any of the
		 * job-specific vector, then add it to the vector. Note: This lop must
		 * be added to ONLY ONE of the job-specific vectors.
		 */

		int numAdded = 0;
		for ( JobType j : JobType.values() ) {
			if ( j.getId() > 0 && hasDirectChildNode(node, arr.get(j.getId()))) {
				if (isCompatible(node, j)) {
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
	private static void removeNodeByJobType(Lop node, ArrayList<ArrayList<Lop>> arr) {
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
	private void handleSingleOutputJobs(ArrayList<Lop> execNodes,
			ArrayList<ArrayList<Lop>> jobNodes, ArrayList<Lop> finishedNodes)
			throws LopsException {
		/*
		 * If the input of a MMCJ/MMRJ job (must have executed in a Mapper) is used
		 * by multiple lops then we should mark it as not-finished.
		 */
		ArrayList<Lop> nodesWithUnfinishedOutputs = new ArrayList<Lop>();
		int[] jobIndices = {JobType.MMCJ.getId()};
		Lop.Type[] lopTypes = { Lop.Type.MMCJ};
		
		// TODO: SortByValue should be treated similar to MMCJ, since it can
		// only sort one file now
		
		for ( int jobi=0; jobi < jobIndices.length; jobi++ ) {
			int jindex = jobIndices[jobi];
			if (!jobNodes.get(jindex).isEmpty()) {
				ArrayList<Lop> vec = jobNodes.get(jindex);

				// first find all nodes with more than one parent that is not finished.
				for (int i = 0; i < vec.size(); i++) {
					Lop node = vec.get(i);
					if (node.getExecLocation() == ExecLocation.MapOrReduce
							|| node.getExecLocation() == ExecLocation.Map) {
						Lop MRparent = getParentNode(node, execNodes, ExecLocation.MapAndReduce);
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
				}

				// need to redo all nodes in nodesWithOutput as well as their children
				for ( Lop node : vec ) {
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
	private static boolean canEliminateLop(Lop node, ArrayList<Lop> execNodes) {
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
	private static void generateInstructionsForInputVariables(ArrayList<Lop> nodes_v, ArrayList<Instruction> inst) throws LopsException, IOException {
		for(Lop n : nodes_v) {
			if (n.getExecLocation() == ExecLocation.Data && !((Data) n).isTransient() 
					&& ((Data) n).getOperationType() == OperationTypes.READ 
					&& (n.getDataType() == DataType.MATRIX || n.getDataType() == DataType.FRAME) ) {
				
				if ( !((Data)n).isLiteral() ) {
					try {
						String inst_string = n.getInstructions();						
						CPInstruction currInstr = CPInstructionParser.parseSingleInstruction(inst_string);
						currInstr.setLocation(n);						
						inst.add(currInstr);
					} catch (DMLRuntimeException e) {
						throw new LopsException(n.printErrorLocation() + "error generating instructions from input variables in Dag -- \n", e);
					}
				}
			}
		}
	}
	
	
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
	private static boolean sendWriteLopToMR(Lop node) 
	{
		if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE )
			return false;
		Lop in = node.getInputs().get(0);
		Format nodeFormat = node.getOutputParameters().getFormat();
		
		// Case of a transient read feeding into only one output persistent binaryblock write
		// Move the temporary file on HDFS to required persistent location, insteadof copying.
		if ( in.getExecLocation() == ExecLocation.Data && in.getOutputs().size() == 1
				&& !((Data)node).isTransient()
				&& ((Data)in).isTransient()
				&& ((Data)in).getOutputParameters().isBlocked()
				&& node.getOutputParameters().isBlocked() ) {
			return false;
		}
		
		//send write lop to MR if (1) it is marked with exec type MR (based on its memory estimate), or
		//(2) if the input lop is in MR and the write format allows to pack it into the same job (this does
		//not apply to csv write because MR csvwrite is a separate MR job type)
		if( node.getExecType() == ExecType.MR || (in.getExecType() == ExecType.MR && nodeFormat != Format.CSV ) )
			return true;
		else
			return false;
	}
	
	/**
	 * Computes the memory footprint required to execute <code>node</code> in the mapper.
	 * It is used only for those nodes that use inputs from distributed cache. The returned 
	 * value is utilized in limiting the number of instructions piggybacked onto a single GMR mapper.
	 */
	private static double computeFootprintInMapper(Lop node) {
		// Memory limits must be checked only for nodes that use distributed cache
		if ( ! node.usesDistributedCache() )
			// default behavior
			return 0.0;
		
		OutputParameters in1dims = node.getInputs().get(0).getOutputParameters();
		OutputParameters in2dims = node.getInputs().get(1).getOutputParameters();
		
		double footprint = 0;
		if ( node instanceof MapMult ) {
			int dcInputIndex = node.distributedCacheInputIndex()[0];
			footprint = AggBinaryOp.getMapmmMemEstimate(
					in1dims.getNumRows(), in1dims.getNumCols(), in1dims.getRowsInBlock(), in1dims.getColsInBlock(), in1dims.getNnz(),
					in2dims.getNumRows(), in2dims.getNumCols(), in2dims.getRowsInBlock(), in2dims.getColsInBlock(), in2dims.getNnz(),
					dcInputIndex, false);
		}
		else if ( node instanceof PMMJ ) {
			int dcInputIndex = node.distributedCacheInputIndex()[0];
			footprint = AggBinaryOp.getMapmmMemEstimate(
					in1dims.getNumRows(), 1, in1dims.getRowsInBlock(), in1dims.getColsInBlock(), in1dims.getNnz(),
					in2dims.getNumRows(), in2dims.getNumCols(), in2dims.getRowsInBlock(), in2dims.getColsInBlock(), in2dims.getNnz(), 
					dcInputIndex, true);
		}
		else if ( node instanceof AppendM ) {
			footprint = BinaryOp.footprintInMapper(
					in1dims.getNumRows(), in1dims.getNumCols(), 
					in2dims.getNumRows(), in2dims.getNumCols(), 
					in1dims.getRowsInBlock(), in1dims.getColsInBlock());
		}
		else if ( node instanceof BinaryM ) {
			footprint = BinaryOp.footprintInMapper(
					in1dims.getNumRows(), in1dims.getNumCols(), 
					in2dims.getNumRows(), in2dims.getNumCols(), 
					in1dims.getRowsInBlock(), in1dims.getColsInBlock());
		}
		else {
			// default behavior
			return 0.0;
		}
		return footprint;
	}
	
	/**
	 * Determines if <code>node</code> can be executed in current round of MR jobs or if it needs to be queued for later rounds.
	 * If the total estimated footprint (<code>node</code> and previously added nodes in GMR) is less than available memory on 
	 * the mappers then <code>node</code> can be executed in current round, and <code>true</code> is returned. Otherwise, 
	 * <code>node</code> must be queued and <code>false</code> is returned. 
	 */
	private static boolean checkMemoryLimits(Lop node, double footprintInMapper) {
		boolean addNode = true;
		
		// Memory limits must be checked only for nodes that use distributed cache
		if ( ! node.usesDistributedCache() )
			// default behavior
			return addNode;
		
		double memBudget = Math.min(AggBinaryOp.MAPMULT_MEM_MULTIPLIER, BinaryOp.APPEND_MEM_MULTIPLIER) * OptimizerUtils.getRemoteMemBudgetMap(true);
		if ( footprintInMapper <= memBudget ) 
			return addNode;
		else
			return !addNode;
	}
	
	/**
	 * Method to group a vector of sorted lops.
	 * 
	 * @param node_v
	 * @throws LopsException
	 * @throws DMLRuntimeException
	 */
	private ArrayList<Instruction> doGreedyGrouping(StatementBlock sb, ArrayList<Lop> node_v)
			throws LopsException, IOException, DMLRuntimeException
	{
		if( LOG.isTraceEnabled() )
			LOG.trace("Grouping DAG ============");

		// nodes to be executed in current iteration
		ArrayList<Lop> execNodes = new ArrayList<Lop>();
		// nodes that have already been processed
		ArrayList<Lop> finishedNodes = new ArrayList<Lop>();
		// nodes that are queued for the following iteration
		ArrayList<Lop> queuedNodes = new ArrayList<Lop>();

		ArrayList<ArrayList<Lop>> jobNodes = createNodeVectors(JobType.getNumJobTypes());
		
		// list of instructions
		ArrayList<Instruction> inst = new ArrayList<Instruction>();

		//ArrayList<Instruction> preWriteDeleteInst = new ArrayList<Instruction>();
		ArrayList<Instruction> writeInst = new ArrayList<Instruction>();
		ArrayList<Instruction> deleteInst = new ArrayList<Instruction>();
		ArrayList<Instruction> endOfBlockInst = new ArrayList<Instruction>();

		// remove files for transient reads that are updated.
		deleteUpdatedTransientReadVariables(sb, node_v, writeInst);
		
		generateRemoveInstructions(sb, endOfBlockInst);

		generateInstructionsForInputVariables(node_v, inst);
		
		
		boolean done = false;
		String indent = "    ";

		while (!done) {
			if( LOG.isTraceEnabled() )
				LOG.trace("Grouping nodes in DAG");

			execNodes.clear();
			queuedNodes.clear();
			clearNodeVectors(jobNodes);
			gmrMapperFootprint=0;

			for ( Lop  node : node_v ) {
				// finished nodes don't need to be processed
				if (finishedNodes.contains(node))
					continue;

				if( LOG.isTraceEnabled() )
					LOG.trace("Processing node (" + node.getID()
							+ ") " + node.toString() + " exec nodes size is " + execNodes.size());
				
				
		        //if node defines MR job, make sure it is compatible with all 
		        //its children nodes in execNodes 
		        if(node.definesMRJob() && !compatibleWithChildrenInExecNodes(execNodes, node))
		        {
		        	if( LOG.isTraceEnabled() )			
			          LOG.trace(indent + "Queueing node "
			                + node.toString() + " (code 1)");
			
		          queuedNodes.add(node);
		          removeNodesForNextIteration(node, finishedNodes, execNodes, queuedNodes, jobNodes);
		          continue;
		        }

				// if child is queued, this node will be processed in the later
				// iteration
				if (hasChildNode(node,queuedNodes)) {

					if( LOG.isTraceEnabled() )
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
				if ( node.getInputs().size() >= 2) {
					int jobid = Integer.MIN_VALUE;
					boolean queueit = false;
					for(int idx=0; idx < node.getInputs().size(); idx++) {
						int input_jobid = jobType(node.getInputs().get(idx), jobNodes);
						if (input_jobid != -1) {
							if ( jobid == Integer.MIN_VALUE )
								jobid = input_jobid;
							else if ( jobid != input_jobid ) { 
								queueit = true;
								break;
							}
						}
					}
					if ( queueit ) {
						if( LOG.isTraceEnabled() )
							LOG.trace(indent + "Queueing node " + node.toString() + " (code 3)");
						queuedNodes.add(node);
						removeNodesForNextIteration(node, finishedNodes, execNodes, queuedNodes, jobNodes);
						continue;
					}
				}

				// See if this lop can be eliminated
				// This check is for "aligner" lops (e.g., group)
				boolean eliminate = false;
				eliminate = canEliminateLop(node, execNodes);
				if (eliminate) {
					if( LOG.isTraceEnabled() )
						LOG.trace(indent + "Adding -"+ node.toString());
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
						if (! (node.getType() == Lop.Type.Grouping && checkDataGenAsChildNode(node,execNodes))  ) {
							if( LOG.isTraceEnabled() )
								LOG.trace(indent + "Queueing node " + node.toString() + " (code 4)");

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
					Lop rr_node = getChildNode(node, execNodes, ExecLocation.RecordReader);

					// all inputs of "node" must be ancestors of rr_node
					boolean queue_it = false;
					for (Lop n : node.getInputs()) {
						// each input should be ancestor of RecordReader lop
						if (!n.equals(rr_node) && !isChild(rr_node, n, IDMap)) {
							queue_it = true; // i.e., "node" must be queued
							break;
						}
					}

					if (queue_it) {
						// queue node
						if( LOG.isTraceEnabled() )
							LOG.trace(indent + "Queueing -" + node.toString() + " (code 5)");
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
					Data dnode = (Data) node;
					boolean dnode_queued = false;
					
					if ( dnode.getOperationType() == OperationTypes.READ ) {
						if( LOG.isTraceEnabled() )
							LOG.trace(indent + "Adding Data -"+ node.toString());

						// TODO: avoid readScalar instruction, and read it on-demand just like the way Matrices are read in control program
						if ( node.getDataType() == DataType.SCALAR 
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
						Lop input = dnode.getInputs().get(0);
						if ( dnode.isTransient() 
								&& input.getExecLocation() == ExecLocation.Data 
								&& ((Data)input).isTransient() 
								&& dnode.getOutputParameters().getLabel().equals(input.getOutputParameters().getLabel()) ) {
							// do nothing, <code>node</code> must not processed any further.
							;
						}
						else if ( execNodes.contains(input) && !isCompatible(node, input) && sendWriteLopToMR(node)) {
							// input is in execNodes but it is not compatible with write lop. So, queue the write lop.
							if( LOG.isTraceEnabled() )
								LOG.trace(indent + "Queueing -" + node.toString());
							queuedNodes.add(node);
							dnode_queued = true;
						}
						else {
							if( LOG.isTraceEnabled() )
								LOG.trace(indent + "Adding Data -"+ node.toString());

							execNodes.add(node);
							if ( sendWriteLopToMR(node) ) {
								addNodeByJobType(node, jobNodes, execNodes, false);
							}
						}
					}
					if (!dnode_queued)
						finishedNodes.add(node);

					continue;
				}

				// map or reduce node, can always be piggybacked with parent
				if (node.getExecLocation() == ExecLocation.MapOrReduce) {
					if( LOG.isTraceEnabled() )
						LOG.trace(indent + "Adding -"+ node.toString());
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
						if( LOG.isTraceEnabled() )
							LOG.trace(indent + "Adding -"+ node.toString());
						execNodes.add(node);
						finishedNodes.add(node);
						addNodeByJobType(node, jobNodes, execNodes, false);
					} else {
						if( LOG.isTraceEnabled() )
							LOG.trace(indent + "Queueing -"+ node.toString() + " (code 6)");
						queuedNodes.add(node);
						removeNodesForNextIteration(node, finishedNodes,
								execNodes, queuedNodes, jobNodes);

					}
					continue;
				}

				// map node, add, if no parent needs reduce, else queue
				if (node.getExecLocation() == ExecLocation.Map) {
					boolean queueThisNode = false;
					int subcode = -1;
					if ( node.usesDistributedCache() ) {
						// if an input to <code>node</code> comes from distributed cache
						// then that input must get executed in one of the previous jobs.
						int[] dcInputIndexes = node.distributedCacheInputIndex();
						for( int dcInputIndex : dcInputIndexes ){
							Lop dcInput = node.getInputs().get(dcInputIndex-1);
							if ( (dcInput.getType() != Lop.Type.Data && dcInput.getExecType()==ExecType.MR)
								  &&  execNodes.contains(dcInput) )
							{
								queueThisNode = true;
								subcode = 1;
							}
						}
						
						// Limit the number of distributed cache inputs based on the available memory in mappers
						double memsize = computeFootprintInMapper(node);
						//gmrMapperFootprint += computeFootprintInMapper(node);
						if ( gmrMapperFootprint>0 && !checkMemoryLimits(node, gmrMapperFootprint+memsize ) ) {
							queueThisNode = true;
							subcode = 2;
						}
						if(!queueThisNode)
							gmrMapperFootprint += memsize;
					}
					if (!queueThisNode && !hasChildNode(node, execNodes,ExecLocation.MapAndReduce)&& !hasMRJobChildNode(node, execNodes)) {
						if( LOG.isTraceEnabled() )
							LOG.trace(indent + "Adding -"+ node.toString());
						execNodes.add(node);
						finishedNodes.add(node);
						addNodeByJobType(node, jobNodes, execNodes, false);
					} else {
						if( LOG.isTraceEnabled() )
							LOG.trace(indent + "Queueing -"+ node.toString() + " (code 7 - " + "subcode " + subcode + ")");
						queuedNodes.add(node);
						removeNodesForNextIteration(node, finishedNodes,
								execNodes, queuedNodes, jobNodes);

					}
					continue;
				}

				// reduce node, make sure no parent needs reduce, else queue
				if (node.getExecLocation() == ExecLocation.MapAndReduce) {
					// TODO: statiko -- keep the middle condition
					// discuss about having a lop that is MapAndReduce but does
					// not define a job
					if( LOG.isTraceEnabled() )
						LOG.trace(indent + "Adding -"+ node.toString());
					execNodes.add(node);
					finishedNodes.add(node);
					addNodeByJobType(node, jobNodes, execNodes, eliminate);
					
					continue;
				}

				// aligned reduce, make sure a parent that is reduce exists
				if (node.getExecLocation() == ExecLocation.Reduce) {
					if (  compatibleWithChildrenInExecNodes(execNodes, node) && 
							(hasChildNode(node, execNodes, ExecLocation.MapAndReduce)
							 || hasChildNode(node, execNodes, ExecLocation.Map) ) ) 
					{ 
						if( LOG.isTraceEnabled() )
							LOG.trace(indent + "Adding -"+ node.toString());
						execNodes.add(node);
						finishedNodes.add(node);
						addNodeByJobType(node, jobNodes, execNodes, false);
					} else {
						if( LOG.isTraceEnabled() )
							LOG.trace(indent + "Queueing -"+ node.toString() + " (code 8)");
						queuedNodes.add(node);
						removeNodesForNextIteration(node, finishedNodes,
								execNodes, queuedNodes, jobNodes);
					}

					continue;

				}

				// add Scalar to execNodes if it has no child in exec nodes
				// that will be executed in a MR job.
				if (node.getExecLocation() == ExecLocation.ControlProgram) {
					for ( Lop lop : node.getInputs() ) {
						if (execNodes.contains(lop)
								&& !(lop.getExecLocation() == ExecLocation.Data)
								&& !(lop.getExecLocation() == ExecLocation.ControlProgram)) {
							if( LOG.isTraceEnabled() )
								LOG.trace(indent + "Queueing -"+ node.toString() + " (code 9)");

							queuedNodes.add(node);
							removeNodesForNextIteration(node, finishedNodes,
									execNodes, queuedNodes, jobNodes);
							break;
						}
					}

					if (queuedNodes.contains(node))
						continue;
					if( LOG.isTraceEnabled() )
						LOG.trace(indent + "Adding - scalar"+ node.toString());
					execNodes.add(node);
					addNodeByJobType(node, jobNodes, execNodes, false);
					finishedNodes.add(node);
					continue;
				}

			}

			// no work to do
			if ( execNodes.isEmpty() ) {
			  
			  if( !queuedNodes.isEmpty() )
			  {
			      //System.err.println("Queued nodes should be 0");
			      throw new LopsException("Queued nodes should not be 0 at this point \n");
			  }
			  
			  if( LOG.isTraceEnabled() )
				LOG.trace("All done! queuedNodes = "+ queuedNodes.size());
				
			  done = true;
			} else {
				// work to do

				if( LOG.isTraceEnabled() )
					LOG.trace("Generating jobs for group -- Node count="+ execNodes.size());

				// first process scalar instructions
				generateControlProgramJobs(execNodes, inst, writeInst, deleteInst);

				// copy unassigned lops in execnodes to gmrnodes
				for (int i = 0; i < execNodes.size(); i++) {
					Lop node = execNodes.get(i);
					if (jobType(node, jobNodes) == -1) {
						if ( isCompatible(node,  JobType.GMR) ) {
							if ( node.hasNonBlockedInputs() ) {
								jobNodes.get(JobType.GMRCELL.getId()).add(node);
								addChildren(node, jobNodes.get(JobType.GMRCELL.getId()), execNodes);
							}
							else {
								jobNodes.get(JobType.GMR.getId()).add(node);
								addChildren(node, jobNodes.get(JobType.GMR.getId()), execNodes);
							}
						}
						else {
							if( LOG.isTraceEnabled() )
								LOG.trace(indent + "Queueing -" + node.toString() + " (code 10)");
							execNodes.remove(i);
							finishedNodes.remove(node);
							queuedNodes.add(node);
							removeNodesForNextIteration(node, finishedNodes,
								execNodes, queuedNodes, jobNodes);
						}
					}
				}

				// next generate MR instructions
				if (!execNodes.isEmpty())
					generateMRJobs(execNodes, inst, writeInst, deleteInst, jobNodes);
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

	private boolean compatibleWithChildrenInExecNodes(ArrayList<Lop> execNodes, Lop node) {
	  for( Lop tmpNode : execNodes ) {
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
	private static void excludeRemoveInstruction(String varName, ArrayList<Instruction> deleteInst) {
		//for(Instruction inst : deleteInst) {
		for(int i=0; i < deleteInst.size(); i++) {
			Instruction inst = deleteInst.get(i);
			if ((inst.getType() == INSTRUCTION_TYPE.CONTROL_PROGRAM  || inst.getType() == INSTRUCTION_TYPE.SPARK)
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
	 */
	private void processConsumersForInputs(Lop node, ArrayList<Instruction> inst, ArrayList<Instruction> delteInst) throws DMLRuntimeException {
		// reduce the consumer count for all input lops
		// if the count becomes zero, then then variable associated w/ input can be removed
		for(Lop in : node.getInputs() ) {
			if(DMLScript.ENABLE_DEBUG_MODE) {
				processConsumers(in, inst, delteInst, node);
			}
			else {
				processConsumers(in, inst, delteInst, null);
			}
		}
	}
	
	private static void processConsumers(Lop node, ArrayList<Instruction> inst, ArrayList<Instruction> deleteInst, Lop locationInfo) throws DMLRuntimeException {
		// reduce the consumer count for all input lops
		// if the count becomes zero, then then variable associated w/ input can be removed
		if ( node.removeConsumer() == 0 ) {
			if ( node.getExecLocation() == ExecLocation.Data && ((Data)node).isLiteral() ) {
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
	 * @param execNodes
	 * @param inst
	 * @param deleteInst
	 * @throws LopsException
	 * @throws DMLRuntimeException
	 */
	private void generateControlProgramJobs(ArrayList<Lop> execNodes,
			ArrayList<Instruction> inst, ArrayList<Instruction> writeInst, ArrayList<Instruction> deleteInst) throws LopsException, DMLRuntimeException {

		// nodes to be deleted from execnodes
		ArrayList<Lop> markedNodes = new ArrayList<Lop>();

		// variable names to be deleted
		ArrayList<String> var_deletions = new ArrayList<String>();
		HashMap<String, Lop> var_deletionsLineNum =  new HashMap<String, Lop>();
		
		boolean doRmVar = false;

		for (int i = 0; i < execNodes.size(); i++) {
			Lop node = execNodes.get(i);
			doRmVar = false;

			// mark input scalar read nodes for deletion
			// TODO: statiko -- check if this condition ever evaluated to TRUE
			if (node.getExecLocation() == ExecLocation.Data
					&& ((Data) node).getOperationType() == Data.OperationTypes.READ
					&& ((Data) node).getDataType() == DataType.SCALAR 
					&& node.getOutputParameters().getFile_name() == null ) {
				markedNodes.add(node);
				continue;
			}
			
			// output scalar instructions and mark nodes for deletion
			if (node.getExecLocation() == ExecLocation.ControlProgram) {

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
						var_deletionsLineNum.put(node.getOutputParameters().getLabel(), node);
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
					else if (node.getInputs().size() == 3 || node.getType() == Type.Ternary) {
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
					else if (node.getInputs().size() == 13) {
						 // Used for im2col and reshape_col
						 inst_string = node.getInstructions(
						 		node.getInputs().get(0).getOutputParameters().getLabel(),
						 		node.getInputs().get(1).getOutputParameters().getLabel(),
						 		node.getInputs().get(2).getOutputParameters().getLabel(),
						 		node.getInputs().get(3).getOutputParameters().getLabel(),
						 		node.getInputs().get(4).getOutputParameters().getLabel(),
						 		node.getInputs().get(5).getOutputParameters().getLabel(),
						 		node.getInputs().get(6).getOutputParameters().getLabel(),
						 		node.getInputs().get(7).getOutputParameters().getLabel(),
						 		node.getInputs().get(8).getOutputParameters().getLabel(),
						 		node.getInputs().get(9).getOutputParameters().getLabel(),
						 		node.getInputs().get(10).getOutputParameters().getLabel(),
						 		node.getInputs().get(11).getOutputParameters().getLabel(),
						 		node.getInputs().get(12).getOutputParameters().getLabel(),
						 		node.getOutputParameters().getLabel());
					}
					else if (node.getInputs().size() == 14) {
						 // Used for pooling_backward
						 inst_string = node.getInstructions(
						 		node.getInputs().get(0).getOutputParameters().getLabel(),
						 		node.getInputs().get(1).getOutputParameters().getLabel(),
						 		node.getInputs().get(2).getOutputParameters().getLabel(),
						 		node.getInputs().get(3).getOutputParameters().getLabel(),
						 		node.getInputs().get(4).getOutputParameters().getLabel(),
						 		node.getInputs().get(5).getOutputParameters().getLabel(),
						 		node.getInputs().get(6).getOutputParameters().getLabel(),
						 		node.getInputs().get(7).getOutputParameters().getLabel(),
						 		node.getInputs().get(8).getOutputParameters().getLabel(),
						 		node.getInputs().get(9).getOutputParameters().getLabel(),
						 		node.getInputs().get(10).getOutputParameters().getLabel(),
						 		node.getInputs().get(11).getOutputParameters().getLabel(),
						 		node.getInputs().get(12).getOutputParameters().getLabel(),
						 		node.getInputs().get(13).getOutputParameters().getLabel(),
						 		node.getOutputParameters().getLabel());
					}
					else {
						throw new LopsException(node.printErrorLocation() + "Node with " + node.getInputs().size() + " inputs is not supported in CP yet! \n");
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
						out = setupNodeOutputs(node, ExecType.CP, false, false);
						if ( dnode.getDataType() == DataType.SCALAR ) {
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
								// In case of persistent write lop, write instruction will be generated 
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
	 * Method to remove all child nodes of a queued node that should be executed
	 * in a following iteration.
	 * 
	 * @param node
	 * @param finishedNodes
	 * @param execNodes
	 * @param queuedNodes
	 * @throws LopsException
	 */
	private void removeNodesForNextIteration(Lop node, ArrayList<Lop> finishedNodes,
			ArrayList<Lop> execNodes, ArrayList<Lop> queuedNodes,
			ArrayList<ArrayList<Lop>> jobvec) throws LopsException {
		
		// only queued nodes with multiple inputs need to be handled.
		if (node.getInputs().size() == 1)
			return;
		
		//if all children are queued, then there is nothing to do.
		boolean allQueued = true;
		for( Lop input : node.getInputs() ) {
			if( !queuedNodes.contains(input) ) {
				allQueued = false;
				break;
			}
		}
		if ( allQueued )
			return; 
		
		if( LOG.isTraceEnabled() )
			LOG.trace("  Before remove nodes for next iteration -- size of execNodes " + execNodes.size());

		// Determine if <code>node</code> has inputs from the same job or multiple jobs
	    int jobid = Integer.MIN_VALUE;
		boolean inputs_in_same_job = true;
		for( Lop input : node.getInputs() ) {
			int input_jobid = jobType(input, jobvec);
			if ( jobid == Integer.MIN_VALUE )
				jobid = input_jobid;
			else if ( jobid != input_jobid ) { 
				inputs_in_same_job = false;
				break;
			}
		}

		// Determine if there exist any unassigned inputs to <code>node</code>
		// Evaluate only those lops that execute in MR.
		boolean unassigned_inputs = false;
		for( Lop input : node.getInputs() ) {
			//if ( input.getExecLocation() != ExecLocation.ControlProgram && jobType(input, jobvec) == -1 ) {
			if ( input.getExecType() == ExecType.MR && !execNodes.contains(input)) { //jobType(input, jobvec) == -1 ) {
				unassigned_inputs = true;
				break;
			}
		}

		// Determine if any node's children are queued
		boolean child_queued = false;
		for( Lop input : node.getInputs() ) {
			if (queuedNodes.contains(input) ) {
				child_queued = true;
				break;
			}
		}
		if (LOG.isTraceEnabled()) {
			LOG.trace("  Property Flags:");
			LOG.trace("    Inputs in same job: " + inputs_in_same_job);
			LOG.trace("    Unassigned inputs: " + unassigned_inputs);
			LOG.trace("    Child queued: " + child_queued);
		}

		// Evaluate each lop in <code>execNodes</code> for removal.
		// Add lops to be removed to <code>markedNodes</code>.
		
		ArrayList<Lop> markedNodes = new ArrayList<Lop>();
		for (Lop tmpNode : execNodes ) {

			if (LOG.isTraceEnabled()) {
				LOG.trace("  Checking for removal (" + tmpNode.getID() + ") " + tmpNode.toString());
			}
			
			// if tmpNode is not a descendant of 'node', then there is no advantage in removing tmpNode for later iterations.
			if(!isChild(tmpNode, node, IDMap))
				continue;
			
			// handle group input lops
			if(node.getInputs().contains(tmpNode) && tmpNode.isAligner()) {
			    markedNodes.add(tmpNode);
			    if( LOG.isTraceEnabled() )
			    	LOG.trace("    Removing for next iteration (code 1): (" + tmpNode.getID() + ") " + tmpNode.toString());
			}
			
			//if (child_queued) {
				// if one of the children are queued, 
				// remove some child nodes on other leg that may be needed later on. 
				// For e.g. Group lop. 
				
				if (!hasOtherQueuedParentNode(tmpNode, queuedNodes, node) 
					&& branchHasNoOtherUnExecutedParents(tmpNode, node, execNodes, finishedNodes)) {
					
					boolean queueit = false;
					int code = -1;
					switch(node.getExecLocation()) {
					case Map:
						if(branchCanBePiggyBackedMap(tmpNode, node, execNodes, queuedNodes, markedNodes))
							queueit = true;
						code=2;
						break;
						
					case MapAndReduce:
						if(branchCanBePiggyBackedMapAndReduce(tmpNode, node, execNodes, queuedNodes)&& !tmpNode.definesMRJob()) 
							queueit = true;
						code=3;
						break;
					case Reduce:
						if(branchCanBePiggyBackedReduce(tmpNode, node, execNodes, queuedNodes))
							queueit = true;
						code=4;
						break;
					default:
						//do nothing
					}
					
					if(queueit) {
						if( LOG.isTraceEnabled() )
							LOG.trace("    Removing for next iteration (code " + code + "): (" + tmpNode.getID() + ") " + tmpNode.toString());
			        		
						markedNodes.add(tmpNode);
					}
				}
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
				//boolean himr = hasOtherMapAndReduceParentNode(tmpNode, execNodes,node);
				//boolean bcbp = branchCanBePiggyBackedMapAndReduce(tmpNode, node, execNodes, finishedNodes);
				//System.out.println("      .. " + inputs_in_same_job + "," + himr + "," + bcbp);
				if ((inputs_in_same_job || unassigned_inputs)
						&& node.getExecLocation() == ExecLocation.MapAndReduce
						&& !hasOtherMapAndReduceParentNode(tmpNode, execNodes,node)  // don't remove since it already piggybacked with a MapReduce node
						&& branchCanBePiggyBackedMapAndReduce(tmpNode, node, execNodes, queuedNodes)
						&& !tmpNode.definesMRJob()) {
					if( LOG.isTraceEnabled() )
						LOG.trace("    Removing for next iteration (code 5): ("+ tmpNode.getID() + ") " + tmpNode.toString());

					markedNodes.add(tmpNode);
				}
		} // for i

		// we also need to delete all parent nodes of marked nodes
		for ( Lop enode : execNodes ) {
			if( LOG.isTraceEnabled() ) {
				LOG.trace("  Checking for removal - ("
							+ enode.getID() + ") " + enode.toString());
			}
			
			if (hasChildNode(enode, markedNodes) && !markedNodes.contains(enode)) {
				markedNodes.add(enode);
				if( LOG.isTraceEnabled() )
					LOG.trace("    Removing for next iteration (code 6) (" + enode.getID() + ") " + enode.toString());
			}
		}

		if ( execNodes.size() != markedNodes.size() ) {
			// delete marked nodes from finishedNodes and execNodes
			// add to queued nodes
			for(Lop n : markedNodes) {
				if ( n.usesDistributedCache() )
					gmrMapperFootprint -= computeFootprintInMapper(n);
				finishedNodes.remove(n);
				execNodes.remove(n);
				removeNodeByJobType(n, jobvec);
				queuedNodes.add(n);
			}
		}
	}

	private boolean branchCanBePiggyBackedReduce(Lop tmpNode, Lop node, ArrayList<Lop> execNodes, ArrayList<Lop> queuedNodes) {
		if(node.getExecLocation() != ExecLocation.Reduce)
			return false;
	    
		// if tmpNode is descendant of any queued child of node, then branch can not be piggybacked
		for(Lop ni : node.getInputs()) {
			if(queuedNodes.contains(ni) && isChild(tmpNode, ni, IDMap))
				return false;
		}
		
		for( Lop n : execNodes ) {
		   if(n.equals(node))
			   continue;
       
		   if(n.equals(tmpNode) && n.getExecLocation() != ExecLocation.Map && n.getExecLocation() != ExecLocation.MapOrReduce)
			   return false;
       
		   // check if n is on the branch tmpNode->*->node
		   if(isChild(n, node, IDMap) && isChild(tmpNode, n, IDMap)) {
			   if(!node.getInputs().contains(tmpNode) // redundant
				   && n.getExecLocation() != ExecLocation.Map && n.getExecLocation() != ExecLocation.MapOrReduce)
				   return false;
		   }
	   }
	   return true;
	}

	private boolean branchCanBePiggyBackedMap(Lop tmpNode, Lop node, ArrayList<Lop> execNodes, ArrayList<Lop> queuedNodes, ArrayList<Lop> markedNodes) {
		if(node.getExecLocation() != ExecLocation.Map)
			return false;
		
		// if tmpNode is descendant of any queued child of node, then branch can not be piggybacked
		for(Lop ni : node.getInputs()) {
			if(queuedNodes != null && queuedNodes.contains(ni) && isChild(tmpNode, ni, IDMap))
				return false;
		}
		
		// since node.location=Map: only Map & MapOrReduce lops must be considered
		if( tmpNode.definesMRJob() || (tmpNode.getExecLocation() != ExecLocation.Map && tmpNode.getExecLocation() != ExecLocation.MapOrReduce))
			return false;

		// if there exist a node "dcInput" that is 
		//   -- a) parent of tmpNode, and b) feeds into "node" via distributed cache
		//   then, tmpNode should not be removed.
		// "dcInput" must be executed prior to "node", and removal of tmpNode does not make that happen.
		if(node.usesDistributedCache() ) {
			for(int dcInputIndex : node.distributedCacheInputIndex()) { 
				Lop dcInput = node.getInputs().get(dcInputIndex-1);
				if(isChild(tmpNode, dcInput, IDMap))
					return false;
			}
		}
		
		// if tmpNode requires an input from distributed cache,
		//   remove tmpNode only if that input can fit into mappers' memory. If not, 
		if ( tmpNode.usesDistributedCache() ) {
			double memsize = computeFootprintInMapper(tmpNode);
			if (node.usesDistributedCache() )
				memsize += computeFootprintInMapper(node);
			if ( markedNodes != null ) {
				for(Lop n : markedNodes) {
					if ( n.usesDistributedCache() ) 
						memsize += computeFootprintInMapper(n);
				}
			}
			if ( !checkMemoryLimits(node, memsize ) ) {
				return false;
			}
		}
		
		if( (tmpNode.getCompatibleJobs() & node.getCompatibleJobs()) > 0)
			return true;
		else
			return false;
			
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
	private boolean branchCanBePiggyBackedMapAndReduce(Lop tmpNode, Lop node,
			ArrayList<Lop> execNodes, ArrayList<Lop> queuedNodes) {

		if (node.getExecLocation() != ExecLocation.MapAndReduce)
			return false;
		JobType jt = JobType.findJobTypeFromLop(node);

		for ( Lop n : execNodes ) {
			if (n.equals(node))
				continue;

			// Evaluate only nodes on the branch between tmpNode->..->node
			if (n.equals(tmpNode) || (isChild(n, node, IDMap) && isChild(tmpNode, n, IDMap))) {
				if ( hasOtherMapAndReduceParentNode(tmpNode, queuedNodes,node) )
					return false;
				ExecLocation el = n.getExecLocation();
				if (el != ExecLocation.Map && el != ExecLocation.MapOrReduce)
					return false;
				else if (!isCompatible(n, jt))
					return false;
			}
		}
		return true;
	}

  private boolean branchHasNoOtherUnExecutedParents(Lop tmpNode, Lop node,
      ArrayList<Lop> execNodes, ArrayList<Lop> finishedNodes) {

	  //if tmpNode has more than one unfinished output, return false 
	  if(tmpNode.getOutputs().size() > 1)
	  {
	    int cnt = 0;
	    for (Lop output : tmpNode.getOutputs() )
	    	if (!finishedNodes.contains(output))
	    		cnt++; 
	    
	    if(cnt != 1)
	      return false;
	  }
	  
	  //check to see if any node between node and tmpNode has more than one unfinished output 
	  for( Lop n : execNodes ) {
	    if(n.equals(node) || n.equals(tmpNode))
	      continue;
	    
	    if(isChild(n, node, IDMap) && isChild(tmpNode, n, IDMap))
	    {      
	      int cnt = 0;
	      for (Lop output : n.getOutputs() ) {
	        if (!finishedNodes.contains(output))    
	          cnt++;
	      } 
	      
	      if(cnt != 1)
	        return false;
	    }
	  }
	  
	  return true;
  }

	/**
	 * Method to return the job index for a lop.
	 * 
	 * @param lops
	 * @param jobvec
	 * @return
	 * @throws LopsException
	 */
	private static int jobType(Lop lops, ArrayList<ArrayList<Lop>> jobvec) throws LopsException {
		for ( JobType jt : JobType.values()) {
			int i = jt.getId();
			if (i > 0 && jobvec.get(i) != null && jobvec.get(i).contains(lops)) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * Method to see if there is a node of type MapAndReduce between tmpNode and node
	 * in given node collection
	 * 
	 * @param tmpNode
	 * @param nodeList
	 * @param node
	 * @return
	 */
	private boolean hasOtherMapAndReduceParentNode(Lop tmpNode,
			ArrayList<Lop> nodeList, Lop node) {
		
		if ( tmpNode.getExecLocation() == ExecLocation.MapAndReduce)
			return true;
		
		for ( Lop n : tmpNode.getOutputs() ) {
			if ( nodeList.contains(n) && isChild(n,node,IDMap)) {
				if(!n.equals(node) && n.getExecLocation() == ExecLocation.MapAndReduce)
					return true;
				else
					return hasOtherMapAndReduceParentNode(n, nodeList, node);
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
	private boolean hasOtherQueuedParentNode(Lop tmpNode, ArrayList<Lop> queuedNodes, Lop node) {
		if ( queuedNodes.isEmpty() )
			return false;
		
		boolean[] nodeMarked = node.get_reachable();
		boolean[] tmpMarked  = tmpNode.get_reachable();
		long nodeid = IDMap.get(node.getID());
		long tmpid = IDMap.get(tmpNode.getID());
		
		for ( Lop qnode : queuedNodes ) {
			int id = IDMap.get(qnode.getID());
			if ((id != nodeid && nodeMarked[id]) && (id != tmpid && tmpMarked[id]) )
				return true;
		}
		
		return false;
	}

	/**
	 * Method to print the lops grouped by job type
	 * 
	 * @param jobNodes
	 * @throws DMLRuntimeException
	 */
	private static void printJobNodes(ArrayList<ArrayList<Lop>> jobNodes)
		throws DMLRuntimeException 
	{
		if (LOG.isTraceEnabled()){
			for ( JobType jt : JobType.values() ) {
				int i = jt.getId();
				if (i > 0 && jobNodes.get(i) != null && !jobNodes.get(i).isEmpty() ) {
					LOG.trace(jt.getName() + " Job Nodes:");
					
					for (int j = 0; j < jobNodes.get(i).size(); j++) {
						LOG.trace("    "
								+ jobNodes.get(i).get(j).getID() + ") "
								+ jobNodes.get(i).get(j).toString());
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
	private static boolean hasANode(ArrayList<Lop> nodes, ExecLocation loc) {
		for ( Lop n : nodes ) {
			if (n.getExecLocation() == ExecLocation.RecordReader)
				return true;
		}
		return false;
	}

	private ArrayList<ArrayList<Lop>> splitGMRNodesByRecordReader(ArrayList<Lop> gmrnodes) 
	{
		// obtain the list of record reader nodes
		ArrayList<Lop> rrnodes = new ArrayList<Lop>();
		for (Lop gmrnode : gmrnodes ) {
			if (gmrnode.getExecLocation() == ExecLocation.RecordReader)
				rrnodes.add(gmrnode);
		}

		// We allocate one extra vector to hold lops that do not depend on any
		// recordreader lops
		ArrayList<ArrayList<Lop>> splitGMR = createNodeVectors(rrnodes.size() + 1);

		// flags to indicate whether a lop has been added to one of the node vectors
		boolean[] flags = new boolean[gmrnodes.size()];
		Arrays.fill(flags, false);
		
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
			if (!flags[i]) {
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
	 */
	private void generateMRJobs(ArrayList<Lop> execNodes,
			ArrayList<Instruction> inst,
			ArrayList<Instruction> writeinst,
			ArrayList<Instruction> deleteinst, ArrayList<ArrayList<Lop>> jobNodes)
			throws LopsException, DMLRuntimeException
	{
		printJobNodes(jobNodes);
		
		ArrayList<Instruction> rmvarinst = new ArrayList<Instruction>();
		for (JobType jt : JobType.values()) { 
			
			// do nothing, if jt = INVALID or ANY
			if ( jt == JobType.INVALID || jt == JobType.ANY )
				continue;
			
			int index = jt.getId(); // job id is used as an index into jobNodes
			ArrayList<Lop> currNodes = jobNodes.get(index);
			
			// generate MR job
			if (currNodes != null && !currNodes.isEmpty() ) {

				if( LOG.isTraceEnabled() )
					LOG.trace("Generating " + jt.getName() + " job");

				if (jt.allowsRecordReaderInstructions() && hasANode(jobNodes.get(index), ExecLocation.RecordReader)) {
					// split the nodes by recordReader lops
					ArrayList<ArrayList<Lop>> rrlist = splitGMRNodesByRecordReader(jobNodes.get(index));
					for (int i = 0; i < rrlist.size(); i++) {
						generateMapReduceInstructions(rrlist.get(i), inst, writeinst, deleteinst, rmvarinst, jt);
					}
				}
				else if ( jt.allowsSingleShuffleInstruction() ) {
					// These jobs allow a single shuffle instruction. 
					// We should split the nodes so that a separate job is produced for each shuffle instruction.
					Lop.Type splittingLopType = jt.getShuffleLopType();
					
					ArrayList<Lop> nodesForASingleJob = new ArrayList<Lop>();
					for (int i = 0; i < jobNodes.get(index).size(); i++) {
						if (jobNodes.get(index).get(i).getType() == splittingLopType) {
							nodesForASingleJob.clear();
							
							// Add the lop that defines the split 
							nodesForASingleJob.add(jobNodes.get(index).get(i));
							
							/*
							 * Add the splitting lop's children. This call is redundant when jt=SORT
							 * because a sort job ALWAYS has a SINGLE lop in the entire job
							 * i.e., there are no children to add when jt=SORT. 
							 */
							addChildren(jobNodes.get(index).get(i), nodesForASingleJob, jobNodes.get(index));
							
							if ( jt.isCompatibleWithParentNodes() ) {
								/*
								 * If the splitting lop is compatible with parent nodes 
								 * then they must be added to the job. For example, MMRJ lop 
								 * may have a Data(Write) lop as its parent, which can be 
								 * executed along with MMRJ.
								 */
								addParents(jobNodes.get(index).get(i), nodesForASingleJob, jobNodes.get(index));
							}
							
							generateMapReduceInstructions(nodesForASingleJob, inst, writeinst, deleteinst, rmvarinst, jt);
						}
					}
				}
				else {
					// the default case
					generateMapReduceInstructions(jobNodes.get(index), inst, writeinst, deleteinst, rmvarinst, jt);
				}
			}
		}
		inst.addAll(rmvarinst);

	}

	/**
	 * Method to add all parents of "node" in exec_n to node_v.
	 * 
	 * @param node
	 * @param node_v
	 * @param exec_n
	 */
	private void addParents(Lop node, ArrayList<Lop> node_v, ArrayList<Lop> exec_n) {
		for (Lop enode : exec_n ) {
			if (isChild(node, enode, IDMap)) {
				if (!node_v.contains(enode)) {
					if( LOG.isTraceEnabled() )
						LOG.trace("Adding parent - " + enode.toString());
					node_v.add(enode);
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
	private static void addChildren(Lop node, ArrayList<Lop> node_v, ArrayList<Lop> exec_n) {

		// add child in exec nodes that is not of type scalar
		if (exec_n.contains(node)
				&& node.getExecLocation() != ExecLocation.ControlProgram) {
			if (!node_v.contains(node)) {
				node_v.add(node);
				if(LOG.isTraceEnabled())
					LOG.trace("      Added child " + node.toString());
			}

		}

		if (!exec_n.contains(node))
			return;

		// recurse
		for (Lop n : node.getInputs() ) {
			addChildren(n, node_v, exec_n);
		}
	}
	
	/**
	 * Method that determines the output format for a given node.
	 * 
	 * @param node
	 * @return
	 * @throws LopsException 
	 */
	private static OutputInfo getOutputInfo(Lop node, boolean cellModeOverride) 
		throws LopsException 
	{
		if ( (node.getDataType() == DataType.SCALAR && node.getExecType() == ExecType.CP) 
				|| node instanceof FunctionCallCP )
			return null;
	
		OutputInfo oinfo = null;
		OutputParameters oparams = node.getOutputParameters();
		
		if (oparams.isBlocked()) {
			if ( !cellModeOverride ) {
				oinfo = OutputInfo.BinaryBlockOutputInfo;
			}
			else {
				// output format is overridden, for example, due to recordReaderInstructions in the job
				oinfo = OutputInfo.BinaryCellOutputInfo;
				
				// record decision of overriding in lop's outputParameters so that 
				// subsequent jobs that use this lop's output know the correct format.
				// TODO: ideally, this should be done by having a member variable in Lop 
				//       which stores the outputInfo.   
				try {
					oparams.setDimensions(oparams.getNumRows(), oparams.getNumCols(), -1, -1, oparams.getNnz(), oparams.getUpdateType());
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
			if( ((SortKeys)node).getOpType() == SortKeys.OperationTypes.Indexes)
				oinfo = OutputInfo.BinaryBlockOutputInfo;
			else
				oinfo = OutputInfo.OutputInfoForSortOutput; 
		} else if (node.getType() == Type.CombineBinary) {
			// Output format of CombineBinary (CB) depends on how the output is consumed
			CombineBinary combine = (CombineBinary) node;
			if ( combine.getOperation() == org.apache.sysml.lops.CombineBinary.OperationTypes.PreSort ) {
				oinfo = OutputInfo.OutputInfoForSortInput; 
			}
			else if ( combine.getOperation() == org.apache.sysml.lops.CombineBinary.OperationTypes.PreCentralMoment 
					  || combine.getOperation() == org.apache.sysml.lops.CombineBinary.OperationTypes.PreCovUnweighted 
					  || combine.getOperation() == org.apache.sysml.lops.CombineBinary.OperationTypes.PreGroupedAggUnweighted ) {
				oinfo = OutputInfo.WeightedPairOutputInfo;
			}
		} else if ( node.getType() == Type.CombineTernary) {
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
	
	private String prepareAssignVarInstruction(Lop input, Lop node) {
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
	 * @throws LopsException 
	 */
	private NodeOutput setupNodeOutputs(Lop node, ExecType et, boolean cellModeOverride, boolean copyTWrite) 
	throws DMLRuntimeException, LopsException {
		
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
			if (node.getDataType() == DataType.SCALAR) {
				oparams.setLabel(Lop.SCALAR_VAR_NAME_PREFIX + var_index.getNextID());
				out.setVarName(oparams.getLabel());
				Instruction currInstr = VariableCPInstruction.prepareRemoveInstruction(oparams.getLabel());
				
				currInstr.setLocation(node);
				
				out.addLastInstruction(currInstr);
			}
			else if(node instanceof ParameterizedBuiltin 
					&& ((ParameterizedBuiltin)node).getOp() == org.apache.sysml.lops.ParameterizedBuiltin.OperationTypes.TRANSFORM) {
				
				ParameterizedBuiltin pbi = (ParameterizedBuiltin)node;
				Lop input = pbi.getNamedInput(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_DATA);
				if(input.getDataType()== DataType.FRAME) {
					
					// Output of transform is in CSV format, which gets subsequently reblocked 
					// TODO: change it to output binaryblock
					
					Data dataInput = (Data) input;
					oparams.setFile_name(getFilePath() + "temp" + job_id.getNextID());
					oparams.setLabel(Lop.MATRIX_VAR_NAME_PREFIX + var_index.getNextID());

					// generate an instruction that creates a symbol table entry for the new variable in CSV format
					Data delimLop = (Data) dataInput.getNamedInputLop(
							DataExpression.DELIM_DELIMITER, DataExpression.DEFAULT_DELIM_DELIMITER);
					
					Instruction createvarInst = VariableCPInstruction.prepareCreateVariableInstruction(
					        oparams.getLabel(), oparams.getFile_name(), true, 
					        DataType.MATRIX, OutputInfo.outputInfoToString(OutputInfo.CSVOutputInfo),
							new MatrixCharacteristics(oparams.getNumRows(), oparams.getNumCols(), -1, -1, oparams.getNnz()), oparams.getUpdateType(), 
							false, delimLop.getStringValue(), true
						);
					
					createvarInst.setLocation(node);
					
					out.addPreInstruction(createvarInst);

					// temp file as well as the variable has to be deleted at the end
					Instruction currInstr = VariableCPInstruction.prepareRemoveInstruction(oparams.getLabel());
					
					currInstr.setLocation(node);
					
					out.addLastInstruction(currInstr);

					// finally, add the generated filename and variable name to the list of outputs
					out.setFileName(oparams.getFile_name());
					out.setVarName(oparams.getLabel());
				}
				else {
					throw new LopsException("Input to transform() has an invalid type: " + input.getDataType() + ", it must be FRAME.");
				}
			}
			else if(!(node instanceof FunctionCallCP)) //general case
			{
				// generate temporary filename and a variable name to hold the
				// output produced by "rootNode"
				oparams.setFile_name(getFilePath() + "temp" + job_id.getNextID());
				oparams.setLabel( (node.getDataType()==DataType.MATRIX ? Lop.MATRIX_VAR_NAME_PREFIX :
						Lop.FRAME_VAR_NAME_PREFIX) + var_index.getNextID());

				// generate an instruction that creates a symbol table entry for the new variable
				//String createInst = prepareVariableInstruction("createvar", node);
				//out.addPreInstruction(CPInstructionParser.parseSingleInstruction(createInst));
				int rpb = (int) oparams.getRowsInBlock();
				int cpb = (int) oparams.getColsInBlock();
				Instruction createvarInst = VariableCPInstruction.prepareCreateVariableInstruction(
									        oparams.getLabel(),
											oparams.getFile_name(), 
											true, node.getDataType(),
											OutputInfo.outputInfoToString(getOutputInfo(node, false)),
											new MatrixCharacteristics(oparams.getNumRows(), oparams.getNumCols(), rpb, cpb, oparams.getNnz()),
											oparams.getUpdateType()
										);
				
				createvarInst.setLocation(node);
				
				out.addPreInstruction(createvarInst);

				// temp file as well as the variable has to be deleted at the end
				Instruction currInstr = VariableCPInstruction.prepareRemoveInstruction(oparams.getLabel());
				
				currInstr.setLocation(node);
				
				out.addLastInstruction(currInstr);

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
								getFilePath() + fnOutParams.getLabel(), 
								true, fnOut.getDataType(),
								OutputInfo.outputInfoToString(getOutputInfo(fnOut, false)),
								new MatrixCharacteristics(fnOutParams.getNumRows(), fnOutParams.getNumCols(), (int)fnOutParams.getRowsInBlock(), (int)fnOutParams.getColsInBlock(), fnOutParams.getNnz()),
								oparams.getUpdateType()
							);
						
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
						Instruction currInstr = VariableCPInstruction.prepareCopyInstruction(inputVarName, constVarName);
						
						currInstr.setLocation(node);
						
						out.addLastInstruction(currInstr);
						out.setFileName(constFileName);
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
						String tempFileName = getFilePath() + "temp" + job_id.getNextID();
						
						//String createInst = prepareVariableInstruction("createvar", tempVarName, node.getDataType(), node.getValueType(), tempFileName, oparams, out.getOutInfo());
						//out.addPreInstruction(CPInstructionParser.parseSingleInstruction(createInst));
						
						int rpb = (int) oparams.getRowsInBlock();
						int cpb = (int) oparams.getColsInBlock();
						Instruction createvarInst = VariableCPInstruction.prepareCreateVariableInstruction(
													tempVarName, 
													tempFileName, 
													true, node.getDataType(),
													OutputInfo.outputInfoToString(out.getOutInfo()), 
													new MatrixCharacteristics(oparams.getNumRows(), oparams.getNumCols(), rpb, cpb, oparams.getNnz()),
													oparams.getUpdateType()
												);
						
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
						
						// rename the temp variable to constant variable (e.g., cpvar tVarAtemp tVarA)
						/*Instruction currInstr = VariableCPInstruction.prepareCopyInstruction(tempVarName, constVarName);
						if(DMLScript.ENABLE_DEBUG_MODE) {
							currInstr.setLineNum(node._beginLine);
						}
						out.addLastInstruction(currInstr);
						Instruction tempInstr = VariableCPInstruction.prepareRemoveInstruction(tempVarName);
						if(DMLScript.ENABLE_DEBUG_MODE) {
							tempInstr.setLineNum(node._beginLine);
						}
						out.addLastInstruction(tempInstr);*/

						// Generate a single mvvar instruction (e.g., mvvar tempA A) 
						//    instead of two instructions "cpvar tempA A" and "rmvar tempA"
						Instruction currInstr = VariableCPInstruction.prepareMoveInstruction(tempVarName, constVarName);
						
						currInstr.setLocation(node);
						
						out.addLastInstruction(currInstr);

						// finally, add the temporary filename and variable name to the list of outputs 
						out.setFileName(tempFileName);
						out.setVarName(tempVarName);
					}
				} 
				// rootNode is not a transient write. It is a persistent write.
				else {
					if(et == ExecType.MR) { //MR PERSISTENT WRITE
						// create a variable to hold the result produced by this "rootNode"
						oparams.setLabel("pVar" + var_index.getNextID() );
						
						//String createInst = prepareVariableInstruction("createvar", node);
						//out.addPreInstruction(CPInstructionParser.parseSingleInstruction(createInst));

						int rpb = (int) oparams.getRowsInBlock();
						int cpb = (int) oparams.getColsInBlock();
						Lop fnameLop = ((Data)node).getNamedInputLop(DataExpression.IO_FILENAME);
						String fnameStr = (fnameLop instanceof Data && ((Data)fnameLop).isLiteral()) ? 
								           fnameLop.getOutputParameters().getLabel() 
								           : Lop.VARIABLE_NAME_PLACEHOLDER + fnameLop.getOutputParameters().getLabel() + Lop.VARIABLE_NAME_PLACEHOLDER;
						
						Instruction createvarInst;
						
						// for MatrixMarket format, the creatvar will output the result to a temporary file in textcell format 
						// the CP write instruction (post instruction) after the MR instruction will merge the result into a single
						// part MM format file on hdfs.
						if (oparams.getFormat() == Format.CSV)  {
							
							String tempFileName = getFilePath() + "temp" + job_id.getNextID();
							
							String createInst = node.getInstructions(tempFileName);
							createvarInst= CPInstructionParser.parseSingleInstruction(createInst);
						
							//NOTE: no instruction patching because final write from cp instruction
							String writeInst = node.getInstructions(oparams.getLabel(), fnameLop.getOutputParameters().getLabel() );
							CPInstruction currInstr = CPInstructionParser.parseSingleInstruction(writeInst);
							
							currInstr.setLocation(node);
							
							out.addPostInstruction(currInstr);
							
							// remove the variable
							CPInstruction tempInstr = CPInstructionParser.parseSingleInstruction(
									"CP" + Lop.OPERAND_DELIMITOR + "rmfilevar" + Lop.OPERAND_DELIMITOR 
									+ oparams.getLabel() + Lop.VALUETYPE_PREFIX + Expression.ValueType.UNKNOWN + Lop.OPERAND_DELIMITOR 
									+ "true" + Lop.VALUETYPE_PREFIX + "BOOLEAN");
							
							tempInstr.setLocation(node);
							
							out.addLastInstruction(tempInstr);
						} 
						else if (oparams.getFormat() == Format.MM )  {
							
							String tempFileName = getFilePath() + "temp" + job_id.getNextID();
							
							createvarInst= VariableCPInstruction.prepareCreateVariableInstruction(
													oparams.getLabel(), 
													tempFileName, 
													false, node.getDataType(),
													OutputInfo.outputInfoToString(getOutputInfo(node, false)), 
													new MatrixCharacteristics(oparams.getNumRows(), oparams.getNumCols(), rpb, cpb, oparams.getNnz()),
													oparams.getUpdateType()
												);

							//NOTE: no instruction patching because final write from cp instruction
							String writeInst = node.getInstructions(oparams.getLabel(), fnameLop.getOutputParameters().getLabel());
							CPInstruction currInstr = CPInstructionParser.parseSingleInstruction(writeInst);
							
							currInstr.setLocation(node);
							
							out.addPostInstruction(currInstr);
							
							// remove the variable
							CPInstruction tempInstr = CPInstructionParser.parseSingleInstruction(
									"CP" + Lop.OPERAND_DELIMITOR + "rmfilevar" + Lop.OPERAND_DELIMITOR 
									+ oparams.getLabel() + Lop.VALUETYPE_PREFIX + Expression.ValueType.UNKNOWN + Lop.OPERAND_DELIMITOR 
									+ "true" + Lop.VALUETYPE_PREFIX + "BOOLEAN");
							
							tempInstr.setLocation(node);
							
							out.addLastInstruction(tempInstr);
						} 
						else {
							createvarInst= VariableCPInstruction.prepareCreateVariableInstruction(
									                oparams.getLabel(), 
									                fnameStr, 
									                false, node.getDataType(),
									                OutputInfo.outputInfoToString(getOutputInfo(node, false)), 
									                new MatrixCharacteristics(oparams.getNumRows(), oparams.getNumCols(), rpb, cpb, oparams.getNnz()),
													oparams.getUpdateType()
								                 );
							// remove the variable
							CPInstruction currInstr = CPInstructionParser.parseSingleInstruction(
									"CP" + Lop.OPERAND_DELIMITOR + "rmfilevar" + Lop.OPERAND_DELIMITOR 
									+ oparams.getLabel() + Lop.VALUETYPE_PREFIX + Expression.ValueType.UNKNOWN + Lop.OPERAND_DELIMITOR 
									+ "false" + Lop.VALUETYPE_PREFIX + "BOOLEAN");
							
							currInstr.setLocation(node);
							
							out.addLastInstruction(currInstr);
							
						}
						
						createvarInst.setLocation(node);
						
						out.addPreInstruction(createvarInst);
						

						// finally, add the filename and variable name to the list of outputs 
						out.setFileName(oparams.getFile_name()); 
						out.setVarName(oparams.getLabel());
					}
					else { //CP PERSISTENT WRITE
						// generate a write instruction that writes matrix to HDFS
						Lop fname = ((Data)node).getNamedInputLop(DataExpression.IO_FILENAME);
						Instruction currInstr = null;
						Lop inputLop = node.getInputs().get(0);
						
						// Case of a transient read feeding into only one output persistent binaryblock write
						// Move the temporary file on HDFS to required persistent location, insteadof copying.
						if (inputLop.getExecLocation() == ExecLocation.Data
								&& inputLop.getOutputs().size() == 1
								&& ((Data)inputLop).isTransient() 
								&& ((Data)inputLop).getOutputParameters().isBlocked()
								&& node.getOutputParameters().isBlocked() ) {
							// transient read feeding into persistent write in blocked representation
							// simply, move the file
							
							//prepare filename (literal or variable in order to support dynamic write) 
							String fnameStr = (fname instanceof Data && ((Data)fname).isLiteral()) ? 
									           fname.getOutputParameters().getLabel() 
									           : Lop.VARIABLE_NAME_PLACEHOLDER + fname.getOutputParameters().getLabel() + Lop.VARIABLE_NAME_PLACEHOLDER;
							
							currInstr = (CPInstruction) VariableCPInstruction.prepareMoveInstruction(
											inputLop.getOutputParameters().getLabel(), 
											fnameStr, "binaryblock" );
						}
						else {
							
							String io_inst = node.getInstructions(
									node.getInputs().get(0).getOutputParameters().getLabel(), 
									fname.getOutputParameters().getLabel());
							
							if(node.getExecType() == ExecType.SPARK)
								// This will throw an exception if the exectype of hop is set incorrectly
								// Note: the exec type and exec location of lops needs to be set to SPARK and ControlProgram respectively
								currInstr = SPInstructionParser.parseSingleInstruction(io_inst);
							else
								currInstr = CPInstructionParser.parseSingleInstruction(io_inst);
						}

						
						if ( !node.getInputs().isEmpty() && node.getInputs().get(0)._beginLine != 0)
							currInstr.setLocation(node.getInputs().get(0));
						else
							currInstr.setLocation(node);
						
						out.addLastInstruction(currInstr);
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
	 * @throws DMLRuntimeException
	 */
	private void generateMapReduceInstructions(ArrayList<Lop> execNodes,
			ArrayList<Instruction> inst, ArrayList<Instruction> writeinst, ArrayList<Instruction> deleteinst, ArrayList<Instruction> rmvarinst, 
			JobType jt) throws LopsException, DMLRuntimeException
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
		ArrayList<Instruction> postInstructions = new ArrayList<Instruction>();
		ArrayList<Integer> MRJobLineNumbers = null;
		if(DMLScript.ENABLE_DEBUG_MODE) {
			MRJobLineNumbers = new ArrayList<Integer>();
		}
		
		ArrayList<Lop> inputLops = new ArrayList<Lop>();
		
		boolean cellModeOverride = false;
		
		/* Find the nodes that produce an output */
		ArrayList<Lop> rootNodes = new ArrayList<Lop>();
		getOutputNodes(execNodes, rootNodes, jt);
		if( LOG.isTraceEnabled() )
			LOG.trace("# of root nodes = " + rootNodes.size());
		
		
		/* Remove transient writes that are simple copy of transient reads */
		if (jt == JobType.GMR || jt == JobType.GMRCELL) {
			ArrayList<Lop> markedNodes = new ArrayList<Lop>();
			// only keep data nodes that are results of some computation.
			for ( Lop rnode : rootNodes ) {
				if (rnode.getExecLocation() == ExecLocation.Data
						&& ((Data) rnode).isTransient()
						&& ((Data) rnode).getOperationType() == OperationTypes.WRITE
						&& ((Data) rnode).getDataType() == DataType.MATRIX) {
					// no computation, just a copy
					if (rnode.getInputs().get(0).getExecLocation() == ExecLocation.Data
							&& ((Data) rnode.getInputs().get(0)).isTransient()
							&& rnode.getOutputParameters().getLabel().equals(
								rnode.getInputs().get(0).getOutputParameters().getLabel())) 
					{
						markedNodes.add(rnode);
					}
				}
			}
			// delete marked nodes
			rootNodes.removeAll(markedNodes);
			markedNodes.clear();
			if ( rootNodes.isEmpty() )
				return;
		}
		
		// structure that maps node to their indices that will be used in the instructions
		HashMap<Lop, Integer> nodeIndexMapping = new HashMap<Lop, Integer>();
		
		/* Determine all input data files */
		
		for ( Lop rnode : rootNodes ) {
			getInputPathsAndParameters(rnode, execNodes, inputs, inputInfos, numRows, numCols, 
				numRowsPerBlock, numColsPerBlock, nodeIndexMapping, inputLabels, inputLops, MRJobLineNumbers);
		}
		
		// In case of RAND job, instructions are defined in the input file
		if (jt == JobType.DATAGEN)
			randInstructions = inputs;
		
		int[] start_index = new int[1];
		start_index[0] = inputs.size();
		
		/* Get RecordReader Instructions */
		
		// currently, recordreader instructions are allowed only in GMR jobs
		if (jt == JobType.GMR || jt == JobType.GMRCELL) {
			for ( Lop rnode : rootNodes ) {
				getRecordReaderInstructions(rnode, execNodes, inputs, recordReaderInstructions, 
					nodeIndexMapping, start_index, inputLabels, inputLops, MRJobLineNumbers);
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
		if ( jt != JobType.REBLOCK && jt != JobType.CSV_REBLOCK && jt != JobType.DATAGEN && jt != JobType.TRANSFORM) {
			for (int i=0; i < inputInfos.size(); i++)
				if ( inputInfos.get(i) == InputInfo.BinaryCellInputInfo || inputInfos.get(i) == InputInfo.TextCellInputInfo )
					cellModeOverride = true;
		}
		
		if ( !recordReaderInstructions.isEmpty() || jt == JobType.GROUPED_AGG )
			cellModeOverride = true;
		
		
		/* Get Mapper Instructions */
	
		for (int i = 0; i < rootNodes.size(); i++) {
			getMapperInstructions(rootNodes.get(i), execNodes, inputs,
					mapperInstructions, nodeIndexMapping, start_index,
					inputLabels, inputLops, MRJobLineNumbers);
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

		for( Lop rn : rootNodes ) {
			int resultIndex = getAggAndOtherInstructions(
					rn, execNodes, shuffleInstructions, aggInstructionsReducer,
					otherInstructionsReducer, nodeIndexMapping, start_index,
					inputLabels, inputLops, MRJobLineNumbers);
			if ( resultIndex == -1)
				throw new LopsException("Unexpected error in piggybacking!");
			
			if ( rn.getExecLocation() == ExecLocation.Data 
					&& ((Data)rn).getOperationType() == Data.OperationTypes.WRITE && ((Data)rn).isTransient() 
					&& rootNodes.contains(rn.getInputs().get(0))
					) {
				// Both rn (a transient write) and its input are root nodes.
				// Instead of creating two copies of the data, simply generate a cpvar instruction 
				NodeOutput out = setupNodeOutputs(rn, ExecType.MR, cellModeOverride, true);
				writeinst.addAll(out.getLastInstructions());
			}
			else {
				resultIndices.add(Byte.valueOf((byte)resultIndex));
			
				// setup output filenames and outputInfos and generate related instructions
				NodeOutput out = setupNodeOutputs(rn, ExecType.MR, cellModeOverride, false);
				outputLabels.add(out.getVarName());
				outputs.add(out.getFileName());
				outputInfos.add(out.getOutInfo());
				if (LOG.isTraceEnabled()) {
					LOG.trace("    Output Info: " + out.getFileName() + ";" + OutputInfo.outputInfoToString(out.getOutInfo()) + ";" + out.getVarName());
				}
				renameInstructions.addAll(out.getLastInstructions());
				variableInstructions.addAll(out.getPreInstructions());
				postInstructions.addAll(out.getPostInstructions());
			}			
		}
		
		/* Determine if the output dimensions are known */
		
		byte[] resultIndicesByte = new byte[resultIndices.size()];
		for (int i = 0; i < resultIndicesByte.length; i++) {
			resultIndicesByte[i] = resultIndices.get(i).byteValue();
		}
		
		if (LOG.isTraceEnabled()) {
			LOG.trace("    Shuffle Instructions: " + getCSVString(shuffleInstructions));
			LOG.trace("    Aggregate Instructions: " + getCSVString(aggInstructionsReducer));
			LOG.trace("    Other instructions =" + getCSVString(otherInstructionsReducer));
			LOG.trace("    Output strings: " + outputs.toString());
			LOG.trace("    ResultIndices = " + resultIndices.toString());
		}
		
		/* Prepare the MapReduce job instruction */
		
		MRJobInstruction mr = new MRJobInstruction(jt);
		
		// check if this is a map-only job. If not, set the number of reducers
		if ( !shuffleInstructions.isEmpty() || !aggInstructionsReducer.isEmpty() || !otherInstructionsReducer.isEmpty() )
			numReducers = total_reducers;
		
		// set inputs, outputs, and other other properties for the job 
		mr.setInputOutputLabels(inputLabels.toArray(new String[0]), outputLabels.toArray(new String[0]));
		mr.setOutputs(resultIndicesByte);
		mr.setDimsUnknownFilePrefix(getFilePath());
		
		mr.setNumberOfReducers(numReducers);
		mr.setReplication(replication);
		
		// set instructions for recordReader and mapper
		mr.setRecordReaderInstructions(getCSVString(recordReaderInstructions));
		mr.setMapperInstructions(getCSVString(mapperInstructions));
		
		//compute and set mapper memory requirements (for consistency of runtime piggybacking)
		if( jt == JobType.GMR ) {
			double mem = 0;
			for( Lop n : execNodes )
				mem += computeFootprintInMapper(n);
			mr.setMemoryRequirements(mem);
		}
			
		if ( jt == JobType.DATAGEN )
			mr.setRandInstructions(getCSVString(randInstructions));
		
		// set shuffle instructions
		mr.setShuffleInstructions(getCSVString(shuffleInstructions));
		
		// set reducer instruction
		mr.setAggregateInstructionsInReducer(getCSVString(aggInstructionsReducer));
		mr.setOtherInstructionsInReducer(getCSVString(otherInstructionsReducer));
		if(DMLScript.ENABLE_DEBUG_MODE) {
			// set line number information for each MR instruction
			mr.setMRJobInstructionsLineNumbers(MRJobLineNumbers);
		}

		/* Add the prepared instructions to output set */
		inst.addAll(variableInstructions);
		inst.add(mr);
		inst.addAll(postInstructions);
		deleteinst.addAll(renameInstructions);
		
		for (Lop l : inputLops) {
			if(DMLScript.ENABLE_DEBUG_MODE) {
				processConsumers(l, rmvarinst, deleteinst, l);
			}
			else {
				processConsumers(l, rmvarinst, deleteinst, null);
			}
		}
	}

	/**
	 * converts an array list into a comma separated string
	 * 
	 * @param inputStrings
	 * @return
	 */
	private static String getCSVString(ArrayList<String> inputStrings) {
		StringBuilder sb = new StringBuilder();
		for ( String str : inputStrings ) {
			if( str != null ) {
				if( sb.length()>0 )
					sb.append(Lop.INSTRUCTION_DELIMITOR);
				sb.append( str ); 
			}
		}
		return sb.toString();
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
	 * @param MRJoblineNumbers
	 * @return
	 * @throws LopsException
	 */
	private int getAggAndOtherInstructions(Lop node, ArrayList<Lop> execNodes,
			ArrayList<String> shuffleInstructions,
			ArrayList<String> aggInstructionsReducer,
			ArrayList<String> otherInstructionsReducer,
			HashMap<Lop, Integer> nodeIndexMapping, int[] start_index,
			ArrayList<String> inputLabels, ArrayList<Lop> inputLops, 
			ArrayList<Integer> MRJobLineNumbers) throws LopsException
	{
		int ret_val = -1;

		if (nodeIndexMapping.containsKey(node))
			return nodeIndexMapping.get(node);

		// if not an input source and not in exec nodes, return.

		if (!execNodes.contains(node))
			return ret_val;

		ArrayList<Integer> inputIndices = new ArrayList<Integer>();

		// recurse
		// For WRITE, since the first element from input is the real input (the other elements
		// are parameters for the WRITE operation), so we only need to take care of the
		// first element.
		if (node.getType() == Lop.Type.Data && ((Data)node).getOperationType() == Data.OperationTypes.WRITE) {
			ret_val = getAggAndOtherInstructions(node.getInputs().get(0),
					execNodes, shuffleInstructions, aggInstructionsReducer,
					otherInstructionsReducer, nodeIndexMapping, start_index,
					inputLabels, inputLops, MRJobLineNumbers);
			inputIndices.add(ret_val);
		}
		else {
			for ( Lop cnode : node.getInputs() ) {
				ret_val = getAggAndOtherInstructions(cnode,
						execNodes, shuffleInstructions, aggInstructionsReducer,
						otherInstructionsReducer, nodeIndexMapping, start_index,
						inputLabels, inputLops, MRJobLineNumbers);
				inputIndices.add(ret_val);
			}
		}

		if (node.getExecLocation() == ExecLocation.Data ) {
			if ( ((Data)node).getFileFormatType() == FileFormatTypes.CSV 
					&& !(node.getInputs().get(0) instanceof ParameterizedBuiltin 
							&& ((ParameterizedBuiltin)node.getInputs().get(0)).getOp() == org.apache.sysml.lops.ParameterizedBuiltin.OperationTypes.TRANSFORM)) {
				// Generate write instruction, which goes into CSV_WRITE Job
				int output_index = start_index[0];
				shuffleInstructions.add(node.getInstructions(inputIndices.get(0), output_index));
				if(DMLScript.ENABLE_DEBUG_MODE) {
					MRJobLineNumbers.add(node._beginLine);
				}
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
				if(DMLScript.ENABLE_DEBUG_MODE) {
					MRJobLineNumbers.add(node._beginLine);
				}
				break;
				
			case ParameterizedBuiltin:
				if( ((ParameterizedBuiltin)node).getOp() == org.apache.sysml.lops.ParameterizedBuiltin.OperationTypes.TRANSFORM ) {
					shuffleInstructions.add(node.getInstructions(output_index));
					if(DMLScript.ENABLE_DEBUG_MODE) {
						MRJobLineNumbers.add(node._beginLine);
					}
				}
				break;
				
			/* Lop types that take two inputs */
			case MMCJ:
			case MMRJ:
			case CombineBinary:
				shuffleInstructions.add(node.getInstructions(inputIndices.get(0), inputIndices.get(1), output_index));
				if(DMLScript.ENABLE_DEBUG_MODE) {
					MRJobLineNumbers.add(node._beginLine);
				}
				break;

			/* Lop types that take three inputs */
			case CombineTernary:
				shuffleInstructions.add(node.getInstructions(inputIndices
						.get(0), inputIndices.get(1), inputIndices.get(2), output_index));
				if(DMLScript.ENABLE_DEBUG_MODE) {
					MRJobLineNumbers.add(node._beginLine);
				}
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
				
				if (node.getType() == Type.Aggregate) {
					aggInstructionsReducer.add(node.getInstructions(
							inputIndices.get(0), output_index));
					if(DMLScript.ENABLE_DEBUG_MODE) {
						MRJobLineNumbers.add(node._beginLine);
					}
				}
				else {
					otherInstructionsReducer.add(node.getInstructions(
							inputIndices.get(0), output_index));
				}
				if(DMLScript.ENABLE_DEBUG_MODE) {
					MRJobLineNumbers.add(node._beginLine);
				}
				nodeIndexMapping.put(node, output_index);

				return output_index;
			} else if (inputIndices.size() == 2) {
				int output_index = start_index[0];
				start_index[0]++;

				otherInstructionsReducer.add(node.getInstructions(inputIndices
						.get(0), inputIndices.get(1), output_index));
				if(DMLScript.ENABLE_DEBUG_MODE) {
					MRJobLineNumbers.add(node._beginLine);
				}
				nodeIndexMapping.put(node, output_index);

				// populate list of input labels.
				// only Unary lops can contribute to labels

				if (node instanceof Unary && node.getInputs().size() > 1) {
					int index = 0;
					for (int i = 0; i < node.getInputs().size(); i++) {
						if (node.getInputs().get(i).getDataType() == DataType.SCALAR) {
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
			} else if (inputIndices.size() == 3 || node.getType() == Type.Ternary) {
				int output_index = start_index[0];
				start_index[0]++;

				if (node.getType() == Type.Ternary ) {
					// in case of CTABLE_TRANSFORM_SCALAR_WEIGHT: inputIndices.get(2) would be -1
					otherInstructionsReducer.add(node.getInstructions(
							inputIndices.get(0), inputIndices.get(1),
							inputIndices.get(2), output_index));
					if(DMLScript.ENABLE_DEBUG_MODE) {
						MRJobLineNumbers.add(node._beginLine);
					}
					nodeIndexMapping.put(node, output_index);
				}
				else if( node.getType() == Type.ParameterizedBuiltin ){
					otherInstructionsReducer.add(node.getInstructions(
							inputIndices.get(0), inputIndices.get(1),
							inputIndices.get(2), output_index));
					if(DMLScript.ENABLE_DEBUG_MODE) {
						MRJobLineNumbers.add(node._beginLine);
					}
					nodeIndexMapping.put(node, output_index);
				}
				else
				{
					otherInstructionsReducer.add(node.getInstructions(
							inputIndices.get(0), inputIndices.get(1),
							inputIndices.get(2), output_index));
					if(DMLScript.ENABLE_DEBUG_MODE) {
						MRJobLineNumbers.add(node._beginLine);
					}
					nodeIndexMapping.put(node, output_index);
					return output_index;
				}

				return output_index;
			}
			else if (inputIndices.size() == 4) {
				int output_index = start_index[0];
				start_index[0]++;
				otherInstructionsReducer.add(node.getInstructions(
						inputIndices.get(0), inputIndices.get(1),
						inputIndices.get(2), inputIndices.get(3), output_index));
				if(DMLScript.ENABLE_DEBUG_MODE) {
					MRJobLineNumbers.add(node._beginLine);
				}
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
	 * @param MRJobLineNumbers
	 * @return
	 * @throws LopsException
	 */
	private static int getRecordReaderInstructions(Lop node, ArrayList<Lop> execNodes,
			ArrayList<String> inputStrings,
			ArrayList<String> recordReaderInstructions,
			HashMap<Lop, Integer> nodeIndexMapping, int[] start_index,
			ArrayList<String> inputLabels, ArrayList<Lop> inputLops,
			ArrayList<Integer> MRJobLineNumbers) throws LopsException
	{
		// if input source, return index
		if (nodeIndexMapping.containsKey(node))
			return nodeIndexMapping.get(node);

		// not input source and not in exec nodes, then return.
		if (!execNodes.contains(node))
			return -1;

		ArrayList<Integer> inputIndices = new ArrayList<Integer>();
		int max_input_index = -1;
		//N child_for_max_input_index = null;

		// get mapper instructions
		for (int i = 0; i < node.getInputs().size(); i++) {
			// recurse
			Lop childNode = node.getInputs().get(i);
			int ret_val = getRecordReaderInstructions(childNode, execNodes,
					inputStrings, recordReaderInstructions, nodeIndexMapping,
					start_index, inputLabels, inputLops, MRJobLineNumbers);

			inputIndices.add(ret_val);

			if (ret_val > max_input_index) {
				max_input_index = ret_val;
				//child_for_max_input_index = childNode;
			}
		}

		// only lops with execLocation as RecordReader can contribute
		// instructions
		if ((node.getExecLocation() == ExecLocation.RecordReader)) {
			int output_index = max_input_index;

			// cannot reuse index if this is true
			// need to add better indexing schemes
			output_index = start_index[0];
			start_index[0]++;

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
			if (node.getInputs().size() == 2) {
				recordReaderInstructions.add(node.getInstructions(inputIndices
						.get(0), inputIndices.get(1), output_index));
				if(DMLScript.ENABLE_DEBUG_MODE) {
					MRJobLineNumbers.add(node._beginLine);
				}
			}
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
	 * @param MRJoblineNumbers
	 * @return
	 * @throws LopsException
	 */
	private int getMapperInstructions(Lop node, ArrayList<Lop> execNodes,
			ArrayList<String> inputStrings,
			ArrayList<String> instructionsInMapper,
			HashMap<Lop, Integer> nodeIndexMapping, int[] start_index,
			ArrayList<String> inputLabels, ArrayList<Lop> inputLops, 
			ArrayList<Integer> MRJobLineNumbers) throws LopsException
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
		for( Lop childNode : node.getInputs()) {
			int ret_val = getMapperInstructions(childNode, execNodes,
					inputStrings, instructionsInMapper, nodeIndexMapping,
					start_index, inputLabels, inputLops, MRJobLineNumbers);

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
					if (node.getInputs().get(i1).getDataType() == DataType.SCALAR) {
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
			
			if(DMLScript.ENABLE_DEBUG_MODE) {
				MRJobLineNumbers.add(node._beginLine);
			}
			return output_index;
		}

		return -1;
	}

	// Method to populate inputs and also populates node index mapping.
	private static void getInputPathsAndParameters(Lop node, ArrayList<Lop> execNodes,
			ArrayList<String> inputStrings, ArrayList<InputInfo> inputInfos,
			ArrayList<Long> numRows, ArrayList<Long> numCols,
			ArrayList<Long> numRowsPerBlock, ArrayList<Long> numColsPerBlock,
			HashMap<Lop, Integer> nodeIndexMapping, ArrayList<String> inputLabels, 
			ArrayList<Lop> inputLops, ArrayList<Integer> MRJobLineNumbers)
			throws LopsException {
		// treat rand as an input.
		if (node.getType() == Type.DataGen && execNodes.contains(node)
				&& !nodeIndexMapping.containsKey(node)) {
			numRows.add(node.getOutputParameters().getNumRows());
			numCols.add(node.getOutputParameters().getNumCols());
			numRowsPerBlock.add(node.getOutputParameters().getRowsInBlock());
			numColsPerBlock.add(node.getOutputParameters().getColsInBlock());
			inputStrings.add(node.getInstructions(inputStrings.size(), inputStrings.size()));
			if(DMLScript.ENABLE_DEBUG_MODE) {
				MRJobLineNumbers.add(node._beginLine);
			}
			inputInfos.add(InputInfo.TextCellInputInfo);
			nodeIndexMapping.put(node, inputStrings.size() - 1);
			
			return;
		}

		// get input file names
		if (!execNodes.contains(node)
				&& !nodeIndexMapping.containsKey(node)
				&& !(node.getExecLocation() == ExecLocation.Data)
				&& (!(node.getExecLocation() == ExecLocation.ControlProgram && node
						.getDataType() == DataType.SCALAR))
				|| (!execNodes.contains(node)
						&& node.getExecLocation() == ExecLocation.Data
						&& ((Data) node).getOperationType() == Data.OperationTypes.READ
						&& ((Data) node).getDataType() != DataType.SCALAR && !nodeIndexMapping
						.containsKey(node))) {
			if (node.getOutputParameters().getFile_name() != null) {
				inputStrings.add(node.getOutputParameters().getFile_name());
			} else {
				// use label name
				inputStrings.add(Lop.VARIABLE_NAME_PLACEHOLDER + node.getOutputParameters().getLabel()
						               + Lop.VARIABLE_NAME_PLACEHOLDER);
			}
			
			inputLabels.add(node.getOutputParameters().getLabel());
			inputLops.add(node);

			numRows.add(node.getOutputParameters().getNumRows());
			numCols.add(node.getOutputParameters().getNumCols());
			numRowsPerBlock.add(node.getOutputParameters().getRowsInBlock());
			numColsPerBlock.add(node.getOutputParameters().getColsInBlock());

			InputInfo nodeInputInfo = null;
			// Check if file format type is binary or text and update infos
			if (node.getOutputParameters().isBlocked()) {
				if (node.getOutputParameters().getFormat() == Format.BINARY)
					nodeInputInfo = InputInfo.BinaryBlockInputInfo;
				else 
					throw new LopsException("Invalid format (" + node.getOutputParameters().getFormat() + ") encountered for a node/lop (ID=" + node.getID() + ") with blocked output.");
			} 
			else {
				if (node.getOutputParameters().getFormat() == Format.TEXT)
					nodeInputInfo = InputInfo.TextCellInputInfo;
				else
					nodeInputInfo = InputInfo.BinaryCellInputInfo;
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
				if ( combine.getOperation() == org.apache.sysml.lops.CombineBinary.OperationTypes.PreSort ) {
					nodeInputInfo = new InputInfo(SequenceFileInputFormat.class,
							DoubleWritable.class, IntWritable.class);
				}
				else if ( combine.getOperation() == org.apache.sysml.lops.CombineBinary.OperationTypes.PreCentralMoment 
						  || combine.getOperation() == org.apache.sysml.lops.CombineBinary.OperationTypes.PreCovUnweighted
						  || combine.getOperation() == org.apache.sysml.lops.CombineBinary.OperationTypes.PreGroupedAggUnweighted ) {
					nodeInputInfo = InputInfo.WeightedPairInputInfo;
				}
			} else if ( node.getType() == Type.CombineTernary ) {
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
		for ( Lop lop : node.getInputs() ) {
			getInputPathsAndParameters(lop, execNodes, inputStrings,
					inputInfos, numRows, numCols, numRowsPerBlock,
					numColsPerBlock, nodeIndexMapping, inputLabels, inputLops, MRJobLineNumbers);
		}

	}

	/**
	 * Method to find all terminal nodes.
	 * 
	 * @param execNodes
	 * @param rootNodes
	 */
	private static void getOutputNodes(ArrayList<Lop> execNodes, ArrayList<Lop> rootNodes, JobType jt) {
		for ( Lop node : execNodes ) {
			// terminal node
			if (node.getOutputs().isEmpty() && !rootNodes.contains(node)) {
				rootNodes.add(node);
			} 
			else {
				// check for nodes with at least one child outside execnodes
				int cnt = 0;
				for (Lop lop : node.getOutputs() ) {
					cnt += (!execNodes.contains(lop)) ? 1 : 0; 
				}

				if (cnt > 0 && !rootNodes.contains(node) // not already a rootnode
					&& !(node.getExecLocation() == ExecLocation.Data
					&& ((Data) node).getOperationType() == OperationTypes.READ 
					&& ((Data) node).getDataType() == DataType.MATRIX) ) // Not a matrix Data READ 
				{
					if ( jt.allowsSingleShuffleInstruction() && node.getExecLocation() != ExecLocation.MapAndReduce)
						continue;
					
					if (cnt < node.getOutputs().size()) {
						if(!node.getProducesIntermediateOutput())	
							rootNodes.add(node);
					} 
					else
						rootNodes.add(node);
				}
			}
		}
	}

	/**
	 * check to see if a is the child of b (i.e., there is a directed path from a to b)
	 * 
	 * @param a
	 * @param b
	 */
	private static boolean isChild(Lop a, Lop b, HashMap<Long, Integer> IDMap) {
		int bID = IDMap.get(b.getID());
		return a.get_reachable()[bID];
	}
	
	/**
	 * Method to topologically sort lops
	 * 
	 * @param v
	 */
	@SuppressWarnings({ "unchecked", "rawtypes" })
	private void doTopologicalSort_strict_order(ArrayList<Lop> v) {
		//int numNodes = v.size();

		/*
		 * Step 1: compute the level for each node in the DAG. Level for each node is 
		 *   computed as lops are created. So, this step is need not be performed here.
		 * Step 2: sort the nodes by level, and within a level by node ID.
		 */
		
		// Step1: Performed at the time of creating Lops
		
		// Step2: sort nodes by level, and then by node ID
		Lop[] nodearray = v.toArray(new Lop[0]);
		Arrays.sort(nodearray, new LopComparator());

		// Copy sorted nodes into "v" and construct a mapping between Lop IDs and sequence of numbers
		v.clear();
		IDMap.clear();
		
		for (int i = 0; i < nodearray.length; i++) {
			v.add(nodearray[i]);
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

		// print the nodes in sorted order
		if (LOG.isTraceEnabled()) {
			for ( Lop vnode : v ) {
				StringBuilder sb = new StringBuilder();
				sb.append(vnode.getID());
				sb.append("(");
				sb.append(vnode.getLevel());
				sb.append(") ");
				sb.append(vnode.getType());
				sb.append("(");
				for(Lop vin : vnode.getInputs()) {
					sb.append(vin.getID());
					sb.append(",");
				}
				sb.append("), ");
				
				LOG.trace(sb.toString());
			}
			
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
	private void dagDFS(Lop root, boolean[] marked) {
		//contains check currently required for globalopt, will be removed when cleaned up
		if( !IDMap.containsKey(root.getID()) )
			return;
		
		int mapID = IDMap.get(root.getID());
		if ( marked[mapID] )
			return;
		marked[mapID] = true;
		for( Lop lop : root.getOutputs() ) {
			dagDFS(lop, marked);
		}
	}
	
	private static boolean hasDirectChildNode(Lop node, ArrayList<Lop> childNodes) {
		if ( childNodes.isEmpty() ) 
			return false;
		for( Lop cnode : childNodes ) {
			if ( cnode.getOutputs().contains(node))
				return true;
		}
		return false;
	}
	
	private boolean hasChildNode(Lop node, ArrayList<Lop> nodes) {
		return hasChildNode(node, nodes, ExecLocation.INVALID);
	}

	private boolean hasChildNode(Lop node, ArrayList<Lop> childNodes, ExecLocation type) {
		if ( childNodes.isEmpty() ) 
			return false;
		int index = IDMap.get(node.getID());
		for( Lop cnode : childNodes ) {
			if ( (type == ExecLocation.INVALID || cnode.getExecLocation() == type) && cnode.get_reachable()[index])
				return true;
		}
		return false;
	}
	
	private Lop getChildNode(Lop node, ArrayList<Lop> childNodes, ExecLocation type) {
		if ( childNodes.isEmpty() )
			return null;
		int index = IDMap.get(node.getID());
		for( Lop cnode : childNodes ) {
			if ( cnode.getExecLocation() == type && cnode.get_reachable()[index])
				return cnode;
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
	private Lop getParentNode(Lop node, ArrayList<Lop> parentNodes, ExecLocation type) {
		if ( parentNodes.isEmpty() )
			return null;
		for( Lop pn : parentNodes ) {
			int index = IDMap.get( pn.getID() );
			if ( pn.getExecLocation() == type && node.get_reachable()[index])
				return pn;
		}
		return null;
	}

	// Checks if "node" has any descendants in nodesVec with definedMRJob flag
	// set to true
	private boolean hasMRJobChildNode(Lop node, ArrayList<Lop> nodesVec) {
		if ( nodesVec.isEmpty() )
			return false;
		
		int index = IDMap.get(node.getID());
		for( Lop n : nodesVec ) {
			if ( n.definesMRJob() && n.get_reachable()[index]) 
				return true;
		}
		return false;
	}

	private boolean checkDataGenAsChildNode(Lop node, ArrayList<Lop> nodesVec) {
		if( nodesVec.isEmpty() )
			return true;
		
		int index = IDMap.get(node.getID());
		boolean onlyDatagen = true;
		for( Lop n : nodesVec ) {
			if ( n.definesMRJob() && n.get_reachable()[index] &&  JobType.findJobTypeFromLop(n) != JobType.DATAGEN )
				onlyDatagen = false;
		}
		// return true also when there is no lop in "nodesVec" that defines a MR job.
		return onlyDatagen;
	}

	private static int getChildAlignment(Lop node, ArrayList<Lop> execNodes, ExecLocation type) 
	{
		for (Lop n : node.getInputs() ) {
			if (!execNodes.contains(n))
				continue;

			if (execNodes.contains(n) && n.getExecLocation() == type) {
				if (n.getBreaksAlignment())
					return MR_CHILD_FOUND_BREAKS_ALIGNMENT;
				else
					return MR_CHILD_FOUND_DOES_NOT_BREAK_ALIGNMENT;
			} 
			else {
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
					throw new RuntimeException("Something wrong in getChildAlignment().");
			}
		}

		return MRCHILD_NOT_FOUND;
	}

	private boolean hasParentNode(Lop node, ArrayList<Lop> parentNodes) {
		if ( parentNodes.isEmpty() )
			return false;		
		for( Lop pnode : parentNodes ) {
			int index = IDMap.get( pnode.getID() );
			if ( node.get_reachable()[index])
				return true;
		}
		return false;
	}
}
