package dml.lops.compile;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Vector;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;

import dml.api.DMLScript;
import dml.lops.CombineBinary;
import dml.lops.Data;
import dml.lops.Lops;
import dml.lops.OutputParameters;
import dml.lops.ParameterizedBuiltin;
import dml.lops.PartitionLop;
import dml.lops.PickByCount;
import dml.lops.Unary;
import dml.lops.Data.OperationTypes;
import dml.lops.LopProperties.ExecLocation;
import dml.lops.LopProperties.ExecType;
import dml.lops.Lops.Type;
import dml.lops.OutputParameters.Format;
import dml.meta.PartitionParams;
import dml.parser.Expression;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.runtime.instructions.CPInstructionParser;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.MRJobInstruction;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.sort.PickFromCompactInputFormat;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;
import dml.utils.LopsException;
import dml.utils.configuration.DMLConfig;

/**
 * 
 * @author aghoting Class to maintain a DAG and compile it into jobs
 * @param <N>
 */

@SuppressWarnings("deprecation")
public class Dag<N extends Lops> {

	static int total_reducers;
	static String scratch = "";

	boolean DEBUG = DMLScript.DEBUG;

	// hash set for all nodes in dag

	ArrayList<N> nodes;

	static int job_id = 0;
	static int var_index = 0;

	static final int CHILD_BREAKS_ALIGNMENT = 2;
	static final int CHILD_DOES_NOT_BREAK_ALIGNMENT = 1;
	static final int MRCHILD_NOT_FOUND = 0;
	static final int MR_CHILD_FOUND_BREAKS_ALIGNMENT = 4;
	static final int MR_CHILD_FOUND_DOES_NOT_BREAK_ALIGNMENT = 5;

	private class NodeOutput {
		String fileName;
		String varName;
		OutputInfo outInfo;
		ArrayList<Instruction> preInstructions;
		ArrayList<Instruction> lastInstructions;
		
		NodeOutput() {
			fileName = null;
			varName = null;
			outInfo = null;
			preInstructions = new ArrayList<Instruction>(); 
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

	public Dag() {
		/**
		 * allocate structures
		 */

		nodes = new ArrayList<N>();

		/** get number of reducers from job config */
		JobConf jConf = new JobConf();
		total_reducers = jConf.getInt("mapred.reduce.tasks", DMLConfig.DEFAULT_NUM_REDUCERS);
	}

	/**
	 * Method to compile a dag generically
	 * 
	 * @param config
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */

	public ArrayList<Instruction> getJobs(boolean debug, DMLConfig config)
			throws LopsException, DMLRuntimeException,
			DMLUnsupportedOperationException {

		if (config != null) {
			if (config.getTextValue("numreducers") != null)
				total_reducers = Integer.parseInt(config
						.getTextValue("numreducers"));

			if (config.getTextValue("scratch") != null)
				scratch = config.getTextValue("scratch") + "/";

		}

		DEBUG = debug;

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
		ArrayList<Instruction> inst = doGreedyGrouping(node_v);

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
		
		if(DEBUG)
		  System.out.println("In delete unwanted variables");

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
					&& ((Data) node).getOperationType() == OperationTypes.WRITE
					&& labelNodeMapping.containsKey(node.getOutputParameters()
							.getLabel())) {
				labelNodeMapping.remove(node.getOutputParameters().getLabel());
			}
		}

		// generate RM instructions

		Iterator<String> it = labelNodeMapping.keySet().iterator();

		while (it.hasNext()) {
			String label = it.next();
			N node = labelNodeMapping.get(label);

			if (((Data) node).get_dataType() == DataType.SCALAR) {
				// if(DEBUG)
				// System.out.println("rmvar" + Lops.OPERAND_DELIMITOR + label);
				// inst.add(new VariableSimpleInstructions("rmvar" +
				// Lops.OPERAND_DELIMITOR + label));

			} else {
				if (DEBUG)
					System.out.println("CP" + Lops.OPERAND_DELIMITOR + "rmfilevar" + Lops.OPERAND_DELIMITOR
							+ label + Lops.VALUETYPE_PREFIX
							+ Expression.ValueType.UNKNOWN
							+ Lops.OPERAND_DELIMITOR + "true"
							+ Lops.VALUETYPE_PREFIX
							+ Expression.ValueType.BOOLEAN);
				inst.add(CPInstructionParser
								.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "rmfilevar"
										+ Lops.OPERAND_DELIMITOR + label
										+ Lops.VALUETYPE_PREFIX
										+ Expression.ValueType.UNKNOWN
										+ Lops.OPERAND_DELIMITOR + "true"
										+ Lops.VALUETYPE_PREFIX
										+ Expression.ValueType.BOOLEAN));
			}
		}

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
				if (DEBUG)
					System.out.println("Not compatible "
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
				JobType jt = JobType.findJobTypeFromLopType(node.getType());
				
				if ( jt == null ) {
					// In case of Reblock, jobType depends not only on LopType but also on the format
					if ( node.getType() == Lops.Type.ReBlock ) {
						// must differentiate between reblock lops that operate on text
						// and binary input
						if (getChildFormat(node) == Format.BINARY) {
							jt = JobType.REBLOCK_BINARY;
						}
						else {
							jt = JobType.REBLOCK_TEXT;
						}
					}
					else {
						throw new LopsException("No matching JobType is found for a the lop type: " + node.getType());
					}
				}
				
				// Add "node" to corresponding job vector
				
				if ( jt == JobType.GMR ) {
					// if "node" (in this case, a group lop) has any inputs from RAND 
					// then add it to RAND job. Otherwise, create a GMR job
					if (hasChildNode(node, arr.get(JobType.RAND.getId()) )) {
						arr.get(JobType.RAND.getId()).add(node);
						// we should NOT call 'addChildren' because appropriate
						// child nodes would have got added to RAND job already
					} else {
						int gmr_index = JobType.GMR.getId();
						arr.get(gmr_index).add(node);
						int from = arr.get(gmr_index).size();
						addChildren(node, arr.get(gmr_index), execNodes);
						int to = arr.get(gmr_index).size();
						if (!isCompatible(arr.get(gmr_index),JobType.GMR, from, to)) {
							throw new LopsException("Error during compatibility check");
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
		int[] jobIndices = {JobType.MMCJ.getId(), JobType.MMRJ.getId()};
		Lops.Type[] lopTypes = { Lops.Type.MMCJ, Lops.Type.MMRJ };
		
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
			throw new RuntimeException("Should not happen.");
	}

	/**
	 * Method to generate instructions that create variables.
	 * i.e., each instructions creates a new entry in the symbol table.
	 * Every "data" input node in a LOPs DAG is translated into an instruction. 
	 *    
	 * @param nodes
	 * @throws LopsException 
	 */
	private void generateInstructionsForInputVariables(Vector<N> nodes_v, ArrayList<Instruction> inst) throws LopsException {
		for(N n : nodes_v) {
			/*
			 * ONLY persistent reads must be considered. For transient reads, previous program block 
			 * would have already created appropriate entires in the symbol table. 
			 * Therefore, no explicit createvar instruction is required.
			 */
			if (n.getExecLocation() == ExecLocation.Data
					&& !((Data) n).isTransient() && ((Data) n).getOperationType() == OperationTypes.READ && n.get_dataType() == DataType.MATRIX) {
				if ( !((Data)n).isLiteral() ) {
					try {
						String inst_string = n.getInstructions();
						inst.add(CPInstructionParser.parseSingleInstruction(inst_string));
					} catch (DMLUnsupportedOperationException e) {
						throw new LopsException(e);
					} catch (DMLRuntimeException e) {
						throw new LopsException(e);
					}
				}
			}
		}
	}
	
	private boolean sendWriteLopToMR(N node) {
		if( node.getInputs().get(0).getExecType() == ExecType.MR 
				|| (node.getInputs().get(0).getExecLocation() == ExecLocation.Data 
						&& node.getInputs().get(0).get_dataType() == DataType.MATRIX) )
			return true;
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

	private ArrayList<Instruction> doGreedyGrouping(Vector<N> node_v)
			throws LopsException, DMLRuntimeException,
			DMLUnsupportedOperationException

	{
		if (DEBUG)
			System.out.println("Grouping DAG ============");

		// nodes to be executed in current iteration
		Vector<N> execNodes = new Vector<N>();
		// nodes that have already been processed
		Vector<N> finishedNodes = new Vector<N>();
		// nodes that are queued for the following iteration
		Vector<N> queuedNodes = new Vector<N>();

		ArrayList<Vector<N>> jobNodes = createNodeVectors(JobType.getNumJobTypes());

		// list of instructions
		ArrayList<Instruction> inst = new ArrayList<Instruction>();

		ArrayList<Instruction> preWriteDeleteInst = new ArrayList<Instruction>();
		ArrayList<Instruction> deleteInst = new ArrayList<Instruction>();
		ArrayList<Instruction> writeInst = new ArrayList<Instruction>();

		// delete transient variables that are no longer needed
		deleteUnwantedTransientReadVariables(node_v, deleteInst);

		// remove files for transient reads that are updated.
		deleteUpdatedTransientReadVariables(node_v, preWriteDeleteInst);

		generateInstructionsForInputVariables(node_v, inst);
		
		boolean done = false;
		String indent = "    ";

		while (!done) {
			if (DEBUG)
				System.out.println("Grouping nodes in DAG");

			execNodes.clear();
			queuedNodes.clear();
			clearNodeVectors(jobNodes);

			for (int i = 0; i < node_v.size(); i++) {
				N node = node_v.elementAt(i);

				// finished nodes don't need to be processed

				if (finishedNodes.contains(node))
					continue;

				if (DEBUG)
					System.out.println("Processing node (" + node.getID()
							+ ") " + node.toString() + " exec nodes size is " + execNodes.size());
				
				
		        //if node defines MR job, make sure it is compatible with all 
		        //its children nodes in execNodes 
		        if(node.definesMRJob() && !compatibleWithChildrenInExecNodes(execNodes, node))
		        {
		          if (DEBUG)
		            System.out.println(indent + "Queueing node "
		                + node.toString() + " (code 1)");
		
		          queuedNodes.add(node);
		          continue;
		
		          //no other nodes need to be dropped.
		          //hence we do not need to invoke removeNodesForNextIteration
		        }

				// if child is queued, this node will be processed in the later
				// iteration
				if (hasChildNode(node, queuedNodes)) {

					if (DEBUG)
						System.out.println(indent + "Queueing node "
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
						if (DEBUG)
							System.out.println(indent + "Queueing node "
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
					if (DEBUG)
						System.out.println(indent + "Adding -"
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
						if (! (node.getType() == Lops.Type.Grouping && getMRJobChildNode(node,execNodes).getType() == Lops.Type.RandLop) ) {
							if (DEBUG)
								System.out.println(indent + "Queueing node "
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
						&& hasChildNode(node, execNodes,
								ExecLocation.RecordReader)) {
					// get the actual RecordReader lop
					N rr_node = getChildNode(node, execNodes,
							ExecLocation.RecordReader);

					// all inputs of "node" must be ancestors of rr_node
					boolean queue_it = false;
					for (int in = 0; in < node.getInputs().size(); in++) {
						// each input should be ancestor of RecordReader lop
						N n = (N) node.getInputs().get(in);
						if (!n.equals(rr_node) && !isChild(rr_node, n)) {
							queue_it = true; // i.e., "node" must be queued
							break;
						}
					}

					if (queue_it) {
						// queue node
						if (DEBUG)
							System.out.println(indent + "Queueing -"
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
					if (DEBUG)
						System.out.println(indent + "Adding Data -"
								+ node.toString());

					finishedNodes.add(node);

					Data.OperationTypes op = ((Data) node).getOperationType(); 
					if ( op == OperationTypes.READ ) {
						// TODO: avoid readScalar instruction, and read it on-demand just like the way Matrices are read in control program
						if ( node.get_dataType() == DataType.SCALAR 
								&& node.getOutputParameters().getFile_name() != null ) {
							// this lop corresponds to reading a scalar from HDFS file
							// add it to execNodes so that "readScalar" instruction gets generated
							execNodes.add(node);
							// note: no need to add it to any job vector
						}
					}
					else if (op == OperationTypes.WRITE) {
						execNodes.add(node);
						/* WRITE lop must go into a mapreduce job if 
						 * a) its input executes in MR; or
						 * b) its input is a Matrix Data READ 
						 * 
						 */
						if ( sendWriteLopToMR(node) ) {
							addNodeByJobType(node, jobNodes, execNodes, false);
						}
					}
					continue;
				}

				// map or reduce node, can always be piggybacked with parent
				if (node.getExecLocation() == ExecLocation.MapOrReduce) {
					if (DEBUG)
						System.out.println(indent + "Adding -"
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
						if (DEBUG)
							System.out.println(indent + "Adding -"
									+ node.toString());
						execNodes.add(node);
						finishedNodes.add(node);
						addNodeByJobType(node, jobNodes, execNodes, false);
					} else {
						if (DEBUG)
							System.out.println(indent + "Queueing -"
									+ node.toString() + " (code 6)");
						queuedNodes.add(node);
						removeNodesForNextIteration(node, finishedNodes,
								execNodes, queuedNodes, jobNodes);

					}
					continue;
				}

				// map node, add, if no parent needs reduce, else queue
				if (node.getExecLocation() == ExecLocation.Map) {
					if (!hasChildNode(node, execNodes,
							ExecLocation.MapAndReduce)) {
						if (DEBUG)
							System.out.println(indent + "Adding -"
									+ node.toString());
						execNodes.add(node);
						finishedNodes.add(node);
						addNodeByJobType(node, jobNodes, execNodes, false);
					} else {
						if (DEBUG)
							System.out.println(indent + "Queueing -"
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
					if (DEBUG)
						System.out.println(indent + "Adding -"
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
					if (hasChildNode(node, execNodes, ExecLocation.MapAndReduce)) {
						if (DEBUG)
							System.out.println(indent + "Adding -"
									+ node.toString());
						execNodes.add(node);
						finishedNodes.add(node);
						addNodeByJobType(node, jobNodes, execNodes, false);
					} else {
						if (DEBUG)
							System.out.println(indent + "Queueing -"
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
							if (DEBUG)
								System.out.println(indent + "Queueing -"
										+ node.toString() + " (code 9)");

							queuedNodes.add(node);
							removeNodesForNextIteration(node, finishedNodes,
									execNodes, queuedNodes, jobNodes);
							break;
						}
					}

					if (queuedNodes.contains(node))
						continue;

					if (DEBUG)
						System.out.println(indent + "Adding - scalar"
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
			    if(DEBUG)
			    {
			      //System.err.println("Queued nodes should be 0");
			      throw new LopsException("Queued nodes should not be 0 at this point");
			    }
			  }
			  
				if (DEBUG)
					System.out.println("All done! queuedNodes = "
							+ queuedNodes.size());
				done = true;
			} else {
				// work to do

				if (DEBUG)
					System.out
							.println("Generating jobs for group -- Node count="
									+ execNodes.size());

				// first process scalar instructions
				generateControlProgramJobs(execNodes, inst, writeInst,
						deleteInst);

				// next generate MR instructions
				if (execNodes.size() > 0)
					generateMRJobs(execNodes, inst, writeInst, deleteInst,
							jobNodes);

				handleSingleOutputJobs(execNodes, jobNodes, finishedNodes);

			}

		}

		// add write and delete inst at the very end.

		inst.addAll(preWriteDeleteInst);
		inst.addAll(writeInst);
		inst.addAll(deleteInst);

		return inst;

	}

	private boolean compatibleWithChildrenInExecNodes(Vector<N> execNodes, N node) {
	  
	  for(int i=0; i < execNodes.size(); i++)
	  {
	    N tmpNode = execNodes.elementAt(i);
	    // for lops that execute in control program, compatibleJobs property is set to LopProperties.INVALID
	    // we should not consider such lops in this check
	    if (isChild(tmpNode, node) 
	    		&& tmpNode.getExecLocation() != ExecLocation.ControlProgram
	    		//&& tmpNode.getCompatibleJobs() != LopProperties.INVALID 
	    		&& (tmpNode.getCompatibleJobs() & node.getCompatibleJobs()) == 0)
	      return false;
	  }
	  
	  return true;
  }

  private void deleteUpdatedTransientReadVariables(Vector<N> nodeV,
			ArrayList<Instruction> inst) throws DMLRuntimeException,
			DMLUnsupportedOperationException {

		HashMap<String, N> labelNodeMapping = new HashMap<String, N>();
		HashSet<String> updatedLabels = new HashSet<String>();
		
    if(DEBUG)
      System.out.println("In delete updated variables");

		// first capture all transient read variables
		for (int i = 0; i < nodeV.size(); i++) {
			N node = nodeV.get(i);

			if (node.getExecLocation() == ExecLocation.Data
					&& ((Data) node).isTransient()
					&& ((Data) node).getOperationType() == OperationTypes.READ
					&& ((Data) node).get_dataType() == DataType.MATRIX) {
				labelNodeMapping.put(node.getOutputParameters().getLabel(),
						node);
			}
		}

		// capture updated transient write variables
		for (int i = 0; i < nodeV.size(); i++) {
			N node = nodeV.get(i);

			if (node.getExecLocation() == ExecLocation.Data
					&& ((Data) node).isTransient()
					&& ((Data) node).getOperationType() == OperationTypes.WRITE
					&& ((Data) node).get_dataType() == DataType.MATRIX
					&& labelNodeMapping.containsKey(node.getOutputParameters()
							.getLabel())
					&& !labelNodeMapping.containsValue(node.getInputs().get(0))) {
				updatedLabels.add(node.getOutputParameters().getLabel());
			}
		}

		// generate RM instructions

		Iterator<String> it = updatedLabels.iterator();

		while (it.hasNext()) {
			String label = it.next();

			if (DEBUG)
				System.out.println("CP" + Lops.OPERAND_DELIMITOR + "rmfilevar" + Lops.OPERAND_DELIMITOR + label
						+ Lops.VALUETYPE_PREFIX + Expression.ValueType.UNKNOWN
						+ Lops.OPERAND_DELIMITOR + "true"
						+ Lops.VALUETYPE_PREFIX + Expression.ValueType.BOOLEAN);
			inst.add(CPInstructionParser.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "rmfilevar"
					+ Lops.OPERAND_DELIMITOR + label + Lops.VALUETYPE_PREFIX
					+ Expression.ValueType.UNKNOWN + Lops.OPERAND_DELIMITOR
					+ "true" + Lops.VALUETYPE_PREFIX
					+ Expression.ValueType.BOOLEAN));
		}
	}
  
	String prepareVariableInstruction(String opcode, String varName, DataType dt, ValueType vt, String fileName, OutputParameters oparams, OutputInfo oinfo) {
  		StringBuilder inst = new StringBuilder();
  		
		inst.append("CP" + Lops.OPERAND_DELIMITOR + opcode);
  		if ( opcode == "createvar" ) {
  			inst.append(Lops.OPERAND_DELIMITOR + varName + Lops.DATATYPE_PREFIX + dt + Lops.VALUETYPE_PREFIX + vt);
  			inst.append(Lops.OPERAND_DELIMITOR + fileName + Lops.DATATYPE_PREFIX + DataType.SCALAR + Lops.VALUETYPE_PREFIX + ValueType.STRING);
  			inst.append(Lops.OPERAND_DELIMITOR + oparams.getNum_rows());
  			inst.append(Lops.OPERAND_DELIMITOR + oparams.getNum_cols());
  			inst.append(Lops.OPERAND_DELIMITOR + oparams.getNum_rows_per_block());
  			inst.append(Lops.OPERAND_DELIMITOR + oparams.getNum_cols_per_block());
  			inst.append(Lops.OPERAND_DELIMITOR + "0"); // TODO: should pass on correct NNZs
  			inst.append(Lops.OPERAND_DELIMITOR + OutputInfo.outputInfoToString(oinfo) ) ;
  		}
  		
  		return inst.toString();
  	}

	String prepareVariableInstruction(String opcode, N node) {
		OutputParameters oparams = node.getOutputParameters();
  		return prepareVariableInstruction(opcode, oparams.getLabel(), node.get_dataType(), node.get_valueType(), oparams.getFile_name(), oparams, getOutputInfo(node));
  	}

	/**
	 * Method to generate instructions that are executed in Control Program. At
	 * this point, this DAG has no dependencies on the MR dag. ie. none of the
	 * inputs are outputs of MR jobs
	 * 
	 * @param execNodes
	 * @param inst
	 * @param writeInst
	 * @param deleteInst
	 * @throws LopsException
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */

	@SuppressWarnings("unchecked")
	private void generateControlProgramJobs(Vector<N> execNodes,
			ArrayList<Instruction> inst, ArrayList<Instruction> writeInst,
			ArrayList<Instruction> deleteInst) throws LopsException,
			DMLUnsupportedOperationException, DMLRuntimeException {

		// nodes to be deleted from execnodes
		Vector<N> markedNodes = new Vector<N>();

		// variable names to be deleted
		Vector<String> var_deletions = new Vector<String>();

		for (int i = 0; i < execNodes.size(); i++) {
			N node = execNodes.get(i);

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
					NodeOutput out = setupNodeOutputs(node, ExecType.CP);
					inst.addAll(out.getPreInstructions()); // dummy
					deleteInst.addAll(out.getLastInstructions());
				} else {
					// Output from lops with non-SCALAR data type must
					// go into Temporary Files (temp0, temp1, etc.)
					
					NodeOutput out = setupNodeOutputs(node, ExecType.CP);
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
					
					if ( !hasTransientWriteParent )
						deleteInst.addAll(out.getLastInstructions());
				}

				String inst_string = "";

				// Since ParameterizedBuiltins have arbitrary number of inputs, we handle it separately
				if (node.getType() == Lops.Type.ParameterizedBuiltin){
					// TODO NEW: check why SCALAR and MATRIX have to be differentiated
					if ( node.get_dataType() == DataType.SCALAR ) {
						inst_string = ((ParameterizedBuiltin) node).getInstructions(node.getOutputParameters().getLabel());
					}
					else {
						inst_string = ((ParameterizedBuiltin) node).getInstructions(node.getOutputParameters().getFile_name());
					}
					
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
					else {
						throw new LopsException("Node with " + node.getInputs().size() + " inputs is not supported in CP yet!");
					}
				}
				
				try {
					if (DEBUG)
						System.out.println("Generating simple instruction - "
								+ inst_string);
					inst.add(CPInstructionParser
							.parseSingleInstruction(inst_string));
				} catch (Exception e) {
					throw new LopsException("Problem generating simple inst - "
							+ inst_string);
				}

				markedNodes.add(node);
				continue;
			}
			else if (node.getExecLocation() == ExecLocation.Data ) {
				Data dnode = (Data)node;
				Data.OperationTypes op = dnode.getOperationType();
				
				if ( op == Data.OperationTypes.WRITE ) {
					NodeOutput out = null;
					if ( sendWriteLopToMR(node) ) {
						// In this case, Data WRITE lop goes into MR, and 
						// we don't have to do anything here
					}
					else {
						out = setupNodeOutputs(node, ExecType.CP);
						if ( dnode.get_dataType() == DataType.SCALAR ) {
							// processing is same for both transient and persistent scalar writes 
							writeInst.addAll(out.getLastInstructions());
						}
						else {
							// setupNodeOutputs() handles both transient and persistent matrix writes 
							if ( dnode.isTransient() ) {
								//inst.addAll(out.getPreInstructions()); // dummy ?
								deleteInst.addAll(out.getLastInstructions());
							}
							else {
								// In case of persistent write lop, write() instruction will be generated 
								// and that instruction must be added to <code>inst</code> so that it gets
								// executed immediately. If it is added to <code>deleteInst</code> then it
								// gets executed at the end of program block's execution
								inst.addAll(out.getLastInstructions());
							}
						}
						markedNodes.add(node);
						continue;
					}
				}
				else {
					// generate a temp label to hold the value that is read from HDFS
					if ( node.get_dataType() == DataType.SCALAR ) {
						node.getOutputParameters().setLabel("Var" + var_index++);
						String io_inst = node.getInstructions(node.getOutputParameters().getLabel(), 
								node.getOutputParameters().getFile_name());
						inst.add(CPInstructionParser.parseSingleInstruction(io_inst));
						
						deleteInst.add(CPInstructionParser.parseSingleInstruction(
								"CP" + Lops.OPERAND_DELIMITOR 
								+ "rmvar" + Lops.OPERAND_DELIMITOR
								+ node.getLevel()));
					}
					else {
						throw new LopsException("Matrix READs are not handled in CP yet!");
					}
					markedNodes.add(node);
					continue;
				}
			}
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
	    
	    if(DEBUG)
	      System.out.println("Before remove nodes for next iteration -- size of execNodes " + execNodes.size());
 
	    

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

			if (DEBUG) {
				System.out.println("Checking for removal (" + tmpNode.getID()
						+ ") " + tmpNode.toString());
				System.out.println("Inputs in same job " + inputs_in_same_job);
				System.out.println("Unassigned inputs " + unassigned_inputs);
				System.out.println("Child queued " + child_queued);

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
			    if (DEBUG)
            System.out.println("Removing for next iteration: ("
                + tmpNode.getID() + ") " + tmpNode.toString());
			  }
			  else
				if (!hasOtherQueuedParentNode(tmpNode, queuedNodes, node)
						&& isChild(tmpNode, node)  && branchHasNoOtherUnExecutedParents(tmpNode, node, execNodes, finishedNodes))

	
				    if( 
				        //e.g. MMCJ
				        (node.getExecLocation() == ExecLocation.MapAndReduce &&
						branchCanBePiggyBackedMapAndReduce(tmpNode, node, execNodes, finishedNodes) && !tmpNode.definesMRJob() )
						||
						//e.g. Binary
						(node.getExecLocation() == ExecLocation.Reduce && branchCanBePiggyBackedReduce(tmpNode, node, execNodes, finishedNodes))  )
						{
					
				      if (DEBUG)
						    System.out.println("Removing for next iteration: ("
								    + tmpNode.getID() + ") " + tmpNode.toString());
					
					
	        				markedNodes.add(tmpNode);
						
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
						&& isChild(tmpNode, node) &&
						branchCanBePiggyBackedMapAndReduce(tmpNode, node, execNodes, finishedNodes)
						&& tmpNode.definesMRJob() != true) {
					if (DEBUG)
						System.out.println("Removing for next iteration:: ("
								+ tmpNode.getID() + ") " + tmpNode.toString());

					markedNodes.add(tmpNode);
				}

				// as this node has inputs coming from different jobs, need to
				// free up everything
				// below and include the closest MapAndReduce lop if this is of
				// type Reduce.
				// if it is of type MapAndReduce, don't need to free any nodes

				if (!inputs_in_same_job && !unassigned_inputs
						&& isChild(tmpNode, node) && 
						(tmpNode == node.getInputs().get(0) || tmpNode == node.getInputs().get(1)) && 
		            tmpNode.isAligner()) 
				{
					if (DEBUG)
						System.out
								.println("Removing for next iteration ("
										+ tmpNode.getID()
										+ ") "
										+ tmpNode.toString());

					markedNodes.add(tmpNode);

				}

			}
		}

		// we also need to delete all parent nodes of marked nodes
		for (int i = 0; i < execNodes.size(); i++) {
			if (DEBUG)
				System.out.println("Checking for removal - ("
						+ execNodes.elementAt(i).getID() + ") "
						+ execNodes.elementAt(i).toString());

			if (hasChildNode(execNodes.elementAt(i), markedNodes)) {
				markedNodes.add(execNodes.elementAt(i));
				if (DEBUG)
					System.out.println("Removing for next iteration - ("
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
       
       if(isChild(n, node) && isChild(tmpNode, n) && !node.getInputs().contains(tmpNode) && n.getExecLocation() != ExecLocation.Map && n.getExecLocation() != ExecLocation.MapOrReduce)
         return false;
         
      }
     
     
     return true;

  }

  private boolean branchCanBePiggyBackedMapAndReduce(N tmpNode, N node,
      Vector<N> execNodes, Vector<N> finishedNodes) {
	   
	  for(int i=0; i < execNodes.size(); i++)
	    {
	      N n = execNodes.elementAt(i);
	      
	      if(n.equals(node))
	        continue;
	      
	      if(n.equals(tmpNode) && n.getExecLocation() != ExecLocation.Map && n.getExecLocation() != ExecLocation.MapOrReduce)
	        return false;
	      
	      if(isChild(n, node) && isChild(tmpNode, n) && n.getExecLocation() != ExecLocation.Map && n.getExecLocation() != ExecLocation.MapOrReduce)
	        return false;
	        
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
	    
	    if(isChild(n, node) && isChild(tmpNode, n))
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
					&& isChild(n, node)) {
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

	private int jobType(Lops lops, ArrayList<Vector<N>> jobvec) throws LopsException {
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
					&& isChild(n, node)) {
				return true;
			} else {
				if (hasOtherMapAndReduceParentNode(n, execNodes, node))
					return true;
			}

		}

		return false;
	}

	/**
	 * Method to check if there is a queued node between tmpNode and node
	 * 
	 * @param tmpNode
	 * @param queuedNodes
	 * @param node
	 * @return
	 */

	@SuppressWarnings("unchecked")
	private boolean hasOtherQueuedParentNode(N tmpNode, Vector<N> queuedNodes,
			N node) {
		for (int i = 0; i < tmpNode.getOutputs().size(); i++) {
			N n = (N) tmpNode.getOutputs().get(i);

			if (queuedNodes.contains(n) && !n.equals(node) && isChild(n, node)) {
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
		if (!DEBUG)
			return;

		for ( JobType jt : JobType.values() ) {
			int i = jt.getId();
			if (i > 0 && jobNodes.get(i) != null && jobNodes.get(i).size() > 0) {
				System.out.println(jt.getName() + " Job Nodes:");
				for (int j = 0; j < jobNodes.get(i).size(); j++) {
					System.out.println("    "
							+ jobNodes.get(i).elementAt(j).getID() + ") "
							+ jobNodes.get(i).elementAt(j).toString());
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
				else if (isChild(rrnodes.get(rrid), gmrnodes.get(j))) {
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
	 * @param writeInst
	 * @param deleteinst
	 * @param jobNodes
	 * @throws LopsException
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */

	public void generateMRJobs(Vector<N> execNodes,
			ArrayList<Instruction> inst, ArrayList<Instruction> writeInst,
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

		if (DEBUG) {
			printJobNodes(jobNodes);
		}

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

				if (DEBUG)
					System.out.println("Generating " + jt.getName() + " job");

				if (jt.allowsRecordReaderInstructions() && hasANode(jobNodes.get(index), ExecLocation.RecordReader)) {
					// split the nodes by recordReader lops
					ArrayList<Vector<N>> rrlist = splitGMRNodesByRecordReader(jobNodes.get(index));
					for (int i = 0; i < rrlist.size(); i++) {
						generateMapReduceInstructions(rrlist.get(i), inst, deleteinst, jt);
					}
				}
				else if ( jt.allowsSingleShuffleInstruction() ) {
					// These jobs allow a single shuffle instruction. 
					// We should split the nodes so that a separate job is produced for each shuffle instruction.
					Lops.Type splittingLopType = jt.getShuffleLopType();
					
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
							
							generateMapReduceInstructions(nodesForASingleJob, inst, deleteinst, jt);
						}
					}
				}
				else {
					// the default case
					generateMapReduceInstructions(jobNodes.get(index), inst, deleteinst, jt);
				}
			}
		}

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
				throw new LopsException("Should only have one child!");
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
			if (isChild(node, exec_n.elementAt(i))) {
				if (!node_v.contains(exec_n.elementAt(i))) {
					if (DEBUG) {
						System.out.println("Adding parent - "
								+ exec_n.elementAt(i).toString());
					}
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
		if (node != null && DEBUG)
			System.out.println("Adding children " + node.toString());

		/** add child in exec nodes that is not of type scalar **/
		if (exec_n.contains(node)
				&& node.getExecLocation() != ExecLocation.ControlProgram) {
			if (!node_v.contains(node))
				node_v.add(node);

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
			throws LopsException {
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
		HashMap<String, dml.runtime.instructions.CPInstructions.Data> outputLabelValueMapping = new HashMap<String, dml.runtime.instructions.CPInstructions.Data>();

		HashMap<N, Integer> nodeIndexMapping = new HashMap<N, Integer>();

		int numReducers = total_reducers;
		int replication = 1;

		Vector<N> rootNodes = new Vector<N>();
		// find all root nodes
		getOutputNodes(execNodes, rootNodes, false);

		if (DEBUG) {
			System.out.println("rootNodes.size() = " + rootNodes.size()
					+ ", execNodes.size() = " + execNodes.size());
			rootNodes.get(0).printMe();
		}

		for (int i = 0; i < rootNodes.size(); i++)
			getInputPathsAndParameters(rootNodes.elementAt(i), execNodes,
					inputs, inputInfos, numRows, numCols, numRowsPerBlock,
					numColsPerBlock, nodeIndexMapping, inputLabels);

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
		
		mr.setPartitionInstructions(getStringArray(inputs), inputInfo,
				outputStrings, numReducers, replication, getLongArray(numRows),
				getLongArray(numCols), getIntArray(numRowsPerBlock),
				getIntArray(numColsPerBlock), pp.getResultIndexes(), pp
						.getResultDimsUnknown(), pp, inputLabels, outputLabels,
				outputLabelValueMapping);
		inst.add(mr);
	}
	
	/**
	 * Method that determines the output format for a given node.
	 * 
	 * @param node
	 * @return
	 */
	OutputInfo getOutputInfo(N node) {
		OutputInfo oinfo = null;
		
		if ( node.get_dataType() == DataType.SCALAR && node.getExecType() == ExecType.CP)
			return null;
		
		if (node.getOutputParameters().isBlocked_representation()) {
			oinfo = OutputInfo.BinaryBlockOutputInfo;
		} else {
			if (node.getOutputParameters().getFormat() == Format.TEXT)
				oinfo = OutputInfo.TextCellOutputInfo;
			else {
				oinfo = OutputInfo.BinaryCellOutputInfo;
			}
		}

		/* Instead of following hardcoding, one must get this information from Lops */
		if (node.getType() == Type.SortKeys) {
			oinfo = OutputInfo.OutputInfoForSortOutput; //new OutputInfo(CompactOutputFormat.class,
					//DoubleWritable.class, IntWritable.class);
		} else if (node.getType() == Type.CombineBinary) {
			// Output format of CombineBinary (CB) depends on how the output is consumed
			CombineBinary combine = (CombineBinary) node;
			if ( combine.getOperation() == dml.lops.CombineBinary.OperationTypes.PreSort ) {
				oinfo = OutputInfo.OutputInfoForSortInput; //new OutputInfo(SequenceFileOutputFormat.class,
						//DoubleWritable.class, IntWritable.class);
			}
			else if ( combine.getOperation() == dml.lops.CombineBinary.OperationTypes.PreCentralMoment 
					  || combine.getOperation() == dml.lops.CombineBinary.OperationTypes.PreCovUnweighted 
					  || combine.getOperation() == dml.lops.CombineBinary.OperationTypes.PreGroupedAggUnweighted ) {
				oinfo = OutputInfo.WeightedPairOutputInfo;
			}
		} else if ( node.getType() == Type.CombineTertiary) {
			oinfo = OutputInfo.WeightedPairOutputInfo;
		} else if (node.getType() == Type.CentralMoment
				|| node.getType() == Type.CoVariance 
				|| node.getType() == Type.GroupedAgg ) {
			// CMMR ang GroupedAggMR always operate in "cell mode",
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
  	String getOutputFormat(N node) {
  		return OutputInfo.outputInfoToString(getOutputInfo(node));
  	}
  	
	/**
	 * Method to setup output filenames and outputInfos, and to generate related instructions
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 * @throws LopsException 
	 */
	private NodeOutput setupNodeOutputs(N node, ExecType et) 
	throws DMLUnsupportedOperationException, DMLRuntimeException, LopsException {
		
		OutputParameters oparams = node.getOutputParameters();
		NodeOutput out = new NodeOutput();
		
		// Compute the output format for this node
		out.setOutInfo(getOutputInfo(node));
		
		// If node is NOT of type Data then we must generate 
		// a variable to hold the value produced by this node
		if (node.getExecLocation() != ExecLocation.Data) {
			
			if ( node.get_dataType() == DataType.SCALAR ) {
				oparams.setLabel("Var"+ var_index++);
				out.setVarName(oparams.getLabel());
				out.addLastInstruction(
						CPInstructionParser.parseSingleInstruction(
								"CP" + Lops.OPERAND_DELIMITOR 
								+ "rmvar" + Lops.OPERAND_DELIMITOR
								+ oparams.getLabel() ) );
			}
			else {
				// generate temporary filename and a variable name to hold the output produced by "rootNode"
				oparams.setFile_name(scratch + "temp" + job_id++);
				oparams.setLabel("mVar" + var_index++ );
				
				
				// generate an instruction that creates a symbol table entry for the new variable
				String createInst = prepareVariableInstruction("createvar", node);
				out.addPreInstruction(CPInstructionParser.parseSingleInstruction(createInst));
				
				// temp file as well as the variable has to be deleted at the end
				out.addLastInstruction(CPInstructionParser.parseSingleInstruction(
						"CP" + Lops.OPERAND_DELIMITOR + "rmfilevar" + Lops.OPERAND_DELIMITOR 
						+ oparams.getLabel() + Lops.VALUETYPE_PREFIX + Expression.ValueType.UNKNOWN + Lops.OPERAND_DELIMITOR 
						+ "true" + Lops.VALUETYPE_PREFIX + "BOOLEAN"));
	
				// finally, add the generated filename and variable name to the list of outputs 
				out.setFileName(oparams.getFile_name());
				out.setVarName(oparams.getLabel());
			}
		} 
		// rootNode is of type Data
		else {
			
			if ( node.get_dataType() == DataType.SCALAR ) {
				// generate assignment operations for final and transient writes
				if ( oparams.getFile_name() == null ) {
					String io_inst = "CP" + Lops.OPERAND_DELIMITOR + "assignvar"
						+ Lops.OPERAND_DELIMITOR
						+ node.getInputs().get(0).getOutputParameters().getLabel()
						+ Lops.DATATYPE_PREFIX + node.getInputs().get(0).get_dataType()
						+ Lops.VALUETYPE_PREFIX + node.getInputs().get(0).get_valueType()
						+ Lops.OPERAND_DELIMITOR
						+ node.getOutputParameters().getLabel()
						+ Lops.DATATYPE_PREFIX + node.get_dataType()
						+ Lops.VALUETYPE_PREFIX + node.get_valueType();
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
						 *     tVarH -> temp21tVarH
						 */
						// rename the temp variable to constant variable (e.g., mvvar mVar1 tVarH)
						out.addLastInstruction(CPInstructionParser
								.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "mvvar"
										+ Lops.OPERAND_DELIMITOR
										+ inputVarName
										+ Lops.OPERAND_DELIMITOR
										+ constVarName));
						
						// remove the old copy of transient file (rm temp21tVarH)
						out.addLastInstruction(CPInstructionParser
								.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "rm"
										+ Lops.OPERAND_DELIMITOR
										+ constFileName));
						
						// rename the new copy's name (e.g., mv temp21 temp21tVarA)
						out.addLastInstruction(CPInstructionParser
								.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "mv"
										+ Lops.OPERAND_DELIMITOR
										+ inputFileName
										+ Lops.OPERAND_DELIMITOR
										+ constFileName));
						
						// Now, tVarA refers to temporary filename temp21
						// we need to update the file name (e.g., setfilename tVarA temp21tVarA remote)
						out.addLastInstruction(CPInstructionParser
								.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "setfilename"
										+ Lops.OPERAND_DELIMITOR
										+ constVarName + Lops.DATATYPE_PREFIX + node.get_dataType() + Lops.VALUETYPE_PREFIX + node.get_valueType()
										+ Lops.OPERAND_DELIMITOR
										+ constFileName
										+ Lops.OPERAND_DELIMITOR
										+ "remote"));
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
						String tempFileName = scratch + "temp" + job_id++;
						
						String createInst = prepareVariableInstruction("createvar", tempVarName, node.get_dataType(), node.get_valueType(), tempFileName, oparams, out.getOutInfo());
						out.addPreInstruction(CPInstructionParser.parseSingleInstruction(createInst));
						
						String constVarName = oparams.getLabel();
						String constFileName = tempFileName + constVarName;
						
						oparams.setFile_name(scratch + constFileName);
						
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
						
						// rename the temp variable to constant variable (e.g., mvvar tVarAtemp tVarA)
						out.addLastInstruction(CPInstructionParser
								.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "mvvar"
										+ Lops.OPERAND_DELIMITOR
										+ tempVarName
										+ Lops.OPERAND_DELIMITOR
										+ constVarName));
						
						// remove the old copy of transient file (rm temp21tVarA)
						out.addLastInstruction(CPInstructionParser
								.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "rm"
										+ Lops.OPERAND_DELIMITOR
										+ constFileName));
						
						// rename the new copy's name (e.g., mv temp21 temp21tVarA)
						out.addLastInstruction(CPInstructionParser
								.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "mv"
										+ Lops.OPERAND_DELIMITOR
										+ tempFileName
										+ Lops.OPERAND_DELIMITOR
										+ constFileName));
						
						// Now, tVarA refers to temporary filename temp21
						// we need to update the file name (e.g., setfilename tVarA temp21tVarA remote)
						out.addLastInstruction(CPInstructionParser
								.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "setfilename"
										+ Lops.OPERAND_DELIMITOR
										+ constVarName + Lops.DATATYPE_PREFIX + node.get_dataType() + Lops.VALUETYPE_PREFIX + node.get_valueType()
										+ Lops.OPERAND_DELIMITOR
										+ constFileName
										+ Lops.OPERAND_DELIMITOR
										+ "remote"));
						
						// finally, add the temporary filename and variable name to the list of outputs 
						out.setFileName(tempFileName);
						out.setVarName(tempVarName);
					}
				} 
				// rootNode is not a transient write. It is a persistent write.
				else {
					if(et == ExecType.MR) {
						// create a variable to hold the result produced by this "rootNode"
						oparams.setLabel("pVar" + var_index++ );
						
						String createInst = prepareVariableInstruction("createvar", node);
						out.addPreInstruction(CPInstructionParser.parseSingleInstruction(createInst));
							
						// remove the variable
						out.addLastInstruction(CPInstructionParser.parseSingleInstruction(
								"CP" + Lops.OPERAND_DELIMITOR + "rmfilevar" + Lops.OPERAND_DELIMITOR 
								+ oparams.getLabel() + Lops.VALUETYPE_PREFIX + Expression.ValueType.UNKNOWN + Lops.OPERAND_DELIMITOR 
								+ "false" + Lops.VALUETYPE_PREFIX + "BOOLEAN"));
							
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
			ArrayList<Instruction> inst, ArrayList<Instruction> deleteinst,
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
		
		
		/* Find the nodes that produce an output */
		Vector<N> rootNodes = new Vector<N>();
		getOutputNodes(execNodes, rootNodes, !jt.producesIntermediateOutput());
		if (DEBUG) {
			System.out.println("# of root nodes = " + rootNodes.size());
		}
		
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
					numColsPerBlock, nodeIndexMapping, inputLabels);
		}
		// In case of RAND job, instructions are defined in the input file
		if (jt == JobType.RAND)
			randInstructions = inputs;
		
		int[] start_index = new int[1];
		start_index[0] = inputs.size();
		
		/* Get RecordReader Instructions */
		
		// currently, recordreader instructions are allowed only in GMR jobs
		if (jt == JobType.GMR) {
			for (int i = 0; i < rootNodes.size(); i++) {
				getRecordReaderInstructions(rootNodes.elementAt(i), execNodes,
						inputs, recordReaderInstructions, nodeIndexMapping,
						start_index, inputLabels);
			}
		}
		
		/* Get Mapper Instructions */
	
		for (int i = 0; i < rootNodes.size(); i++) {
			getMapperInstructions(rootNodes.elementAt(i), execNodes, inputs,
					mapperInstructions, nodeIndexMapping, start_index,
					inputLabels);
		}
		
		if (DEBUG) {
			System.out.println("    Input strings: " + inputs.toString());
			if (jt == JobType.RAND)
				System.out.println("    Rand instructions: " + getCSVString(randInstructions));
			if (jt == JobType.GMR)
				System.out.println("    RecordReader instructions: " + getCSVString(recordReaderInstructions));
			System.out.println("    Mapper instructions: " + getCSVString(mapperInstructions));
		}

		/* Get Shuffle and Reducer Instructions */
		
		ArrayList<String> shuffleInstructions = new ArrayList<String>();
		ArrayList<String> aggInstructionsReducer = new ArrayList<String>();
		ArrayList<String> otherInstructionsReducer = new ArrayList<String>();

		for (int i = 0; i < rootNodes.size(); i++) {
			int resultIndex = getAggAndOtherInstructions(
					rootNodes.elementAt(i), execNodes, shuffleInstructions, aggInstructionsReducer,
					otherInstructionsReducer, nodeIndexMapping, start_index,
					inputLabels);
			resultIndices.add(new Byte((byte) resultIndex));
			
			// setup output filenames and outputInfos and generate related instructions
			NodeOutput out = setupNodeOutputs(rootNodes.elementAt(i), ExecType.MR);
			outputLabels.add(out.getVarName());
			outputs.add(out.getFileName());
			//if ( out.getFileName() == null )
			//	System.err.println("error here .. ");
			outputInfos.add(out.getOutInfo());
			renameInstructions.addAll(out.getLastInstructions());
			variableInstructions.addAll(out.getPreInstructions());
			
		}
		
		// a sanity check
		if (resultIndices.size() != rootNodes.size()) {
			throw new LopsException("Unexected error in piggybacking: "
					+ resultIndices.size() + " != " + rootNodes.size());
		}
		
		/* Determine if the output dimensions are known */
		
		byte[] resultIndicesByte = new byte[resultIndices.size()];
		byte[] resultDimsUnknown = new byte[resultIndices.size()];
		for (int i = 0; i < resultIndicesByte.length; i++) {
			resultIndicesByte[i] = resultIndices.get(i).byteValue();
			// check if the output matrix dimensions are known at compile time
			if (rootNodes.elementAt(i).getOutputParameters().getNum_rows() == -1
					&& rootNodes.elementAt(i).getOutputParameters()
							.getNum_cols() == -1) {
				resultDimsUnknown[i] = (byte) 1;
			} else {
				resultDimsUnknown[i] = (byte) 0;
			}
		}
		
		if (DEBUG) {
			System.out.println("    Shuffle/Aggregate Instructions: " + getCSVString(aggInstructionsReducer));
			System.out.println("    Other instructions =" + getCSVString(otherInstructionsReducer));
			System.out.println("    Output strings: " + outputs.toString());
			System.out.println("    ResultIndices = " + resultIndices.toString());
			System.out.println("    ResultDimsUnknown = " + resultDimsUnknown.toString());
		}
		
		/* Prepare the MapReduce job instruction */
		
		MRJobInstruction mr = new MRJobInstruction(jt);
		
		// check if this is a map-only job. If not, set the number of reducers
		//if (getCSVString(shuffleInstructions).compareTo("") != 0 || getCSVString(aggInstructionsReducer).compareTo("") != 0
		//		|| getCSVString(otherInstructionsReducer).compareTo("") != 0)
		//	numReducers = total_reducers;
		if ( shuffleInstructions.size() > 0 || aggInstructionsReducer.size() > 0 || otherInstructionsReducer.size() > 0 )
			numReducers = total_reducers;
		
		// set inputs, outputs, and other other properties for the job 
		mr.setInputs(getStringArray(inputs),getInputInfoArray(inputInfos));
		mr.setInputDimensions(getLongArray(numRows), getIntArray(numRowsPerBlock), getLongArray(numCols), getIntArray(numColsPerBlock));
		
		mr.setOutputs(getStringArray(outputs), getOutputInfoArray(outputInfos), resultIndicesByte);
		mr.setOutputDimensions(resultDimsUnknown);
		
		mr.setNumberOfReducers(numReducers);
		mr.setReplication(replication);
		
		mr.setInputOutputLabels(inputLabels, outputLabels);
		
		// set instructions for recordReader and mapper
		mr.setRecordReaderInstructions(getCSVString(recordReaderInstructions));
		mr.setMapperInstructions(getCSVString(mapperInstructions));
		
		if ( jt == JobType.RAND )
			mr.setRandInstructions(getCSVString(randInstructions));
		
		// set shuffle instructions
		mr.setShuffleInstructions(getCSVString(shuffleInstructions));
		
		// set reducer instruction
		mr.setAggregateInstructionsInReducer(getCSVString(aggInstructionsReducer));
		mr.setOtherInstructionsInReducer(getCSVString(otherInstructionsReducer));

		/* Add the prepared instructions to output set */
		inst.addAll(variableInstructions);
		inst.add(mr);
		deleteinst.addAll(renameInstructions);

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
					s += "," + inputStrings.get(i);
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
			ArrayList<String> inputLabels) throws LopsException

	{

		int ret_val = -1;

		if (nodeIndexMapping.containsKey(node))
			return nodeIndexMapping.get(node);

		// if not an input source and not in exec nodes, return.

		if (!execNodes.contains(node))
			return ret_val;

		ArrayList<Integer> inputIndices = new ArrayList<Integer>();

		// recurse

		for (int i = 0; i < node.getInputs().size(); i++) {
			ret_val = getAggAndOtherInstructions((N) node.getInputs().get(i),
					execNodes, shuffleInstructions, aggInstructionsReducer,
					otherInstructionsReducer, nodeIndexMapping, start_index,
					inputLabels);
			inputIndices.add(ret_val);
		}

		// have to verify if this is needed
		if (node.getExecLocation() == ExecLocation.Data) {
			return ret_val;
		}

		if (node.getExecLocation() == ExecLocation.MapAndReduce) {
			
			/* Generate Shuffle Instruction for "node", and return the index associated with produced output */
			
			boolean flag = false;
			int output_index = start_index[0];
			switch(node.getType()) {
			
			/* Lop types that take a single input */
			case ReBlock:
			case SortKeys:
			case CentralMoment:
			case CoVariance:
			case GroupedAgg:
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
				flag = true;
				break;
			}
			
			if ( !flag ) { 
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
				|| hasChildNode(node, execNodes, ExecLocation.MapAndReduce)) {

			if (inputIndices.size() == 1) {
				int output_index = start_index[0];
				start_index[0]++;

				if (node.getType() == Type.Aggregate)
					aggInstructionsReducer.add(node.getInstructions(
							inputIndices.get(0), output_index));
				else
					otherInstructionsReducer.add(node.getInstructions(
							inputIndices.get(0), output_index));

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
							&& !((Data) (node.getInputs().get(index)))
									.isLiteral())
						inputLabels.add(node.getInputs().get(index)
								.getOutputParameters().getLabel());

					if (node.getInputs().get(index).getExecLocation() != ExecLocation.Data)
						inputLabels.add(node.getInputs().get(index)
								.getOutputParameters().getLabel());

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

				return output_index;

			} else
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
			ArrayList<String> inputLabels) throws LopsException

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
					start_index, inputLabels);

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
							&& !((Data) (node.getInputs().get(scalarIndex)))
									.isLiteral())
						inputLabels.add(node.getInputs().get(scalarIndex)
								.getOutputParameters().getLabel());

					// if not data lop, then this is an intermediate variable.
					if (node.getInputs().get(scalarIndex).getExecLocation() != ExecLocation.Data)
						inputLabels.add(node.getInputs().get(scalarIndex)
								.getOutputParameters().getLabel());
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
			ArrayList<String> inputLabels) throws LopsException

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
					start_index, inputLabels);

			inputIndices.add(ret_val);

			if (ret_val > max_input_index) {
				max_input_index = ret_val;
			}
		}

		// only map and map-or-reduce without a reduce child node can contribute
		// to mapper instructions.
		if ((node.getExecLocation() == ExecLocation.Map || node
				.getExecLocation() == ExecLocation.MapOrReduce)
				&& !hasChildNode(node, execNodes, ExecLocation.MapAndReduce)) {
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
						&& !((Data) (node.getInputs().get(index))).isLiteral())
					inputLabels.add(node.getInputs().get(index)
							.getOutputParameters().getLabel());

				// if not data lop, then this is an intermediate variable.
				if (node.getInputs().get(index).getExecLocation() != ExecLocation.Data)
					inputLabels.add(node.getInputs().get(index)
							.getOutputParameters().getLabel());

			}

			// get mapper instruction.
			if (node.getInputs().size() == 1)
				instructionsInMapper.add(node.getInstructions(inputIndices
						.get(0), output_index));
			if (node.getInputs().size() == 2)
				instructionsInMapper.add(node.getInstructions(inputIndices
						.get(0), inputIndices.get(1), output_index));
			if (node.getInputs().size() == 3)
				instructionsInMapper.add(node.getInstructions(inputIndices.get(0), 
															  inputIndices.get(1), 
															  inputIndices.get(2), 
															  output_index));
			
			if ( node.getInputs().size() == 5) {
				// Example: RangeBasedReIndex A[row_l:row_u, col_l:col_u]
				instructionsInMapper.add(node.getInstructions(
						inputIndices.get(0),
						inputIndices.get(1),
						inputIndices.get(2),
						inputIndices.get(3),
						inputIndices.get(4),
						output_index ));
			}
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
			HashMap<N, Integer> nodeIndexMapping, ArrayList<String> inputLabels)
			throws LopsException {
		// treat rand as an input.
		if (node.getType() == Type.RandLop && execNodes.contains(node)
				&& !nodeIndexMapping.containsKey(node)) {
			numRows.add(node.getOutputParameters().getNum_rows());
			numCols.add(node.getOutputParameters().getNum_cols());
			numRowsPerBlock.add(node.getOutputParameters()
					.getNum_rows_per_block());
			numColsPerBlock.add(node.getOutputParameters()
					.getNum_cols_per_block());
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
				inputStrings.add("##" + node.getOutputParameters().getLabel()
						+ "##");
			}
			//if ( node.getType() == Lops.Type.Data && ((Data)node).isTransient())
			//	inputStrings.add("##" + node.getOutputParameters().getLabel() + "##");
			//else
			//	inputStrings.add(node.getOutputParameters().getLabel());
			inputLabels.add(node.getOutputParameters().getLabel());

			numRows.add(node.getOutputParameters().getNum_rows());
			numCols.add(node.getOutputParameters().getNum_cols());
			numRowsPerBlock.add(node.getOutputParameters()
					.getNum_rows_per_block());
			numColsPerBlock.add(node.getOutputParameters()
					.getNum_cols_per_block());

			InputInfo nodeInputInfo = null;
			// Check if file format type is binary or text and update infos
			if (node.getOutputParameters().isBlocked_representation()) {
				if (node.getOutputParameters().getFormat() == Format.BINARY)
					nodeInputInfo = InputInfo.BinaryBlockInputInfo;
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
				if ( combine.getOperation() == dml.lops.CombineBinary.OperationTypes.PreSort ) {
					nodeInputInfo = new InputInfo(SequenceFileInputFormat.class,
							DoubleWritable.class, IntWritable.class);
				}
				else if ( combine.getOperation() == dml.lops.CombineBinary.OperationTypes.PreCentralMoment 
						  || combine.getOperation() == dml.lops.CombineBinary.OperationTypes.PreCovUnweighted
						  || combine.getOperation() == dml.lops.CombineBinary.OperationTypes.PreGroupedAggUnweighted ) {
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
					numColsPerBlock, nodeIndexMapping, inputLabels);
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
			if (node.getType() == Lops.Type.PartitionLop)
				partitionNodes.add(node);
		}
	}

	/**
	 * check to see if a is the child of b
	 * 
	 * @param a
	 * @param b
	 */

	public static boolean isChild(Lops a, Lops b) {
		for (int i = 0; i < b.getInputs().size(); i++) {
			if (b.getInputs().get(i).equals(a))
				return true;

			/**
			 * dfs search
			 */
			boolean return_val = isChild(a, b.getInputs().get(i));

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
		int numNodes = v.size();

		/*
		 * Step 1: compute the level for each node in the DAG.
		 * Step 2: sort the nodes by level, and within a level by node ID.
		 */
		
		/*
		 * Source nodes with no inputs are at level zero.
		 * level(v) = max( levels(v.inputs) ) + 1.
		 */
		// initialize
		for (int i = 0; i < numNodes; i++) {
			v.get(i).setLevel(-1);
		}

		// BFS (breadth-first) style of algorithm.
		Queue<N> q = new LinkedList<N>();

		/*
		 * A node is marked visited only afterall its inputs are visited.
		 * A node is added to sortedNodes and its levelmap is updated only when
		 * it is visited.
		 */
		int numSourceNodes = 0;
		for (int i = 0; i < numNodes; i++) {
			if (v.get(i).getInputs().size() == 0) {
				v.get(i).setLevel(0);
				q.add(v.get(i)); // add to queue
				numSourceNodes++;
			}
		}

		N n, parent;
		int maxLevel, inputLevel;
		boolean markAsVisited;
		while (q.size() != 0) {
			n = q.remove();

			// check if outputs of "n" can be marked as visited
			for (int i = 0; i < n.getOutputs().size(); i++) {
				parent = (N) n.getOutputs().get(i);

				markAsVisited = true;
				maxLevel = -1;
				for (int j = 0; j < parent.getInputs().size(); j++) {
					inputLevel = parent.getInputs().get(j).getLevel(); 
					if (inputLevel == -1) {
						// "parent" can not be visited if any of its inputs are
						// not visited
						markAsVisited = false;
						break;
					}

					if (maxLevel < inputLevel)
						maxLevel = inputLevel;
				}

				if (markAsVisited == true) {
					// mark "parent" as visited
					parent.setLevel(maxLevel + 1);
					q.add(parent);
				}
			}
		}

		// Step2: sort nodes by level, and then by node ID
		Object[] nodearray = v.toArray();
		Arrays.sort(nodearray, new LopComparator());

		// Copy sorted nodes into "v"
		v.clear();
		for (int i = 0; i < nodearray.length; i++) {
			v.add((N) nodearray[i]);
		}

		// Sanity check -- can be removed
		for (int i = 1; i < v.size(); i++) {
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

		// print the nodes in sorted order
		if (DEBUG) {
			for (int i = 0; i < v.size(); i++) {
				// System.out.print(sortedNodes.get(i).getID() + "("
				// + levelmap.get(sortedNodes.get(i).getID()) + "), ");
				System.out.print(v.get(i).getID() + "(" + v.get(i).getLevel()
						+ "), ");
			}
			System.out.println("");
			System.out.println("topological sort -- done");
		}

	}

	@SuppressWarnings("unchecked")
	private boolean hasChildNode(N node, Vector<N> childNodes, ExecLocation type) {

		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);
			if (childNodes.contains(n) && n.getExecLocation() == type) {
				return true;
			} else {
				if (hasChildNode(n, childNodes, type))
					return true;
			}

		}

		return false;

	}

	@SuppressWarnings("unchecked")
	private N getChildNode(N node, Vector<N> childNodes, ExecLocation type) {

		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);
			if (childNodes.contains(n) && n.getExecLocation() == type) {
				return n;
			} else {
				return getChildNode(n, childNodes, type);
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

		for (int i = 0; i < node.getOutputs().size(); i++) {
			N n = (N) node.getOutputs().get(i);
			if (parentNodes.contains(n) && n.getExecLocation() == type) {
				return n;
			} else {
				return getParentNode(n, parentNodes, type);
			}

		}

		return null;

	}

	// Checks if "node" has any descendants in nodesVec with definedMRJob flag
	// set to true
	@SuppressWarnings("unchecked")
	private boolean hasMRJobChildNode(N node, Vector<N> nodesVec) {

		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);
			if (nodesVec.contains(n) && n.definesMRJob()) {
				return true;
			} else {
				if (hasMRJobChildNode(n, nodesVec))
					return true;
			}

		}

		return false;

	}

	// Find the descendant of "node" in "nodeVec" that has its definedMRJob flag set to true
	// returns null if no such descendant exists in nodeVec
	@SuppressWarnings("unchecked")
	private N getMRJobChildNode(N node, Vector<N> nodesVec) {

		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);
			if (nodesVec.contains(n) && n.definesMRJob()) {
				return n;
			} else {
				N ret = getMRJobChildNode(n, nodesVec);
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

		for (int i = 0; i < node.getInputs().size(); i++) {
			N n = (N) node.getInputs().get(i);
			if (nodes.contains(n)) {
				return true;
			} else {
				if (hasChildNode(n, nodes))
					return true;
			}

		}

		return false;

	}

	@SuppressWarnings("unchecked")
	private boolean hasParentNode(N node, Vector<N> childNodes) {

		for (int i = 0; i < node.getOutputs().size(); i++) {
			N n = (N) node.getOutputs().get(i);
			if (childNodes.contains(n)) {
				return true;
			} else {
				if (hasParentNode(n, childNodes))
					return true;
			}

		}

		return false;
	}
}
