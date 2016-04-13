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

package org.apache.sysml.runtime.controlprogram.context;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.ConfigConstants;
import org.apache.flink.runtime.util.EnvironmentInformation;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.flink.data.DataSetObject;
import org.apache.sysml.runtime.instructions.flink.functions.CopyBinaryCellFunction;
import org.apache.sysml.runtime.instructions.flink.functions.CopyBlockPairFunction;
import org.apache.sysml.runtime.instructions.flink.functions.CopyTextInputFunction;
import org.apache.sysml.runtime.instructions.flink.utils.DataSetAggregateUtils;
import org.apache.sysml.runtime.instructions.flink.utils.IOUtils;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.utils.Statistics;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class FlinkExecutionContext extends ExecutionContext {

	// fraction of memory the unmanaged memory that should be used as upper bound
	// when allocating, e.g. buffers for matrix blocks, in UDFs
	private static final double USER_MEMORY_FRACTION = 0.9;

	//executor memory and relative fractions as obtained from the flink configuration
	private static long _memTaskManagerTotal = -1; // total heap memory per taskmanager
	private static double _memTaskManagerManagedFraction = 0.7; // ratio that flink uses for managed memory
	private static double _memTaskManagerManaged = -1; // absolute managed memory for flink operators
	private static int _numTaskmanagers = -1; //total executors
	private static int _defaultPar = -1; //total vcores
	private static int _slotsPerTaskManager = -1;
	private static long _memNetworkBuffers = -1;
	private static boolean _confOnly = false; //infrastructure info based on config

	private static final Log LOG = LogFactory.getLog(FlinkExecutionContext.class.getName());

	private static ExecutionEnvironment _execEnv = null;


	protected FlinkExecutionContext(Program prog) {
		this(true, prog);
	}

	protected FlinkExecutionContext(boolean allocateVars, Program prog) {
		super(allocateVars, prog);

		//if (OptimizerUtils.isHybridExecutionMode())
		initFlinkContext();
	}

	public ExecutionEnvironment getFlinkContext() {
		return _execEnv;
	}

	public DataSet<Tuple2<MatrixIndexes, MatrixBlock>> getBinaryBlockDataSetHandleForVariable(String varname)
			throws DMLRuntimeException {

		return (DataSet<Tuple2<MatrixIndexes, MatrixBlock>>) getDataSetHandleForVariable(varname,
				InputInfo.BinaryBlockInputInfo);
	}

	public DataSet<?> getDataSetHandleForVariable(String varname, InputInfo inputInfo)
			throws DMLRuntimeException {

		MatrixObject mo = getMatrixObject(varname);
		return getDataSetHandleForMatrixObject(mo, inputInfo);
	}

	public void setDataSetHandleForVariable(String varname,
											DataSet<Tuple2<MatrixIndexes, MatrixBlock>> ds) throws DMLRuntimeException {
		MatrixObject mo = getMatrixObject(varname);
		DataSetObject dsHandle = new DataSetObject(ds, varname);
		mo.setDataSetHandle(dsHandle);
	}

	public void addLineageDataSet(String varParent, String varChild) throws DMLRuntimeException {
		DataSetObject parent = getMatrixObject(varParent).getDataSetHandle();
		DataSetObject child = getMatrixObject(varChild).getDataSetHandle();

		parent.addLineageChild(child);
	}


	private DataSet<?> getDataSetHandleForMatrixObject(MatrixObject mo, InputInfo inputInfo)
			throws DMLRuntimeException {

		//FIXME this logic should be in matrix-object (see spark version of this method for more info)
		DataSet<?> dataSet = null;

		//CASE 1: dataset already existing (reuse if checkpoint or trigger
		//pending dataset operations if not yet cached but prevent to re-evaluate
		//dataset operations if already executed and cached
		if (mo.getDataSetHandle() != null
				&& (mo.getDataSetHandle().isCheckpointed() || !mo.isCached(false))) {
			//return existing dataset handling (w/o input format change)
			dataSet = mo.getDataSetHandle().getDataSet();
		}
		//CASE 2: dirty in memory data or cached result of dataset operations
		else if (mo.isDirty() || mo.isCached(false)) {
			//get in-memory matrix block and parallelize it
			//w/ guarded parallelize (fallback to export, dataset from file if too large)
			boolean fromFile = false;
			// TODO (see spark case for large matrices)

			//default case
			MatrixBlock mb = mo.acquireRead(); //pin matrix in memory
			dataSet = toDataSet(getFlinkContext(), mb, (int) mo.getNumRowsPerBlock(), (int) mo.getNumColumnsPerBlock());
			mo.release(); //unpin matrix


			//keep dataset handle for future operations on it
			DataSetObject dshandle = new DataSetObject(dataSet, mo.getVarName());
			dshandle.setHDFSFile(fromFile);
			mo.setDataSetHandle(dshandle);
		}
		//CASE 3: non-dirty (file exists on HDFS)
		else {

			// parallelize hdfs-resident file
			// For binary block, these are: SequenceFileInputFormat.class, MatrixIndexes.class, MatrixBlock.class
			if (inputInfo == InputInfo.BinaryBlockInputInfo) {
				dataSet = IOUtils.hadoopFile(getFlinkContext(), mo.getFileName(), inputInfo.inputFormatClass,
						inputInfo.inputKeyClass, inputInfo.inputValueClass);
				dataSet = ((DataSet<Tuple2<MatrixIndexes, MatrixBlock>>) dataSet).map(
						new CopyBlockPairFunction()); //cp is workaround for read bug
			} else if (inputInfo == InputInfo.TextCellInputInfo || inputInfo == InputInfo.CSVInputInfo || inputInfo == InputInfo.MatrixMarketInputInfo) {
				dataSet = IOUtils.hadoopFile(getFlinkContext(), mo.getFileName(), inputInfo.inputFormatClass,
						inputInfo.inputKeyClass, inputInfo.inputValueClass);
				dataSet = ((DataSet<Tuple2<LongWritable, Text>>) dataSet).map(
						new CopyTextInputFunction()); //cp is workaround for read bug
			} else if (inputInfo == InputInfo.BinaryCellInputInfo) {
				dataSet = IOUtils.hadoopFile(getFlinkContext(), mo.getFileName(), inputInfo.inputFormatClass,
						inputInfo.inputKeyClass, inputInfo.inputValueClass);
				dataSet = ((DataSet<Tuple2<MatrixIndexes, MatrixCell>>) dataSet).map(
						new CopyBinaryCellFunction()); //cp is workaround for read bug
			} else {
				throw new DMLRuntimeException("Incorrect input format in getDatasetHandleForVariable");
			}

			//keep dataset handle for future operations on it
			DataSetObject dataSetHandle = new DataSetObject(dataSet, mo.getVarName());
			dataSetHandle.setHDFSFile(true);
			mo.setDataSetHandle(dataSetHandle);
		}
		return dataSet;
	}

	private synchronized static void initFlinkContext() {
		_execEnv = ExecutionEnvironment.getExecutionEnvironment();
	}

	/**
	 * Utility method for creating an dataset out of an in-memory matrix block.
	 *
	 * @param env
	 * @param src
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static DataSet<Tuple2<MatrixIndexes, MatrixBlock>> toDataSet(ExecutionEnvironment env, MatrixBlock src,
																		int brlen, int bclen)
			throws DMLRuntimeException {
		LinkedList<Tuple2<MatrixIndexes, MatrixBlock>> list = new LinkedList<Tuple2<MatrixIndexes, MatrixBlock>>();

		if (src.getNumRows() <= brlen
				&& src.getNumColumns() <= bclen) {
			list.addLast(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(1, 1), src));
		} else {
			boolean sparse = src.isInSparseFormat();

			//create and write subblocks of matrix
			for (int blockRow = 0; blockRow < (int) Math.ceil(src.getNumRows() / (double) brlen); blockRow++)
				for (int blockCol = 0; blockCol < (int) Math.ceil(src.getNumColumns() / (double) bclen); blockCol++) {
					int maxRow = (blockRow * brlen + brlen < src.getNumRows()) ? brlen : src.getNumRows() - blockRow * brlen;
					int maxCol = (blockCol * bclen + bclen < src.getNumColumns()) ? bclen : src.getNumColumns() - blockCol * bclen;

					MatrixBlock block = new MatrixBlock(maxRow, maxCol, sparse);

					int row_offset = blockRow * brlen;
					int col_offset = blockCol * bclen;

					//copy submatrix to block
					src.sliceOperations(row_offset, row_offset + maxRow - 1,
							col_offset, col_offset + maxCol - 1, block);

					//append block to sequence file
					MatrixIndexes indexes = new MatrixIndexes(blockRow + 1, blockCol + 1);
					list.addLast(new Tuple2<MatrixIndexes, MatrixBlock>(indexes, block));
				}
		}

		return env.fromCollection(list);
	}

	/**
	 * @param dataset
	 * @param oinfo
	 */
	@SuppressWarnings("unchecked")
	public static long writeDataSetToHDFS(DataSetObject dataset, String path,
										  OutputInfo oinfo) throws DMLRuntimeException {
		DataSet<Tuple2<MatrixIndexes, MatrixBlock>> ldataset = (DataSet<Tuple2<MatrixIndexes, MatrixBlock>>) dataset.getDataSet();

		//recompute nnz
		long nnz = DataSetAggregateUtils.computeNNZFromBlocks(ldataset);

		//save file is an action which also triggers nnz maintenance
		IOUtils.saveAsHadoopFile(ldataset,
				path,
				oinfo.outputKeyClass,
				oinfo.outputValueClass,
				oinfo.outputFormatClass);

		//return nnz aggregate of all blocks
		return nnz;
	}

	/**
	 * This method is a generic abstraction for calls from the buffer pool.
	 * See toMatrixBlock(DataSet<Tuple2<MatrixIndexes,MatrixBlock>> dataset, int numRows, int numCols);
	 *
	 * @param dataset
	 * @param numRows
	 * @param numCols
	 * @return
	 * @throws DMLRuntimeException
	 */
	@SuppressWarnings("unchecked")
	public static MatrixBlock toMatrixBlock(DataSetObject dataset, int rlen, int clen, int brlen, int bclen, long nnz)
			throws DMLRuntimeException {
		return toMatrixBlock(
				(DataSet<Tuple2<MatrixIndexes, MatrixBlock>>) dataset.getDataSet(),
				rlen, clen, brlen, bclen, nnz);
	}

	/**
	 * Utility method for creating a single matrix block out of a binary block RDD.
	 * Note that this collect call might trigger execution of any pending transformations.
	 * <p>
	 * NOTE: This is an unguarded utility function, which requires memory for both the output matrix
	 * and its collected, blocked representation.
	 *
	 * @param dataSet
	 * @param numRows
	 * @param numCols
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock toMatrixBlock(DataSet<Tuple2<MatrixIndexes, MatrixBlock>> dataSet, int rlen, int clen,
											int brlen, int bclen, long nnz)
			throws DMLRuntimeException {

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;

		MatrixBlock out = null;

		if (rlen <= brlen && clen <= bclen) //SINGLE BLOCK
		{
			//special case without copy and nnz maintenance
			List<Tuple2<MatrixIndexes, MatrixBlock>> list = null;
			try {
				list = dataSet.collect();
			} catch (Exception e) {
				e.printStackTrace();
			}

			if (list.size() > 1)
				throw new DMLRuntimeException("Expecting no more than one result block.");
			else if (list.size() == 1)
				out = list.get(0).f1;
			else //empty (e.g., after ops w/ outputEmpty=false)
				out = new MatrixBlock(rlen, clen, true);
		} else //MULTIPLE BLOCKS
		{
			//determine target sparse/dense representation
			long lnnz = (nnz >= 0) ? nnz : (long) rlen * clen;
			boolean sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, lnnz);

			//create output matrix block (w/ lazy allocation)
			out = new MatrixBlock(rlen, clen, sparse);

			List<Tuple2<MatrixIndexes, MatrixBlock>> list = null;
			try {
				list = dataSet.collect();
			} catch (Exception e) {
				e.printStackTrace();
			}

			//copy blocks one-at-a-time into output matrix block
			for (Tuple2<MatrixIndexes, MatrixBlock> keyval : list) {
				//unpack index-block pair
				MatrixIndexes ix = keyval.f0;
				MatrixBlock block = keyval.f1;

				//compute row/column block offsets
				int row_offset = (int) (ix.getRowIndex() - 1) * brlen;
				int col_offset = (int) (ix.getColumnIndex() - 1) * bclen;
				int rows = block.getNumRows();
				int cols = block.getNumColumns();

				if (sparse) { //SPARSE OUTPUT
					//append block to sparse target in order to avoid shifting
					//note: this append requires a final sort of sparse rows
					out.appendToSparse(block, row_offset, col_offset);
				} else { //DENSE OUTPUT
					out.copy(row_offset, row_offset + rows - 1,
							col_offset, col_offset + cols - 1, block, false);
				}
			}

			//post-processing output matrix
			if (sparse)
				out.sortSparseRows();
			out.recomputeNonZeros();
			out.examSparsity();
		}

		if (DMLScript.STATISTICS) {
			Statistics.accSparkCollectTime(System.nanoTime() - t0);
			Statistics.incSparkCollectCount(1);
		}

		return out;
	}

	@SuppressWarnings("unchecked")
	public static MatrixBlock toMatrixBlock(DataSetObject dataset, int rlen, int clen, long nnz)
			throws DMLRuntimeException {
		return toMatrixBlock(
				(DataSet<Tuple2<MatrixIndexes, MatrixCell>>) dataset.getDataSet(),
				rlen, clen, nnz);
	}

	/**
	 * Utility method for creating a single matrix block out of a binary cell dataset.
	 * Note that this collect call might trigger execution of any pending transformations.
	 *
	 * @param dataset
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock toMatrixBlock(DataSet<Tuple2<MatrixIndexes, MatrixCell>> dataset, int rlen, int clen,
											long nnz)
			throws DMLRuntimeException {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;

		MatrixBlock out = null;

		//determine target sparse/dense representation
		long lnnz = (nnz >= 0) ? nnz : (long) rlen * clen;
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, lnnz);

		//create output matrix block (w/ lazy allocation)
		out = new MatrixBlock(rlen, clen, sparse);

		List<Tuple2<MatrixIndexes, MatrixCell>> list = null;
		try {
			list = dataset.collect();
		} catch (Exception e) {
			throw new DMLRuntimeException(e);
		}

		//copy blocks one-at-a-time into output matrix block
		for (Tuple2<MatrixIndexes, MatrixCell> keyval : list) {
			//unpack index-block pair
			MatrixIndexes ix = keyval.f0;
			MatrixCell cell = keyval.f1;

			//append cell to dense/sparse target in order to avoid shifting for sparse
			//note: this append requires a final sort of sparse rows
			out.appendValue((int) ix.getRowIndex() - 1, (int) ix.getColumnIndex() - 1, cell.getValue());
		}

		//post-processing output matrix
		if (sparse)
			out.sortSparseRows();
		out.recomputeNonZeros();
		out.examSparsity();

		if (DMLScript.STATISTICS) {
			Statistics.accSparkCollectTime(System.nanoTime() - t0);
			Statistics.incSparkCollectCount(1);
		}

		return out;
	}

	/**
	 * This is a wrapper for flink execute() method. It has to becalled to trigger the execution of remaining
	 * dataflows. Since we currently don't know if a flink dataflow is constructed (e.g. in hybrid_flink mode),
	 * a call to execute leads to an error if no sinks are defined.
	 */
	public void execute() {
		// FIXME track number of sinks
		// HACK ALERT !!!
		try {
			_execEnv.execute();
		} catch (Exception e) {
			System.out.println("Called execute() without defined sinks. Find a better solution");
		}
	}

	/**
	 * Flink reserves a fixed size of heap memory for operators (sort, hashing, ...).
	 * This memory is not available for broadcast and user object. In some cases SystemML creates large buffers.
	 * The problem is that flink statically assigns the remaining memory to every running task per slot.
	 * This can lead to the situation in which not enough memory for the buffers remains (even though the statically
	 * reserved Flink-part is not completely occupied).
	 *
	 * It is not easy to get the configuration parameters from Flink and to read taskmanager configuration.
	 * Therefore, we use a very conservative estimation based on Flink's default setting.
	 *
	 * @return
	 */

	/**
	 * Returns an estimated upper bound of the memory (in bytes) available in a task manager's user space (UDFs).
	 * <p>
	 * Flink manages its on memory for certain operations (sorting, hashing, ...). Therefore,
	 * it allocates a fraction of the currently available heap memory (default) on the task manager startup.
	 * This fraction can lazily grow up to a (user) defined limit of the heap. As we can not access the
	 * concrete memory settings easily, this method returns a conservative fraction ({@link #USER_MEMORY_FRACTION})
	 * of the memory not managed by Flink (maximal fraction of memory that can be allocated by Flink).
	 * <p>
	 * NOTE: The available estimate returned is SHARED among all slots in a task manager! To get a max. estimate
	 * for a single slot, this value has to be divided by the number of slots.
	 *
	 * @return an estimated upper bound of memory (in bytes) available in a task manager's user space (UDFs).
	 */
	public static long getUDFMemoryBudget() {
		long freeMemory = EnvironmentInformation.getSizeOfFreeHeapMemoryWithDefrag();
		// FIXME the 0.7 here is flinks default for static memory allocation - should be read from the config
		// As we can not get the set value, we rely on the default...
		double flinkManagedMemory = freeMemory * ConfigConstants.DEFAULT_MEMORY_MANAGER_MEMORY_FRACTION;
		double maxFreeMemory = freeMemory - flinkManagedMemory;

		return (long) (maxFreeMemory * USER_MEMORY_FRACTION);
	}

	/**
	 * Computes the memory that is available to every thread in the taskmanager
	 *
	 * @return
	 */

	public static int getDefaultParallelism() {
		return getDefaultParallelism(false);
	}

	public static int getDefaultParallelism(boolean refresh) {
		if (_defaultPar < 0 || refresh) {
			_defaultPar = _execEnv.getParallelism();
		}
		return _defaultPar;
	}

	// FIXME this method does not fully work right now since we cannot parse the configuration without a Jobmanager running
	public static void analyzeFlinkConfiguration(ExecutionConfig conf) {

		Map<String, String> parameters = conf.getGlobalJobParameters().toMap();

		// get the total memory for the taskmanager
		_memTaskManagerTotal = Integer.parseInt(parameters.getOrDefault("taskmanager.heap.mb", "512")) * 1024 * 1024;
		// get the memory that is managed by flink for the operators (sorting, hashing, caching)
		int taskManagerMemSize = Integer.parseInt(parameters.getOrDefault("taskmanager.memory.size", "-1"));

		// if the task manager memory size is not set, it is evaluated with the fraction
		if (taskManagerMemSize > 0) {
			_memTaskManagerManaged = taskManagerMemSize * 1024 * 1024;
		} else {
			_memTaskManagerManagedFraction = Double.parseDouble(
					parameters.getOrDefault("taskmanager.memory.fraction", "0.7"));
			_memTaskManagerManaged = _memTaskManagerManagedFraction * _memTaskManagerTotal;
		}

		// number of slots per taskmanager
		_slotsPerTaskManager = Integer.parseInt(parameters.getOrDefault("taskmanager.numberOfTaskSlots", "1"));
		// total number of slots
		_defaultPar = Integer.parseInt(parameters.getOrDefault("parallelization.degree.default", "1"));
		// number of taskmanagers = ceil(total slots / number of slots per taskmanager)
		_numTaskmanagers = (int) Math.ceil(_defaultPar / _slotsPerTaskManager); // calculate the number of taskmanagers

		// network buffers (for boradcasts/shuffles)
		int numNetworkBuffers = Integer.parseInt(
				parameters.getOrDefault("taskmanager.network.numberOfBuffers", "2048"));
		int networkBufferSize = Integer.parseInt(
				parameters.getOrDefault("taskmanager.network.bufferSizeInBytes", "32768"));
		_memNetworkBuffers = numNetworkBuffers * networkBufferSize;
	}
}
