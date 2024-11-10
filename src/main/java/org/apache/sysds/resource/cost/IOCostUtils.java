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

package org.apache.sysds.resource.cost;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class IOCostUtils {

	// empirical factor to scale down theoretical peak CPU performance
	private static final double COMPUTE_EFFICIENCY = 0.5;
	private static final double READ_DENSE_FACTOR = 0.5;
	private static final double WRITE_DENSE_FACTOR = 0.3;
	private static final double SPARSE_FACTOR = 0.5;
	private static final double TEXT_FACTOR = 0.5;
	// empirical value for data transfer between S3 and EC2 instances
	private static final double S3_COMPUTE_BOUND = 1.2 * 1E+9; // GFLOP/MB
	private static final double SERIALIZATION_FACTOR = 0.5;
	private static final double DESERIALIZATION_FACTOR = 0.8;
	public static final long DEFAULT_FLOPS = 2L * 1024 * 1024 * 1024; // 2 GFLOPS

	public static class IOMetrics {
		// FLOPS value not directly related to I/O metrics,
		// but it is not worth it to store it separately
		long cpuFLOPS;
		int cpuCores;
		// All metrics here use MB/s bandwidth unit
		// Metrics for disk I/O operations
		double localDiskReadBandwidth;
		double localDiskWriteBandwidth;
		double hdfsReadBinaryDenseBandwidth;
		double hdfsReadBinarySparseBandwidth;
		double hdfsWriteBinaryDenseBandwidth;
		double hdfsWriteBinarySparseBandwidth;
		double hdfsReadTextDenseBandwidth;
		double hdfsReadTextSparseBandwidth;
		double hdfsWriteTextDenseBandwidth;
		double hdfsWriteTextSparseBandwidth;
		double s3BandwidthEfficiency;
		// Metrics for main memory I/O operations
		double memReadBandwidth;
		double memWriteBandwidth;
		// Metrics for networking operations
		double networkingBandwidth;
		// Metrics for (de)serialization
		double serializationBandwidth;
		double deserializationBandwidth;

		public IOMetrics(CloudInstance instance) {
			this(
					instance.getFLOPS(),
					instance.getVCPUs(),
					instance.getMemoryBandwidth(),
					instance.getDiskReadBandwidth(),
					instance.getDiskWriteBandwidth(),
					instance.getNetworkBandwidth()
			);
		}
		public IOMetrics(long flops, int cores, double memoryBandwidth, double diskReadBandwidth, double diskWriteBandwidth, double networkBandwidth) {
			// CPU metrics
			cpuFLOPS = (long) (flops * COMPUTE_EFFICIENCY);
			cpuCores = cores;
			// Metrics for main memory I/O operations
			memReadBandwidth = memoryBandwidth;
			memWriteBandwidth = memoryBandwidth;
			// Metrics for networking operations
			networkingBandwidth = networkBandwidth;
			// Metrics for disk I/O operations
			localDiskReadBandwidth = diskReadBandwidth;
			localDiskWriteBandwidth = diskReadBandwidth;
			// Assume that the HDFS I/O operations is done always by accessing local blocks
			hdfsReadBinaryDenseBandwidth = diskReadBandwidth * READ_DENSE_FACTOR;
			hdfsReadBinarySparseBandwidth = hdfsReadBinaryDenseBandwidth * SPARSE_FACTOR;
			hdfsWriteBinaryDenseBandwidth = diskWriteBandwidth * WRITE_DENSE_FACTOR;
			hdfsWriteBinarySparseBandwidth = hdfsWriteBinaryDenseBandwidth * SPARSE_FACTOR;
			hdfsReadTextDenseBandwidth = hdfsReadBinaryDenseBandwidth * TEXT_FACTOR;
			hdfsReadTextSparseBandwidth = hdfsReadBinarySparseBandwidth * TEXT_FACTOR;
			hdfsWriteTextDenseBandwidth = hdfsWriteBinaryDenseBandwidth * TEXT_FACTOR;
			hdfsWriteTextSparseBandwidth = hdfsWriteBinarySparseBandwidth * TEXT_FACTOR;
			s3BandwidthEfficiency = (S3_COMPUTE_BOUND / cpuFLOPS); // [s/MB]
			// Metrics for (de)serialization
			double currentFlopsFactor = (double) DEFAULT_FLOPS / cpuFLOPS;
			serializationBandwidth = memReadBandwidth * SERIALIZATION_FACTOR * currentFlopsFactor;
			deserializationBandwidth = memWriteBandwidth * DESERIALIZATION_FACTOR * currentFlopsFactor;
		}

		// ----- Testing default -----
		public static final int DEFAULT_NUM_CPU_CORES = 8;
		//IO Read
		public static final double DEFAULT_MBS_MEMORY_BANDWIDTH = 21328.0; // e.g. DDR4-2666
		public static final double DEFAULT_MBS_DISK_BANDWIDTH = 600; // e.g. m5.4xlarge, baseline bandwidth: 4750Mbps = 593.75 MB/s
		public static final double DEFAULT_MBS_NETWORK_BANDWIDTH = 640; // e.g. m5.4xlarge, baseline bandwidth: 5Gbps = 640MB/s
		public static final double DEFAULT_MBS_HDFS_READ_BINARY_DENSE = 150;
		public static final double DEFAULT_MBS_HDFS_READ_BINARY_SPARSE = 75;
		public static final double DEFAULT_MBS_HDFS_READ_TEXT_DENSE = 75;
		public static final double DEFAULT_MBS_HDFS_READ_TEXT_SPARSE = 50;
		// IO Write
		public static final double DEFAULT_MBS_HDFS_WRITE_BINARY_DENSE = 120;
		public static final double DEFAULT_MBS_HDFS_WRITE_BINARY_SPARSE = 60;
		public static final double DEFAULT_MBS_HDFS_WRITE_TEXT_DENSE = 40;
		public static final double DEFAULT_MBS_HDFS_WRITE_TEXT_SPARSE = 30;

		/**
		 * Meant to be used for testing by setting known
		 * default values for each metric
		 */
		public IOMetrics() {
			cpuFLOPS = DEFAULT_FLOPS;
			cpuCores = DEFAULT_NUM_CPU_CORES;
			// Metrics for disk I/O operations
			localDiskReadBandwidth = DEFAULT_MBS_DISK_BANDWIDTH;
			localDiskWriteBandwidth = DEFAULT_MBS_DISK_BANDWIDTH;
			// Assume that the HDFS I/O operations is done always by accessing local blocks
			hdfsReadBinaryDenseBandwidth = DEFAULT_MBS_HDFS_READ_BINARY_DENSE;
			hdfsReadBinarySparseBandwidth = DEFAULT_MBS_HDFS_READ_BINARY_SPARSE;
			hdfsWriteBinaryDenseBandwidth = DEFAULT_MBS_HDFS_WRITE_BINARY_DENSE;
			hdfsWriteBinarySparseBandwidth = DEFAULT_MBS_HDFS_WRITE_BINARY_SPARSE;
			hdfsReadTextDenseBandwidth = DEFAULT_MBS_HDFS_READ_TEXT_DENSE;
			hdfsReadTextSparseBandwidth = DEFAULT_MBS_HDFS_READ_TEXT_SPARSE;
			hdfsWriteTextDenseBandwidth = DEFAULT_MBS_HDFS_WRITE_TEXT_DENSE;
			hdfsWriteTextSparseBandwidth = DEFAULT_MBS_HDFS_WRITE_TEXT_SPARSE;
			s3BandwidthEfficiency = (S3_COMPUTE_BOUND / cpuFLOPS);
			// Metrics for main memory I/O operations
			memReadBandwidth = DEFAULT_MBS_MEMORY_BANDWIDTH;
			memWriteBandwidth = DEFAULT_MBS_MEMORY_BANDWIDTH;
			// Metrics for networking operations
			networkingBandwidth = DEFAULT_MBS_NETWORK_BANDWIDTH;
			// Metrics for (de)serialization,
			double currentFlopsFactor = (double) DEFAULT_FLOPS / cpuFLOPS;
			serializationBandwidth = memReadBandwidth * SERIALIZATION_FACTOR * currentFlopsFactor;
			deserializationBandwidth = memWriteBandwidth * DESERIALIZATION_FACTOR * currentFlopsFactor;
		}
	}

	protected static final String S3_SOURCE_IDENTIFIER = "s3";
	protected static final String HDFS_SOURCE_IDENTIFIER = "hdfs";


	/**
	 * Estimate time to scan object in memory in CP.
	 *
	 * @param stats object statistics
	 * @param metrics CP node's metrics
	 * @return estimated time in seconds
	 */
	public static double getMemReadTime(VarStats stats, IOMetrics metrics) {
		if (stats.isScalar()) return 0; // scalars
		if (stats.allocatedMemory < 0)
			throw new RuntimeException("VarStats.allocatedMemory should carry the estimated size before getting read time");
		double sizeMB = (double) stats.allocatedMemory / (1024 * 1024);
		return sizeMB / metrics.memReadBandwidth;
	}

	/**
	 * Estimate time to scan distributed data sets in memory on Spark.
	 * It integrates a mechanism to account for scanning
	 * spilled-over data sets on the local disk.
	 *
	 * @param stats object statistics
	 * @param metrics CP node's metrics
	 * @return estimated time in seconds
	 */
	public static double getMemReadTime(RDDStats stats, IOMetrics metrics) {
		// no scalars expected
		double size = (double) stats.distributedSize;
		if (size < 0)
			throw new RuntimeException("RDDStats.distributedMemory should carry the estimated size before getting read time");
		// define if/what a fraction is spilled over to disk
		double minExecutionMemory = SparkExecutionContext.getDataMemoryBudget(true, false); // execution mem = storage mem
		double spillOverFraction = minExecutionMemory >= size? 0 : (size - minExecutionMemory) / size;
		// for simplification define an average read bandwidth combination form memory and disk bandwidths
		double mixedBandwidthPerCore = (spillOverFraction * metrics.localDiskReadBandwidth +
				(1-spillOverFraction) * metrics.memReadBandwidth) / metrics.cpuCores;
		double numWaves = Math.ceil((double) stats.numPartitions / SparkExecutionContext.getDefaultParallelism(false));
		double sizeMB = size / (1024 * 1024);
		double partitionSizeMB = sizeMB / stats.numPartitions;
		return numWaves * (partitionSizeMB / mixedBandwidthPerCore);
	}

	/**
	 * Estimate time to write object to memory in CP.
	 *
	 * @param stats object statistics
	 * @param metrics CP node's metrics
	 * @return estimated time in seconds
	 */
	public static double getMemWriteTime(VarStats stats, IOMetrics metrics) {
		if (stats == null) return 0; // scalars
		if (stats.allocatedMemory < 0)
			throw new DMLRuntimeException("VarStats.allocatedMemory should carry the estimated size before getting write time");
		double sizeMB = (double) stats.allocatedMemory / (1024 * 1024);

		return sizeMB / metrics.memWriteBandwidth;
	}

	/**
	 * Estimate time to write distributed data set on memory in CP.
	 * It does NOT integrate mechanism to account for spill-overs.
	 *
	 * @param stats object statistics
	 * @param metrics CP node's metrics
	 * @return estimated time in seconds
	 */
	public static double getMemWriteTime(RDDStats stats, IOMetrics metrics) {
		// no scalars expected
		if (stats.distributedSize < 0)
			throw new RuntimeException("RDDStats.distributedMemory should carry the estimated size before getting write time");
		double numWaves = Math.ceil((double) stats.numPartitions / SparkExecutionContext.getDefaultParallelism(false));
		double sizeMB = (double) stats.distributedSize / (1024 * 1024);
		double partitionSizeMB = sizeMB / stats.numPartitions;
		return numWaves * partitionSizeMB / (metrics.memWriteBandwidth / metrics.cpuCores);
	}

	/**
	 * Estimates the read time for a file on HDFS or S3 by the Control Program
	 * @param stats stats for the input matrix/object
	 * @param metrics I/O metrics for the driver node
	 * @return estimated time in seconds
	 */
	public static double getFileSystemReadTime(VarStats stats, IOMetrics metrics) {
		String sourceType = (String) stats.fileInfo[0];
		Types.FileFormat format = (Types.FileFormat) stats.fileInfo[1];
		double sizeMB = getFileSizeInMB(stats);
		boolean isSparse = MatrixBlock.evalSparseFormatOnDisk(stats.getM(), stats.getN(), stats.getNNZ());
		return getStorageReadTime(sizeMB, isSparse, sourceType, format, metrics);
	}

	/**
	 * Estimates the read time for a file on HDFS or S3 by Spark cluster.
	 * It doesn't directly calculate the execution time regarding the object size
	 * but regarding full executor utilization and maximum block size to be read by
	 * an executor core (HDFS block size). The estimated time for "fully utilized"
	 * reading is then multiplied by the slot execution round since even not fully utilized,
	 * the last round should take approximately the same time as if all slots are assigned
	 * to an active reading task.
	 * This function cannot rely on the {@code RDDStats} since they would not be
	 * initialized for the input object.
	 * @param stats stats for the input matrix/object
	 * @param metrics I/O metrics for the executor node
	 * @return estimated time in seconds
	 */
	public static double getHadoopReadTime(VarStats stats, IOMetrics metrics) {
		String sourceType = (String) stats.fileInfo[0];
		Types.FileFormat format = (Types.FileFormat) stats.fileInfo[1];
		long size =  getPartitionedFileSize(stats);
		// since getDiskReadTime() computes the write time utilizing the whole executor resources
		// use the fact that <partition size> / <bandwidth per slot> = <partition size> * <slots per executor> / <bandwidth per executor>
		long hdfsBlockSize = InfrastructureAnalyzer.getHDFSBlockSize();

		double numPartitions = Math.ceil((double) size / hdfsBlockSize);
		double sizePerExecutorMB;
		if (size < hdfsBlockSize) {
			// emulate full executor utilization
			sizePerExecutorMB = (double) (metrics.cpuCores * size) / (1024 * 1024);
		} else {
			sizePerExecutorMB =(double) (metrics.cpuCores * hdfsBlockSize) / (1024 * 1024);
		}
		boolean isSparse = MatrixBlock.evalSparseFormatOnDisk(stats.getM(), stats.getN(), stats.getNNZ());
		double timePerCore = getStorageReadTime(sizePerExecutorMB, isSparse, sourceType, format, metrics); // same as time per executor
		// number of execution waves (maximum task to execute per core)
		double numWaves = Math.ceil(numPartitions / (SparkExecutionContext.getDefaultParallelism(false)));
		return numWaves * timePerCore;
	}

	private static double getStorageReadTime(double sizeMB, boolean isSparse, String source, Types.FileFormat format, IOMetrics metrics)
	{
		double time;
		if (format == null || format.isTextFormat()) {
			if (source.equals(S3_SOURCE_IDENTIFIER)) {
				if (isSparse) {
					time = SPARSE_FACTOR * metrics.s3BandwidthEfficiency * sizeMB;
				}
				else {// dense
					time = metrics.s3BandwidthEfficiency * sizeMB;
				}
			} else { // HDFS
				if (isSparse)
					time = sizeMB / metrics.hdfsReadTextSparseBandwidth;
				else //dense
					time = sizeMB / metrics.hdfsReadTextDenseBandwidth;
			}
		} else if (format == Types.FileFormat.BINARY) {
			if (source.equals(HDFS_SOURCE_IDENTIFIER)) {
				if (isSparse)
					time = sizeMB / metrics.hdfsReadBinarySparseBandwidth;
				else //dense
					time = sizeMB / metrics.hdfsReadBinaryDenseBandwidth;
			} else { // S3
				throw new RuntimeException("Reading binary files from S3 is not supported");
			}
		} else { // compressed
			throw new RuntimeException("Format " + format + " is not supported for estimation yet.");
		}
		return time;
	}

	/**
	 * Estimates the time for writing a file to HDFS or S3.
	 *
	 * @param stats stats for the input matrix/object
	 * @param metrics I/O metrics for the driver node
	 * @return estimated time in seconds
	 */
	public static double getFileSystemWriteTime(VarStats stats, IOMetrics metrics) {
		String sourceType = (String) stats.fileInfo[0];
		Types.FileFormat format = (Types.FileFormat) stats.fileInfo[1];
		double sizeMB = getFileSizeInMB(stats);
		boolean isSparse = MatrixBlock.evalSparseFormatOnDisk(stats.getM(), stats.getN(), stats.getNNZ());
		return getStorageWriteTime(sizeMB, isSparse, sourceType, format, metrics);
	}

	/**
	 * Estimates the write time for a file on HDFS or S3 by Spark cluster.
	 * Follows the same logic as {@code getHadoopReadTime}, but here
	 * it can be relied on the {@code RDDStats} since the input object
	 * should be initialized by the prior instruction
	 * @param stats stats for the input matrix/object
	 * @param metrics I/O metrics for the executor node
	 * @return estimated time in seconds
	 */
	public static double getHadoopWriteTime(VarStats stats, IOMetrics metrics) {
		if (stats.rddStats == null) {
			throw new RuntimeException("Estimation for hadoop write time required VarStats object with assigned 'rddStats' member");
		}
		String sourceType = (String) stats.fileInfo[0];
		Types.FileFormat format = (Types.FileFormat) stats.fileInfo[1];
		long size = getPartitionedFileSize(stats);
		// time = <num. waves> * <partition size> / <bandwidth per slot>
		// here it cannot be assumed that the partition size is equal to the HDFS block size
		double sizePerPartitionMB = (double) size / stats.rddStats.numPartitions / (1024*1024);
		// since getDiskWriteTime() computes the write time utilizing the whole executor resources
		// use the fact that <partition size> / <bandwidth per slot> = <partition size> * <slots per executor> / <bandwidth per executor>
		double sizePerExecutor = sizePerPartitionMB * metrics.cpuCores;
		boolean isSparse = MatrixBlock.evalSparseFormatOnDisk(stats.getM(), stats.getN(), stats.getNNZ());
		double timePerCore = getStorageWriteTime(sizePerExecutor, isSparse, sourceType, format, metrics); // same as time per executor
		// number of execution waves (maximum task to execute per core)
		double numWaves = Math.ceil((double) stats.rddStats.numPartitions /
				(SparkExecutionContext.getNumExecutors() * metrics.cpuCores));
		return numWaves * timePerCore;
	}

	protected static double getStorageWriteTime(double sizeMB, boolean isSparse, String source, Types.FileFormat format, IOMetrics metrics) {
		if (format == null || isInvalidDataSource(source)) {
			throw new RuntimeException("Estimation not possible without source identifier and file format");
		}
		double time;
		if (format.isTextFormat()) {
			if (source.equals(S3_SOURCE_IDENTIFIER)) {
				if (isSparse)
					time = SPARSE_FACTOR * metrics.s3BandwidthEfficiency * sizeMB;
				else // dense
					time = metrics.s3BandwidthEfficiency * sizeMB;
			} else { // HDFS
				if (isSparse)
					time = sizeMB / metrics.hdfsWriteTextSparseBandwidth;
				else //dense
					time = sizeMB / metrics.hdfsWriteTextDenseBandwidth;
			}
		} else if (format == Types.FileFormat.BINARY) {
			if (source.equals(HDFS_SOURCE_IDENTIFIER)) {
				if (isSparse)
					time = sizeMB / metrics.hdfsWriteBinarySparseBandwidth;
				else //dense
					time = sizeMB / metrics.hdfsWriteBinaryDenseBandwidth;
			} else { // S3
				throw new RuntimeException("Writing binary files from S3 is not supported");
			}
		} else { // compressed
			throw new RuntimeException("Format " + format + " is not supported for estimation yet.");
		}
		return time;
	}

	/**
	 * Estimates the time to parallelize a local object to Spark.
	 *
	 * @param output RDD statistics for the object to be collected/transferred.
	 * @param driverMetrics I/O metrics for the receiver - driver node
	 * @param executorMetrics I/O metrics for the executor nodes
	 * @return estimated time in seconds
	 */
	public static double getSparkParallelizeTime(RDDStats output, IOMetrics driverMetrics, IOMetrics executorMetrics) {
		// it is assumed that the RDD object is already created/read
		// general idea: time = <serialization time> + <transfer time>;
		// NOTE: currently it is assumed that ht serialized data has the same size as the original data what may not be true in the general case
		double sizeMB = (double) output.distributedSize / (1024 * 1024);
		// 1. serialization time
		double serializationTime = sizeMB / driverMetrics.serializationBandwidth;
		// 2. transfer time
		double effectiveBandwidth = Math.min(driverMetrics.networkingBandwidth,
				SparkExecutionContext.getNumExecutors() * executorMetrics.networkingBandwidth);
		double transferTime = sizeMB / effectiveBandwidth;
		// sum the time for the steps since they cannot overlap
		return serializationTime + transferTime;
	}

	/**
	 * Estimates the time for collecting Spark Job output;
	 * The output RDD is transferred to the Spark driver at the end of each ResultStage;
	 * time = transfer time (overlaps and dominates the read and deserialization times);
	 *
	 * @param output RDD statistics for the object to be collected/transferred.
	 * @param driverMetrics I/O metrics for the receiver - driver node
	 * @param executorMetrics I/O metrics for the executor nodes
	 * @return estimated time in seconds
	 */
	public static double getSparkCollectTime(RDDStats output, IOMetrics driverMetrics, IOMetrics executorMetrics) {
		double sizeMB = (double) output.distributedSize / (1024 * 1024);
		double numWaves = Math.ceil((double) output.numPartitions / SparkExecutionContext.getDefaultParallelism(false));
		int currentParallelism = Math.min(output.numPartitions, SparkExecutionContext.getDefaultParallelism(false));
		double bandwidthPerCore = executorMetrics.networkingBandwidth / executorMetrics.cpuCores;
		double effectiveBandwidth = Math.min(numWaves * driverMetrics.networkingBandwidth,
				currentParallelism * bandwidthPerCore);
		// transfer time
		return  sizeMB / effectiveBandwidth;
	}

	/**
	 * Estimates the time for reading distributed RDD input at the beginning of a Stage;
	 * time = transfer time (overlaps and dominates the read and deserialization times);
	 * For simplification it is assumed that the whole dataset is shuffled;
	 *
	 * @param input RDD statistics for the object to be shuffled at the begging of a Stage.
	 * @param metrics I/O metrics for the executor nodes
	 * @return estimated time in seconds
	 */
	public static double getSparkShuffleReadTime(RDDStats input, IOMetrics metrics) {
		double sizeMB = (double) input.distributedSize / (1024 * 1024);
		// edge case: 1 executor only would not trigger any data
		if (SparkExecutionContext.getNumExecutors() < 2) {
			// even without shuffling the data needs to be read from the intermediate shuffle files
			double diskBandwidthPerCore = metrics.localDiskWriteBandwidth / metrics.cpuCores;
			// disk read time
			return sizeMB / diskBandwidthPerCore;
		}
		int currentParallelism = Math.min(input.numPartitions, SparkExecutionContext.getDefaultParallelism(false));
		double networkBandwidthPerCore = metrics.networkingBandwidth / metrics.cpuCores;
		// transfer time
		return sizeMB / (currentParallelism * networkBandwidthPerCore);
	}

	/**
	 * Estimates the time for reading distributed RDD input at the beginning of a Stage
	 * when a wide-transformation is partition preserving: only local disk reads
	 *
	 * @param input RDD statistics for the object to be shuffled (read) at the begging of a Stage.
	 * @param metrics I/O metrics for the executor nodes
	 * @return estimated time in seconds
	 */
	public static double getSparkShuffleReadStaticTime(RDDStats input, IOMetrics metrics) {
		double sizeMB = (double) input.distributedSize / (1024 * 1024);
		int currentParallelism = Math.min(input.numPartitions, SparkExecutionContext.getDefaultParallelism(false));
		double readBandwidthPerCore = metrics.memReadBandwidth / metrics.cpuCores;
		// read time
		return sizeMB / (currentParallelism * readBandwidthPerCore);
	}

	/**
	 * Estimates the time for writing the RDD output to the local system at the end of a ShuffleMapStage;
	 * time = disk write time (overlaps and dominates the serialization time)
	 * The whole data set is being written to shuffle files even if 1 executor is utilized;
	 *
	 * @param output RDD statistics for the output each ShuffleMapStage
	 * @param metrics I/O metrics for the executor nodes
	 * @return estimated time in seconds
	 */
	public static double getSparkShuffleWriteTime(RDDStats output, IOMetrics metrics) {
		double sizeMB = (double) output.distributedSize / (1024 * 1024);
		int currentParallelism = Math.min(output.numPartitions, SparkExecutionContext.getDefaultParallelism(false));
		double bandwidthPerCore = metrics.localDiskWriteBandwidth / metrics.cpuCores;
		// disk write time
		return sizeMB / (currentParallelism * bandwidthPerCore);
	}

	/**
	 * Combines the shuffle write and read time since these are being typically
	 * added in one place to the general data transmission for instruction estimation.
	 *
	 * @param output RDD statistics for the output each ShuffleMapStage
	 * @param metrics I/O metrics for the executor nodes
	 * @param withDistribution flag if the data is indeed reshuffled (default case),
	 *                         false in case of co-partitioned wide-transformation
	 * @return estimated time in seconds
	 */
	public static double getSparkShuffleTime(RDDStats output, IOMetrics metrics, boolean withDistribution) {
		double totalTime = getSparkShuffleWriteTime(output, metrics);
		if (withDistribution)
			totalTime += getSparkShuffleReadTime(output, metrics);
		else
			totalTime += getSparkShuffleReadStaticTime(output, metrics);
		return totalTime;
	}

	/**
	 * Estimates the time for broadcasting an object;
	 * This function takes into account the torrent-like mechanism
	 * for broadcast distribution across all executors;
	 *
	 * @param stats statistics for the object for broadcasting
	 * @param driverMetrics I/O metrics for the driver node
	 * @param executorMetrics I/O metrics for the executor nodes
	 * @return estimated time in seconds
	 */
	protected static double getSparkBroadcastTime(VarStats stats, IOMetrics driverMetrics, IOMetrics executorMetrics) {
		// TODO: ensure the object related to stats is read in memory already ot add logic to account for its read time
		// it is assumed that the Cp broadcast object is already created/read
		// general idea: time = <serialization time> + <transfer time>;
		double sizeMB = (double) OptimizerUtils.estimatePartitionedSizeExactSparsity(stats.characteristics) / (1024 * 1024);
		// 1. serialization time
		double serializationTime = sizeMB / driverMetrics.serializationBandwidth;
		// 2. transfer time considering the torrent-like mechanism: time to transfer the whole object to a single node
		double effectiveBandwidth = Math.min(driverMetrics.networkingBandwidth, executorMetrics.networkingBandwidth);
		double transferTime = sizeMB / effectiveBandwidth;
		// sum the time for the steps since they cannot overlap
		return serializationTime + transferTime;
	}

	/**
	 * Extracts the data source for a given file name: e.g. "hdfs" or "s3"
	 *
	 * @param fileName filename to parse
	 * @return data source type
	 */
	public static String getDataSource(String fileName) {
		String[] fileParts = fileName.split("://");
		if (fileParts.length > 1) {
			String filesystem = fileParts[0].toLowerCase();
			if (filesystem.matches("\\b(s3|s3a)\\b"))
				return S3_SOURCE_IDENTIFIER;
			return filesystem;
		}
		return HDFS_SOURCE_IDENTIFIER;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	// Helpers 																				    //
	//////////////////////////////////////////////////////////////////////////////////////////////

	private static double getFileSizeInMB(VarStats fileStats) {
		Types.FileFormat format = (Types.FileFormat) fileStats.fileInfo[1];
		double sizeMB;
		if (format == Types.FileFormat.BINARY) {
			sizeMB = (double) MatrixBlock.estimateSizeOnDisk(fileStats.getM(), fileStats.getM(), fileStats.getNNZ()) / (1024*1024);
		} else if (format.isTextFormat()) {
			sizeMB = (double) OptimizerUtils.estimateSizeTextOutput(fileStats.getM(), fileStats.getM(), fileStats.getNNZ(), format)  / (1024*1024);
		} else { // compressed
			throw new RuntimeException("Format " + format + " is not supported for estimation yet.");
		}
		return sizeMB;
	}

	private static long getPartitionedFileSize(VarStats fileStats) {
		Types.FileFormat format = (Types.FileFormat) fileStats.fileInfo[1];
		long size;
		if (format == Types.FileFormat.BINARY) {
			size = MatrixBlock.estimateSizeOnDisk(fileStats.getM(), fileStats.getN(), fileStats.getNNZ());
		} else if (format.isTextFormat()) {
			size = OptimizerUtils.estimateSizeTextOutput(fileStats.getM(), fileStats.getN(), fileStats.getNNZ(), format);
		} else { // compressed
			throw new RuntimeException("Format " + format + " is not supported for estimation yet.");
		}
		return size;
	}

	public static boolean isInvalidDataSource(String identifier) {
		return !identifier.equals(HDFS_SOURCE_IDENTIFIER) && !identifier.equals(S3_SOURCE_IDENTIFIER);
	}
}
