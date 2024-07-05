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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class IOCostUtils {
	// NOTE: this class does NOT include methods for estimating IO time
	//  for operation ot the local file system since they are not relevant at the moment
	protected static final String S3_SOURCE_IDENTIFIER = "s3";
	protected static final String HDFS_SOURCE_IDENTIFIER = "hdfs";
	//IO READ throughput
	private static final double DEFAULT_MBS_S3READ_BINARYBLOCK_DENSE = 200;
	private static final double DEFAULT_MBS_S3READ_BINARYBLOCK_SPARSE = 100;
	private static final double DEFAULT_MBS_HDFSREAD_BINARYBLOCK_DENSE = 150;
	public static final double DEFAULT_MBS_HDFSREAD_BINARYBLOCK_SPARSE = 75;
	private static final double DEFAULT_MBS_S3READ_TEXT_DENSE = 50;
	private static final double DEFAULT_MBS_S3READ_TEXT_SPARSE = 25;
	private static final double DEFAULT_MBS_HDFSREAD_TEXT_DENSE = 75;
	private static final double DEFAULT_MBS_HDFSREAD_TEXT_SPARSE = 50;
	//IO WRITE throughput
	private static final double DEFAULT_MBS_S3WRITE_BINARYBLOCK_DENSE = 150;
	private static final double DEFAULT_MBS_S3WRITE_BINARYBLOCK_SPARSE = 75;
	private static final double DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE = 120;
	private static final double DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE = 60;
	private static final double DEFAULT_MBS_S3WRITE_TEXT_DENSE = 30;
	private static final double DEFAULT_MBS_S3WRITE_TEXT_SPARSE = 20;
	private static final double DEFAULT_MBS_HDFSWRITE_TEXT_DENSE = 40;
	private static final double DEFAULT_MBS_HDFSWRITE_TEXT_SPARSE = 30;
	// New -> Spark cost estimation
	private static final double DEFAULT_NETWORK_BANDWIDTH = 100; // bandwidth for shuffling data

	//private static final double DEFAULT_DISK_BANDWIDTH = 1000; // bandwidth for shuffling data
	private static final double DEFAULT_NETWORK_LATENCY = 0.001; // latency for data transfer in seconds
	//private static final double DEFAULT_META_TO_DRIVER_MS = 10; // cost in ms to account for the metadata transmitted to the driver at the end of each stage
	private static final double SERIALIZATION_FACTOR = 10; // virtual unit - MB/(GFLOPS*s)
	private static final double MIN_TRANSFER_TIME = 0.001; // 1ms
	private static final double MIN_SERIALIZATION_TIME = 0.001; // 1ms (intended to include serialization and deserialization time)
	private static final double DEFAULT_MBS_MEM_READ_BANDWIDTH = 32000; // TODO: dynamic value later
	private static final double DEFAULT_MBS_MEM_WRITE_BANDWIDTH = 32000; // TODO: dynamic value later
	protected static double getMemReadTime(VarStats stats) {
		if (stats == null) return 0; // scalars
		if (stats._memory < 0)
			throw new DMLRuntimeException("VarStats should have estimated size before getting read time");
		long size = stats._memory;
		double sizeMB = (double) size / (1024 * 1024);

		return sizeMB / DEFAULT_MBS_MEM_READ_BANDWIDTH;
	}

	protected static double getMemWriteTime(VarStats stats) {
		if (stats == null) return 0; // scalars
		if (stats._memory < 0)
			throw new DMLRuntimeException("VarStats should have estimated size before getting write time");
		long size = stats._memory;
		double sizeMB = (double) size / (1024 * 1024);

		return sizeMB / DEFAULT_MBS_MEM_WRITE_BANDWIDTH;
	}

	/**
	 * Returns the estimated read time from HDFS.
	 * NOTE: Does not handle unknowns.
	 *
	 * @param dm rows?
	 * @param dn columns?
	 * @param ds sparsity factor?
	 * @param source data source (S3 or HDFS)
	 * @param format file format (null for binary)
	 * @return estimated HDFS read time
	 */
	protected static double getReadTime(long dm, long dn, double ds, String source, Types.FileFormat format)
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		double ret = ((double)MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn))) / (1024*1024);

		if (format == null || !format.isTextFormat()) {
			if (source.equals(S3_SOURCE_IDENTIFIER)) {
				if (sparse)
					ret /= DEFAULT_MBS_S3READ_BINARYBLOCK_SPARSE;
				else //dense
					ret /= DEFAULT_MBS_S3READ_BINARYBLOCK_DENSE;
			} else { //HDFS
				if (sparse)
					ret /= DEFAULT_MBS_HDFSREAD_BINARYBLOCK_SPARSE;
				else //dense
					ret /= DEFAULT_MBS_HDFSREAD_BINARYBLOCK_DENSE;
			}
		} else {
			if (source.equals(S3_SOURCE_IDENTIFIER)) {
				if (sparse)
					ret /= DEFAULT_MBS_S3READ_TEXT_SPARSE;
				else //dense
					ret /= DEFAULT_MBS_S3READ_TEXT_DENSE;
			} else { //HDFS
				if (sparse)
					ret /= DEFAULT_MBS_HDFSREAD_TEXT_SPARSE;
				else //dense
					ret /= DEFAULT_MBS_HDFSREAD_TEXT_DENSE;
			}
		}
		return ret;
	}

	protected static double getWriteTime(long dm, long dn, double ds, String source, Types.FileFormat format) {
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		double bytes = MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn));
		double mbytes = bytes / (1024*1024);
		double ret;

		if (source == S3_SOURCE_IDENTIFIER) {
			if (format.isTextFormat()) {
				if (sparse)
					ret = mbytes / DEFAULT_MBS_S3WRITE_TEXT_SPARSE;
				else //dense
					ret = mbytes / DEFAULT_MBS_S3WRITE_TEXT_DENSE;
				ret *= 2.75; //text commonly 2x-3.5x larger than binary
			} else {
				if (sparse)
					ret = mbytes / DEFAULT_MBS_S3WRITE_BINARYBLOCK_SPARSE;
				else //dense
					ret = mbytes / DEFAULT_MBS_S3WRITE_BINARYBLOCK_DENSE;
			}
		} else { //HDFS
			if (format.isTextFormat()) {
				if (sparse)
					ret = mbytes / DEFAULT_MBS_HDFSWRITE_TEXT_SPARSE;
				else //dense
					ret = mbytes / DEFAULT_MBS_HDFSWRITE_TEXT_DENSE;
				ret *= 2.75; //text commonly 2x-3.5x larger than binary
			} else {
				if (sparse)
					ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE;
				else //dense
					ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE;
			}
		}
		return ret;
	}

	/**
	 * Returns the estimated cost for transmitting a packet of size bytes.
	 * This function is supposed to be used for parallelize and result data transfer.
	 * Driver <-> Executors interaction.
	 * @param size
	 * @param numExecutors
	 * @return
	 */
	protected static double getSparkTransmissionCost(long size, int numExecutors) {
		double transferTime = Math.max(((double) size / (DEFAULT_NETWORK_BANDWIDTH * numExecutors)), MIN_TRANSFER_TIME);
		double serializationTime = Math.max((size * SERIALIZATION_FACTOR) / CostEstimator.CP_FLOPS, MIN_SERIALIZATION_TIME);
		return DEFAULT_NETWORK_LATENCY +  transferTime + serializationTime;
	}

	/**
	 * Returns the estimated cost for shuffling the records of an RDD of given size.
	 * This function assumes that all the records would be reshuffled what often not the case
	 * but this approximation is good enough for estimating the shuffle cost with higher skewness.
	 * Executors <-> Executors interaction.
	 * @param size
	 * @param numExecutors
	 * @return
	 */
	protected static double getShuffleCost(long size, int numExecutors) {
		double transferTime = Math.max(((double) size / (DEFAULT_NETWORK_BANDWIDTH * numExecutors)), MIN_TRANSFER_TIME);
		double serializationTime = Math.max((size * SERIALIZATION_FACTOR) / CostEstimator.SP_FLOPS, MIN_SERIALIZATION_TIME) / numExecutors;
		return DEFAULT_NETWORK_LATENCY * numExecutors +  transferTime + serializationTime;
	}

	/**
	 * Returns the estimated cost for  broadcasting a packet of size bytes.
	 * This function takes into account the torrent-like trnasmission of the
	 * broadcast data packages.
	 * Executors <-> Driver <-> Executors interaction.
	 * @param size
	 * @param numExecutors
	 * @return
	 */
	protected static double getBroadcastCost(long size, int numExecutors) {
		double transferTime = Math.max(((double) size / (DEFAULT_NETWORK_BANDWIDTH)), MIN_TRANSFER_TIME);
		double serializationTime = Math.max((size * SERIALIZATION_FACTOR) / CostEstimator.CP_FLOPS, MIN_SERIALIZATION_TIME);
		return DEFAULT_NETWORK_LATENCY * numExecutors +  transferTime + serializationTime;
	}

	public static String getDataSource(String fileName) {
		String[] fileParts = fileName.split("://");
		if (fileParts.length > 1) {
			return fileParts[0].toLowerCase();
		}
		return HDFS_SOURCE_IDENTIFIER;
	}
}
