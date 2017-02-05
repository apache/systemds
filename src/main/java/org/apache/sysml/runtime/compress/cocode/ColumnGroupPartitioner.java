package org.apache.sysml.runtime.compress.cocode;

import java.util.HashMap;
import java.util.List;

import org.apache.sysml.runtime.compress.cocode.PlanningCoCoder.GroupableColInfo;

public abstract class ColumnGroupPartitioner 
{
	/**
	 * Partitions a list of columns into a list of partitions that contains subsets of columns.
	 * Note that this call must compute a complete and disjoint partitioning.
	 * 
	 * @param groupCols list of columns 
	 * @param groupColsInfo list of column infos
	 * @return list of partitions (where each partition is a list of columns)
	 */
	public abstract List<List<Integer>> partitionColumns(List<Integer> groupCols, HashMap<Integer, GroupableColInfo> groupColsInfo);
}
