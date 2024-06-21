package org.apache.sysds.api.ropt.cost;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;

public class RDDStats {
    private static int blockSize; // TODO: think of more efficient way (does not changes, init once)
    public long totalSize;
    private static long hdfsBlockSize;
    public long numPartitions;
    public long numBlocks;
    public long numValues;
    public long rlen;
    public long clen;
    public double sparsity;
    public VarStats cpVar;
    public int numParallelTasks;

    public static void setDefaults() {
        blockSize = ConfigurationManager.getBlocksize();
        hdfsBlockSize = InfrastructureAnalyzer.getHDFSBlockSize();
    }

    public RDDStats(VarStats cpVar) {
        totalSize = OptimizerUtils.estimateSizeExactSparsity(cpVar.getM(), cpVar.getN(), cpVar.getS());
        numPartitions = (int) Math.max(Math.min(totalSize / hdfsBlockSize, cpVar._mc.getNumBlocks()), 1);
        numBlocks = cpVar._mc.getNumBlocks();
        this.cpVar = cpVar;
        rlen = cpVar.getM();
        clen = cpVar.getN();
        numValues = rlen*clen;
        sparsity = cpVar.getS();
        numParallelTasks = (int) Math.min(numPartitions, SparkExecutionContext.getDefaultParallelism(false));
    }

    public static RDDStats transformNumPartitions(RDDStats oldRDD, long newNumPartitions) {
        if (oldRDD.cpVar == null) {
            throw new DMLRuntimeException("Cannot transform RDDStats without VarStats");
        }
        RDDStats newRDD = new RDDStats(oldRDD.cpVar);
        newRDD.numPartitions = newNumPartitions;
        return newRDD;
    }

    public static RDDStats transformNumBlocks(RDDStats oldRDD, long newNumBlocks) {
        if (oldRDD.cpVar == null) {
            throw new DMLRuntimeException("Cannot transform RDDStats without VarStats");
        }
        RDDStats newRDD = new RDDStats(oldRDD.cpVar);
        newRDD.numBlocks = newNumBlocks;
        return newRDD;
    }
}
