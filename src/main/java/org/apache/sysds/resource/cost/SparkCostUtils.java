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
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.*;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.apache.sysds.runtime.instructions.spark.*;
import org.apache.sysds.runtime.instructions.spark.SPInstruction.SPType;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

import static org.apache.sysds.lops.DataGen.*;
import static org.apache.sysds.resource.cost.IOCostUtils.*;

public class SparkCostUtils {

    public static double getReblockInstTime(String opcode, VarStats input, VarStats output, IOMetrics executorMetrics) {
        // Reblock triggers a new stage
        // old stage: read text file + shuffle the intermediate text rdd
        double readTime = getHadoopReadTime(input, executorMetrics);
        long sizeTextFile = OptimizerUtils.estimateSizeTextOutput(input.getM(), input.getN(), input.getNNZ(), (Types.FileFormat) input.fileInfo[1]);
        RDDStats textRdd = new RDDStats(sizeTextFile, -1);
        double shuffleTime = getSparkShuffleTime(textRdd, executorMetrics, true);
        double timeStage1 = readTime + shuffleTime;
        // new stage: transform partitioned shuffled text object into partitioned binary object
        long nflop = getInstNFLOP(SPType.Reblock, opcode, output);
        double timeStage2 = getCPUTime(nflop, textRdd.numPartitions, executorMetrics, output.rddStats, textRdd);
        return timeStage1 + timeStage2;
    }

    public static double getRandInstTime(String opcode, int randType, VarStats output, IOMetrics executorMetrics) {
        if (opcode.equals(SAMPLE_OPCODE)) {
            // sample uses sortByKey() op. and it should be handled differently
            throw new RuntimeException("Spark operation Rand with opcode " + SAMPLE_OPCODE + " is not supported yet");
        }

        long nflop;
        if (opcode.equals(RAND_OPCODE) || opcode.equals(FRAME_OPCODE)) {
            if (randType == 0) return 0; // empty matrix
            else if (randType == 1) nflop = 8; // allocate, array fill
            else if (randType == 2) nflop = 32; // full rand
            else throw new RuntimeException("Unknown type of random instruction");
        } else if (opcode.equals(SEQ_OPCODE)) {
            nflop = 1;
        } else {
            throw new DMLRuntimeException("Rand operation with opcode '" + opcode + "' is not supported by SystemDS");
        }
        nflop *= output.getCells();
        // no shuffling required -> only computation time
        return getCPUTime(nflop, output.rddStats.numPartitions, executorMetrics, output.rddStats);
    }

    public static double getUnaryInstTime(String opcode, VarStats input, VarStats output, IOMetrics executorMetrics) {
        // handles operations of type Builtin as Unary
        // Unary adds a map() to an open stage
        long nflop = getInstNFLOP(SPType.Unary, opcode, output, input);
        double mapTime = getCPUTime(nflop, input.rddStats.numPartitions, executorMetrics, output.rddStats, input.rddStats);
        // the resulting rdd is being hash-partitioned depending on the input one
        output.rddStats.hashPartitioned = input.rddStats.hashPartitioned;
        return mapTime;
    }

    public static double getAggUnaryInstTime(UnarySPInstruction inst, VarStats input, VarStats output, IOMetrics executorMetrics) {
        // AggregateUnary results in different Spark execution plan depending on the output dimensions
        String opcode = inst.getOpcode();
        AggBinaryOp.SparkAggType aggType = (inst instanceof AggregateUnarySPInstruction)?
                ((AggregateUnarySPInstruction) inst).getAggType():
                ((AggregateUnarySketchSPInstruction) inst).getAggType();
        double shuffleTime;
        if (inst instanceof CumulativeAggregateSPInstruction) {
            shuffleTime = getSparkShuffleTime(output.rddStats, executorMetrics, true);
            output.rddStats.hashPartitioned = true;
        } else {
            if (aggType == AggBinaryOp.SparkAggType.SINGLE_BLOCK) {
                // loading RDD to the driver (CP) explicitly (not triggered by CP instruction)
                output.rddStats.isCollected = true;
                // cost for transferring result values (result from fold()) is negligible -> cost = computation time
                shuffleTime = 0;
            } else if (aggType == AggBinaryOp.SparkAggType.MULTI_BLOCK) {
                // combineByKey() triggers a new stage -> cost = computation time + shuffle time (combineByKey);
                if (opcode.equals("uaktrace")) {
                    long diagonalBlockSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(
                            input.characteristics.getBlocksize() * input.getM(),
                            input.characteristics.getBlocksize(),
                            input.characteristics.getBlocksize(),
                            input.getNNZ()
                    );
                    RDDStats filteredRDD = new RDDStats(diagonalBlockSize, input.rddStats.numPartitions);
                    shuffleTime = getSparkShuffleTime(filteredRDD, executorMetrics, true);
                } else {
                    shuffleTime = getSparkShuffleTime(input.rddStats, executorMetrics, true);
                }
                output.rddStats.hashPartitioned = true;
                output.rddStats.numPartitions = input.rddStats.numPartitions;
            } else {  // aggType == AggBinaryOp.SparkAggType.NONE
                output.rddStats.hashPartitioned = input.rddStats.hashPartitioned;
                output.rddStats.numPartitions = input.rddStats.numPartitions;
                // only mapping transformation -> cost = computation time
                shuffleTime = 0;
            }
        }
        long nflop = getInstNFLOP(SPType.AggregateUnary, opcode, output, input);
        double mapTime = getCPUTime(nflop, input.rddStats.numPartitions, executorMetrics, output.rddStats, input.rddStats);
        return shuffleTime + mapTime;
    }

    public static double getIndexingInstTime(IndexingSPInstruction inst, VarStats input1, VarStats input2, VarStats output, IOMetrics driverMetrics, IOMetrics executorMetrics) {
        String opcode = inst.getOpcode();
        double dataTransmissionTime;
        if (opcode.equals(RightIndex.OPCODE)) {
            // assume direct collecting if output dimensions not larger than block size
            int blockSize = ConfigurationManager.getBlocksize();
            if (output.getM() <= blockSize && output.getN() <= blockSize) {
                // represents single block and multi block cases
                dataTransmissionTime = getSparkCollectTime(output.rddStats, driverMetrics, executorMetrics);
                output.rddStats.isCollected = true;
            } else {
                // represents general indexing: worst case: shuffling required
                dataTransmissionTime = getSparkShuffleTime(output.rddStats, executorMetrics, true);
            }
        } else if (opcode.equals(LeftIndex.OPCODE)) {
            // model combineByKey() with shuffling the second input
            dataTransmissionTime = getSparkShuffleTime(input2.rddStats, executorMetrics, true);
        } else { // mapLeftIndex
            dataTransmissionTime = getSparkBroadcastTime(input2, driverMetrics, executorMetrics);
        }
        long nflop = getInstNFLOP(SPType.MatrixIndexing, opcode, output);
        // scan only the size of the output since filter is applied first
        RDDStats[] objectsToScan = (input2 == null)? new RDDStats[]{output.rddStats} :
                new RDDStats[]{output.rddStats, output.rddStats};
        double mapTime = getCPUTime(nflop, input1.rddStats.numPartitions, executorMetrics, output.rddStats, objectsToScan);
        return dataTransmissionTime + mapTime;
    }

    public static double getBinaryInstTime(SPInstruction inst, VarStats input1, VarStats input2, VarStats output, IOMetrics driverMetrics, IOMetrics executorMetrics) {
        SPType opType = inst.getSPInstructionType();
        String opcode = inst.getOpcode();
        // binary, builtin binary (log and log_nz)
        // for the NFLOP calculation if the function is executed as map is not relevant
        if (opcode.startsWith("map")) {
            opcode = opcode.substring(3);
        }
        double dataTransmissionTime;
        if (inst instanceof BinaryMatrixMatrixSPInstruction) {
            if (inst instanceof BinaryMatrixBVectorSPInstruction) {
                // the second matrix is always the broadcast one
                dataTransmissionTime = getSparkBroadcastTime(input2, driverMetrics, executorMetrics);
                // flatMapToPair() or ()mapPartitionsToPair invoked -> no shuffling
                output.rddStats.numPartitions = input1.rddStats.numPartitions;
                output.rddStats.hashPartitioned = input1.rddStats.hashPartitioned;
            } else { // regular BinaryMatrixMatrixSPInstruction
                // join() input1 and input2
                dataTransmissionTime = getSparkShuffleWriteTime(input1.rddStats, executorMetrics) +
                        getSparkShuffleWriteTime(input2.rddStats, executorMetrics);
                if (input1.rddStats.hashPartitioned) {
                    output.rddStats.numPartitions = input1.rddStats.numPartitions;
                    if (!input2.rddStats.hashPartitioned || !(input1.rddStats.numPartitions == input2.rddStats.numPartitions)) {
                        // shuffle needed for join() -> actual shuffle only for input2
                        dataTransmissionTime += getSparkShuffleReadStaticTime(input1.rddStats, executorMetrics) +
                                getSparkShuffleReadTime(input2.rddStats, executorMetrics);
                    } else { // no shuffle needed for join() -> only read from local disk
                        dataTransmissionTime += getSparkShuffleReadStaticTime(input1.rddStats, executorMetrics) +
                                getSparkShuffleReadStaticTime(input2.rddStats, executorMetrics);
                    }
                } else if (input2.rddStats.hashPartitioned) {
                    output.rddStats.numPartitions = input2.rddStats.numPartitions;
                    // input1 not hash partitioned: shuffle needed for join() -> actual shuffle only for input2
                    dataTransmissionTime += getSparkShuffleReadStaticTime(input1.rddStats, executorMetrics) +
                            getSparkShuffleReadTime(input2.rddStats, executorMetrics);
                } else {
                    // repartition all data needed
                    output.rddStats.numPartitions = 2 * output.rddStats.numPartitions;
                    dataTransmissionTime += getSparkShuffleReadTime(input1.rddStats, executorMetrics) +
                            getSparkShuffleReadTime(input2.rddStats, executorMetrics);
                }
                output.rddStats.hashPartitioned = true;
            }
        } else if (inst instanceof BinaryMatrixScalarSPInstruction) {
            // only mapValues() invoked -> no shuffling
            dataTransmissionTime = 0;
            output.rddStats.hashPartitioned = (input2.isScalar())? input1.rddStats.hashPartitioned : input2.rddStats.hashPartitioned;
        } else if (inst instanceof BinaryFrameMatrixSPInstruction || inst instanceof BinaryFrameFrameSPInstruction) {
            throw new RuntimeException("Handling binary instructions for frames not handled yet.");
        } else {
            throw new RuntimeException("Not supported binary instruction: "+inst);
        }
        long nflop = getInstNFLOP(opType, opcode, output, input1, input2);
        double mapTime = getCPUTime(nflop, output.rddStats.numPartitions, executorMetrics, output.rddStats, input1.rddStats, input2.rddStats);
        return dataTransmissionTime + mapTime;
    }

    public static double getAppendInstTime(AppendSPInstruction inst, VarStats input1, VarStats input2, VarStats output, IOMetrics driverMetrics, IOMetrics executorMetrics) {
        double dataTransmissionTime;
        if (inst instanceof AppendMSPInstruction) {
            dataTransmissionTime = getSparkBroadcastTime(input2, driverMetrics, executorMetrics);
            output.rddStats.hashPartitioned = true;
        } else if (inst instanceof AppendRSPInstruction) {
            dataTransmissionTime = getSparkShuffleTime(output.rddStats, executorMetrics, false);
        } else if (inst instanceof AppendGAlignedSPInstruction) {
            // only changing matrix indexing
            dataTransmissionTime = 0;
        } else { // AppendGSPInstruction
            // shuffle the whole appended matrix
            dataTransmissionTime = getSparkShuffleTime(input2.rddStats, executorMetrics, true);
            output.rddStats.hashPartitioned = true;
        }
        // opcode not relevant for the nflop estimation of append instructions;
        long nflop = getInstNFLOP(inst.getSPInstructionType(), "append", output, input1, input2);
        double mapTime = getCPUTime(nflop, output.rddStats.numPartitions, executorMetrics, output.rddStats, input1.rddStats, input2.rddStats);
        return dataTransmissionTime + mapTime;
    }

    public static double getReorgInstTime(UnarySPInstruction inst, VarStats input, VarStats output, IOMetrics executorMetrics) {
        // includes logic for MatrixReshapeSPInstruction
        String opcode = inst.getOpcode();
        double dataTransmissionTime;
        switch (opcode) {
            case "rshape":
                dataTransmissionTime = getSparkShuffleTime(input.rddStats, executorMetrics, true);
                output.rddStats.hashPartitioned = true;
                break;
            case "r'":
                dataTransmissionTime = 0;
                output.rddStats.hashPartitioned = input.rddStats.hashPartitioned;
                break;
            case "rev":
                dataTransmissionTime = getSparkShuffleTime(output.rddStats, executorMetrics, true);
                output.rddStats.hashPartitioned = true;
                break;
            case "rdiag":
                dataTransmissionTime = 0;
                output.rddStats.numPartitions = input.rddStats.numPartitions;
                output.rddStats.hashPartitioned = input.rddStats.hashPartitioned;
                break;
            default:  // rsort
                String ixretAsString = InstructionUtils.getInstructionParts(inst.getInstructionString())[4];
                boolean ixret = ixretAsString.equalsIgnoreCase("true");
                int shuffleFactor;
                if (ixret) { // index return
                    shuffleFactor = 2; // estimate cost for 2 shuffles
                } else {
                    shuffleFactor = 4;// estimate cost for 2 shuffles
                }
                // assume case: 4 times shuffling the output
                dataTransmissionTime = getSparkShuffleWriteTime(output.rddStats, executorMetrics) +
                        getSparkShuffleReadTime(output.rddStats, executorMetrics);
                dataTransmissionTime *= shuffleFactor;
                break;
        }
        long nflop = getInstNFLOP(inst.getSPInstructionType(), opcode, output); // uses output only
        double mapTime = getCPUTime(nflop, output.rddStats.numPartitions, executorMetrics, output.rddStats, input.rddStats);
        return dataTransmissionTime + mapTime;
    }

    public static double getTSMMInstTime(UnarySPInstruction inst, VarStats input, VarStats output, IOMetrics driverMetrics, IOMetrics executorMetrics) {
        String opcode = inst.getOpcode();
        MMTSJ.MMTSJType type;

        double dataTransmissionTime;
        if (inst instanceof TsmmSPInstruction) {
            type = ((TsmmSPInstruction) inst).getMMTSJType();
            // fold() used but result is still a whole matrix block
            dataTransmissionTime = getSparkCollectTime(output.rddStats, driverMetrics, executorMetrics);
            output.rddStats.isCollected = true;
        } else { // Tsmm2SPInstruction
            type = ((Tsmm2SPInstruction) inst).getMMTSJType();
            // assumes always default output with collect
            long rowsRange = (type == MMTSJ.MMTSJType.LEFT)? input.getM() :
                    input.getM() - input.characteristics.getBlocksize();
            long colsRange = (type != MMTSJ.MMTSJType.LEFT)? input.getN() :
                    input.getN() - input.characteristics.getBlocksize();
            VarStats broadcast = new VarStats("tmp1", new MatrixCharacteristics(rowsRange, colsRange));
            broadcast.rddStats = new RDDStats(broadcast);
            dataTransmissionTime = getSparkCollectTime(broadcast.rddStats, driverMetrics, executorMetrics);
            dataTransmissionTime += getSparkBroadcastTime(broadcast, driverMetrics, executorMetrics);
            dataTransmissionTime += getSparkCollectTime(output.rddStats, driverMetrics, executorMetrics);
        }
        opcode += type.isLeft() ? "_left" : "_right";
        long nflop = getInstNFLOP(inst.getSPInstructionType(), opcode, output, input);
        double mapTime = getCPUTime(nflop, input.rddStats.numPartitions, executorMetrics, output.rddStats, input.rddStats);
        return dataTransmissionTime + mapTime;
    }

    public static double getCentralMomentInstTime(CentralMomentSPInstruction inst, VarStats input, VarStats weights, VarStats output, IOMetrics executorMetrics) {
        CMOperator.AggregateOperationTypes opType = ((CMOperator) inst.getOperator()).getAggOpType();
        String opcode = inst.getOpcode() + "_" + opType.name().toLowerCase();

        double dataTransmissionTime = 0;
        if (weights != null) {
            dataTransmissionTime = getSparkShuffleWriteTime(weights.rddStats, executorMetrics) +
                    getSparkShuffleReadTime(weights.rddStats, executorMetrics);

        }
        output.rddStats.isCollected = true;

        RDDStats[] RDDInputs = (weights == null)? new RDDStats[]{input.rddStats} : new RDDStats[]{input.rddStats, weights.rddStats};
        long nflop = getInstNFLOP(inst.getSPInstructionType(), opcode, output, input);
        double mapTime = getCPUTime(nflop, input.rddStats.numPartitions, executorMetrics, output.rddStats, RDDInputs);
        return dataTransmissionTime + mapTime;
    }

    public static double getCastInstTime(CastSPInstruction inst, VarStats input, VarStats output, IOMetrics executorMetrics) {
        double shuffleTime = 0;
        if (input.getN() > input.characteristics.getBlocksize()) {
            shuffleTime = getSparkShuffleWriteTime(input.rddStats, executorMetrics) +
                    getSparkShuffleReadTime(input.rddStats, executorMetrics);
            output.rddStats.hashPartitioned = true;
        }
        long nflop = getInstNFLOP(inst.getSPInstructionType(), inst.getOpcode(), output, input);
        double mapTime = getCPUTime(nflop, input.rddStats.numPartitions, executorMetrics, output.rddStats, input.rddStats);
        return shuffleTime + mapTime;
    }

    public static double getQSortInstTime(QuantileSortSPInstruction inst, VarStats input, VarStats weights, VarStats output, IOMetrics executorMetrics) {
        String opcode = inst.getOpcode();
        double shuffleTime = 0;
        if (weights != null) {
            opcode += "_wts";
            shuffleTime += getSparkShuffleWriteTime(weights.rddStats, executorMetrics) +
                    getSparkShuffleReadTime(weights.rddStats, executorMetrics);
        }
        shuffleTime += getSparkShuffleWriteTime(output.rddStats, executorMetrics) +
                getSparkShuffleReadTime(output.rddStats, executorMetrics);
        output.rddStats.hashPartitioned = true;

        long nflop = getInstNFLOP(SPType.QSort, opcode, output, input, weights);
        RDDStats[] RDDInputs = (weights == null)? new RDDStats[]{input.rddStats} : new RDDStats[]{input.rddStats, weights.rddStats};
        double mapTime = getCPUTime(nflop, input.rddStats.numPartitions, executorMetrics, output.rddStats, RDDInputs);
        return shuffleTime + mapTime;
    }

    public static double getMatMulInstTime(BinarySPInstruction inst, VarStats input1, VarStats input2, VarStats output, IOMetrics driverMetrics, IOMetrics executorMetrics) {
        double dataTransmissionTime;
        int numPartitionsForMapping;
        if (inst instanceof CpmmSPInstruction) {
            CpmmSPInstruction cpmminst = (CpmmSPInstruction) inst;
            AggBinaryOp.SparkAggType aggType = cpmminst.getAggType();
            // estimate for in1.join(in2)
            long joinedSize = input1.rddStats.distributedSize + input2.rddStats.distributedSize;
            RDDStats joinedRDD = new RDDStats(joinedSize, -1);
            dataTransmissionTime = getSparkShuffleTime(joinedRDD, executorMetrics, true);
            if (aggType == AggBinaryOp.SparkAggType.SINGLE_BLOCK) {
                dataTransmissionTime += getSparkCollectTime(output.rddStats, driverMetrics, executorMetrics);
                output.rddStats.isCollected = true;
            } else {
                dataTransmissionTime += getSparkShuffleTime(output.rddStats, executorMetrics, true);
                output.rddStats.hashPartitioned = true;
            }
            numPartitionsForMapping = joinedRDD.numPartitions;
        } else if (inst instanceof RmmSPInstruction) {
            // estimate for in1.join(in2)
            long joinedSize = input1.rddStats.distributedSize + input2.rddStats.distributedSize;
            RDDStats joinedRDD = new RDDStats(joinedSize, -1);
            dataTransmissionTime = getSparkShuffleTime(joinedRDD, executorMetrics, true);
            // estimate for out.combineByKey() per partition
            dataTransmissionTime += getSparkShuffleTime(output.rddStats, executorMetrics, false);
            output.rddStats.hashPartitioned = true;
            numPartitionsForMapping = joinedRDD.numPartitions;
        } else if (inst instanceof MapmmSPInstruction) {
            dataTransmissionTime = getSparkBroadcastTime(input2, driverMetrics, executorMetrics);
            MapmmSPInstruction mapmminst = (MapmmSPInstruction) inst;
            AggBinaryOp.SparkAggType aggType = mapmminst.getAggType();
            if (aggType == AggBinaryOp.SparkAggType.SINGLE_BLOCK) {
                dataTransmissionTime += getSparkCollectTime(output.rddStats, driverMetrics, executorMetrics);
                output.rddStats.isCollected = true;
            } else {
                dataTransmissionTime += getSparkShuffleTime(output.rddStats, executorMetrics, true);
                output.rddStats.hashPartitioned = true;
            }
            numPartitionsForMapping = input1.rddStats.numPartitions;
        } else if (inst instanceof PmmSPInstruction) {
            dataTransmissionTime = getSparkBroadcastTime(input2, driverMetrics, executorMetrics);
            output.rddStats.numPartitions = input1.rddStats.numPartitions;
            dataTransmissionTime += getSparkShuffleTime(output.rddStats, executorMetrics, true);
            output.rddStats.hashPartitioned = true;
            numPartitionsForMapping = input1.rddStats.numPartitions;
        } else if (inst instanceof ZipmmSPInstruction) {
            // assume always a shuffle without data re-distribution
            dataTransmissionTime = getSparkShuffleTime(output.rddStats, executorMetrics, false);
            dataTransmissionTime += getSparkCollectTime(output.rddStats, driverMetrics, executorMetrics);
            numPartitionsForMapping = input1.rddStats.numPartitions;
            output.rddStats.isCollected = true;
        } else if (inst instanceof PMapmmSPInstruction) {
            throw new RuntimeException("PMapmmSPInstruction instruction is still experimental and not supported yet");
        } else {
            throw new RuntimeException(inst.getClass().getName() + " instruction is not handled by the current method");
        }
        long nflop = getInstNFLOP(inst.getSPInstructionType(), inst.getOpcode(), output, input1, input2);
        double mapTime;
        if (inst instanceof MapmmSPInstruction || inst instanceof PmmSPInstruction) {
            // scan only first input
            mapTime = getCPUTime(nflop, numPartitionsForMapping, executorMetrics, output.rddStats, input1.rddStats);
        } else {
            mapTime = getCPUTime(nflop, numPartitionsForMapping, executorMetrics, output.rddStats, input1.rddStats, input2.rddStats);
        }
        return dataTransmissionTime + mapTime;
    }

    public static double getMatMulChainInstTime(MapmmChainSPInstruction inst, VarStats input1, VarStats input2, VarStats input3, VarStats output,
                                                IOMetrics driverMetrics, IOMetrics executorMetrics) {
        double dataTransmissionTime = 0;
        if (input3 != null) {
            dataTransmissionTime += getSparkBroadcastTime(input3, driverMetrics, executorMetrics);
        }
        dataTransmissionTime += getSparkBroadcastTime(input2, driverMetrics, executorMetrics);
        dataTransmissionTime += getSparkCollectTime(output.rddStats, driverMetrics, executorMetrics);
        output.rddStats.isCollected = true;

        long nflop = getInstNFLOP(SPType.MAPMMCHAIN, inst.getOpcode(), output, input1, input2);
        double mapTime = getCPUTime(nflop, input1.rddStats.numPartitions, executorMetrics, output.rddStats, input1.rddStats);
        return dataTransmissionTime + mapTime;
    }

    public static double getCtableInstTime(CtableSPInstruction tableInst, VarStats input1, VarStats input2, VarStats input3, VarStats output, IOMetrics executorMetrics) {
        String opcode = tableInst.getOpcode();
        double shuffleTime;
        if (opcode.equals("ctableexpand") || !input2.isScalar() && input3.isScalar()) { // CTABLE_EXPAND_SCALAR_WEIGHT/CTABLE_TRANSFORM_SCALAR_WEIGHT
            // in1.join(in2)
            shuffleTime = getSparkShuffleTime(input2.rddStats, executorMetrics, true);
        } else if (input2.isScalar() && input3.isScalar()) { // CTABLE_TRANSFORM_HISTOGRAM
            // no joins
            shuffleTime = 0;
        } else if (input2.isScalar() && !input3.isScalar()) { // CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM
            // in1.join(in3)
            shuffleTime = getSparkShuffleTime(input3.rddStats, executorMetrics, true);
        } else { // CTABLE_TRANSFORM
            // in1.join(in2).join(in3)
            shuffleTime = getSparkShuffleTime(input2.rddStats, executorMetrics, true);
            shuffleTime += getSparkShuffleTime(input3.rddStats, executorMetrics, true);
        }
        // combineByKey()
        shuffleTime += getSparkShuffleTime(output.rddStats, executorMetrics, true);
        output.rddStats.hashPartitioned = true;

        long nflop = getInstNFLOP(SPType.Ctable, opcode, output, input1, input2, input3);
        double mapTime = getCPUTime(nflop, output.rddStats.numPartitions, executorMetrics,
                output.rddStats, input1.rddStats, input2.rddStats, input3.rddStats);

        return shuffleTime + mapTime;
    }

    public static double getParameterizedBuiltinInstTime(ParameterizedBuiltinSPInstruction paramInst, VarStats input1, VarStats input2, VarStats output, IOMetrics driverMetrics, IOMetrics executorMetrics) {
        String opcode = paramInst.getOpcode();
        double dataTransmissionTime;
        switch (opcode) {
            case "rmempty":
                if (input2.rddStats == null) // broadcast
                    dataTransmissionTime = getSparkBroadcastTime(input2, driverMetrics, executorMetrics);
                else // join
                    dataTransmissionTime = getSparkShuffleTime(input1.rddStats, executorMetrics, true);
                dataTransmissionTime += getSparkShuffleTime(output.rddStats, executorMetrics, true);
                break;
            case "contains":
                if (input2.isScalar()) {
                    dataTransmissionTime = 0;
                } else {
                    dataTransmissionTime = getSparkBroadcastTime(input2, driverMetrics, executorMetrics);
                    // ignore reduceByKey() cost
                }
                output.rddStats.isCollected = true;
                break;
            case "replace":
            case "lowertri":
            case "uppertri":
                dataTransmissionTime = 0;
                break;
            default:
                throw new RuntimeException("Spark operation ParameterizedBuiltin with opcode " + opcode + " is not supported yet");
        }

        long nflop = getInstNFLOP(paramInst.getSPInstructionType(), opcode, output, input1);
        double mapTime = getCPUTime(nflop, input1.rddStats.numPartitions, executorMetrics, output.rddStats, input1.rddStats);

        return dataTransmissionTime + mapTime;
    }

    /**
     * Computes an estimate for the time needed by the CPU to execute (including memory access)
     * an instruction by providing number of floating operations.
     *
     * @param nflop number FLOP to execute a target CPU operation
     * @param numPartitions number partitions used to execute the target operation;
     *                      not bound to any of the input/output statistics object to allow more
     *                      flexibility depending on the corresponding instruction
     * @param executorMetrics metrics for the executor utilized by the Spark cluster
     * @param output statistics for the output variable
     * @param inputs arrays of statistics for the output variable
     * @return time estimate
     */
    public static double getCPUTime(long nflop, int numPartitions, IOMetrics executorMetrics, RDDStats output, RDDStats...inputs) {
        double memScanTime = 0;
        for (RDDStats input: inputs) {
            if (input == null) continue;
            // compensates for spill-overs to account for non-compute bound operations
            memScanTime += getMemReadTime(input, executorMetrics);
        }
        double numWaves = Math.ceil((double) numPartitions / SparkExecutionContext.getDefaultParallelism(false));
        double scaledNFLOP = (numWaves * nflop) / numPartitions;
        double cpuComputationTime = scaledNFLOP / executorMetrics.cpuFLOPS;
        double memWriteTime = output != null? getMemWriteTime(output, executorMetrics) : 0;
        return Math.max(memScanTime, cpuComputationTime) + memWriteTime;
    }

    public static void assignOutputRDDStats(SPInstruction inst, VarStats output, VarStats...inputs) {
        if (!output.isScalar()) {
            SPType instType = inst.getSPInstructionType();
            String opcode = inst.getOpcode();
            if (output.getCells() < 0) {
                inferStats(instType, opcode, output, inputs);
            }
        }
        output.rddStats = new RDDStats(output);
    }

    private static void inferStats(SPType instType, String opcode, VarStats output, VarStats...inputs) {
        switch (instType) {
            case Unary:
            case Builtin:
                CPCostUtils.inferStats(CPType.Unary, opcode, output, inputs);
                break;
            case AggregateUnary:
            case AggregateUnarySketch:
                CPCostUtils.inferStats(CPType.AggregateUnary, opcode, output, inputs);
            case MatrixIndexing:
                CPCostUtils.inferStats(CPType.MatrixIndexing, opcode, output, inputs);
                break;
            case Reorg:
                CPCostUtils.inferStats(CPType.Reorg, opcode, output, inputs);
                break;
            case Binary:
                CPCostUtils.inferStats(CPType.Binary, opcode, output, inputs);
                break;
            case CPMM:
            case RMM:
            case MAPMM:
            case PMM:
            case ZIPMM:
                CPCostUtils.inferStats(CPType.AggregateBinary, opcode, output, inputs);
                break;
            case ParameterizedBuiltin:
                CPCostUtils.inferStats(CPType.ParameterizedBuiltin, opcode, output, inputs);
                break;
            case Rand:
                CPCostUtils.inferStats(CPType.Rand, opcode, output, inputs);
                break;
            case Ctable:
                CPCostUtils.inferStats(CPType.Ctable, opcode, output, inputs);
                break;
            default:
                throw new RuntimeException("Operation of type "+instType+" with opcode '"+opcode+"' has no formula for inferring dimensions");
        }
        if (output.getCells() < 0) {
            throw new RuntimeException("Operation of type "+instType+" with opcode '"+opcode+"' has incomplete formula for inferring dimensions");
        }
    }

    private static long getInstNFLOP(
            SPType instructionType,
            String opcode,
            VarStats output,
            VarStats...inputs
    ) {
        opcode = opcode.toLowerCase();
        double costs;
        switch (instructionType) {
            case Reblock:
                if (opcode.startsWith("libsvm")) {
                    return output.getCellsWithSparsity();
                } else { // starts with "rblk" or "csvrblk"
                    return output.getCells();
                }
            case Unary:
            case Builtin:
                return CPCostUtils.getInstNFLOP(CPType.Unary, opcode, output, inputs);
            case AggregateUnary:
            case AggregateUnarySketch:
                switch (opcode) {
                    case "uacdr":
                    case "uacdc":
                        throw new DMLRuntimeException(opcode + " opcode is not implemented by SystemDS");
                    default:
                        return CPCostUtils.getInstNFLOP(CPType.AggregateUnary, opcode, output, inputs);
                }
            case CumsumAggregate:
                switch (opcode) {
                    case "ucumack+":
                    case "ucumac*":
                    case "ucumacmin":
                    case "ucumacmax":
                        costs = 1; break;
                    case "ucumac+*":
                        costs = 2; break;
                    default:
                        throw new DMLRuntimeException(opcode + " opcode is not implemented by SystemDS");
                }
                return (long) (costs * inputs[0].getCells() + costs * output.getN());
            case TSMM:
            case TSMM2:
                return CPCostUtils.getInstNFLOP(CPType.MMTSJ, opcode, output, inputs);
            case Reorg:
            case MatrixReshape:
                return CPCostUtils.getInstNFLOP(CPType.Reorg, opcode, output, inputs);
            case MatrixIndexing:
                // the actual opcode value is not used at the moment
                return CPCostUtils.getInstNFLOP(CPType.MatrixIndexing, opcode, output, inputs);
            case Cast:
                return output.getCellsWithSparsity();
            case QSort:
                return CPCostUtils.getInstNFLOP(CPType.QSort, opcode, output, inputs);
            case CentralMoment:
                return CPCostUtils.getInstNFLOP(CPType.CentralMoment, opcode, output, inputs);
            case UaggOuterChain:
            case Dnn:
                throw new RuntimeException("Spark operation type'" + instructionType + "' is not supported yet");
            // types corresponding to BinaryCPInstruction
            case Binary:
                switch (opcode) {
                    case "+*":
                    case "-*":
                        // original "map+*" and "map-*"
                        // "+*" and "-*" defined as ternary
                        throw new RuntimeException("Spark operation with opcode '" + opcode + "' is not supported yet");
                    default:
                        return CPCostUtils.getInstNFLOP(CPType.Binary, opcode, output, inputs);
                }
            case CPMM:
            case RMM:
            case MAPMM:
            case PMM:
            case ZIPMM:
            case PMAPMM:
                // do not reduce by factor of 2: not explicit matrix multiplication
                return 2 * CPCostUtils.getInstNFLOP(CPType.AggregateBinary, opcode, output, inputs);
            case MAPMMCHAIN:
                return 2 * inputs[0].getCells() * inputs[0].getN() // ba(+*)
                        + 2 * inputs[0].getM() * inputs[1].getN() // cellwise b(*) + r(t)
                        + 2 * inputs[0].getCellsWithSparsity() * inputs[1].getN() // ba(+*)
                        + inputs[1].getM() * output.getM() ; //r(t)
            case BinUaggChain:
                break;
            case MAppend:
            case RAppend:
            case GAppend:
            case GAlignedAppend:
                // the actual opcode value is not used at the moment
                return CPCostUtils.getInstNFLOP(CPType.Append, opcode, output, inputs);
            case BuiltinNary:
                return CPCostUtils.getInstNFLOP(CPType.BuiltinNary, opcode, output, inputs);
            case Ctable:
                return CPCostUtils.getInstNFLOP(CPType.Ctable, opcode, output, inputs);
            case ParameterizedBuiltin:
                return CPCostUtils.getInstNFLOP(CPType.ParameterizedBuiltin, opcode, output, inputs);
            default:
                // all existing cases should have been handled above
                throw new DMLRuntimeException("Spark operation type'" + instructionType + "' is not supported by SystemDS");
        }
        throw new RuntimeException();
    }


//        //ternary aggregate operators
//        case "tak+*":
//            break;
//        case "tack+*":
//            break;
//        // Neural network operators
//        case "conv2d":
//        case "conv2d_bias_add":
//        case "maxpooling":
//        case "relu_maxpooling":
//        case RightIndex.OPCODE:
//        case LeftIndex.OPCODE:
//        case "mapLeftIndex":
//        case "_map",:
//            break;
//        // Spark-specific instructions
//        case Checkpoint.DEFAULT_CP_OPCODE,:
//            break;
//        case Checkpoint.ASYNC_CP_OPCODE,:
//            break;
//        case Compression.OPCODE,:
//            break;
//        case DeCompression.OPCODE,:
//            break;
//        // Parameterized Builtin Functions
//        case "autoDiff",:
//            break;
//        case "contains",:
//            break;
//        case "groupedagg",:
//            break;
//        case "mapgroupedagg",:
//            break;
//        case "rmempty",:
//            break;
//        case "replace",:
//            break;
//        case "rexpand",:
//            break;
//        case "lowertri",:
//            break;
//        case "uppertri",:
//            break;
//        case "tokenize",:
//            break;
//        case "transformapply",:
//            break;
//        case "transformdecode",:
//            break;
//        case "transformencode",:
//            break;
//        case "mappend",:
//            break;
//        case "rappend",:
//            break;
//        case "gappend",:
//            break;
//        case "galignedappend",:
//            break;
//        //ternary instruction opcodes
//        case "ctable",:
//            break;
//        case "ctableexpand",:
//            break;
//
//        //ternary instruction opcodes
//        case "+*",:
//            break;
//        case "-*",:
//            break;
//        case "ifelse",:
//            break;
//
//        //quaternary instruction opcodes
//        case WeightedSquaredLoss.OPCODE,:
//            break;
//        case WeightedSquaredLossR.OPCODE,:
//            break;
//        case WeightedSigmoid.OPCODE,:
//            break;
//        case WeightedSigmoidR.OPCODE,:
//            break;
//        case WeightedDivMM.OPCODE,:
//            break;
//        case WeightedDivMMR.OPCODE,:
//            break;
//        case WeightedCrossEntropy.OPCODE,:
//            break;
//        case WeightedCrossEntropyR.OPCODE,:
//            break;
//        case WeightedUnaryMM.OPCODE,:
//            break;
//        case WeightedUnaryMMR.OPCODE,:
//            break;
//        case "bcumoffk+":
//            break;
//        case "bcumoff*":
//            break;
//        case "bcumoff+*":
//            break;
//        case "bcumoffmin",:
//            break;
//        case "bcumoffmax",:
//            break;
//
//        //central moment, covariance, quantiles (sort/pick)
//        case "cm"     ,:
//            break;
//        case "cov"    ,:
//            break;
//        case "qsort"  ,:
//            break;
//        case "qpick"  ,:
//            break;
//
//        case "binuaggchain",:
//            break;
//
//        case "write"	,:
//            break;
//
//
//        case "spoof":
//            break;
//        default:
//            throw RuntimeException("No complexity factor for op. code: " + opcode);
//    }
}
