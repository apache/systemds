package org.apache.sysds.resource.cost;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.spark.rdd.RDD;
import org.apache.sysds.lops.*;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;

import static org.apache.sysds.resource.cost.IOCostUtils.*;

public class SparkCostUtils {

    /** Getting the cost for transforming the text data into double value */
    public static double getReblockInstTime(String opcode, RDDStats output, IOMetrics metrics) {
        long nflop;
        if (opcode.startsWith("libsvm")) {
            nflop = (long) (output.m * output.n * output.sparsity);
        } else { // starts with "rblk" or "csvrblk"
            nflop = output.m * output.n;
        }
        double computeTime = getScaledNFLOP(nflop, output) / metrics.cpuFLOPS;
        // TODO: scan time for the input is more accurate?
        double scanTime = getMemReadTime(output, metrics);
        double memWriteTime = getMemWriteTime(output, metrics);
        return Math.max(computeTime, scanTime) + memWriteTime;
    }

    public static double getUnaryInstTime(String opcode, RDDStats output, IOMetrics metrics) {
        long nflop;
        boolean scale = true;
        switch (opcode) {
            case "abs": case "round": case "ceil": case "floor": case "sign":
            case "!": case "isna": case "isnan": case "isinf":
                nflop = 1; break;
            case "sprop": case "sqrt":
                nflop = 2; break;
            case "sin": case "exp":
                nflop = 18; break;
            case "sigmoid":
                nflop = 21; break;
            case "cos":
                nflop = 22; break;
            case "tan":
                nflop = 42; break;
            case "asin": case "sinh":
                nflop = 93; break;
            case "acos": case "cosh":
                nflop = 103; break;
            case "atan": case "tanh":
                nflop = 40; break;
            // frames only
            case "detectSchema":
                nflop = output.m;
                scale = false;
                break;
            case "colnames":
                nflop = output.n;
                scale = false;
                break;
            default:
                throw new RuntimeException("No complexity factor for op. code: " + opcode);
        }
        // scale
        if (scale) nflop *= output.m * output.n;

        double computeTime = getScaledNFLOP(nflop, output) / metrics.cpuFLOPS;
        double scanTime = getMemReadTime(output, metrics);
        double memWriteTime = getMemWriteTime(output, metrics);
        return Math.max(computeTime, scanTime) + memWriteTime;
    }

    public static double getAggUnaryInstTime(String opcode, RDDStats input, RDDStats output, IOMetrics metrics) {
        // use input sparsity since this is the one of the matrix for aggregation
        // TODO: think of AggregateUnaryOperator.sparseSafe when accounting for the sparsity
        long nflop;
        long cellsOut = output.m * output.n;
        switch (opcode) {
            // Unary aggregate operators
            case "uak+":
            case "uark+":
            case "uack+":
                nflop = 4 * cellsOut;
                break;
            case "uasqk+":
            case "uarsqk+":
            case "uacsqk+":
                nflop = 5 * cellsOut;
                break;
            case "uamean":
            case "uarmean":
            case "uacmean":
                nflop = 7 * cellsOut;
                break;
            case "uavar":
            case "uarvar":
            case "uacvar":
                nflop = 14 * cellsOut;
                break;
            case "uamax":
            case "uarmax":
            case "uarimax":
            case "uacmax":
            case "uamin":
            case "uarmin":
            case "uarimin":
            case "uacmin":
                nflop = cellsOut;
                break;
            case "ua+":
            case "uar+":
            case "uac+":
                nflop = (long) (1 * cellsOut * input.sparsity);
                break;
            case "ua*":
            case "uar*":
            case "uac*":
                nflop = (long) (2 * cellsOut * input.sparsity); // TODO: refine
                break;
            // Aggregation over the diagonal of a square matrix
            case "uatrace":
            case "uaktrace":
                nflop = input.m;
                break;
            // Aggregate Unary Sketch operators
            case "uacd":
                nflop = (long) (cellsOut * input.sparsity);
                break;
            case "uacdap":
            case "uacdapr":
            case "uacdapc":
                nflop = (long) (0.5  * cellsOut * input.sparsity); // do not iterate through all the cells
                break;
            case "uacdr":
            case "uacdc":
                throw new DMLRuntimeException(opcode+ " opcode is not implemented yet");
            default:
                throw new RuntimeException("No complexity factor for op. code: " + opcode);
        }

        if (!(opcode.equals("uatrace") || opcode.equals("uaktrace"))) {
            // scale
            if (opcode.startsWith("uar")) {
                nflop *= input.m;
            } else if (opcode.startsWith("uac")) {
                nflop *= input.n;
            } else {
                nflop *= input.m * input.n;
            }
        }
        double computeTime = getScaledNFLOP(nflop, input) / metrics.cpuFLOPS;
        double scanTime = getMemReadTime(input, metrics);
        double memWriteTime = getMemWriteTime(output, metrics);
        return Math.max(scanTime, computeTime) + memWriteTime;
    }

    public static double getBinaryInstTime(String opcode, RDDStats input1, RDDStats input2, RDDStats output, IOMetrics metrics) {
        // binary, builtin binary, builtin log and log_nz (unary or binary), // todo: and binagg too (cbin, rbin ...)?
        // for the NFLOP calculation if the function is executed as map is not relevant
        if (opcode.startsWith("map")) {
            opcode = opcode.substring(3);
        }
        double factor;
        switch (opcode) {
            case "+": case "-":
                factor = output.sparsity; // sparse safe
                break;
            case "*": case "^2": case "*2": case "max": case "min":
            case "==": case "!=": case "<": case ">": case "<=": case ">=":
            case "&&": case "||": case "xor": case "bitwAnd": case "bitwOr": case "bitwXor": case "bitwShiftL": case "bitwShiftR":
                factor = 1; break;
            case "%/%":
                factor = 6; break;
            case "%%":
                factor = 8; break;
            case "/":
                factor = 22; break;
            case "log": case "log_nz":
                factor = 32; break;
            case "^":
                factor = 16; break;
            case "1-*":
                factor = 2; break;
            case "dropInvalidType":
            case "freplicate":
            case "dropInvalidLength": // original "mapdropInvalidLength"
            case "valueSwap":
            case "+*": case "-*": // original "map+*" and "map-*"
                // "+*" and "-*" defined as ternary
                // TODO: implement
                throw new RuntimeException("Not sure how to handle " + opcode);
            default:
                throw new RuntimeException("No complexity factor for op. code: " + opcode);
        }
        long nflop = (long) (factor * output.m * output.n);
        double computeTime = getScaledNFLOP(nflop, output) / metrics.cpuFLOPS;
        // handle scanning time for any type inputs
        double scanTime = ((input1 != null)? getMemReadTime(input1, metrics) : 0)
                + ((input2 != null)? getMemReadTime(input2, metrics) : 0);
        double memWriteTime = getMemWriteTime(output, metrics);
        return Math.max(computeTime, scanTime) + memWriteTime;
    }

    public static double getAggBinaryInstNFLOP(String opcode, long colsIn1, long rowsIn1, double sparsityIn1,
                                             long colsIn2, long rowsIn2, double sparsityIn2,
                                             long rowsOut, long colsOut, double sparsityOut) {
        switch (opcode) {
            // Matrix multiplication operators
            case "cpmm":
            case "rmm":
            case "mapmm":
                return rowsIn1 * colsIn1 * sparsityIn1 * colsIn2 * (colsIn2 > 1? sparsityIn1 : 1);
            case "mapmmchain":
                return 2 * rowsIn1 * colsIn2 * colsIn1 * (colsIn2 > 1? sparsityIn1 : 1) //ba(+*)
                        + 2 * rowsIn1 * colsIn2 // cellwise b(*) + r(t)
                        + 2 * colsIn2 * colsIn1 * rowsIn1 * sparsityIn1 //ba(+*)
                        + colsIn1 * colsIn2; //r(t)
            case "tsmm":
            case "tsmm2":
                return rowsIn1 * colsIn1 * sparsityIn1 * colsIn1 * sparsityIn1 /2; // TODO: refine
            case "pmm":
            case "zipmm":
            case "pmapmm":
                return colsOut * rowsOut * sparsityOut; // TODO: implement
            // Rest operators
            case "uaggouterchain":
                throw new RuntimeException("Not sure how to handle " + opcode);
        }
        throw new RuntimeException("No complexity factor for op. code: " + opcode);
    }

    public static double getReorgInstNFLOP(String opcode, long rowsIn, long colsIn, double sparsityIn) {
        // includes MatrixReshape instruction "rshape"
        double costs;
        switch (opcode) {
            case "r'": case "rdiag": case "rshape":
                costs = sparsityIn;
                break;
            case "rev":
            case "rsort":
                throw new RuntimeException("Not sure how to handle " + opcode);
            default:
                throw new RuntimeException("No complexity factor for op. code: " + opcode);
        }
        return costs * colsIn * rowsIn;
    }

    public static double getBuiltinNaryInstNFLOP(String opcode, int numIns, long rowsOut, long colsOut, double sparsityOut) {
        // NOTE: very dummy implementation
        double costs;
        switch (opcode) {
            case "cbind": case "rbind":
                return colsOut * rowsOut * sparsityOut;
            case "nmin": case "nmax": case "n+":
                return numIns * colsOut * rowsOut * sparsityOut;
        }
        throw new RuntimeException("No complexity factor for op. code: " + opcode);
    }

    public static double getRandInstTime(String opcode, RDDStats output, int randType, IOMetrics metrics) {
        long nflop;
        switch (opcode) {
            case DataGen.RAND_OPCODE:
            case DataGen.FRAME_OPCODE:
                if (randType == 0) {
                    return 0; // empty matrix
                } else if (randType == 1) {
                    nflop = 8; // allocate, array fill
                } else if (randType == 2) {
                    nflop = 32; // full rand
                } else {
                    throw new RuntimeException("Unknown type of random instruction");
                }
                break;
            case DataGen.SEQ_OPCODE:
                nflop = 1; break;
            case DataGen.SAMPLE_OPCODE:
                nflop = 16; break; // NOTE: dummy
            default:
                throw new RuntimeException("No complexity factor for op. code: " + opcode);
        }
        nflop *= output.m * output.n;
        double computeTime = getScaledNFLOP(nflop, output) / metrics.cpuFLOPS;
        double memWriteTime = getMemWriteTime(output, metrics);
        return computeTime + memWriteTime;
    }

    public static double getCastInstNFLOP(long rowsOut, long colsOut) {
        return (double) colsOut * rowsOut;
    }

    public static double getIndexingInstNFLOP(String opcode, long rowsOut, long colsOut, double sparsityOut) {
        // NOTE: dummy
//        case RightIndex.OPCODE:
//        case LeftIndex.OPCODE:
//        case "mapLeftIndex":
        return colsOut * rowsOut * sparsityOut;
    }

    public static double getCumAggInstNFLOP(String opcode, long rowsOut, long colsOut, double sparsityOut) {
        double costs;
        switch (opcode) {
            case "ucumack+":
            case "ucumac*":
            case "ucumacmin":
            case "ucumacmax":
                costs = 1; break;
            case "ucumac+*":
                costs = 2; break;
            default:
                throw new RuntimeException("No complexity factor for op. code: " + opcode);
        }
        return costs * colsOut * rowsOut * sparsityOut;
    }

    public static double getScaledNFLOP(long nflop, RDDStats stats) {
        double numWaves = Math.ceil((double) stats.numPartitions / SparkExecutionContext.getDefaultParallelism(false));
        return (numWaves * nflop) / stats.numPartitions;
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
