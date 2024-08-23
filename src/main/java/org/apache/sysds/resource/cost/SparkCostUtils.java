package org.apache.sysds.resource.cost;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.function.TriFunction;
import org.apache.spark.api.java.function.Function3;
import org.apache.spark.sql.sources.In;
import org.apache.sysds.lops.*;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.runtime.instructions.spark.UnarySPInstruction;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import scala.Function2;

import java.util.HashMap;
import java.util.function.BiFunction;
import java.util.function.Function;

public class SparkCostUtils {

    public static double getReblockInstNFLOP(String opcode, long colsOut, long rowsOut, double sparsity) {
        if (opcode.startsWith("libsvm")) { // case "libsvmrblk"
            return colsOut * rowsOut * sparsity;
        }
        // case "rblk" and case "csvrblk"
        return colsOut * rowsOut;
    }

    public static double getUnaryInstNFLOP(String opcode, long colsOut, long rowsOut) {
        double costs;
        switch (opcode) {
            case "abs": case "round": case "ceil": case "floor": case "sign":
            case "!": case "isna": case "isnan": case "isinf":
                costs = 1; break;
            case "sprop": case "sqrt":
                costs = 2; break;
            case "sin": case "exp":
                costs = 18; break;
            case "plogp":
                costs = 32; break; // TODO: refine (now the log() cost)
            case "sigmoid":
                costs = 21; break;
            case "cos":
                costs = 22; break;
            case "tan":
                costs = 42; break;
            case "asin": case "sinh":
                costs = 93; break;
            case "acos": case "cosh":
                costs = 103; break;
            case "atan": case "tanh":
                costs = 40; break;
            case "detectSchema":
                return rowsOut;
            case "colnames":
                return colsOut;
            default:
                throw new RuntimeException("No complexity factor for op. code: " + opcode);
        }
        return costs * colsOut * rowsOut;
    }

    public static double getAggUnaryInstNFLOP(String opcode, long colsIn, long rowsIn, double sparsityIn,
                                           long colsOut, long rowsOut) {
        // use input sparsity since this is the one of the matrix for aggregation
        long cellsOut = colsOut * rowsOut;
        double costs;
        switch (opcode) {
            // Unary aggregate operators
            case "uak+":
            case "uark+":
            case "uack+":
                costs = 4 * cellsOut;
                break;
            case "uasqk+":
            case "uarsqk+":
            case "uacsqk+":
                costs = 5 * cellsOut;
                break;
            case "uamean":
            case "uarmean":
            case "uacmean":
                costs = 7 * cellsOut;
                break;
            case "uavar":
            case "uarvar":
            case "uacvar":
                costs = 14 * cellsOut;
                break;
            case "uamax":
            case "uarmax":
            case "uarimax":
            case "uacmax":
            case "uamin":
            case "uarmin":
            case "uarimin":
            case "uacmin":
                costs = cellsOut;
                break;
            case "ua+":
            case "uar+":
            case "uac+":
                costs = 1 * cellsOut * sparsityIn;
                break;
            case "ua*":
            case "uar*":
            case "uac*":
                costs = 2 * cellsOut * sparsityIn; // TODO: refine
                break;
            case "uatrace":
            case "uaktrace":
                costs = 2 * cellsOut;
                break;
            // Aggregate Unary Sketch operators
            case "uacd":
                costs = 1 * cellsOut * sparsityIn; // TODO: refine
                break;
            case "uacdap":
            case "uacdapr":
            case "uacdapc":
                costs = 0.5  * cellsOut * sparsityIn; // TODO: refine
                break;
            case "uacdr":
            case "uacdc":
                throw new NotImplementedException(opcode+ " opcode is not implemented yet");
            default:
                throw new RuntimeException("No complexity factor for op. code: " + opcode);
        }

        if (opcode.startsWith("uar")) {
            costs *= rowsIn;
        } else if (opcode.startsWith("uac")) {
            costs *= colsIn;
        } else {
            costs *= colsIn * rowsIn;
        }
        return costs;
    }

    public static double getBinaryInstNFLOP(String opcode, long colsOut, long rowsOut, double sparsityOut) {
        // binary, builtin binary, builtin log and log_nz (unary or binary), // todo: and binagg too (cbin, rbin ...)?
        // for the NFLOP calculation if the function is executed as map is not relevant
        if (opcode.startsWith("map")) {
            opcode = opcode.substring(3);
        }
        double costs;
        switch (opcode) {
            case "+": case "-":
                costs = sparsityOut; // sparse safe
            case "*": case "^2": case "*2": case "max": case "min":
            case "==": case "!=": case "<": case ">": case "<=": case ">=":
            case "&&": case "||": case "xor": case "bitwAnd": case "bitwOr": case "bitwXor": case "bitwShiftL": case "bitwShiftR":
                costs = 1; break;
            case "%/%":
                costs = 6; break;
            case "%%":
                costs = 8; break;
            case "/":
                costs = 22; break;
            case "log": case "log_nz":
                costs = 32; break;
            case "^":
                costs = 16; break;
            case "1-*":
                costs = 2; break;
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
        return costs * colsOut * rowsOut;
    }

    public static double getAggBinaryInstNFLOP(String opcode, long colsIn1, long rowsIn1, double sparsityIn1,
                                             long colsIn2, long rowsIn2, double sparsityIn2,
                                             long colsOut, long rowsOut, double sparsityOut) {
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

    public static double getReorgInstNFLOP(String opcode, long colsIn, long rowsIn, double sparsityIn) {
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

    public static double getBuiltinNaryInstNFLOP(String opcode, int numIns, long colsOut, long rowsOut, double sparsityOut) {
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

    public static double getComplexityDataGen(String opcode, long colsOut, long rowsOut) {
        double costs;
        switch (opcode) {
            case DataGen.RAND_OPCODE:
                // TODO: extend for constants and only fill
                costs = 32; break;
            case DataGen.SEQ_OPCODE:
                costs = 1; break;
            case DataGen.SAMPLE_OPCODE:
                costs = 16; break; // NOTE: dummy
            case DataGen.FRAME_OPCODE:
                costs = 2; break; // NOTE: dummy
            default:
                throw new RuntimeException("No complexity factor for op. code: " + opcode);
        }
        return costs * colsOut * rowsOut;
    }

    public static double getCastInstNFLOP(long colsOut, long rowsOut) {
        return (double) colsOut * rowsOut;
    }

    public static double getIndexingInstNFLOP(String opcode, long colsOut, long rowsOut, double sparsityOut) {
        // NOTE: dummy
//        case RightIndex.OPCODE:
//        case LeftIndex.OPCODE:
//        case "mapLeftIndex":
        return colsOut * rowsOut * sparsityOut;
    }

    public static double getCumAggInstNFLOP(String opcode, long colsOut, long rowsOut, double sparsityOut) {
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

    public static final HashMap<String, Object[]> SPARK_COST_MAP;
    

    static {
        SPARK_COST_MAP = new HashMap<>();
        // ----- Reblock/CSVReblock/LIBSVMReblock -----
        Function<RDDStats, Double> reblockComputeCost = (input) -> (double) input.cpStats.getCellsWithSparsity();
    }

    @SuppressWarnings("unchecked")
    public static double callFunc(String instruction, int funcIndex, RDDStats... args) {
        switch (args.length) {
            case 1:
                return ((Function<RDDStats, Double>) SPARK_COST_MAP.get(instruction)[funcIndex]).apply(args[0]);
            case 2:
                return ((BiFunction<RDDStats, RDDStats, Double>) SPARK_COST_MAP.get(instruction)[funcIndex]).apply(args[0], args[1]);
            case 3:
                return ((TriFunction<RDDStats, RDDStats, RDDStats, Double>) SPARK_COST_MAP.get(instruction)[funcIndex]).apply(args[0], args[1], args[2]);
            default:
                throw new IllegalStateException("SPARK_COST_MAP does not contain functions with " + args.length + " arguments.");
        }
    }

    public static void main(String[] args) throws Exception {
        // test the utilities at dev process

        Function<RDDStats, Double> unaryExample = (input) -> (double) input.cpStats.getCells() / input.numParallelTasks;
        BiFunction<RDDStats, RDDStats, Double> add = (input1, input2) -> (double) (input1.cpStats.getCells() * input1.cpStats.getCells()) / input1.numParallelTasks;
        TriFunction<RDDStats, RDDStats, RDDStats, Double> dummy = (input1, input2, output) -> (double) output.cpStats.getCells();
        SPARK_COST_MAP.put("example", new Object[]{unaryExample, add, dummy});

        MatrixCharacteristics mc = new MatrixCharacteristics(1000, 1000, 1000, 1000000);
        VarStats varStats = new VarStats(mc);
        RDDStats rddStats = new RDDStats(varStats);

        double result1 = callFunc("example", 0, rddStats);
        double result2 = callFunc("example", 1, rddStats, rddStats);
        double result3 = callFunc("example", 2, rddStats, rddStats, rddStats);

        System.out.println(result1);
        System.out.println(result2);
        System.out.println(result3);
    }
}
