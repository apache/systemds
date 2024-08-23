package org.apache.sysds.resource.cost;

import org.apache.commons.lang3.function.TriFunction;
import org.apache.spark.api.java.function.Function3;
import org.apache.spark.sql.sources.In;
import org.apache.sysds.runtime.instructions.spark.UnarySPInstruction;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import scala.Function2;

import java.util.HashMap;
import java.util.function.BiFunction;
import java.util.function.Function;

public class SparkCostUtils {
    public static final HashMap<String, Object[]> SPARK_COST_MAP;

    static {
        SPARK_COST_MAP = new HashMap<>();
        // ----- Reblock -----

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
