package org.apache.sysds.resource.enumeration;

import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.runtime.controlprogram.*;
import org.apache.sysds.resource.enumeration.EnumerationUtils.InstanceSearchSpace;

import java.util.*;
import java.util.stream.Collectors;

public class InterestBasedEnumerator extends AnEnumerator {
    public final static long MINIMUM_RELEVANT_MEM_ESTIMATE = 2L * 1024 * 1024 * 1024; // 2GB
    // different instance families can have slightly different memory characteristics (e.g. EC2 Graviton (arm) instances)
    public final static double MEMORY_DELTA_FRACTION = 0.05; // 5%
    public final static double DATA_MEMORY_FACTOR = 0.3; // minimum memory fraction for storage
    public final static double BROADCAST_MEMORY_FACTOR = 0.7; // fraction of the minimum memory fraction for storage
    // marks if memory estimates should be used at deciding
    // for the search space of the instance for the driver nodes
    private final boolean fitDriverMemory;
    // marks if memory estimates should be used at deciding
    // for the search space of the instance for the executor nodes
    private final boolean fitBroadcastMemory;
    // marks if the estimation of the range of number of executors
    // for consideration should exclude single node execution mode
    // if any of the estimates cannot fit in the driver's memory
    private final boolean checkSingleNodeExecution;
    // marks if the estimated output size should be
    // considered as interesting point at deciding the
    // number of executors - checkpoint storage level
    private final boolean fitCheckpointMemory;
    // marks if the number of executors should
    // be increased exponentially
    // (single node execution mode is not excluded)
    // -1 marks no exp. increasing
    int expBaseExecutors;

    private List<Long> memoryEstimatesCP;
    private List<Long> memoryEstimatesOutput;
    public InterestBasedEnumerator(
            Builder builder,
            boolean fitDriverMemory,
            boolean fitBroadcastMemory,
            boolean checkSingleNodeExecution,
            boolean fitCheckpointMemory,
            int expBaseExecutors
    ) {
        super(builder);
        this.fitDriverMemory = fitDriverMemory;
        this.fitBroadcastMemory = fitBroadcastMemory;
        this.checkSingleNodeExecution = checkSingleNodeExecution;
        this.fitCheckpointMemory = fitCheckpointMemory;
        this.expBaseExecutors = expBaseExecutors;
    }

    @Override
    public void preprocessing() {
        List<Long> availableNodesMemory = null;
        InstanceSearchSpace searchSpace = new InstanceSearchSpace();
        searchSpace.initSpace(instances);

        if (fitDriverMemory || checkSingleNodeExecution) {
            memoryEstimatesCP = getMemoryEstimates(program, false);
        }
        if (fitBroadcastMemory || fitCheckpointMemory) {
            memoryEstimatesOutput = getMemoryEstimates(program, true);
        }

        if (fitDriverMemory) {
            availableNodesMemory = new ArrayList<>(searchSpace.keySet());
            List<Long> driverMemoryPoints = getMemoryPoints(memoryEstimatesCP, availableNodesMemory);
            for (long dMemory: driverMemoryPoints) {
                driverSpace.put(dMemory, searchSpace.get(dMemory));
            }
            // in case no big enough memory estimates exist set the instances with minimal memory
            if (driverSpace.isEmpty()) {
                long minMemory = availableNodesMemory.get(0);
                driverSpace.put(minMemory, searchSpace.get(minMemory));
            }
        } else {
            driverSpace.putAll(searchSpace);
        }

        if (fitBroadcastMemory) {
            if (availableNodesMemory == null)
                availableNodesMemory = new ArrayList<>(searchSpace.keySet());
            List<Long> memoryEstimatesBroadcast = memoryEstimatesOutput.stream()
                    .map(mem -> Math.round(mem * BROADCAST_MEMORY_FACTOR))
                    .collect(Collectors.toList());
            List<Long> executorMemoryPoints = getMemoryPoints(memoryEstimatesBroadcast, availableNodesMemory);
            for (long eMemory: executorMemoryPoints) {
                executorSpace.put(eMemory, searchSpace.get(eMemory));
            }
            // in case no big enough memory estimates exist set the instances with minimal memory
            if (executorSpace.isEmpty()) {
                long minMemory = availableNodesMemory.get(0);
                executorSpace.put(minMemory, searchSpace.get(minMemory));
            }
        } else {
            executorSpace.putAll(searchSpace);
        }
    }

    @Override
    public List<Integer> estimateRangeExecutors(long driverMemory, long executorMemory, int executorCores) {
        // consider the maximum level of parallelism and
        // based on the initiated flags decides on the following methods
        // for enumeration of the number of executors:
        // 1. Such a number that leads to combined distributed memory
        //    close to the output size of the HOPs
        // 2. Exponentially increasing number of executors based on
        //    a given exponent base - with additional option for 0 executors
        // 3. Enumerating all options with the established range
        // Checking if single node execution should be excluded is optional.
        int min = minExecutors;
        int max = Math.min(maxExecutors, (MAX_LEVEL_PARALLELISM / executorCores));
        
        if (checkSingleNodeExecution && min == 0 && !memoryEstimatesCP.isEmpty()) {
            long maxEstimate = memoryEstimatesCP.get(memoryEstimatesCP.size()-1);
            if (maxEstimate > driverMemory) {
                min = 1;
            }
        }
        ArrayList<Integer> result = new ArrayList<>();
        if (fitCheckpointMemory) {
            double ratio = (double) driverMemory / memoryEstimatesOutput.get(0);
            int lastNumExecutors = (int) Math.floor(1/ratio);
            result.add(Math.max(min, lastNumExecutors));
            for (long estimate: memoryEstimatesOutput) {
                ratio = (double) driverMemory / estimate;
                int numExecutors = (int) Math.ceil(1/ratio);
                if (numExecutors <= max) {
                    if (lastNumExecutors < numExecutors) {
                        result.add(numExecutors);
                        lastNumExecutors = numExecutors;
                    }
                } else {
                    break;
                }
            }
        } else if (expBaseExecutors > 1) {
            if (min == 0) {
                result.add(0);
            }
            int exponent = 0;
            int numExecutors;
            while ((numExecutors = (int) Math.pow(expBaseExecutors, exponent)) <= max) {
                result.add(numExecutors);
            }
        } else { // enumerate all options within the min-max range
            for (int n = min; n <= max; n++) {
                result.add(n);
            }
        }
        return result;
    }

    // Static helper methods -------------------------------------------------------------------------------------------

    private static List<Long> getMemoryPoints(List<Long> estimates, List<Long> availableMemory) {
        ArrayList<Long> result = new ArrayList<>();

        List<Long> relevantPoints = new ArrayList<>(availableMemory);
        Collections.sort(relevantPoints);
        long lastAdded = -1;
        for (long estimate: estimates) {
            if (availableMemory.isEmpty()) {
                break;
            }
            long memoryDelta = Math.round(estimate*MEMORY_DELTA_FRACTION);
            // divide list on bigger and smaller by partitioning - partitioning preserve the order
            Map<Boolean, List<Long>> divided = relevantPoints.stream()
                    .collect(Collectors.partitioningBy(n -> n < estimate));
            List<Long> smallerPoints = divided.get(true);
            // get points smaller than the current memory estimate
            long biggestOfTheSmaller = smallerPoints.get(smallerPoints.size() - 1);
            for (long point : smallerPoints) {
                if (point >= (biggestOfTheSmaller - memoryDelta) && point > lastAdded) {
                    result.add(point);
                    lastAdded = point;
                }
            }
            // reduce the list of relevant points - equal or bigger than the estimate
            relevantPoints = divided.get(false);
            // get points bigger than the current memory estimate
            long smallestOfTheBigger = relevantPoints.get(0);
            for (long point : relevantPoints) {
                if (point <= (smallestOfTheBigger + memoryDelta) && point > lastAdded) {
                    result.add(point);
                    lastAdded = point;
                } else {
                    break;
                }
            }
        }
        return result;
    }

    private static List<Long> getMemoryEstimates(Program currentProgram, boolean outputOnly) {
        HashSet<Long> estimates = new HashSet<>();
        getMemoryEstimates(currentProgram.getProgramBlocks(), estimates, outputOnly);
        double currentFactor = outputOnly? DATA_MEMORY_FACTOR : OptimizerUtils.MEM_UTIL_FACTOR;
        return estimates.stream()
                .filter(mem -> mem > MINIMUM_RELEVANT_MEM_ESTIMATE)
                .map(mem -> (long) (mem / currentFactor))
                .sorted()
                .collect(Collectors.toList());
    }

    private static void getMemoryEstimates(ArrayList<ProgramBlock> pbs, HashSet<Long> mem, boolean outputOnly) {
        for( ProgramBlock pb : pbs ) {
            getMemoryEstimates(pb, mem, outputOnly);
        }
    }

    private static void getMemoryEstimates(ProgramBlock pb, HashSet<Long> mem, boolean outputOnly) {
        if (pb instanceof FunctionProgramBlock)
        {
            FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
            getMemoryEstimates(fpb.getChildBlocks(), mem, outputOnly);
        }
        else if (pb instanceof WhileProgramBlock)
        {
            WhileProgramBlock fpb = (WhileProgramBlock)pb;
            getMemoryEstimates(fpb.getChildBlocks(), mem, outputOnly);
        }
        else if (pb instanceof IfProgramBlock)
        {
            IfProgramBlock fpb = (IfProgramBlock)pb;
            getMemoryEstimates(fpb.getChildBlocksIfBody(), mem, outputOnly);
            getMemoryEstimates(fpb.getChildBlocksElseBody(), mem, outputOnly);
        }
        else if (pb instanceof ForProgramBlock) // including parfor
        {
            ForProgramBlock fpb = (ForProgramBlock)pb;
            getMemoryEstimates(fpb.getChildBlocks(), mem, outputOnly);
        }
        else
        {
            StatementBlock sb = pb.getStatementBlock();
            if( sb != null && sb.getHops() != null ){
                Hop.resetVisitStatus(sb.getHops());
                for( Hop hop : sb.getHops() )
                    getMemoryEstimates(hop, mem, outputOnly);
            }
        }
    }

    private static void getMemoryEstimates(Hop hop, HashSet<Long> mem, boolean outputOnly)
    {
        if( hop.isVisited() )
            return;
        //process children
        for(Hop hi : hop.getInput())
            getMemoryEstimates(hi, mem, outputOnly);

        if (outputOnly) {
            long estimate = (long) hop.getOutputMemEstimate(0);
            if (estimate > 0)
                mem.add(estimate);
        } else {
            mem.add((long) hop.getMemEstimate());
        }
        hop.setVisited();
    }
}
