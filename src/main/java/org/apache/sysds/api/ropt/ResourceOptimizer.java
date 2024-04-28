package org.apache.sysds.api.ropt;

import org.apache.spark.SparkConf;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.api.ropt.SearchSpace.SearchPoint;
import org.apache.sysds.api.ropt.CloudOptimizerUtils;
import org.apache.sysds.api.ropt.strategies.GridSearch;
import org.apache.sysds.api.ropt.strategies.SearchStrategy;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.cost.CostEstimationWrapper;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.parser.*;
import org.apache.sysds.runtime.controlprogram.*;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.utils.Explain;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import static org.apache.sysds.hops.recompile.Recompiler.recompileProgramBlockInstructions;

public class ResourceOptimizer {
    private static final boolean DEBUGGING_ON = true;
    // TODO: think about how to replace it
    public static final boolean INCLUDE_PREDICATES = true;
    protected static class OptimalSolution {
        private SearchPoint _searchPoint = null;
        private double _cost = Double.MAX_VALUE;

        protected boolean update(SearchPoint searchPoint, double cost) {
            if (cost < _cost) {
                _searchPoint = searchPoint;
                _cost = cost;
                return true;
            }
            return false;
        }

        protected SearchPoint getSearchPoint() {
            return _searchPoint;
        }

        protected double getCost() {
            return _cost;
        }

        public String getDriverInstanceName() {
            return SearchSpace.getInstanceName(
                    _searchPoint.getInstanceTypeDriver(),
                    _searchPoint.getInstanceSizeDriver()
            );
        }

        public String getExecutorInstanceName() {
            return SearchSpace.getInstanceName(
                    _searchPoint.getInstanceTypeExecutor(),
                    _searchPoint.getInstanceSizeExecutor()
            );
        }

        public int getExecutorInstanceNumber() {
            return _searchPoint.getNumberExecutors();
        }
    }
    private Program _runtimeProgram;
    private SearchSpace _searchSpace;
    private SearchStrategy _searchStrategy;
    private HashMap<String, CloudInstance> _availableInstances;
    private OptimalSolution _optimalSolution;

    public ResourceOptimizer(Program precompledProgram, String AWSMetaFilePath) throws IOException {
        // TODO: init dml config
        _runtimeProgram = precompledProgram;
        _availableInstances = new HashMap<>();
        _optimalSolution = new OptimalSolution();
        // get the AWS meta data
        loadInstanceInfoTable(AWSMetaFilePath);
        _searchSpace = new SearchSpace(_availableInstances);
        _searchStrategy = new GridSearch(_searchSpace);
    }

    private void loadInstanceInfoTable(String instanceTablePath) throws IOException {
        int lineCount = 1;
        // try to open the file
        BufferedReader br = new BufferedReader(new FileReader(instanceTablePath));
        String parsedLine;
        // validate the file header
        parsedLine = br.readLine();
        if (!parsedLine.equals("API_Name,Memory,vCPUs,Family,Price"))
            throw new IOException("Invalid CSV header inside: " + instanceTablePath);


        while ((parsedLine = br.readLine()) != null) {
            String[] values = parsedLine.split(",");
            if (values.length != 5)
                throw new IOException(String.format("Invalid CSV line(%d) inside: %s", lineCount, instanceTablePath));
            CloudInstance parsedInstance = new CloudInstance(
                    values[0],
                    (long)Double.parseDouble(values[1])*1024,
                    Integer.parseInt(values[2]),
                    Double.parseDouble(values[4])
            );
            _availableInstances.put(values[0], parsedInstance);
            lineCount++;
        }
    }

    protected OptimalSolution execute() {
        while (_searchStrategy.hasNext()) {
            // Step 1
            SearchPoint searchPoint = _searchStrategy.enumerateNext();
            // Step 2
            Program recompiledProgram = recompile(searchPoint);

            System.out.println(Explain.explain(recompiledProgram.getDMLProg()));
            System.out.println();
            System.out.println(Explain.explain(recompiledProgram));
            // Step 3
            double cost = estimateCostMonetary(searchPoint, recompiledProgram);
            // Step 4
            _optimalSolution.update(searchPoint, cost);
        }
        System.out.println("Optimal solution: " + _optimalSolution.getSearchPoint().toString());
        return _optimalSolution;
    }

    private void initSearchStrategy(SearchStrategy strategy) throws IOException {
        // load the instances' info first
        loadInstanceInfoTable("");
        // init the search space representation (object)
        // TODO: decide if keeping reference to the search space in here is a good idea
        SearchSpace searchSpace = new SearchSpace(_availableInstances);
        _searchStrategy = new GridSearch(searchSpace);
    }

    public Program recompile(SearchPoint searchPoint) {
        setClusterConfigs(searchPoint);
        // TODO: Ensure that for many recompilation the name incrementing would not lead to an issue
        ArrayList<ProgramBlock> B = new ArrayList<>();
        compileArrayBlocks(_runtimeProgram.getProgramBlocks(), B);

        return B.get(0).getProgram();
    }

    private static void compileArrayBlocks(ArrayList<ProgramBlock> arrayPb, ArrayList<ProgramBlock> B) {
        for (ProgramBlock pb: arrayPb) {
            compileSingleBlock(pb, B);
        }
    }

    private static void compileSingleBlock(ProgramBlock pb, ArrayList<ProgramBlock> B)
    {
        if (pb instanceof FunctionProgramBlock)
        {
            FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
            compileArrayBlocks(fpb.getChildBlocks(), B);
        }
        else if (pb instanceof WhileProgramBlock)
        {
            WhileProgramBlock wpb = (WhileProgramBlock)pb;
            WhileStatementBlock sb = (WhileStatementBlock) pb.getStatementBlock();
            if(sb!=null && sb.getPredicateHops()!=null ){
                ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb.getPredicateHops(), (LocalVariableMap) null, null, true, true, 0);
                wpb.setPredicate(inst);
                B.add(wpb);
            }
            compileArrayBlocks(wpb.getChildBlocks(), B);
        }
        else if (pb instanceof IfProgramBlock)
        {
            IfProgramBlock ipb = (IfProgramBlock)pb;
            IfStatementBlock sb = (IfStatementBlock) ipb.getStatementBlock();
            if(sb!=null && sb.getPredicateHops()!=null ){
                ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb.getPredicateHops(), (LocalVariableMap) null, null, true, true, 0);
                ipb.setPredicate(inst);
                B.add(ipb);
            }
            compileArrayBlocks(ipb.getChildBlocksIfBody(), B);
            compileArrayBlocks(ipb.getChildBlocksElseBody(), B);
        }
        else if (pb instanceof ForProgramBlock) //incl parfor
        {
            ForProgramBlock fpb = (ForProgramBlock)pb;
            ForStatementBlock sb = (ForStatementBlock) fpb.getStatementBlock();
            if(sb!=null){
                if( sb.getFromHops()!=null ){
                    ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb.getFromHops(), (LocalVariableMap) null, null, true, true, 0);
                    fpb.setFromInstructions( inst );
                }
                if(sb.getToHops()!=null){
                    ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb.getToHops(), (LocalVariableMap) null, null, true, true, 0);
                    fpb.setToInstructions( inst );
                }
                if(sb.getIncrementHops()!=null){
                    ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb.getIncrementHops(), (LocalVariableMap) null, null, true, true, 0);
                    fpb.setIncrementInstructions(inst);
                }
                B.add(fpb);

            }
            compileArrayBlocks(fpb.getChildBlocks(), B);
        }
        else
        {
            BasicProgramBlock bpb = (BasicProgramBlock)pb;
            StatementBlock sb = bpb.getStatementBlock();
            ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb, sb.getHops(), new LocalVariableMap(), null, true, true, 0);
            bpb.setInstructions(inst);
            B.add(pb);
        }
    }

    public void setClusterConfigs(SearchPoint searchPoint) {
        CloudInstance driverInstance = _availableInstances.get(searchPoint.getInstanceNameDriver());
        CloudInstance executorInstance = _availableInstances.get(searchPoint.getInstanceNameExecutor());

        InfrastructureAnalyzer.setLocalMaxMemory(CloudOptimizerUtils.toB(driverInstance.getMemoryMB())); // the CP mem budget = Core instance mem budget [bytes]
        InfrastructureAnalyzer.setLocalPar(driverInstance.getVCPUCores()); // the CP available log. cores = Core instance log. cores
        OptimizerUtils.resetDefaultSize();

        if (executorInstance != null) {
            DMLScript.setGlobalExecMode(Types.ExecMode.HYBRID);

            SparkConf sparkConf = SparkExecutionContext.createSystemDSSparkConf();
            sparkConf.set("spark.master", "spark://localhost");
            sparkConf.set("spark.app.name", "SystemDS");
            sparkConf.set("spark.driver.maxResultSize", "1g"); // TODO: consider its meaning
            sparkConf.set("spark.memory.useLegacyMode", "false");
            sparkConf.set("spark.executor.memory", executorInstance.getMemoryMB()+"m");
            sparkConf.set("spark.storage.memoryFraction", "0.6"); // TODO: consider it better
            sparkConf.set("spark.executor.instances", Integer.toString(searchPoint.getNumberExecutors()));
            sparkConf.set("spark.executor.cores", Integer.toString(executorInstance.getVCPUCores()));
            sparkConf.set("spark.default.parallelism", Integer.toString(searchPoint.getNumberExecutors())); // NOTE: What rule should be applied here?

            SparkExecutionContext.initVirtualSparkContext(sparkConf);
        } else {
            DMLScript.setGlobalExecMode(Types.ExecMode.SINGLE_NODE);
        }



    }


    private double estimateCostTime(SearchPoint searchPoint, Program program) {
        ExecutionContext execContext = ExecutionContextFactory.createContext();
        // TODO: update or rewrite the cost class
        double time = CostEstimationWrapper.getTimeEstimate(program, execContext);
        System.out.println("RUNTIME PROGRAM EVALUATED " + program.toString());
        if (DEBUGGING_ON) {
            System.out.println("Execution time for " + searchPoint.toString() + " is " + time + "s");
        }
        return time;
    }

    private double estimateCostMonetary(SearchPoint searchPoint, Program program) {
        double time = estimateCostTime(searchPoint, program); // seconds
        double costPerSecond = getCostPerHour(searchPoint)/3600;
        if (DEBUGGING_ON) {
            System.out.println("Monetary cost per second: " + costPerSecond);
        }
        return time*costPerSecond;
    }

    private double getCostPerHour(SearchPoint searchPoint) {
        String driverInstanceName = searchPoint.getInstanceNameDriver();
        String executorInstanceName = searchPoint.getInstanceNameExecutor();
        if (executorInstanceName.equals("")) {
            return _availableInstances.get(driverInstanceName).getPricePerHour();
        }
        return _availableInstances.get(driverInstanceName).getPricePerHour() +
                _availableInstances.get(executorInstanceName).getPricePerHour()*searchPoint.getNumberExecutors();
    }
}
