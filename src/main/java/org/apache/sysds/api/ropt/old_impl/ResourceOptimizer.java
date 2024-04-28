package org.apache.sysds.api.ropt.old_impl;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.SparkConf;
import org.apache.sysds.api.DMLOptions;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.CompilerConfig;
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

public class ResourceOptimizer
{
    private static final Log LOG = LogFactory.getLog(ResourceOptimizer.class);

    //internal configuration parameters
    // TODO: think about how to replace it
    public static final boolean INCLUDE_PREDICATES = true;
    public static final boolean COSTS_MAX_PARALLELISM = true;
    public static final boolean COST_INDIVIDUAL_BLOCKS = true;

    private static final int MAXIMUM_SPARK_INSTANCES = 2;

    private static final String CONFIG_PATH = System.getProperty("user.dir") + "/scripts/ropt/";
    // NOTE: predefined args for testing only
    private static final String[] UNIVAR_STATS_DML_SCRIPT= {"-f", "/Users/lachezarnikolov/my_projects/thesis/systemds/scripts/aws/systemds/Univar-Stats.dml", "-config", "/Users/lachezarnikolov/my_projects/thesis/systemds/scripts/ropt/SystemDS-config.xml", "-nvargs", "X=data/haberman.data", "TYPES=data/types.csv", "STATS=data/univarOut.mtx", "CONSOLE_OUTPUT=TRUE"};
    private static final String[] HELLO_DML_SCRIPT= {"-f", "/Users/lachezarnikolov/my_projects/thesis/systemds/scripts/aws/systemds/hello.dml", "-config", "/Users/lachezarnikolov/my_projects/thesis/systemds/scripts/ropt/SystemDS-config.xml"};
    private static final int executorCores = 2; // NOTE: us for more advanced executor mapping

    private static void setArtificialConfigs(CloudClusterConfig cc) {
        CloudInstanceConfig cpConfig = cc.getCpInstance();
        CloudInstanceConfig spGroupConfig = cc.getSpGroupInstance();
        // TODO: consider is all the resources can be given or the OS resources need to be first subtracted
        // set the artificial local infrastructure configuration
        InfrastructureAnalyzer.setLocalMaxMemory(CloudOptimizerUtils.toB(cpConfig.getAvailableMemoryMB())); // the CP mem budget = Core instance mem budget [bytes]
        InfrastructureAnalyzer.setLocalPar(cpConfig.getVCPUCores()); // the CP available log. cores = Core instance log. cores
        // NOTE: Here is used simple executor mapping: 1 executor for each task instance logical core; TODO: implement more advances functionality
        // create the artificial spark configuration
        SparkConf sparkConf = SparkExecutionContext.createSystemDSSparkConf();
        sparkConf.set("spark.master", "local[*]");
        sparkConf.set("spark.app.name", "SystemDS");
        sparkConf.set("spark.driver.maxResultSize", "1g"); // TODO: consider its meaning
        sparkConf.set("spark.memory.useLegacyMode", "false");
        if (spGroupConfig != null) {
            sparkConf.set("spark.executor.memory", spGroupConfig.getMaxMemoryPerCore()+"m");
            sparkConf.set("spark.storage.memoryFraction", "0.6");
            sparkConf.set("spark.executor.instances", Integer.toString(spGroupConfig.getVCPUCores()*cc.getSpGroupSize()));
            sparkConf.set("spark.executor.cores", Integer.toString(spGroupConfig.getVCPUCores()));
            sparkConf.set("spark.default.parallelism", Integer.toString(cc.getSpGroupSize())); // NOTE: What rule should be applied here?
        } else {
            // TODO: Think of reasonable minimum or a way for configuring executing only on driver container in Spark
            sparkConf.set("spark.executor.memory", "512m");
            sparkConf.set("spark.storage.memoryFraction", "0.6");
            sparkConf.set("spark.executor.instances", "0");
            sparkConf.set("spark.executor.cores", "1");
            sparkConf.set("spark.default.parallelism", "0"); // NOTE: What rule should be applied here?
        }

        // set the artificial spark configuration
        SparkExecutionContext.initVirtualSparkContext(sparkConf);

//        JavaSparkContext context = SparkExecutionContext.getSparkContextStatic();
//        SparkConf sparkConf = context.getConf();
//        for (scala.Tuple2<String, String> tuple : sparkConf.getAll()) {
//            System.out.println(tuple._1 + ":" + tuple._2);
//        }
//        System.out.println("Remote parallelism " + InfrastructureAnalyzer.getCkMaxMR());
//        System.out.println("Remote memory budget: " + SparkExecutionContext.getBroadcastMemoryBudget());
    }

    // check the if the required configs are set all correct
    private static void validateCompilerConfigs(CompilerConfig config) {
        if (config.getInt(CompilerConfig.ConfigType.OPT_LEVEL) !=
                OptimizerUtils.OptimizationLevel.O4_GLOBAL_TIME_MEMORY.ordinal())
            throw new RuntimeException("Opt level not set properly");
        if (config.getInt(CompilerConfig.ConfigType.BLOCK_SIZE) != 1000)
            throw new RuntimeException("Block size not set properly");
        if (!config.getBool(CompilerConfig.ConfigType.REJECT_READ_WRITE_UNKNOWNS))
            throw new RuntimeException("Ignoring metadata not allowed");
    }

    private static double getCost(Program program) {
        ExecutionContext execContext = ExecutionContextFactory.createContext();
        return CostEstimationWrapper.getTimeEstimate(program, execContext);
    }

    private static double getMonetaryCost(Program program, CloudClusterConfig cc) {
        double timeCost = getCost(program);
        return cc.getFinalPrice(timeCost);
    }

    private static void recompile( ArrayList<ProgramBlock> pbs, CloudClusterConfig cc) {
        //init compiler memory budget
        setArtificialConfigs(cc);
        OptimizerUtils.resetDefaultSize();

        for (ProgramBlock pb : pbs) {
            recompile(pb);
        }
    }

    private static void recompile( ProgramBlock pb)
    {
        //recompile instructions (incl predicates)
        if (pb instanceof WhileProgramBlock) {
            WhileProgramBlock wpb = (WhileProgramBlock)pb;
            WhileStatementBlock sb = (WhileStatementBlock) pb.getStatementBlock();
            if( INCLUDE_PREDICATES && sb!=null && sb.getPredicateHops()!=null ){
                ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
                        sb.getPredicateHops(), new LocalVariableMap(), null, false, false, 0);
                //inst = annotateMRJobInstructions(inst, cp, mr);
                wpb.setPredicate( inst );
            }
        }
        else if (pb instanceof IfProgramBlock) {
            IfProgramBlock ipb = (IfProgramBlock)pb;
            IfStatementBlock sb = (IfStatementBlock) ipb.getStatementBlock();
            if( INCLUDE_PREDICATES && sb!=null && sb.getPredicateHops()!=null ){
                ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
                        sb.getPredicateHops(), new LocalVariableMap(), null, false, false, 0);
                //inst = annotateMRJobInstructions(inst, cp, mr);
                ipb.setPredicate( inst );
            }
        }
        else if (pb instanceof ForProgramBlock) { //incl parfor
            ForProgramBlock fpb = (ForProgramBlock)pb;
            ForStatementBlock sb = (ForStatementBlock) fpb.getStatementBlock();
            if( INCLUDE_PREDICATES && sb!=null ){
                if( sb.getFromHops()!=null ){
                    ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
                            sb.getFromHops(), new LocalVariableMap(), null, false, false, 0);
                    //inst = annotateMRJobInstructions(inst, cp, mr);
                    fpb.setFromInstructions( inst );
                }
                if( sb.getToHops()!=null ){
                    ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
                            sb.getToHops(), new LocalVariableMap(), null, false, false, 0);
                    //inst = annotateMRJobInstructions(inst, cp, mr);
                    fpb.setToInstructions( inst );
                }
                if( sb.getIncrementHops()!=null ){
                    ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
                            sb.getIncrementHops(), new LocalVariableMap(), null, false, false, 0);
                    //inst = annotateMRJobInstructions(inst, cp, mr);
                    fpb.setIncrementInstructions( inst );
                }
            }
        }
        else { //last-level program blocks
            StatementBlock sb = pb.getStatementBlock();
            // NOTE: inplace=true
            ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
                    sb, sb.getHops(), new LocalVariableMap(), null, true, false, 0);
            //inst = annotateMRJobInstructions(inst, cp, mr);
            pb.setInstructions( inst );
        }
    }

    public static void main(String[] args) throws IOException {
        // extract the info for all available instance
        CloudInstanceAnalyzer analyzer = new CloudInstanceAnalyzer(2.0);
        ArrayList<CloudInstanceConfig> availableInstance = analyzer.getListInstancesSorted();
        // build init cluster config
        CloudClusterConfig optCluster = new CloudClusterConfig(availableInstance.get(0), availableInstance.get(0));


        String dmlScriptPath;
        String dmlScriptStr;
        Map<String, String> argVals;

        DMLOptions dmlOptions = null;
        try{
            dmlOptions = DMLOptions.parseCLArguments(UNIVAR_STATS_DML_SCRIPT);
        }
        catch(org.apache.commons.cli.ParseException e) {
            throw new RuntimeException("Parsing arguments failed");
        }

        dmlScriptPath = dmlOptions.filePath;
        if (dmlScriptPath == null)
            throw new RuntimeException("No file path given");

        argVals = dmlOptions.argVals;

        // parse the script into string
        dmlScriptStr = DMLScript.readDMLScript(true, dmlScriptPath);

        //Step 1: parse configuration files & write any configuration specific global variables
        DMLScript.loadConfiguration(dmlOptions.configFile);
        validateCompilerConfigs(ConfigurationManager.getCompilerConfig());

        //Step X: configure codegen: probably not needed
        //DMLScript.configureCodeGen();

        // TODO: find the most appropriate place for applying the cluster settings
        // Step 2: set initial local/remote configurations if requested
        setArtificialConfigs(optCluster);

        //Step 3: parse dml script
        ParserWrapper parserWrapper = ParserFactory.createParser();
        DMLProgram dmlProgram = parserWrapper.parse(dmlScriptPath, dmlScriptStr, argVals);

        //Step 4: construct HOP DAGs (incl LVA, validate, and setup)
        DMLTranslator dmlt = new DMLTranslator(dmlProgram);
        dmlt.liveVariableAnalysis(dmlProgram);
        dmlt.validateParseTree(dmlProgram);
        dmlt.constructHops(dmlProgram);

        //Step 5: rewrite HOP DAGs (incl IPA and memory estimates)
        dmlt.rewriteHopsDAG(dmlProgram);

        //Step 6: construct lops (incl exec type and op selection)
        dmlt.constructLops(dmlProgram);

        //Step 7: rewrite LOP DAGs (incl adding new LOPs s.a. prefetch, broadcast)
        dmlt.rewriteLopDAG(dmlProgram);

        //Step 8: generate runtime program, incl codegen
        Program runtimeProgram = dmlt.getRuntimeProgram(dmlProgram, ConfigurationManager.getDMLConfig());

        // get initial cost
        double optCost = 2.0;// NOTE: bug - getMonetaryCost(runtimeProgram, optCluster);
        // recompilation loop
        CloudClusterConfig currentCluster = new CloudClusterConfig(availableInstance.get(0), null);
        double newCost = Double.MAX_VALUE;
        // FIXME: Currently the old runtime program is used for calculating the new cost
        // TODO: Clear out if the recompilation is doing the job needed of internally replacing the runtime program
        for (CloudInstanceConfig cpInstance: availableInstance) {
            // outer loop: iterate once over each instance type
            currentCluster.setCpInstance(cpInstance);
            newCost = 3.0;// NOTE: getMonetaryCost(runtimeProgram, currentCluster);
            if (optCost > newCost) {
                optCost = newCost;
                optCluster = currentCluster;
            }
            for (CloudInstanceConfig spInstance: availableInstance) {
                // middle loop: iterate over each instance type separately from the outer loop
                currentCluster.setSpGroupInstance(spInstance);
                recompile(runtimeProgram.getProgramBlocks(), currentCluster);
                newCost = 3.0;// NOTE: getMonetaryCost(runtimeProgram, currentCluster);
                if (optCost > newCost) {
                    optCost = newCost;
                    optCluster = currentCluster;
                }
                for (int i = 2; i <= MAXIMUM_SPARK_INSTANCES; i++) {
                    // inner loop: iterate over possible number of spark instances
                    currentCluster.setSpGroupSize(i);
                    recompile(runtimeProgram.getProgramBlocks(), currentCluster);
                    newCost = 3.0;// NOTE: getMonetaryCost(runtimeProgram, currentCluster);
                    if (optCost > newCost) {
                        optCost = newCost;
                        optCluster = currentCluster;
                    }
                    // NOTE: Maybe break the loop if the cost doesn't improve
                }
            }
        }

        // FIXME: current recompilation breaks the explanation and does not contain the optimal plan
        //Step 9: prepare statistics [and optional explain output]
        //count number compiled MR jobs / SP instructions
        // Explain.ExplainCounts counts = Explain.countDistributedOperations(runtimeProgram);
        //explain plan of program (hops or runtime)
        //System.out.println(Explain.display(dmlProgram, runtimeProgram, Explain.ExplainType.NONE, counts));

        //Step 10: generate the config files for the optimal cluster
        (new EMRConfig(optCluster)).generateConfigsFiles(CONFIG_PATH);
    }
}

