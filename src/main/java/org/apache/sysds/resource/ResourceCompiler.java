package org.apache.sysds.resource;

import org.apache.spark.SparkConf;
import org.apache.sysds.api.DMLOptions;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.lops.rewrite.LopRewriter;
import org.apache.sysds.parser.*;
import org.apache.sysds.runtime.controlprogram.*;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.instructions.Instruction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.sysds.api.DMLScript.*;

/**
 * This class does full or partial program recompilation
 * based on given runtime program. It uses the methods provided
 * by {@code hops.recompile.Recompiler}).
 * It keeps a state of the current recompilation phase in order
 * to decide when to do full recompilation and when not.
 */
public class ResourceCompiler {
    public static final long DEFAULT_DRIVER_MEMORY = 512*1024*1024; // 0.5GB
    public static final int DEFAULT_DRIVER_THREADS = 1; // 0.5GB
    public static final long DEFAULT_EXECUTOR_MEMORY = 512*1024*1024; // 0.5GB
    public static final int DEFAULT_EXECUTOR_THREADS = 1; // 0.5GB
    public static final int DEFAULT_NUMBER_EXECUTORS = 1; // 0.5GB
    static {
        // TODO: consider moving to the executable of the resource optimizer once implemented
        USE_LOCAL_SPARK_CONFIG = true;
        ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.ALLOW_DYN_RECOMPILATION, false);
        ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.RESOURCE_OPTIMIZATION, true);
    }
    private static final LopRewriter _lopRewriter = new LopRewriter();

    public static Program compile(String filePath, Map<String, String> args) throws IOException {
        // setting the dynamic recompilation flags during resource optimization is obsolete
        DMLOptions dmlOptions =DMLOptions.defaultOptions;
        dmlOptions.argVals = args;

        String dmlScriptStr = readDMLScript(true, filePath);
        Map<String, String> argVals = dmlOptions.argVals;

        Dag.resetUniqueMembers();
        // NOTE: skip configuring code generation
        // NOTE: expects setting up the initial cluster configs before calling
        ParserWrapper parser = ParserFactory.createParser();
        DMLProgram dmlProgram = parser.parse(null, dmlScriptStr, argVals);
        DMLTranslator dmlTranslator = new DMLTranslator(dmlProgram);
        dmlTranslator.liveVariableAnalysis(dmlProgram);
        dmlTranslator.validateParseTree(dmlProgram);
        dmlTranslator.constructHops(dmlProgram);
        dmlTranslator.rewriteHopsDAG(dmlProgram);
        dmlTranslator.constructLops(dmlProgram);
        dmlTranslator.rewriteLopDAG(dmlProgram);
        return dmlTranslator.getRuntimeProgram(dmlProgram, ConfigurationManager.getDMLConfig());
    }

    private static ArrayList<Instruction> recompile(StatementBlock sb, ArrayList<Hop> hops) {
        // construct new lops
        ArrayList<Lop> lops = new ArrayList<>(hops.size());
        Hop.resetVisitStatus(hops);
        for( Hop hop : hops ){
            Recompiler.rClearLops(hop);
            lops.add(hop.constructLops());
        }
        // apply hop-lop rewrites to cover the case of changed lop operators
        _lopRewriter.rewriteLopDAG(sb, lops);

        Dag<Lop> dag = new Dag<>();
        for (Lop l : lops) {
            l.addToDag(dag);
        }

        return dag.getJobs(sb, ConfigurationManager.getDMLConfig());
    }

    /**
     * Recompiling a given program for resource optimization for single node execution
     * @param program program to be recompiled
     * @param driverMemory target driver memory
     * @param driverCores target driver threads/cores
     * @return the recompiled program as a new {@code Program} instance
     */
    public static Program doFullRecompilation(Program program, long driverMemory, int driverCores) {
        setDriverConfigurations(driverMemory, driverCores);
        setSingleNodeExecution();
        return doFullRecompilation(program);
    }

    /**
     * Recompiling a given program for resource optimization for Spark execution
     * @param program program to be recompiled
     * @param driverMemory target driver memory
     * @param driverCores target driver threads/cores
     * @param numberExecutors target number of executor nodes
     * @param executorMemory target executor memory
     * @param executorCores target executor threads/cores
     * @return the recompiled program as a new {@code Program} instance
     */
    public static Program doFullRecompilation(Program program, long driverMemory, int driverCores, int numberExecutors, long executorMemory, int executorCores) {
        setDriverConfigurations(driverMemory, driverCores);
        setExecutorConfigurations(numberExecutors, executorMemory, executorCores);
        return doFullRecompilation(program);
    }

    private static Program doFullRecompilation(Program program) {
        Dag.resetUniqueMembers();
        Program newProgram = new Program();
        ArrayList<ProgramBlock> B = Stream.concat(
                        program.getProgramBlocks().stream(),
                        program.getFunctionProgramBlocks().values().stream())
                .collect(Collectors.toCollection(ArrayList::new));
        doRecompilation(B, newProgram);
        return newProgram;
    }

    private static void doRecompilation(ArrayList<ProgramBlock> origin, Program target) {
        for (ProgramBlock originBlock : origin) {
            doRecompilation(originBlock, target);
        }
    }

    private static void doRecompilation(ProgramBlock originBlock, Program target) {
        if (originBlock instanceof FunctionProgramBlock)
        {
            FunctionProgramBlock fpb = (FunctionProgramBlock)originBlock;
            doRecompilation(fpb.getChildBlocks(), target);
        }
        else if (originBlock instanceof WhileProgramBlock)
        {
            WhileProgramBlock wpb = (WhileProgramBlock)originBlock;
            WhileStatementBlock sb = (WhileStatementBlock) originBlock.getStatementBlock();
            if(sb!=null && sb.getPredicateHops()!=null ){
                ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb.getPredicateHops(), null, null, true, true, 0);
                wpb.setPredicate(inst);
                target.addProgramBlock(wpb);
            }
            doRecompilation(wpb.getChildBlocks(), target);
        }
        else if (originBlock instanceof IfProgramBlock)
        {
            IfProgramBlock ipb = (IfProgramBlock)originBlock;
            IfStatementBlock sb = (IfStatementBlock) ipb.getStatementBlock();
            if(sb!=null && sb.getPredicateHops()!=null ){
                ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb.getPredicateHops(), null, null, true, true, 0);
                ipb.setPredicate(inst);
                target.addProgramBlock(ipb);
            }
            doRecompilation(ipb.getChildBlocksIfBody(), target);
            doRecompilation(ipb.getChildBlocksElseBody(), target);
        }
        else if (originBlock instanceof ForProgramBlock) //incl parfor
        {
            ForProgramBlock fpb = (ForProgramBlock)originBlock;
            ForStatementBlock sb = (ForStatementBlock) fpb.getStatementBlock();
            if(sb!=null){
                if( sb.getFromHops()!=null ){
                    ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb.getFromHops(), null, null, true, true, 0);
                    fpb.setFromInstructions( inst );
                }
                if(sb.getToHops()!=null){
                    ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb.getToHops(), null, null, true, true, 0);
                    fpb.setToInstructions( inst );
                }
                if(sb.getIncrementHops()!=null){
                    ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb.getIncrementHops(), null, null, true, true, 0);
                    fpb.setIncrementInstructions(inst);
                }
                target.addProgramBlock(fpb);

            }
            doRecompilation(fpb.getChildBlocks(), target);
        }
        else
        {
            BasicProgramBlock bpb = (BasicProgramBlock)originBlock;
            StatementBlock sb = bpb.getStatementBlock();
            ArrayList<Instruction> inst = recompile(sb, sb.getHops());
            bpb.setInstructions(inst);
            target.addProgramBlock(bpb);
        }
    }

    public static void setDriverConfigurations(long nodeMemory, int nodeNumCores) {
        // TODO: think of reasonable factor for the JVM heap as prt of the node's memory
        InfrastructureAnalyzer.setLocalMaxMemory(nodeMemory);
        InfrastructureAnalyzer.setLocalPar(nodeNumCores);
    }

    public static void setExecutorConfigurations(int numExecutors, long nodeMemory, int nodeNumCores) {
        // TODO: think of reasonable factor for the JVM heap as prt of the node's memory
        if (numExecutors > 0) {
            DMLScript.setGlobalExecMode(Types.ExecMode.HYBRID);
            SparkConf sparkConf = SparkExecutionContext.createSystemDSSparkConf();
            // ------------------ Static Configurations -------------------
            // TODO: think how to avoid setting them every time
            sparkConf.set("spark.master", "local[*]");
            sparkConf.set("spark.app.name", "SystemDS");
            sparkConf.set("spark.memory.useLegacyMode", "false");
            // ------------------ Static Configurations -------------------
            // ------------------ Dynamic Configurations -------------------
            sparkConf.set("spark.executor.memory", (nodeMemory/(1024*1024))+"m");
            sparkConf.set("spark.executor.instances", Integer.toString(numExecutors));
            sparkConf.set("spark.executor.cores", Integer.toString(nodeNumCores));
            // ------------------ Dynamic Configurations -------------------
            SparkExecutionContext.initLocalSparkContext(sparkConf);
        } else {
            throw new RuntimeException("The given number of executors was 0");
        }
    }

    public static void setSingleNodeExecution() {
        DMLScript.setGlobalExecMode(Types.ExecMode.SINGLE_NODE);
    }
}
