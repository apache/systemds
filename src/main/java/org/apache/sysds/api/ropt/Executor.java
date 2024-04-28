package org.apache.sysds.api.ropt;

import org.apache.commons.cli.*;
import org.apache.commons.cli.ParseException;
import org.apache.sysds.api.DMLOptions;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.parser.*;
import org.apache.sysds.api.ropt.ResourceOptimizer.OptimalSolution;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.utils.Explain;
import org.apache.wink.json4j.JSONException;

import java.io.IOException;
import java.util.Map;

public class Executor {
    public static final String AWS_META_INPUT_PATH = "scripts/ropt/ec2_instances.csv";
    private static final String INSTANCE_CONFIG_OUTPUT_PATH = "scripts/ropt/instances.json";
    private static final String SPARK_CONFIG_OUTPUT_PATH = "scripts/ropt/configs.json";

    static {
        // TODO: think of reasonable init values
        // avoid unnecessary initializing with current system parameters
        InfrastructureAnalyzer.setLocalMaxMemory(2L *1024*1024*1024); // the CP mem budget = Core instance mem budget [bytes]
        InfrastructureAnalyzer.setLocalPar(8);
        // avoid unnecessary initializing of SparkContext at this point
        DMLScript.setGlobalExecMode(Types.ExecMode.SINGLE_NODE);
    }

    /**
     * Validates the required CLI arguments needed for resource optimization:
     *  <li>-f: path to the dml script</li>
     *  <li>-config: path to the SystemDS configurations</li>
     *  <li>-nvargs: arguments for the dml script</li>
     *
     * @param args: arguments for parsing
     * @throws RuntimeException if the arguments are not missing
     */
    private static void validateCLIArguments(String[] args) {
        Option argsOpt = OptionBuilder.create("f");
        Option configOpt = OptionBuilder.create("config");
        Option nvargsOpt = OptionBuilder.create("nvargs");

        Options options = new Options();
        options.addOption(argsOpt);
        options.addOption(configOpt);
        options.addOption(nvargsOpt);

        CommandLineParser clParser = new PosixParser();
        try {
            clParser.parse(options, args);
        } catch (ParseException e) {
            throw new RuntimeException("Error at parsing arguments: " + e);
        }
    }

    /**
     * Validates the required compiler configurations:
     *  <li>OptLevel: O4_GLOBAL_TIME_MEMORY</li>
     *  <li>BlockSize: 1000</li>
     *  <li>RejectReadWriteUnknowns: true</li>
     * @param config
     * @throws RuntimeException if the configurations do not follow the requirements
     */
    private static void validateAndCompleteCompilerConfigs(CompilerConfig config) throws ParseException {
        // TODO: decide for actual required domains of the options
        if (config.getInt(CompilerConfig.ConfigType.OPT_LEVEL) !=
                OptimizerUtils.OptimizationLevel.O4_GLOBAL_TIME_MEMORY.ordinal())
            throw new ParseException("Opt level not set properly");
        if (config.getInt(CompilerConfig.ConfigType.BLOCK_SIZE) != 1000)
            throw new ParseException("Block size not set properly");
        if (!config.getBool(CompilerConfig.ConfigType.REJECT_READ_WRITE_UNKNOWNS))
            throw new ParseException("Ignoring metadata not allowed");

        // NOTE: Dynamic recompilation is rather obsolete since the optimizer does NOT execute the script
        ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.ALLOW_DYN_RECOMPILATION, false);
    }

    private static String validateAndLoadDMLOptions(DMLOptions dmlOptions) throws ParseException, IOException {
        if (dmlOptions.configFile != null) {
            DMLScript.loadConfiguration(dmlOptions.configFile);
        } else {
            throw new ParseException("Config filepath is missing.");
        }

        String dmlScriptPath = dmlOptions.filePath;
        if (dmlScriptPath == null)
            throw new ParseException("DML filepath is missing.");

        return dmlScriptPath;
    }

    private static Program runCompilationChain(String[] args) throws IOException {
        String dmlScriptPath;
        String dmlScriptStr;

        // validate and load all options and configurations
        DMLOptions dmlOptions;
        try {
            dmlOptions = DMLOptions.parseCLArguments(args);
            dmlScriptPath = validateAndLoadDMLOptions(dmlOptions);
            validateAndCompleteCompilerConfigs(ConfigurationManager.getCompilerConfig());
        }
        catch(ParseException e) {
            throw new RuntimeException("Parsing arguments failed: "+e.getMessage());
        }

        // generate IR (DMLProgram)
        dmlScriptStr = DMLScript.readDMLScript(true, dmlScriptPath);
        ParserWrapper parserWrapper = ParserFactory.createParser();
        DMLProgram dmlProgram = parserWrapper.parse(dmlScriptPath, dmlScriptStr, dmlOptions.argVals);
        DMLTranslator dmlTranslator = new DMLTranslator(dmlProgram);
        dmlTranslator.liveVariableAnalysis(dmlProgram);
        dmlTranslator.validateParseTree(dmlProgram);
        dmlTranslator.constructHops(dmlProgram);
        dmlTranslator.rewriteHopsDAG(dmlProgram);
        dmlTranslator.constructLops(dmlProgram);
        dmlTranslator.rewriteLopDAG(dmlProgram);
        System.out.println(Explain.explain(dmlProgram));
        // compile IR to runtime program (Program)
        Program precompiledProgram = dmlTranslator.getRuntimeProgram(dmlProgram, ConfigurationManager.getDMLConfig());
        System.out.println(Explain.display(dmlProgram, precompiledProgram, Explain.ExplainType.RUNTIME, Explain.countDistributedOperations(precompiledProgram)));

        return precompiledProgram;
    }

    public static void main(String[] args) throws IOException, JSONException {
        // NOTE: For now all the specific Ropt arguments (file paths) are hardcoded
        // parse the expected arguments or fail; Expected arguments:
        // * DML Script: flag "-f"
        // * SystemDS configs: flag "-config"
        // * Script Arguments: flag "-nvargs"
        validateCLIArguments(args);
        // generate IR (DMLProgram) or fail
        Program precompiledProgram = runCompilationChain(args);
        // resource optimization
        ResourceOptimizer optimizer = new ResourceOptimizer(precompiledProgram, AWS_META_INPUT_PATH);
        OptimalSolution optSolution = optimizer.execute();
        // output preparation
//        AWSUtils.generateInstancesConfigsFile("", optSolution.getDriverInstanceName(),
//                optSolution.getExecutorInstanceName(), optSolution.getExecutorInstanceNumber());
//        AWSUtils.generateSparkConfigsFile("");
    }
}
