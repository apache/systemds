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
 
package org.apache.sysds.api;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstructionUtils;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.LineageCachePolicy;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.utils.Explain;
import org.apache.sysds.utils.Explain.ExplainType;

/**
 * Set of DMLOptions that can be set through the command line
 * and {@link org.apache.sysds.api.mlcontext.MLContext}
 * The values have been initialized with the default values
 * Despite there being a DML and PyDML, this class is named DMLOptions
 * to keep it consistent with {@link DMLOptions} and {@link DMLOptions}
 */
public class DMLOptions {
	// private static final Log LOG = LogFactory.getLog(DMLOptions.class.getName());

	public final Options        options;
	public Map<String, String>  argVals       = new HashMap<>();  // Arguments map containing either named arguments or arguments by position for a DML program
	public String               configFile    = null;             // Path to config file if default config and default config is to be overridden
	public boolean              clean         = false;            // Whether to clean up all SystemDS working directories (FS, DFS)
	public boolean              stats         = false;            // Whether to record and print the statistics
	public boolean              statsNGrams  = false;            // Whether to record and print the statistics n-grams
	public int                  statsCount    = 10;               // Default statistics count
	public int[]                statsNGramSizes = { 3 };          // Default n-gram tuple sizes
	public int                  statsTopKNGrams = 10;             // How many of the most heavy hitting n-grams are displayed
	public boolean              statsNGramsUseLineage = true;     // If N-Grams use lineage for data-dependent tracking
	public boolean              fedStats      = false;            // Whether to record and print the federated statistics
	public int                  fedStatsCount = 10;               // Default federated statistics count
	public boolean              memStats      = false;            // max memory statistics
	public Explain.ExplainType  explainType   = Explain.ExplainType.NONE;  // Whether to print the "Explain" and if so, what type
	public ExecMode             execMode      = OptimizerUtils.getDefaultExecutionMode();  // Execution mode standalone, MR, Spark or a hybrid
	public boolean              gpu           = false;            // Whether to use the GPU
	public boolean              forceGPU      = false;            // Whether to ignore memory & estimates and always use the GPU
	public boolean              debug         = false;            // to go into debug mode to be able to step through a program
	public String               filePath      = null;             // path to script
	public String               script        = null;             // the script itself
	public boolean              help          = false;            // whether to print the usage option
	public boolean              lineage       = false;            // whether compute lineage trace
	public boolean              lineage_dedup = false;            // whether deduplicate lineage items
	public ReuseCacheType       linReuseType  = ReuseCacheType.NONE; // reuse type (full, partial, hybrid)
	public LineageCachePolicy   linCachePolicy= LineageCachePolicy.COSTNSIZE; // lineage cache eviction policy
	public boolean              lineage_estimate = false;         // whether estimate reuse benefits
	public boolean              lineage_debugger = false;         // whether enable lineage debugger
	public boolean              fedWorker     = false;
	public int                  fedWorkerPort = -1;
	public boolean              fedMonitoring = false;
	public int                  fedMonitoringPort = -1;
	public String               fedMonitoringAddress = null;
	public int                  pythonPort    = -1;
	public boolean              checkPrivacy  = false;            // Check which privacy constraints are loaded and checked during federated execution 
	public boolean              federatedCompilation = false;     // Compile federated instructions based on input federation state and privacy constraints.
	public boolean              noFedRuntimeConversion = false;   // If activated, no runtime conversion of CP instructions to FED instructions will be performed.
	public int                  seed          = -1;               // The general seed for the execution, if -1 random (system time).

	public final static DMLOptions defaultOptions = new DMLOptions(null);

	public DMLOptions(Options opts) {
		options = opts;
	}
	
	@Override
	public String toString() {
		return "DMLOptions{" +
			"argVals=" + argVals +
			", configFile='" + configFile + '\'' +
			", clean=" + clean +
			", stats=" + stats +
			", statsCount=" + statsCount +
			", fedStats=" + fedStats +
			", fedStatsCount=" + fedStatsCount +
			", fedMonitoring=" + fedMonitoring +
			", fedMonitoringAddress" + fedMonitoringAddress +
			", memStats=" + memStats +
			", explainType=" + explainType +
			", execMode=" + execMode +
			", gpu=" + gpu +
			", forceGPU=" + forceGPU +
			", debug=" + debug +
			", filePath='" + filePath + '\'' +
			", script='" + script + '\'' +
			", help=" + help +
			", lineage=" + lineage +
			", w=" + fedWorker +
			", federatedCompilation=" + federatedCompilation +
			", noFedRuntimeConversion=" + noFedRuntimeConversion +
			", seed=" + seed + 
			'}';
	}
	
	/**
	 * Parses command line arguments to create a {@link DMLOptions} instance with the correct options
	 * @param args arguments from the command line
	 * @return an instance of {@link DMLOptions} that contain the correct {@link Option}s.
	 * @throws org.apache.commons.cli.ParseException if there is an incorrect option specified in the CLI
	 */
	public static DMLOptions parseCLArguments(String[] args)
		throws org.apache.commons.cli.ParseException
	{
		Options options = createCLIOptions();
		CommandLineParser clParser = new PosixParser();
		CommandLine line = clParser.parse(options, args);

		DMLOptions dmlOptions = new DMLOptions(options);
		dmlOptions.help = line.hasOption("help");
		if (line.hasOption("lineage")){
			dmlOptions.lineage = true;
			String lineageTypes[] = line.getOptionValues("lineage");
			if (lineageTypes != null) {
				for (String lineageType : lineageTypes) {
					if (lineageType != null){
						if (lineageType.equalsIgnoreCase("dedup"))
							dmlOptions.lineage_dedup = lineageType.equalsIgnoreCase("dedup");
						else if (lineageType.equalsIgnoreCase("reuse_full")
							|| lineageType.equalsIgnoreCase("reuse"))
							dmlOptions.linReuseType = ReuseCacheType.REUSE_FULL;
						else if (lineageType.equalsIgnoreCase("reuse_partial"))
							dmlOptions.linReuseType = ReuseCacheType.REUSE_PARTIAL;
						else if (lineageType.equalsIgnoreCase("reuse_multilevel"))
							dmlOptions.linReuseType = ReuseCacheType.REUSE_MULTILEVEL;
						else if (lineageType.equalsIgnoreCase("reuse_hybrid"))
							dmlOptions.linReuseType = ReuseCacheType.REUSE_HYBRID;
						else if (lineageType.equalsIgnoreCase("none"))
							dmlOptions.linReuseType = ReuseCacheType.NONE;
						else if (lineageType.equalsIgnoreCase("policy_lru"))
							dmlOptions.linCachePolicy = LineageCachePolicy.LRU;
						else if (lineageType.equalsIgnoreCase("policy_costnsize"))
							dmlOptions.linCachePolicy = LineageCachePolicy.COSTNSIZE;
						else if (lineageType.equalsIgnoreCase("policy_dagheight"))
							dmlOptions.linCachePolicy = LineageCachePolicy.DAGHEIGHT;
						else if (lineageType.equalsIgnoreCase("estimate"))
							dmlOptions.lineage_estimate = lineageType.equalsIgnoreCase("estimate");
						else if (lineageType.equalsIgnoreCase("debugger"))
							dmlOptions.lineage_debugger = lineageType.equalsIgnoreCase("debugger");							
						else
							throw new org.apache.commons.cli.ParseException(
								"Invalid argument specified for -lineage option: " + lineageType);
					}
				}
			}
		}
		dmlOptions.debug = line.hasOption("debug");
		dmlOptions.gpu = line.hasOption("gpu");
		if (dmlOptions.gpu) {
			String force = line.getOptionValue("gpu");
			if (force != null) {
				if (force.equalsIgnoreCase("force")) {
					dmlOptions.forceGPU = true;
				} else {
					throw new org.apache.commons.cli.ParseException("Invalid argument specified for -gpu option");
				}
			}
		}
		if (line.hasOption("exec")){
			String execMode = line.getOptionValue("exec");
			if (execMode != null){
				if (execMode.equalsIgnoreCase("singlenode")) dmlOptions.execMode = ExecMode.SINGLE_NODE;
				else if (execMode.equalsIgnoreCase("hybrid")) dmlOptions.execMode = ExecMode.HYBRID;
				else if (execMode.equalsIgnoreCase("spark")) dmlOptions.execMode = ExecMode.SPARK;
				else throw new org.apache.commons.cli.ParseException("Invalid argument specified for -exec option, must be one of [hadoop, singlenode, hybrid, HYBRID, spark]");
			}
		}
		if (line.hasOption("explain")) {
			dmlOptions.explainType = ExplainType.RUNTIME;
			String explainType = line.getOptionValue("explain");
			if (explainType != null){
				if (explainType.equalsIgnoreCase("hops")) dmlOptions.explainType = ExplainType.HOPS;
				else if (explainType.equalsIgnoreCase("runtime")) dmlOptions.explainType = ExplainType.RUNTIME;
				else if (explainType.equalsIgnoreCase("recompile_hops")) dmlOptions.explainType = ExplainType.RECOMPILE_HOPS;
				else if (explainType.equalsIgnoreCase("recompile_runtime")) dmlOptions.explainType = ExplainType.RECOMPILE_RUNTIME;
				else if (explainType.equalsIgnoreCase("codegen")) dmlOptions.explainType = ExplainType.CODEGEN;
				else if (explainType.equalsIgnoreCase("codegen_recompile")) dmlOptions.explainType = ExplainType.CODEGEN_RECOMPILE;
				else throw new org.apache.commons.cli.ParseException("Invalid argument specified for -hops option, must be one of [hops, runtime, recompile_hops, recompile_runtime, codegen, codegen_recompile]");
			}
		}

		dmlOptions.stats = line.hasOption("stats");
		if (dmlOptions.stats){
			String statsCount = line.getOptionValue("stats");
			if (statsCount != null) {
				try {
					dmlOptions.statsCount = Integer.parseInt(statsCount);
				} catch (NumberFormatException e) {
					throw new org.apache.commons.cli.ParseException("Invalid argument specified for -stats option, must be a valid integer");
				}
			}
		}

		dmlOptions.statsNGrams = line.hasOption("ngrams");
		if (dmlOptions.statsNGrams){
			String[] nGramArgs = line.getOptionValues("ngrams");
			if (nGramArgs.length >= 2) {
				try {
					String[] nGramSizeSplit = nGramArgs[0].split(",");
					dmlOptions.statsNGramSizes = new int[nGramSizeSplit.length];

					for (int i = 0; i < nGramSizeSplit.length; i++) {
						dmlOptions.statsNGramSizes[i] = Integer.parseInt(nGramSizeSplit[i]);
					}

					dmlOptions.statsTopKNGrams = Integer.parseInt(nGramArgs[1]);

					if (nGramArgs.length == 3) {
						dmlOptions.statsNGramsUseLineage = Boolean.parseBoolean(nGramArgs[2]);
					}
				} catch (NumberFormatException e) {
					throw new org.apache.commons.cli.ParseException("Invalid argument specified for -ngrams option, must be a valid integer");
				}
			}

			if (dmlOptions.statsNGramsUseLineage) {
				dmlOptions.lineage = true;
			}
		}

		dmlOptions.fedStats = line.hasOption("fedStats");
		if (dmlOptions.fedStats) {
			String fedStatsCount = line.getOptionValue("fedStats");
			if(fedStatsCount != null) {
				try {
					dmlOptions.fedStatsCount = Integer.parseInt(fedStatsCount);
				} catch (NumberFormatException e) {
					throw new org.apache.commons.cli.ParseException("Invalid argument specified for -fedStats option, must be a valid integer");
				}
			}
		}

		dmlOptions.memStats = line.hasOption("mem");

		dmlOptions.clean = line.hasOption("clean");
		
		if (line.hasOption("config")){
			dmlOptions.configFile = line.getOptionValue("config");
		}
		
		if (line.hasOption("w")){
			dmlOptions.fedWorker = true;
			dmlOptions.fedWorkerPort = Integer.parseInt(line.getOptionValue("w"));
		}

		if (line.hasOption("fedMonitoring")) {
			dmlOptions.fedMonitoring= true;
			dmlOptions.fedMonitoringPort = Integer.parseInt(line.getOptionValue("fedMonitoring"));
		}

		if (line.hasOption("fedMonitoringAddress")) {
			dmlOptions.fedMonitoringAddress = line.getOptionValue("fedMonitoringAddress");
		}

		if (line.hasOption("f")){
			dmlOptions.filePath = line.getOptionValue("f");
		}

		if (line.hasOption("s")){
			dmlOptions.script = line.getOptionValue("s");
		}

		// Positional arguments map is created as ("$1", "a"), ("$2", 123), etc
		if (line.hasOption("args")){
			String[] argValues = line.getOptionValues("args");
			for (int k=0; k<argValues.length; k++){
				String str = argValues[k];
				if (!str.isEmpty()) {
					dmlOptions.argVals.put("$" + (k+1), str);
				}
			}
		}

		if (line.hasOption("python"))
			dmlOptions.pythonPort = Integer.parseInt(line.getOptionValue("python"));

		// Named arguments map is created as ("$K, 123), ("$X", "X.csv"), etc
		if (line.hasOption("nvargs")){
			String varNameRegex = "^[a-zA-Z]([a-zA-Z0-9_])*$";
			String[] nvargValues = line.getOptionValues("nvargs");
			for (String str : nvargValues){
				if (!str.isEmpty()){
					String[] kv = str.split("=");
					if (kv.length != 2){
						throw new org.apache.commons.cli.ParseException("Invalid argument specified for -nvargs option, must be a list of space separated K=V pairs, where K is a valid name of a variable in the DML/PyDML program");
					}
					if (!kv[0].matches(varNameRegex)) {
						throw new org.apache.commons.cli.ParseException("Invalid argument specified for -nvargs option, " + kv[0] + " does not seem like a valid variable name in DML. Valid variable names in DML start with upper-case or lower-case letter, and contain only letters, digits, or underscores");
					}
					dmlOptions.argVals.put("$" + kv[0], kv[1]);
				}
			}
		}

		dmlOptions.checkPrivacy = line.hasOption("checkPrivacy");

		if (line.hasOption("federatedCompilation")){
			OptimizerUtils.FEDERATED_COMPILATION = true;
			dmlOptions.federatedCompilation = true;
			String[] fedCompSpecs = line.getOptionValues("federatedCompilation");
			if ( fedCompSpecs != null && fedCompSpecs.length > 0 ){
				for ( String spec : fedCompSpecs ){
					String[] specPair = spec.split("=");
					if (specPair.length != 2){
						throw new org.apache.commons.cli.ParseException("Invalid argument specified for -federatedCompilation option, must be a list of space separated K=V pairs, where K is a line number of the DML script and V is a federated output value");
					}
					int dmlLineNum = Integer.parseInt(specPair[0]);
					FEDInstruction.FederatedOutput fedOutSpec = FEDInstruction.FederatedOutput.valueOf(specPair[1]);
					OptimizerUtils.FEDERATED_SPECS.put(dmlLineNum,fedOutSpec);
				}
			}
		}

		if ( line.hasOption("noFedRuntimeConversion") ){
			FEDInstructionUtils.noFedRuntimeConversion = true;
			dmlOptions.noFedRuntimeConversion = true;
		}

		if(line.hasOption("seed")){
			dmlOptions.seed = Integer.parseInt(line.getOptionValue("seed"));
		}

		return dmlOptions;
	}
	
	@SuppressWarnings("static-access")
	private static Options createCLIOptions() {
		Options options = new Options();
		Option nvargsOpt = OptionBuilder.withArgName("key=value")
			.withDescription("parameterizes DML script with named parameters of the form <key=value>; <key> should be a valid identifier in DML/PyDML")
			.hasArgs().create("nvargs");
		Option argsOpt = OptionBuilder.withArgName("argN")
			.withDescription("specifies positional parameters; first value will replace $1 in DML program; $2 will replace 2nd and so on")
			.hasArgs().create("args");
		Option configOpt = OptionBuilder.withArgName("filename")
			.withDescription("uses a given configuration file (can be on local/hdfs/gpfs; default values in SystemDS-config.xml")
			.hasArg().create("config");
		Option cleanOpt = OptionBuilder
			.withDescription("cleans up all SystemDS working directories (FS, DFS); all other flags are ignored in this mode.")
			.create("clean");
		Option statsOpt = OptionBuilder.withArgName("count")
			.withDescription("monitors and reports summary execution statistics; heavy hitter <count> is 10 unless overridden; default off")
			.hasOptionalArg().create("stats");
		Option ngramsOpt = OptionBuilder//.withArgName("ngrams")
			.withDescription("monitors and reports the most occurring n-grams; -ngrams <comma separated n's> <topK>")
			.hasOptionalArgs(2).create("ngrams");
		Option fedStatsOpt = OptionBuilder.withArgName("count")
			.withDescription("monitors and reports summary execution statistics of federated workers; heavy hitter <count> is 10 unless overridden; default off")
			.hasOptionalArg().create("fedStats");
		Option memOpt = OptionBuilder.withDescription("monitors and reports max memory consumption in CP; default off")
			.create("mem");
		Option explainOpt = OptionBuilder.withArgName("level")
			.withDescription("explains plan levels; can be 'hops' / 'runtime'[default] / 'recompile_hops' / 'recompile_runtime' / 'codegen' / 'codegen_recompile'")
			.hasOptionalArg().create("explain");
		Option execOpt = OptionBuilder.withArgName("mode")
			.withDescription("sets execution mode; can be 'hadoop' / 'singlenode' / 'hybrid'[default] / 'HYBRID' / 'spark'")
			.hasArg().create("exec");
		Option gpuOpt = OptionBuilder.withArgName("force")
			.withDescription("uses CUDA instructions when reasonable; set <force> option to skip conservative memory estimates and use GPU wherever possible; default off")
			.hasOptionalArg().create("gpu");
		Option debugOpt = OptionBuilder.withDescription("runs in debug mode; default off")
			.create("debug");
		Option pythonOpt = OptionBuilder
			.withDescription("Python Context start with port argument for communication to from python to java")
			.isRequired().hasArg().create("python");
		Option monitorIdOpt = OptionBuilder
				.withDescription("Coordinator context start with monitorId argument for monitoring registration")
				.hasOptionalArg().create("monitorId");
		Option fileOpt = OptionBuilder.withArgName("filename")
			.withDescription("specifies dml/pydml file to execute; path can be local/hdfs/gpfs (prefixed with appropriate URI)")
			.isRequired().hasArg().create("f");
		Option scriptOpt = OptionBuilder.withArgName("script_contents")
			.withDescription("specified script string to execute directly")
			.isRequired().hasArg().create("s");
		Option helpOpt = OptionBuilder
			.withDescription("shows usage message")
			.create("help");
		Option lineageOpt = OptionBuilder
			.withDescription("computes lineage traces")
			.hasOptionalArgs().create("lineage");
		Option fedOpt = OptionBuilder
			.withDescription("starts a federated worker with the given argument as the port.")
			.hasOptionalArg().create("w");
		Option monitorOpt = OptionBuilder
			.withDescription("Starts a federated monitoring backend with the given argument as the port.")
			.hasOptionalArg().create("fedMonitoring");
		Option registerMonitorOpt = OptionBuilder
				.withDescription("Registers the coordinator for monitoring with the specified address of the monitoring tool.")
				.hasOptionalArg().create("fedMonitoringAddress");
		Option checkPrivacy = OptionBuilder
			.withDescription("Check which privacy constraints are loaded and checked during federated execution")
			.create("checkPrivacy");
		Option federatedCompilation = OptionBuilder
			.withArgName("key=value")
			.withDescription("Compile federated instructions based on input federation state and privacy constraints.")
			.hasOptionalArgs()
			.create("federatedCompilation");
		Option noFedRuntimeConversion = OptionBuilder
			.withDescription("If activated, no runtime conversion of CP instructions to FED instructions will be performed.")
			.create("noFedRuntimeConversion");
		Option commandlineSeed = OptionBuilder
			.withDescription("A general seed for the execution through the commandline")
			.hasArg().create("seed");
		
		options.addOption(configOpt);
		options.addOption(cleanOpt);
		options.addOption(statsOpt);
		options.addOption(ngramsOpt);
		options.addOption(fedStatsOpt);
		options.addOption(memOpt);
		options.addOption(explainOpt);
		options.addOption(execOpt);
		options.addOption(gpuOpt);
		options.addOption(debugOpt);
		options.addOption(lineageOpt);
		options.addOption(fedOpt);
		options.addOption(monitorOpt);
		options.addOption(registerMonitorOpt);
		options.addOption(monitorIdOpt);
		options.addOption(checkPrivacy);
		options.addOption(federatedCompilation);
		options.addOption(noFedRuntimeConversion);
		options.addOption(commandlineSeed);

		// Either a clean(-clean), a file(-f), a script(-s) or help(-help) needs to be specified
		OptionGroup fileOrScriptOpt = new OptionGroup()
			.addOption(scriptOpt)
			.addOption(fileOpt)
			.addOption(cleanOpt)
			.addOption(helpOpt)
			.addOption(fedOpt)
			.addOption(monitorOpt)
			.addOption(pythonOpt);
		fileOrScriptOpt.setRequired(true);
		options.addOptionGroup(fileOrScriptOpt);
		
		// Either -args or -nvargs
		options.addOptionGroup(new OptionGroup()
			.addOption(nvargsOpt).addOption(argsOpt));
		options.addOption(helpOpt);
		
		return options;
	}
}
