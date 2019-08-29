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

package org.tugraz.sysds.api;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.utils.Explain;
import org.tugraz.sysds.utils.Explain.ExplainType;

/**
 * Set of DMLOptions that can be set through the command line
 * and {@link org.tugraz.sysds.api.mlcontext.MLContext}
 * The values have been initialized with the default values
 * Despite there being a DML and PyDML, this class is named DMLOptions
 * to keep it consistent with {@link DMLOptions} and {@link DMLOptions}
 */
public class DMLOptions {
	public final Options        options;
	public Map<String, String>  argVals       = new HashMap<>();  // Arguments map containing either named arguments or arguments by position for a DML program
	public String               configFile    = null;             // Path to config file if default config and default config is to be overriden
	public boolean              clean         = false;            // Whether to clean up all SystemDS working directories (FS, DFS)
	public boolean              stats         = false;            // Whether to record and print the statistics
	public int                  statsCount    = 10;               // Default statistics count
	public boolean              memStats      = false;            // max memory statistics
	public Explain.ExplainType  explainType   = Explain.ExplainType.NONE;  // Whether to print the "Explain" and if so, what type
	public ExecMode             execMode = OptimizerUtils.getDefaultExecutionMode();  // Execution mode standalone, MR, Spark or a hybrid
	public boolean              gpu           = false;            // Whether to use the GPU
	public boolean              forceGPU      = false;            // Whether to ignore memory & estimates and always use the GPU
	public boolean              debug         = false;            // to go into debug mode to be able to step through a program
	public String               filePath      = null;             // path to script
	public String               script        = null;             // the script itself
	public boolean              help          = false;            // whether to print the usage option
	public boolean              lineage       = false;            // whether compute lineage trace
	public boolean              lineage_dedup = false;            // whether deduplicate lineage items
	public boolean              lineage_reuse = false;            // whether lineage-based reuse of intermediates

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
						else if (lineageType.equalsIgnoreCase("reuse"))
							dmlOptions.lineage_reuse = lineageType.equalsIgnoreCase("reuse");
						else
							throw new org.apache.commons.cli.ParseException("Invalid argument specified for -lineage option");
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
				else throw new org.apache.commons.cli.ParseException("Invalid argument specified for -hops option, must be one of [hops, runtime, recompile_hops, recompile_runtime]");
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
		dmlOptions.memStats = line.hasOption("mem");

		dmlOptions.clean = line.hasOption("clean");

		if (line.hasOption("config")){
			dmlOptions.configFile = line.getOptionValue("config");
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
		Option cleanOpt = OptionBuilder.withDescription("cleans up all SystemDS working directories (FS, DFS); all other flags are ignored in this mode. \n")
			.create("clean");
		Option statsOpt = OptionBuilder.withArgName("count")
			.withDescription("monitors and reports summary execution statistics; heavy hitter <count> is 10 unless overridden; default off")
			.hasOptionalArg().create("stats");
		Option memOpt = OptionBuilder.withDescription("monitors and reports max memory consumption in CP; default off")
			.create("mem");
		Option explainOpt = OptionBuilder.withArgName("level")
			.withDescription("explains plan levels; can be 'hops' / 'runtime'[default] / 'recompile_hops' / 'recompile_runtime'")
			.hasOptionalArg().create("explain");
		Option execOpt = OptionBuilder.withArgName("mode")
			.withDescription("sets execution mode; can be 'hadoop' / 'singlenode' / 'hybrid'[default] / 'HYBRID' / 'spark'")
			.hasArg().create("exec");
		Option gpuOpt = OptionBuilder.withArgName("force")
			.withDescription("uses CUDA instructions when reasonable; set <force> option to skip conservative memory estimates and use GPU wherever possible; default off")
			.hasOptionalArg().create("gpu");
		Option debugOpt = OptionBuilder.withDescription("runs in debug mode; default off")
			.create("debug");
		Option pythonOpt = OptionBuilder.withDescription("parses Python-like DML")
			.create("python");
		Option fileOpt = OptionBuilder.withArgName("filename")
			.withDescription("specifies dml/pydml file to execute; path can be local/hdfs/gpfs (prefixed with appropriate URI)")
			.isRequired().hasArg().create("f");
		Option scriptOpt = OptionBuilder.withArgName("script_contents")
			.withDescription("specified script string to execute directly")
			.isRequired().hasArg().create("s");
		Option helpOpt = OptionBuilder.withDescription("shows usage message")
			.create("help");
		Option lineageOpt = OptionBuilder.withDescription("computes lineage traces")
			.hasOptionalArgs().create("lineage");

		options.addOption(configOpt);
		options.addOption(cleanOpt);
		options.addOption(statsOpt);
		options.addOption(memOpt);
		options.addOption(explainOpt);
		options.addOption(execOpt);
		options.addOption(gpuOpt);
		options.addOption(debugOpt);
		options.addOption(pythonOpt);
		options.addOption(lineageOpt);

		// Either a clean(-clean), a file(-f), a script(-s) or help(-help) needs to be specified
		OptionGroup fileOrScriptOpt = new OptionGroup()
			.addOption(scriptOpt).addOption(fileOpt).addOption(cleanOpt).addOption(helpOpt);
		fileOrScriptOpt.setRequired(true);
		options.addOptionGroup(fileOrScriptOpt);
		
		// Either -args or -nvargs
		options.addOptionGroup(new OptionGroup()
			.addOption(nvargsOpt).addOption(argsOpt));
		options.addOption(helpOpt);
		
		return options;
	}
}
