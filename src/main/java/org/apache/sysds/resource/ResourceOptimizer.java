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

package org.apache.sysds.resource;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.configuration2.PropertiesConfiguration;
import org.apache.commons.configuration2.ex.ConfigurationException;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.resource.enumeration.EnumerationUtils;
import org.apache.sysds.resource.enumeration.Enumerator;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.commons.configuration2.io.FileHandler;

import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;


import static org.apache.sysds.resource.CloudUtils.DEFAULT_CLUSTER_LAUNCH_TIME;

public class ResourceOptimizer {
	private static final String DEFAULT_OPTIONS_FILE = "./options.properties";

	static {
		ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.RESOURCE_OPTIMIZATION, true);
		ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.REJECT_READ_WRITE_UNKNOWNS, true);
	}
	private static final String EMR_INSTANCE_GROUP_FILENAME = "emr_instance_groups.json";
	private static final String EMR_CONFIGURATIONS_FILENAME = "emr_configurations.json";
	private static final String EC2_ARGUMENTS_FILENAME = "ec2_configurations.json";

	@SuppressWarnings("static-access")
	public static Options createOptions() {
		Options options = new Options();

		Option fileOpt = OptionBuilder.withArgName("filename")
				.withDescription("specifies DML file to execute; path should be local")
				.hasArg().create("f");
		Option optionsOpt = OptionBuilder
				.withDescription("specifies options file for the resource optimization")
				.hasArg().create("options");
		Option nvargsOpt = OptionBuilder.withArgName("key=value")
				.withDescription("parameterizes DML script with named parameters of the form <key=value>; " +
						"<key> should be a valid identifier in DML")
				.hasArgs().create("nvargs");
		Option argsOpt = OptionBuilder.withArgName("argN")
				.withDescription("specifies positional parameters; " +
						"first value will replace $1 in DML program, $2 will replace 2nd and so on")
				.hasArgs().create("args");
		Option helpOpt = OptionBuilder
				.withDescription("shows usage message")
				.create("help");

		options.addOption(fileOpt);
		options.addOption(optionsOpt);
		options.addOption(nvargsOpt);
		options.addOption(argsOpt);
		options.addOption(helpOpt);

		OptionGroup helpOrFile = new OptionGroup()
				.addOption(helpOpt)
				.addOption(fileOpt);
		helpOrFile.setRequired(true);
		options.addOptionGroup(helpOrFile);

		options.addOptionGroup(new OptionGroup()
				.addOption(nvargsOpt).addOption(argsOpt));
		options.addOption(helpOpt);

		return options;
	}

	public static Enumerator initEnumerator(CommandLine line, PropertiesConfiguration options) throws ParseException, IOException {
		// parse script arguments
		HashMap <String, String> argsMap = new HashMap<>();
		if (line.hasOption("args")){
			String[] argValues = line.getOptionValues("args");
			for (int k=0; k<argValues.length; k++){
				String str = argValues[k];
				if (!str.isEmpty()) {
					argsMap.put("$" + (k+1), str);
				}
			}
		}
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
					argsMap.put("$" + kv[0], kv[1]);
				}
			}
		}

		// load the rest of the options variables
		String regionOpt = getOrDefault(options, "REGION", "");
		String infoTablePathOpt = getOrDefault(options, "INFO_TABLE", "");
		String regionTablePathOpt = getOrDefault(options, "REGION_TABLE", "");
		String localInputsOpt = getOrDefault(options, "LOCAL_INPUTS", "");

		String enumerationOpt = getOrDefault(options, "ENUMERATION", "");
		String optimizationOpt = getOrDefault(options, "OPTIMIZATION_FUNCTION", "");
		String costsWeightOpt = getOrDefault(options, "COSTS_WEIGHT", "");
		String maxTimeOpt = getOrDefault(options, "MAX_TIME", "");
		String maxPriceOpt = getOrDefault(options, "MAX_PRICE", "");
		String cpuQuotaOpt = getOrDefault(options, "CPU_QUOTA", "");
		String minExecutorsOpt = getOrDefault(options, "MIN_EXECUTORS", "");
		String maxExecutorsOpt = getOrDefault(options, "MAX_EXECUTORS", "");
		String instanceFamiliesOpt = getOrDefault(options, "INSTANCE_FAMILIES", "");
		String instanceSizesOpt = getOrDefault(options, "INSTANCE_SIZES", "");
		String stepSizeOpt = getOrDefault(options, "STEP_SIZE", "");
		String expBaseOpt = getOrDefault(options, "EXPONENTIAL_BASE", "");
		String useLargestEstOpt = getOrDefault(options, "USE_LARGEST_ESTIMATE", "");
		String useCpEstOpt = getOrDefault(options, "USE_CP_ESTIMATES", "");
		String useBroadcastOpt = getOrDefault(options, "USE_BROADCASTS", "");
		String useOutputsOpt = getOrDefault(options, "USE_OUTPUTS", "");

		// replace declared S3 files with local path
		HashMap<String, String> localInputMap = new HashMap<>();
		if (!localInputsOpt.isEmpty()) {
			String[] inputParts = localInputsOpt.split(",");
			for (String var : inputParts){
				String[] varParts = var.split("=");
				if (varParts.length != 2) {
					throw new RuntimeException("Invalid local variable pairs declaration: " + var);
				}
				if (!argsMap.containsValue(varParts[0])) {
					throw new RuntimeException("Option for local input does not match any given argument: " + varParts[0]);
				}
				String argName = getKeyByValue(argsMap, varParts[0]);
				// update variables for compilation
				argsMap.put(argName, varParts[1]);
				// fill a map for later replacement back after first compilation
				localInputMap.put(varParts[1], varParts[0]);
			}
		}
		// replace S3 filesystem identifier to match the available hadoop connector if needed
		if (argsMap.values().stream().anyMatch(var -> var.startsWith("s3"))) {
			String s3Filesystem = getAvailableHadoopS3Filesystem();
			replaceS3Filesystem(argsMap, s3Filesystem);
		}

		// materialize the options

		Enumerator.EnumerationStrategy strategy;
		if (enumerationOpt.isEmpty()) {
			strategy = Enumerator.EnumerationStrategy.GridBased; // default
		} else {
			switch (enumerationOpt) {
				case "grid":
					strategy = Enumerator.EnumerationStrategy.GridBased;
					break;
				case "interest":
					strategy = Enumerator.EnumerationStrategy.InterestBased;
					break;
				case "prune":
					strategy = Enumerator.EnumerationStrategy.PruneBased;
					break;
				default:
					throw new ParseException("Unsupported identifier for enumeration strategy: " + line.getOptionValue("enum"));
			}
		}

		Enumerator.OptimizationStrategy optimizedFor;
		if (optimizationOpt.isEmpty()) {
			optimizedFor = Enumerator.OptimizationStrategy.MinCosts;
		} else {
			switch (optimizationOpt) {
				case "costs":
					optimizedFor = Enumerator.OptimizationStrategy.MinCosts;
					break;
				case "time":
					optimizedFor = Enumerator.OptimizationStrategy.MinTime;
					break;
				case "price":
					optimizedFor = Enumerator.OptimizationStrategy.MinPrice;
					break;
				default:
					throw new ParseException("Unsupported identifier for optimization strategy: " + line.getOptionValue("optimizeFor"));
			}
		}

		if (optimizedFor == Enumerator.OptimizationStrategy.MinCosts && !costsWeightOpt.isEmpty()) {
			double costsWeighFactor = Double.parseDouble(costsWeightOpt);
			if (costsWeighFactor < 0.0 || costsWeighFactor > 1.0) {
				throw new ParseException("The provided option 'price' for -enum requires additionally an option for -maxTime");
			}
			Enumerator.setCostsWeightFactor(costsWeighFactor);
		} else if (!costsWeightOpt.isEmpty()) {
			System.err.println("Warning: option MAX_PRICE is relevant only for OPTIMIZATION_FUNCTION 'time'");
		}

		if (optimizedFor == Enumerator.OptimizationStrategy.MinTime) {
			if (maxPriceOpt.isEmpty()) {
				throw new ParseException("Providing the option MAX_PRICE value is required " +
						"when OPTIMIZATION_FUNCTION is set to 'time'");
			}
			double priceConstraint = Double.parseDouble(maxPriceOpt);
			if (priceConstraint <= 0) {
				throw new ParseException("Invalid value for option MIN_PRICE " +
						"when option OPTIMIZATION_FUNCTION is set to  'time'");
			}
			Enumerator.setMinPrice(priceConstraint);
		} else if (!maxPriceOpt.isEmpty()) {
			System.err.println("Warning: option MAX_PRICE is relevant only for OPTIMIZATION_FUNCTION 'time'");
		}

		if (optimizedFor == Enumerator.OptimizationStrategy.MinPrice) {
			if (maxTimeOpt.isEmpty()) {
				throw new ParseException("Providing the option MAX_TIME value is required " +
						"when OPTIMIZATION_FUNCTION is set to 'price'");
			}
			double timeConstraint = Double.parseDouble(maxTimeOpt);
			if (timeConstraint <= 0) {
				throw new ParseException("Missing or invalid value for option MIN_TIME " +
						"when option OPTIMIZATION_FUNCTION is set to 'price'");
			}
			Enumerator.setMinTime(timeConstraint);
		} else if (!maxTimeOpt.isEmpty()) {
			System.err.println("Warning: option MAX_TIME is relevant only for OPTIMIZATION_FUNCTION 'price'");
		}

		if (!cpuQuotaOpt.isEmpty()) {
			int quotaForNumCores = Integer.parseInt(cpuQuotaOpt);
			if (quotaForNumCores < 32) {
				throw new ParseException("CPU quota of under 32 number of cores is not allowed");
			}
			Enumerator.setCpuQuota(quotaForNumCores);
		}

		int minExecutors = minExecutorsOpt.isEmpty()? -1 : Integer.parseInt(minExecutorsOpt);
		int maxExecutors = maxExecutorsOpt.isEmpty()? -1 : Integer.parseInt(maxExecutorsOpt);
		String[] instanceFamilies = instanceFamiliesOpt.isEmpty()? null : instanceFamiliesOpt.split(",");
		String[] instanceSizes = instanceSizesOpt.isEmpty()? null : instanceSizesOpt.split(",");
		// parse arguments specific to enumeration strategies
		int stepSize = 1;
		int expBase = -1;
		if (strategy == Enumerator.EnumerationStrategy.GridBased) {
			if (!stepSizeOpt.isEmpty())
				stepSize = Integer.parseInt(stepSizeOpt);
			if (!expBaseOpt.isEmpty())
				expBase = Integer.parseInt(expBaseOpt);
		} else {
			if (!stepSizeOpt.isEmpty())
				System.err.println("Warning: option STEP_SIZE is relevant only for option ENUMERATION 'grid'");
			if (line.hasOption("expBase"))
				System.err.println("Warning: option EXPONENTIAL_BASE is relevant only for option ENUMERATION 'grid'");
		}
		boolean interestLargestEstimate = true;
		boolean interestEstimatesInCP = true;
		boolean interestBroadcastVars = true;
		boolean interestOutputCaching = false;
		if (strategy == Enumerator.EnumerationStrategy.InterestBased) {
			if (!useLargestEstOpt.isEmpty())
				interestLargestEstimate = Boolean.parseBoolean(useLargestEstOpt);
			if (!useCpEstOpt.isEmpty())
				interestEstimatesInCP = Boolean.parseBoolean(useCpEstOpt);
			if (!useBroadcastOpt.isEmpty())
				interestBroadcastVars = Boolean.parseBoolean(useBroadcastOpt);
			if (!useOutputsOpt.isEmpty())
				interestOutputCaching = Boolean.parseBoolean(useOutputsOpt);
		} else {
			if (!useLargestEstOpt.isEmpty())
				System.err.println("Warning: option -useLargestEst is relevant only for -enum 'interest'");
			if (!useCpEstOpt.isEmpty())
				System.err.println("Warning: option -useCpEstimates is relevant only for -enum 'interest'");
			if (!useBroadcastOpt.isEmpty())
				System.err.println("Warning: option -useBroadcasts is relevant only for -enum 'interest'");
			if (!useOutputsOpt.isEmpty())
				System.err.println("Warning: option -useOutputs is relevant only for -enum 'interest'");
		}

		double[] regionalPrices = CloudUtils.loadRegionalPrices(regionTablePathOpt, regionOpt);
		HashMap<String, CloudInstance> allInstances = CloudUtils.loadInstanceInfoTable(infoTablePathOpt, regionalPrices[0], regionalPrices[1]);

		// step 2: compile the initial runtime program
		Program sourceProgram = ResourceCompiler.compile(line.getOptionValue("f"), argsMap, localInputMap);
		// step 3: initialize the enumerator
		// set the mandatory setting
		Enumerator.Builder builder = new Enumerator.Builder()
				.withRuntimeProgram(sourceProgram)
				.withAvailableInstances(allInstances)
				.withEnumerationStrategy(strategy)
				.withOptimizationStrategy(optimizedFor);
		// set min and max number of executors
		if (maxExecutors >= 0 && minExecutors > maxExecutors) {
			throw new ParseException("Option for MAX_EXECUTORS should be always greater or equal the option for -minExecutors");
		}
		builder.withNumberExecutorsRange(minExecutors, maxExecutors);
		// set range of instance types
		try {
			if (instanceFamilies != null)
				builder.withInstanceFamilyRange(instanceFamilies);
		} catch (IllegalArgumentException e) {
			throw new ParseException("Not all provided options for INSTANCE_FAMILIES are supported or valid. Error thrown at:\n"+e.getMessage());
		}
		// set range of instance sizes
		try {
			if (instanceSizes != null)
				builder.withInstanceSizeRange(instanceSizes);
		} catch (IllegalArgumentException e) {
			throw new ParseException("Not all provided options for INSTANCE_SIZES are supported or valid. Error thrown at:\n"+e.getMessage());
		}

		// set step size for grid-based enum.
		if (strategy == Enumerator.EnumerationStrategy.GridBased && stepSize > 1) {
			builder.withStepSizeExecutor(stepSize);
		} else if (stepSize < 1) {
			throw new ParseException("Invalid option for -stepSize");
		}
		// set exponential base for grid-based enum.
		if (strategy == Enumerator.EnumerationStrategy.GridBased) {
			builder.withExpBaseExecutors(expBase);
		}
		// set flags for interest-based enum.
		if (strategy == Enumerator.EnumerationStrategy.InterestBased) {
			builder.withInterestLargestEstimate(interestLargestEstimate)
					.withInterestEstimatesInCP(interestEstimatesInCP)
					.withInterestBroadcastVars(interestBroadcastVars)
					.withInterestOutputCaching(interestOutputCaching);

		}
		// build the enumerator
		return builder.build();
	}

	public static void execute(CommandLine line, PropertiesConfiguration options) throws ParseException, IOException {
		String outputPath = getOrDefault(options, "OUTPUT_FOLDER", "");
		// validate the given output path now to avoid errors after the whole optimization process
		Path folderPath;
		try {
			folderPath = Paths.get(outputPath);
		} catch (InvalidPathException e) {
			throw new RuntimeException("Given value for option 'OUTPUT_FOLDER' is not a valid path");
		}
		try {
			Files.createDirectory(folderPath);
		} catch (FileAlreadyExistsException e) {
			System.err.printf("Folder '%s' already exists on the given path. Files will be overwritten!\n", folderPath);
		} catch (IOException e) {
			throw new RuntimeException("Given value for option 'OUTPUT_FOLDER' is not a valid path: "+e);
		}

		// initialize the enumerator (including initial program compilation)
		Enumerator enumerator = initEnumerator(line, options);
		if (enumerator == null) {
			// help requested
			return;
		}
		System.out.println("Number instances to be used for enumeration: " + enumerator.getInstances().size());
		System.out.println("All options are set! Enumeration is now running...");

		long startTime = System.currentTimeMillis();
		// pre-processing (generating search space according to the enumeration strategy)
		enumerator.preprocessing();
		// processing (finding the optimal solution) + postprocessing (retrieving the solution)
		enumerator.processing();
		// processing (currently only fetching the optimal solution)
		EnumerationUtils.SolutionPoint optConfig = enumerator.postprocessing();
		long endTime = System.currentTimeMillis();
		System.out.println("...enumeration finished for " + ((double) (endTime-startTime))/1000 + " seconds\n");
		System.out.println("The resulted runtime plan for the optimal configurations is the following:");

		// generate configuration files according the optimal solution (if solution not empty)
		if (optConfig.getTimeCost() < Double.MAX_VALUE) {
			if (optConfig.numberExecutors == 0) {
				String filePath = Paths.get(folderPath.toString(), EC2_ARGUMENTS_FILENAME).toString();
				CloudUtils.generateEC2ConfigJson(optConfig.driverInstance, filePath);
			} else {
				String instanceGroupsPath = Paths.get(folderPath.toString(), EMR_INSTANCE_GROUP_FILENAME).toString();
				String configurationsPath = Paths.get(folderPath.toString(), EMR_CONFIGURATIONS_FILENAME).toString();
				CloudUtils.generateEMRInstanceGroupsJson(optConfig, instanceGroupsPath);
				CloudUtils.generateEMRConfigurationsJson(optConfig, configurationsPath);
			}
		} else {
			System.err.println("Error: The provided combination of target instances and constraints leads to empty solution.");
			return;
		}
		// step 7: provide final info to the user
		String prompt = String.format(
				"\nEstimated optimal execution time: %.2fs (%.1fs static bootstrap time), price: %.2f$" +
						"\n\nCluster configuration:\n" + optConfig +
						"\n\nGenerated configurations stored in folder %s\n",
				optConfig.getTimeCost(), DEFAULT_CLUSTER_LAUNCH_TIME, optConfig.getMonetaryCost(), outputPath);
		System.out.println(prompt);
		System.out.println("Execution suggestions:\n");
		String executionSuggestions;
		if (optConfig.numberExecutors == 0) {
			executionSuggestions =String.format(
					"Launch the EC2 instance using the script %s.\nUse -help to check the options.\n\n" +
							"SystemDS rely on memory only for all computations but in debugging more or longer estimated execution time,\n" +
							"please adapt the root EBS volume in case no NVMe storage is attached.\n" +
							"Note that the storage configurations for EBS from the instance info table are relevant only for EMR cluster executions.\n" +
							"Increasing the EBS root volume size for larger instances is also recommended.\n" +
							"Adjusting the root volume configurations is done manually in the %s file.\n" +
							"\nMore details can be found in the README.md file."
					, "<SystemDS root>/script/resource/launch_ec2.sh", EC2_ARGUMENTS_FILENAME
			);
		} else {
			executionSuggestions =String.format(
					"Launch the EMR cluster using the script %s.\nUse -help to check the options.\n\n" +
							"If you you decide to run in debug mode and/or the estimated execution time is significantly long,\n" +
							"please adjust the default EBS root volume size to account for larger log files!\n" +
							"Currently the Resource Optimizer does not adapt the storage configurations\n" +
							"and the defaults from the instance info table are used.\n" +
							"In case of constraining the available instances for enumeration with the provided optional arguments and large input datasets,\n" +
							"please adjust the EBS configurations in the %s file manually following the instructions from the README.md file!\n\n" +
							"Disable the automatic cluster termination if you want to access the cluster logs or the any file not exported to S3 by the DML script.\n" +
							"\nMore details can be found in the README.md file."
					, "<SystemDS root>/script/resource/launch_emr.sh", EMR_INSTANCE_GROUP_FILENAME
			);
		}
		System.out.println(executionSuggestions);
	}

	public static void main(String[] args) throws ParseException, IOException, ConfigurationException {
		// load directly passed options
		Options cliOptions = createOptions();
		CommandLineParser clParser = new PosixParser();
		CommandLine line = clParser.parse(cliOptions, args);
		if (line.hasOption("help")) {
			(new HelpFormatter()).printHelp("SystemDS Resource Optimizer", cliOptions);
			return;
		}
		String optionsFile;
		if (line.hasOption("options")) {
			optionsFile = line.getOptionValue("options");
		} else {
			Path defaultOptions = Paths.get(DEFAULT_OPTIONS_FILE);
			if (Files.exists(defaultOptions)) {
				optionsFile = defaultOptions.toString();
			} else {
				throw new ParseException("File with options was neither provided or " +
						"found in the current execution directory: "+DEFAULT_OPTIONS_FILE);
			}
		}
		// load options
		PropertiesConfiguration options = new PropertiesConfiguration();
		FileHandler handler = new FileHandler(options);
		handler.load(optionsFile);
		// execute the actual main logic
		execute(line, options);
	}

	// Helpers ---------------------------------------------------------------------------------------------------------

	private static String getKeyByValue(HashMap <String, String> hashmap, String value) {
		for (Map.Entry<String, String> pair : hashmap.entrySet()) {
			if (pair.getValue().equals(value)) {
				return pair.getKey();
			}
		}
		return null;
	}

	private static void replaceS3Filesystem(HashMap <String, String> argsMap, String filesystem) {
		for (Map.Entry<String, String> pair : argsMap.entrySet()) {
			String[] currentFileParts = pair.getValue().split(":");
			if (currentFileParts.length != 2) continue;
			if (!currentFileParts[0].startsWith("s3")) continue;
			pair.setValue(String.format("%s:%s", filesystem, currentFileParts[1]));
		}
	}

	private static String getAvailableHadoopS3Filesystem() {
		try {
			Class.forName("org.apache.hadoop.fs.s3a.S3AFileSystem");
			return "s3a";
		} catch (ClassNotFoundException ignored) {}
		try {
			Class.forName("org.apache.hadoop.fs.s3.S3AFileSystem");
			return "s3";
		} catch (ClassNotFoundException ignored) {}
		throw new RuntimeException("No Hadoop S3 Filesystem connector installed");
	}

	public static String getOrDefault(PropertiesConfiguration config, String key, String defaultValue) {
		return config.containsKey(key) ? config.getString(key) : defaultValue;
	}
}
