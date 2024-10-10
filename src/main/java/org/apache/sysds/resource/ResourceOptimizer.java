package org.apache.sysds.resource;

import org.apache.commons.cli.*;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.resource.enumeration.EnumerationUtils;
import org.apache.sysds.resource.enumeration.Enumerator;
import org.apache.sysds.runtime.controlprogram.Program;

import java.io.DataOutput;
import java.io.IOException;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;
import java.util.HashMap;

import static org.apache.sysds.resource.CloudUtils.DEFAULT_CLUSTER_LAUNCH_TIME;

public class ResourceOptimizer {
    static {
        ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.ALLOW_DYN_RECOMPILATION, false);
        ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.RESOURCE_OPTIMIZATION, true);
    }

    private static final String DEFAULT_REGIONAL_PRICE_TABLE_PATH = "./aws_regional_prices.csv";
    private static final String DEFAULT_INSTANCE_INFO_TABLE_PATH = "./ec2_stats.csv";
    private static final String DEFAULT_OUTPUT_PATH = "./ec2_stats.csv";
    private static final String DEFAULT_REGION = "us-east-1";

    @SuppressWarnings("static-access")
    private static Options createOptions() {
        Options options = new Options();

        Option fileOpt = OptionBuilder.withArgName("filename")
                .withDescription("specifies dml/pydml file to execute; path should be local")
                .hasArg().create("f");
        Option infoTableOpt = OptionBuilder
                .withDescription("specifies filename of CSV table containing the meta data about all available cloud VM instance")
                .hasArg().create("infoTable");
        Option regionOpt = OptionBuilder
                .withDescription("specifies cloud region (using the corresponding abbreviation)")
                .hasArg().create("region");
        Option regionPriceTableOpt = OptionBuilder
                .withDescription("specifies filename of CSV table containing the extra price metrics depending on the target cloud region")
                .hasArg().create("regionTable");
        Option nvargsOpt = OptionBuilder.withArgName("key=value")
                .withDescription("parameterizes DML script with named parameters of the form <key=value>; <key> should be a valid identifier in DML/PyDML")
                .hasArgs().create("nvargs");
        Option argsOpt = OptionBuilder.withArgName("argN")
                .withDescription("specifies positional parameters; first value will replace $1 in DML program; $2 will replace 2nd and so on")
                .hasArgs().create("args");
        Option enumOpt = OptionBuilder.withArgName("strategy")
                .withDescription("specifies enumeration strategy; it should be one of the following: 'grid', 'interest', 'prune'; default 'grid'")
                .hasArg().create("enum");
        Option optimizeForOpt = OptionBuilder.withArgName("mode")
                .withDescription("specifies optimization strategy (scoring function); it should be one of the following: 'costs', 'time', 'price'; default 'costs'")
                .hasArg().create("optimizeFor");
        Option maxTimeOpt = OptionBuilder
                .withDescription("specifies constraint for maximum execution time")
                .hasArg().create("maxTime");
        Option maxPriceOpt = OptionBuilder
                .withDescription("specifies constraint for maximum price")
                .hasArg().create("maxPrice");
        Option minExecutorsOpt = OptionBuilder
                .withDescription("specifies minimum desired executors; default 0 (single node execution allowed); a negative value lead to setting the default")
                .hasArg().create("minExecutors");
        Option maxExecutorsOpt = OptionBuilder
                .withDescription("specifies maximum desired executors; default 200; a negative value leads to setting the default")
                .hasArg().create("maxExecutors");
        Option instanceTypesOpt = OptionBuilder
                .withDescription("specifies VM instance types for consideration and searching for optimal configuration; if not specified, all instances form the table with instance metadata are considered")
                .hasArg().create("instanceTypes");
        Option instanceSizesOpt = OptionBuilder
                .withDescription("specifies VM instance sizes for consideration and searching for optimal configuration; if not specified, all instances form the table with instance metadata are considered")
                .hasArg().create("instanceSizes");
        Option stepSizeOpt = OptionBuilder
                .withDescription("specific to grid-based enum. strategy; specifies step size for enumerating number of executors; default 1")
                .hasArg().create("stepSize");
        Option expBaseOpt = OptionBuilder
                .withDescription("specific to grid-based enum. strategy; specifies exponential base for increasing the number of executors exponentially; apply only if specified as larger than 1")
                .hasArg().create("expBase");
        Option fitDriverMemOpt = OptionBuilder
                .withDescription("specific to interest-based enum. strategy; boolean ('true'/'false') to indicate if the driver memory is an interest for the enumeration; default true")
                .hasArg().create("fitDriverMem");
        Option fitBroadcastMemOpt = OptionBuilder
                .withDescription("specific to interest-based enum. strategy; boolean ('true'/'false') to indicate if potential broadcasts' size is an interest for the enumeration; default true")
                .hasArg().create("fitBroadcastMem");
        Option checkSingleNodeOpt = OptionBuilder
                .withDescription("specific to interest-based enum. strategy; boolean ('true'/'false') to indicate if single node execution should be considered only in case of sufficient memory budget for the driver; default false")
                .hasArg().create("checkSingleNode");
        Option fitCheckPointMemory = OptionBuilder
                .withDescription("specific to interest-based enum. strategy; boolean ('true'/'false') to indicate if the size of the outputs is an interest for the enumeration; default false")
                .hasArg().create("checkSingleNode");
        Option outputOpt = OptionBuilder
                .hasArg().withDescription("output folder for configurations files")
                .create("output");
        Option helpOpt = OptionBuilder
                .withDescription("shows usage message")
                .create("help");

        options.addOption(fileOpt);
        options.addOption(infoTableOpt);
        options.addOption(regionOpt);
        options.addOption(regionPriceTableOpt);
        options.addOption(nvargsOpt);
        options.addOption(argsOpt);
        options.addOption(enumOpt);
        options.addOption(optimizeForOpt);
        options.addOption(maxTimeOpt);
        options.addOption(maxPriceOpt);
        options.addOption(minExecutorsOpt);
        options.addOption(maxExecutorsOpt);
        options.addOption(instanceTypesOpt);
        options.addOption(instanceSizesOpt);
        options.addOption(stepSizeOpt);
        options.addOption(expBaseOpt);
        options.addOption(fitDriverMemOpt);
        options.addOption(fitBroadcastMemOpt);
        options.addOption(checkSingleNodeOpt);
        options.addOption(fitCheckPointMemory);
        options.addOption(outputOpt);
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


    private static HashMap<String, String> parseArguments(String[] args) {



        HashMap<String, String> arguments = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("-")) {
                String key = args[i].substring(2);
                if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    arguments.put(key, args[i + 1]);
                    i++; // skip the next argument as it's a value
                } else {
                    arguments.put(key, "true"); // if it's a flag
                }
            }
        }
        return arguments;
    }

    public static void main(String[] args) throws ParseException, IOException {
        // step 1: parse options and arguments
        Options options = createOptions();
        CommandLineParser clParser = new PosixParser();
        CommandLine line = clParser.parse(options, args);

        if (line.hasOption("help")) {
            (new HelpFormatter()).printHelp("Main", options);
            return;
        }
        // 1a: parse script arguments
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

        // 1b: read the required files (according to the parsed parameters)
        String infoTablePath;
        if (line.hasOption("infoTable")) {
            infoTablePath = line.getOptionValue("infoTable");
        } else {
            infoTablePath = DEFAULT_INSTANCE_INFO_TABLE_PATH;
        }
        String region;
        if (line.hasOption("region")) {
            region = line.getOptionValue("region");
        } else {
            region = DEFAULT_REGION;
        }
        String regionTablePath;
        if (line.hasOption("regionTable")) {
            regionTablePath = line.getOptionValue("regionTable");
        } else {
            regionTablePath = DEFAULT_REGIONAL_PRICE_TABLE_PATH;
        }
        String outputPath;
        if (line.hasOption("output")) {
            outputPath = line.getOptionValue("output");
        } else {
            outputPath = DEFAULT_OUTPUT_PATH;
        }
        // validate the given output path now to avoid errors after the whole optimization process
        try {
            Paths.get(line.getOptionValue("output"));
        } catch (InvalidPathException e) {
            throw new MissingOptionException("Given value for option 'output' is not a valid path");
        }

        double[] regionalPrices = CloudUtils.loadRegionalPrices(line.getOptionValue("regionTable"), region);
        HashMap<String, CloudInstance> allInstances = CloudUtils.loadInstanceInfoTable(infoTablePath, regionalPrices[0], regionalPrices[1]);

        // 1c: parse strategy parameters
        Enumerator.EnumerationStrategy strategy;
        if (!line.hasOption("enum")) {
            strategy = Enumerator.EnumerationStrategy.GridBased; // default
        } else {
            switch (line.getOptionValue("enum")) {
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
        Enumerator.OptimizationStrategy mode;
        if (!line.hasOption("optimizeFor")) {
            mode = Enumerator.OptimizationStrategy.MinCosts;
        } else {
            switch (line.getOptionValue("optimizeFor")) {
                case "costs":
                    mode = Enumerator.OptimizationStrategy.MinCosts;
                    break;
                case "time":
                    mode = Enumerator.OptimizationStrategy.MinTime;
                    break;
                case "price":
                    mode = Enumerator.OptimizationStrategy.MinPrice;
                    break;
                default:
                    throw new ParseException("Unsupported identifier for optimization strategy: " + line.getOptionValue("optimizeFor"));
            }
        }
        double priceConstraint = -1;
        if (mode == Enumerator.OptimizationStrategy.MinTime) {
            String parsedValue = line.getOptionValue("maxPrice");
            if (parsedValue == null) {
                throw new ParseException("The provided option 'time' for -enum requires additionally an option for -maxPrice");
            }
            priceConstraint = Double.parseDouble(parsedValue);
        } else if (line.hasOption("maxPrice")) {
            System.err.println("Warning: option -maxPrice is relevant only for -optimizeFor 'time'");
        }
        double timeConstraint = -1;
        if (mode == Enumerator.OptimizationStrategy.MinPrice) {
            String parsedValue = line.getOptionValue("maxTime");
            if (parsedValue == null) {
                throw new ParseException("The provided option 'price' for -enum requires additionally an option for -maxTime");
            }
            timeConstraint = Double.parseDouble(parsedValue);
        } else if (line.hasOption("maxTime")) {
            System.err.println("Warning: option -maxTime is relevant only for -optimizeFor 'price'");
        }
        // 1d: parse search space range/limits
        int minExecutors = line.hasOption("minExecutors")? Integer.parseInt(line.getOptionValue("minExecutors")) : -1;
        int maxExecutors = line.hasOption("maxExecutors")? Integer.parseInt(line.getOptionValue("maxExecutors")) : -1;
        String[] instanceTypes = line.hasOption("instanceTypes")? line.getOptionValues("instanceTypes") : null;
        String[] instanceSizes = line.hasOption("instanceSizes")? line.getOptionValues("instanceSizes") : null;
        // 1e: parse arguments specific to enumeration strategies
        int stepSize = 1;
        int expBase = -1;
        if (strategy == Enumerator.EnumerationStrategy.GridBased) {
            if (line.hasOption("stepSize"))
                stepSize = Integer.parseInt(line.getOptionValue("stepSize"));
            if (line.hasOption("expBase"))
                stepSize = Integer.parseInt(line.getOptionValue("expBase"));
        } else {
            if (line.hasOption("stepSize"))
                System.err.println("Warning: option -stepSize is relevant only for -enum 'grid'");
            if (line.hasOption("expBase"))
                System.err.println("Warning: option -expBase is relevant only for -enum 'grid'");
        }
        boolean fitDriverMem = true;
        boolean fitBroadcastMem = true;
        boolean checkSingleNode = false;
        boolean fitCheckpointMem = false;
        if (strategy == Enumerator.EnumerationStrategy.InterestBased) {
            if (line.hasOption("fitDriverMem"))
                fitDriverMem = Boolean.parseBoolean(line.getOptionValue("fitDriverMem"));
            if (line.hasOption("fitBroadcastMem"))
                fitBroadcastMem = Boolean.parseBoolean(line.getOptionValue("fitBroadcastMem"));
            if (line.hasOption("checkSingleNode"))
                checkSingleNode = Boolean.parseBoolean(line.getOptionValue("checkSingleNode"));
            if (line.hasOption("fitCheckpointMem"))
                fitCheckpointMem = Boolean.parseBoolean(line.getOptionValue("fitCheckpointMem"));
        } else {
            if (line.hasOption("fitDriverMem"))
                System.err.println("Warning: option -fitDriverMem is relevant only for -enum 'interest'");
            if (line.hasOption("fitBroadcastMem"))
                System.err.println("Warning: option -fitBroadcastMem is relevant only for -enum 'interest'");
            if (line.hasOption("checkSingleNode"))
                System.err.println("Warning: option -checkSingleNode is relevant only for -enum 'interest'");
            if (line.hasOption("fitCheckpointMem"))
                System.err.println("Warning: option -fitCheckpointMem is relevant only for -enum 'interest'");
        }

        // step 2: compile the initial runtime program
        Program sourceProgram = ResourceCompiler.compile(line.getOptionValue("f"), argsMap);
        // step 3: initialize the enumerator
        // set the mandatory setting
        Enumerator.Builder builder = new Enumerator.Builder()
                .withRuntimeProgram(sourceProgram)
                .withAvailableInstances(allInstances)
                .withEnumerationStrategy(strategy)
                .withOptimizationStrategy(mode);
        // set min and max number of executors
        if (minExecutors > maxExecutors) {
            throw new ParseException("Option for -maxExecutors should be always greater or equal the option for -minExecutors");
        }
        builder.withNumberExecutorsRange(minExecutors, maxExecutors);
        // set range of instance types
        try {
            if (instanceTypes != null)
                builder.withInstanceTypeRange(instanceTypes);
        } catch (IllegalArgumentException e) {
            throw new ParseException("Not all provided options for -instanceTypes are supported or valid. Error thrown at:\n"+e.getMessage());
        }
        // set range of instance sizes
        try {
            if (instanceSizes != null)
                builder.withInstanceSizeRange(instanceSizes);
        } catch (IllegalArgumentException e) {
            throw new ParseException("Not all provided options for -instanceSizes are supported or valid. Error thrown at:\n"+e.getMessage());
        }
        // set budget if optimizing for time
        if (mode == Enumerator.OptimizationStrategy.MinTime && priceConstraint <= 0) {
            throw new ParseException("Missing or invalid option for -minPrice when -optimizeFor 'time'");
        } else if (mode == Enumerator.OptimizationStrategy.MinTime && priceConstraint > 0) {
            builder.withBudget(priceConstraint);
        }
        // set time limit if optimizing for price
        if (mode == Enumerator.OptimizationStrategy.MinPrice && timeConstraint <= 0) {
            throw new ParseException("Missing or invalid option for -minPrice when -optimizeFor 'time'");
        } else if (mode == Enumerator.OptimizationStrategy.MinPrice && timeConstraint > 0) {
            builder.withTimeLimit(timeConstraint);
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
            builder.withFitDriverMemory(fitDriverMem)
                    .withFitBroadcastMemory(fitBroadcastMem)
                    .withFitCheckpointMemory(fitCheckpointMem)
                    .withCheckSingleNodeExecution(checkSingleNode);
        }
        // build the enumerator
        Enumerator enumerator = builder.build();
        System.out.println("Number instances to be used for enumeration: " + enumerator.getInstances().size());
        System.out.println("All options are set! Enumeration is now running...");
        // step 4: pre-processing (generating search space according to the enumeration strategy)
        long startTime = System.currentTimeMillis();
        enumerator.preprocessing();
        // step 5: processing (finding the optimal solution) + postprocessing (retrieving the solution)
        enumerator.processing();
        EnumerationUtils.SolutionPoint optConfig = enumerator.postprocessing();
        long endTime = System.currentTimeMillis();
        System.out.println("Enumeration finished for " + ((double) (endTime-startTime))/1000 + " seconds");
        // step 6: generate configuration files according the optimal solution (if solution not empty)
        if (optConfig.getTimeCost() < Double.MAX_VALUE) {
            if (optConfig.numberExecutors == 0) {
                String filePath = Paths.get(outputPath, "ec2Arguments.json").toString();
                CloudUtils.generateEC2ConfigJson(optConfig.driverInstance, filePath);
            } else {
                String instanceGroupsPath = Paths.get(outputPath, "emrInstanceGroups.json").toString();
                String configurationsPath = Paths.get(outputPath, "emrConfigurations.json").toString();
                CloudUtils.generateEMRInstanceGroupsJson(
                        optConfig.driverInstance,
                        optConfig.numberExecutors,
                        optConfig.executorInstance,
                        instanceGroupsPath
                );
                CloudUtils.generateEMRConfigurationsJson(configurationsPath);
            }
        } else {
            System.err.println("Error: The provided combination of target instances and constraints leads to empty solution.");
            return;
        }
        // step 7: provide final info to the user
        String prompt = String.format(
                "Estimated optimal execution time: %.2fs (%.1fs static bootstrap time), price: %.2f$" +
                        "\nCluster configuration:\n" + optConfig +
                        "\nGenerated configuration stored in files: %s and %s",
                optConfig.getTimeCost(), DEFAULT_CLUSTER_LAUNCH_TIME, optConfig.getMonetaryCost(), "todo1.json", "todo2.json");
        System.out.println(prompt);
    }
}
