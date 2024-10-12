package org.apache.sysds.resource;

import org.apache.commons.cli.*;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.resource.enumeration.EnumerationUtils;
import org.apache.sysds.resource.enumeration.Enumerator;
import org.apache.sysds.runtime.controlprogram.Program;

import java.io.IOException;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;
import java.util.HashMap;

import static org.apache.sysds.resource.CloudUtils.DEFAULT_CLUSTER_LAUNCH_TIME;

public class ResourceOptimizer {
    static {
        ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.RESOURCE_OPTIMIZATION, true);
    }
    private static final String RESOURCE_SCRIPT_DIR = "./scripts/resource/";
    private static final String DEFAULT_REGIONAL_PRICE_TABLE_PATH = RESOURCE_SCRIPT_DIR + "aws_regional_prices.csv";
    private static final String DEFAULT_INSTANCE_INFO_TABLE_PATH = RESOURCE_SCRIPT_DIR + "ec2_stats.csv";
    private static final String DEFAULT_OUTPUT_PATH = RESOURCE_SCRIPT_DIR + "output";
    private static final String EMR_INSTANCE_GROUP_FILENAME = "emrInstanceGroups.json";
    private static final String EMR_CONFIGURATIONS_FILENAME = "emrConfigurations.json";
    private static final String EC2_ARGUMENTS_FILENAME = "ec2Arguments.json";
    private static final String DEFAULT_REGION = "us-east-1";

    @SuppressWarnings("static-access")
    public static Options createOptions() {
        Options options = new Options();

        Option fileOpt = OptionBuilder.withArgName("filename")
                .withDescription("specifies dml/pydml file to execute; path should be local")
                .hasArg().create("f");
        Option infoTableOpt = OptionBuilder
                .withDescription("specifies filename of CSV table containing the meta data " +
                        "about all available cloud VM instance")
                .hasArg().create("infoTable");
        Option regionOpt = OptionBuilder
                .withDescription("specifies cloud region (using the corresponding abbreviation)")
                .hasArg().create("region");
        Option regionPriceTableOpt = OptionBuilder
                .withDescription("specifies filename of CSV table containing the extra price metrics " +
                        "depending on the target cloud region")
                .hasArg().create("regionTable");
        Option nvargsOpt = OptionBuilder.withArgName("key=value")
                .withDescription("parameterizes DML script with named parameters of the form <key=value>; " +
                        "<key> should be a valid identifier in DML/PyDML")
                .hasArgs().create("nvargs");
        Option argsOpt = OptionBuilder.withArgName("argN")
                .withDescription("specifies positional parameters; " +
                        "first value will replace $1 in DML program, $2 will replace 2nd and so on")
                .hasArgs().create("args");
        Option enumOpt = OptionBuilder.withArgName("strategy")
                .withDescription("specifies enumeration strategy; " +
                        "it should be one of the following: 'grid', 'interest', 'prune'; default 'grid'")
                .hasArg().create("enum");
        Option optimizeForOpt = OptionBuilder.withArgName("mode")
                .withDescription("specifies optimization strategy (scoring function); " +
                        "it should be one of the following: 'costs', 'time', 'price'; default 'costs'")
                .hasArg().create("optimizeFor");
        Option maxTimeOpt = OptionBuilder
                .withDescription("specifies constraint for maximum execution time")
                .hasArg().create("maxTime");
        Option maxPriceOpt = OptionBuilder
                .withDescription("specifies constraint for maximum price")
                .hasArg().create("maxPrice");
        Option minExecutorsOpt = OptionBuilder
                .withDescription("specifies minimum desired executors; " +
                        "default 0 (single node execution allowed); " +
                        "a negative value lead to setting the default")
                .hasArg().create("minExecutors");
        Option maxExecutorsOpt = OptionBuilder
                .withDescription("specifies maximum desired executors; " +
                        "default 200; a negative value leads to setting the default")
                .hasArg().create("maxExecutors");
        Option instanceFamiliesOpt = OptionBuilder
                .withDescription("specifies VM instance types for consideration " +
                        "at searching for optimal configuration; " +
                        "if not specified, all instances form the table with instance metadata are considered")
                .hasArgs().create("instanceFamilies");
        Option instanceSizesOpt = OptionBuilder
                .withDescription("specifies VM instance sizes for consideration " +
                        "at searching for optimal configuration; " +
                        "if not specified, all instances form the table with instance metadata are considered")
                .hasArgs().create("instanceSizes");
        Option stepSizeOpt = OptionBuilder
                .withDescription("specific to grid-based enum. strategy; " +
                        "specifies step size for enumerating number of executors; default 1")
                .hasArg().create("stepSize");
        Option expBaseOpt = OptionBuilder
                .withDescription("specific to grid-based enum. strategy; " +
                        "specifies exponential base for increasing the number of executors exponentially; apply only if specified as larger than 1")
                .hasArg().create("expBase");
        Option interestLargestEstimateOpt = OptionBuilder
                .withDescription("specific to interest-based enum. strategy; " +
                        "boolean ('true'/'false') to indicate if single node execution should be considered only " +
                        "in case of sufficient memory budget for the driver; default true")
                .hasArg().create("useLargestEst");
        Option interestEstimatesInCPOpt = OptionBuilder
                .withDescription("specific to interest-based enum. strategy; " +
                        "boolean ('true'/'false') to indicate if the CP memory is an interest for the enumeration; " +
                        "default true")
                .hasArg().create("useCpEstimates");
        Option interestBroadcastVarsOpt = OptionBuilder
                .withDescription("specific to interest-based enum. strategy; " +
                        "boolean ('true'/'false') to indicate if potential broadcast variables' size is an interest " +
                        "for driver and executors memory budget; default true")
                .hasArg().create("useBroadcasts");
        Option interestOutputCachingOpt = OptionBuilder
                .withDescription("specific to interest-based enum. strategy; " +
                        "boolean ('true'/'false') to indicate if the size of the outputs (potentially cached) " +
                        "is an interest for the enumerated number of executors; default false")
                .hasArg().create("useOutputs");
        Option outputOpt = OptionBuilder
                .hasArg().withDescription("output folder for configurations files; " +
                        "existing configurations files will be overwritten")
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
        options.addOption(instanceFamiliesOpt);
        options.addOption(instanceSizesOpt);
        options.addOption(stepSizeOpt);
        options.addOption(expBaseOpt);
        options.addOption(interestLargestEstimateOpt);
        options.addOption(interestEstimatesInCPOpt);
        options.addOption(interestBroadcastVarsOpt);
        options.addOption(interestOutputCachingOpt);
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

    public static Enumerator initEnumeratorFromArgs(CommandLine line, Options options) throws ParseException, IOException {
        if (line.hasOption("help")) {
            (new HelpFormatter()).printHelp("Main", options);
            return null;
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

        double[] regionalPrices = CloudUtils.loadRegionalPrices(regionTablePath, region);
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
        String[] instanceFamilies = line.hasOption("instanceFamilies")? line.getOptionValues("instanceFamilies") : null;
        String[] instanceSizes = line.hasOption("instanceSizes")? line.getOptionValues("instanceSizes") : null;
        // 1e: parse arguments specific to enumeration strategies
        int stepSize = 1;
        int expBase = -1;
        if (strategy == Enumerator.EnumerationStrategy.GridBased) {
            if (line.hasOption("stepSize"))
                stepSize = Integer.parseInt(line.getOptionValue("stepSize"));
            if (line.hasOption("expBase"))
                expBase = Integer.parseInt(line.getOptionValue("expBase"));
        } else {
            if (line.hasOption("stepSize"))
                System.err.println("Warning: option -stepSize is relevant only for -enum 'grid'");
            if (line.hasOption("expBase"))
                System.err.println("Warning: option -expBase is relevant only for -enum 'grid'");
        }
        boolean interestLargestEstimate = true;
        boolean interestEstimatesInCP = true;
        boolean interestBroadcastVars = true;
        boolean interestOutputCaching = false;
        if (strategy == Enumerator.EnumerationStrategy.InterestBased) {
            if (line.hasOption("useLargestEst"))
                interestLargestEstimate = Boolean.parseBoolean(line.getOptionValue("useLargestEst"));
            if (line.hasOption("useCpEstimates"))
                interestEstimatesInCP = Boolean.parseBoolean(line.getOptionValue("useCpEstimates"));
            if (line.hasOption("useBroadcasts"))
                interestBroadcastVars = Boolean.parseBoolean(line.getOptionValue("useBroadcasts"));
            if (line.hasOption("useOutputs"))
                interestOutputCaching = Boolean.parseBoolean(line.getOptionValue("useOutputs"));
        } else {
            if (line.hasOption("useLargestEst"))
                System.err.println("Warning: option -useLargestEst is relevant only for -enum 'interest'");
            if (line.hasOption("useCpEstimates"))
                System.err.println("Warning: option -useCpEstimates is relevant only for -enum 'interest'");
            if (line.hasOption("useBroadcasts"))
                System.err.println("Warning: option -useBroadcasts is relevant only for -enum 'interest'");
            if (line.hasOption("useOutputs"))
                System.err.println("Warning: option -useOutputs is relevant only for -enum 'interest'");
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
            if (instanceFamilies != null)
                builder.withInstanceFamilyRange(instanceFamilies);
        } catch (IllegalArgumentException e) {
            throw new ParseException("Not all provided options for -instanceFamilies are supported or valid. Error thrown at:\n"+e.getMessage());
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
            builder.withInterestLargestEstimate(interestLargestEstimate)
                    .withInterestEstimatesInCP(interestEstimatesInCP)
                    .withInterestBroadcastVars(interestBroadcastVars)
                    .withInterestOutputCaching(interestOutputCaching);

        }
        // build the enumerator
        return builder.build();
    }

    public static void main(String[] args) throws ParseException, IOException {
        Options options = createOptions();
        CommandLineParser clParser = new PosixParser();
        CommandLine line = clParser.parse(options, args);

        Enumerator enumerator = initEnumeratorFromArgs(line, options);
        if (enumerator == null) {
            // help requested
            return;
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

        System.out.println("Number instances to be used for enumeration: " + enumerator.getInstances().size());
        System.out.println("All options are set! Enumeration is now running...");
        // step 4: pre-processing (generating search space according to the enumeration strategy)
        long startTime = System.currentTimeMillis();
        enumerator.preprocessing();
        // step 5: processing (finding the optimal solution) + postprocessing (retrieving the solution)
        enumerator.processing();
        EnumerationUtils.SolutionPoint optConfig = enumerator.postprocessing();
        long endTime = System.currentTimeMillis();
        System.out.println("...enumeration finished for " + ((double) (endTime-startTime))/1000 + " seconds\n");
        // step 6: generate configuration files according the optimal solution (if solution not empty)
        if (optConfig.getTimeCost() < Double.MAX_VALUE) {
            if (optConfig.numberExecutors == 0) {
                String filePath = Paths.get(outputPath, EC2_ARGUMENTS_FILENAME).toString();
                CloudUtils.generateEC2ConfigJson(optConfig.driverInstance, filePath);
            } else {
                String instanceGroupsPath = Paths.get(outputPath, EMR_INSTANCE_GROUP_FILENAME).toString();
                String configurationsPath = Paths.get(outputPath, EMR_CONFIGURATIONS_FILENAME).toString();
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
}
