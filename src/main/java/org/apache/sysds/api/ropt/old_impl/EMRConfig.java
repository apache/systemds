package org.apache.sysds.api.ropt.old_impl;

import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;


public class EMRConfig {
    private static final String INSTANCE_CONFIG_NAME = "instances.json";
    private static final String SPARK_CONFIG_NAME = "configurations.json";

    private final String primaryNodeType;
    private final String coreNodeType;
    private final int numberCoreNodes;
    private final String taskNodeType;
    private final int numberTaskNodes;

    private final String JAVA_HOME = "/usr/lib/jvm/jre-11";

    public EMRConfig(String primaryNodeType, String coreNodeType, int numberCoreNodes, String taskNodeType, int numberTaskNodes) {
        this.primaryNodeType = primaryNodeType;
        this.numberCoreNodes = numberCoreNodes;
        if (numberCoreNodes < 1) {
            throw new RuntimeException("Minimum one core instance is required");
        }
        this.coreNodeType = coreNodeType;
        this.numberTaskNodes = numberTaskNodes;
        if (numberTaskNodes > 0) {
            this.taskNodeType = taskNodeType;
        } else {
            // for consistency
            this.taskNodeType = "";
        }
    }

    public EMRConfig(CloudClusterConfig cc) {
        primaryNodeType = cc.getManagingInstance().getInstanceType();
        numberCoreNodes = 1;
        coreNodeType = cc.getCpInstance().getInstanceType();
        numberTaskNodes = cc.getSpGroupSize();
        if (numberTaskNodes > 0) {
            taskNodeType = cc.getSpGroupInstance().getInstanceType();
        } else {
            // for consistency
            this.taskNodeType = "";
        }
    }

    private void generateInstancesConfigsFile(String filepath) throws JSONException {
        JSONArray instanceConfigsArray = new JSONArray();
        Map<String, String> primaryNodeConfigMap = new HashMap<>(){{
            put("Name", "MasterInstanceGroup");
            put("InstanceGroupType", "MASTER");
            put("InstanceType", primaryNodeType);
            put("InstanceCount", "1");
        }};
        instanceConfigsArray.put(primaryNodeConfigMap);

        Map<String, String> coreNodeConfigMap = new HashMap<>(){{
            put("Name", "CoreInstanceGroup");
            put("InstanceGroupType", "CORE");
            put("InstanceType", coreNodeType);
            put("InstanceCount", Integer.toString(numberCoreNodes));
        }};
        instanceConfigsArray.put(coreNodeConfigMap);

        if (numberTaskNodes > 0) {
            Map<String, String> taskNodeConfigMap = new HashMap<>(){{
                put("Name", "TaskInstanceGroup");
                put("InstanceGroupType", "TASK");
                put("InstanceType", taskNodeType);
                put("InstanceCount", Integer.toString(numberTaskNodes));
            }};
            instanceConfigsArray.put(taskNodeConfigMap);
        }

        try {
            FileWriter jsonFileWriter = new FileWriter(filepath);
            instanceConfigsArray.write(jsonFileWriter);
        } catch (IOException e) {
            throw new RuntimeException();
        }
    }

    private void generateSparkConfigsFile(String filepath) throws JSONException {
        Map<String, String> propertiesSparkDefaults = new HashMap<>(){{
            put("spark.driver.cores", "SomeNumber");
            put("spark.driver.memory", "SomeNumber");
            put("spark.driver.memoryOverheadFactor", "SomeNumber");
            put("spark.executor.instances", "SomeNumber");
            put("spark.executor.cores", "SomeNumber");
            put("spark.executor.memory", "SomeNumber");
            put("spark.executor.memoryOverheadFactor", "SomeNumber");
            put("spark.default.parallelism", "SomeNumber");
        }};

        JSONObject javaHomeObject = new JSONObject() {{
            put("Classification", "export");
            put("Properties", new JSONObject() {{
                put("JAVA_HOME", JAVA_HOME);
            }});
        }};
        JSONArray javaHomeArray =  new JSONArray();
        javaHomeArray.add(javaHomeObject);
        JSONObject sparkEnvObject = new JSONObject() {{
            put("Classification", "spark-env");
            put("Configurations", javaHomeArray);

        }};

        JSONArray configurationArray = new JSONArray() {{
            put(propertiesSparkDefaults);
            put(sparkEnvObject);
        }};

        try {
            FileWriter jsonFileWriter = new FileWriter(filepath);
            configurationArray.write(jsonFileWriter);
        } catch (IOException e) {
            throw new RuntimeException();
        }
    }

    public void generateConfigsFiles(String outputDir) {
        Path directory = Paths.get(outputDir);
        Path filePathInstanceConfig = directory.resolve(INSTANCE_CONFIG_NAME);
        Path filePathSparkConfig = directory.resolve(SPARK_CONFIG_NAME);
        try {
            generateInstancesConfigsFile(filePathInstanceConfig.toString());
            generateSparkConfigsFile(filePathSparkConfig.toString());
        } catch (JSONException e) {
            throw new RuntimeException("Generating JSON configs failed");
        }
    }

    /**
     * Testing only
     */
    public static void main(String[] args) throws JSONException {
        EMRConfig configs = new EMRConfig("m5.xlarge", "m5.xlarge", 1, "m5.xlarge", 0);

        configs.generateInstancesConfigsFile("/Users/lachezarnikolov/my_projects/thesis/systemds/scripts/ropt/instances.json");
        configs.generateSparkConfigsFile("/Users/lachezarnikolov/my_projects/thesis/systemds/scripts/ropt/configurations.json");
    }

}
