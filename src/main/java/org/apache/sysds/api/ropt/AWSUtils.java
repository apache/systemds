package org.apache.sysds.api.ropt;

import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.io.FileWriter;
import java.io.IOException;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class AWSUtils {
    // NOTE: does NOT match additional capabilities: '-flex' and 'q'
    public static final String EC2_REGEX = "^([a-z]+)([0-9])(a|g|i?)([bdnez]*)\\.([a-z0-9]*)$";

    private static final String EMR_JAVA_HOME_JDK11 = "/usr/lib/jvm/jre-11";

    public static void generateInstancesConfigsFile(String filepath, String driverNodeType, String executorNodeType, int numberExecutorNodes) throws JSONException {
        // NOTE: For now always the driver runs on the primary node
        //  the first executors runs on a core node
        //  and the rest executors on task nodes
        int numberCoreNodes = 1;
        int numberTaskNodes = numberExecutorNodes - 1;

        JSONArray instanceConfigsArray = new JSONArray();
        Map<String, String> primaryNodeConfigMap = new HashMap<>(){{
            put("Name", "MasterInstanceGroup");
            put("InstanceGroupType", "MASTER");
            put("InstanceType", driverNodeType);
            put("InstanceCount", "1");
        }};
        instanceConfigsArray.put(primaryNodeConfigMap);

        Map<String, String> coreNodeConfigMap = new HashMap<>(){{
            put("Name", "CoreInstanceGroup");
            put("InstanceGroupType", "CORE");
            put("InstanceType", executorNodeType);
            put("InstanceCount", Integer.toString(numberCoreNodes));
        }};
        instanceConfigsArray.put(coreNodeConfigMap);

        if (numberTaskNodes > 0) {
            Map<String, String> taskNodeConfigMap = new HashMap<>(){{
                put("Name", "TaskInstanceGroup");
                put("InstanceGroupType", "TASK");
                put("InstanceType", executorNodeType);
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

    public static void generateSparkConfigsFile(String filepath) throws JSONException {
//        Map<String, String> propertiesSparkDefaults = new HashMap<>(){{
//            put("spark.driver.cores", "SomeNumber");
//            put("spark.driver.memory", "SomeNumber");
//            put("spark.driver.memoryOverheadFactor", "SomeNumber");
//            put("spark.executor.instances", "SomeNumber");
//            put("spark.executor.cores", "SomeNumber");
//            put("spark.executor.memory", "SomeNumber");
//            put("spark.executor.memoryOverheadFactor", "SomeNumber");
//            put("spark.default.parallelism", "SomeNumber");
//        }};

        JSONObject javaHomeObject = new JSONObject() {{
            put("Classification", "export");
            put("Properties", new JSONObject() {{
                put("JAVA_HOME", EMR_JAVA_HOME_JDK11);
            }});
        }};
        JSONArray javaHomeArray =  new JSONArray();
        javaHomeArray.add(javaHomeObject);
        JSONObject sparkEnvObject = new JSONObject() {{
            put("Classification", "spark-env");
            put("Configurations", javaHomeArray);

        }};

        JSONArray configurationArray = new JSONArray() {{
            //put(propertiesSparkDefaults);
            put(sparkEnvObject);
        }};

        try {
            FileWriter jsonFileWriter = new FileWriter(filepath);
            configurationArray.write(jsonFileWriter);
        } catch (IOException e) {
            throw new RuntimeException();
        }
    }
}
