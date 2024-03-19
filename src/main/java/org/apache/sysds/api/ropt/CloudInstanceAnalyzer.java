package org.apache.sysds.api.ropt;

import scala.xml.dtd.ValidationException;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CloudInstanceAnalyzer {

    public static class InstanceTypeSpecification {
        public final String family;
        public final int generation;
        public final String processor;
        public final String capabilities;
        public final String size;

        public InstanceTypeSpecification(String instanceTypeName) throws ValidationException {
            Pattern pattern = Pattern.compile(EMRUtils.EC2_REGEX);
            Matcher matcher = pattern.matcher(instanceTypeName);
            if (matcher.matches()) {
                family = matcher.group(1);
                generation = Integer.parseInt(matcher.group(2));
                processor = matcher.group(3);
                capabilities = matcher.group(4);
                size = matcher.group(5);
            } else {
                throw new ValidationException(instanceTypeName + " is not a valid instance type name.");
            }
        }
    }
    private static String INSTANCE_TABLE_PATH = System.getProperty("user.dir") + "/scripts/ropt/ec2_types.csv";
    private static HashMap<String, CloudInstanceConfig> infoTable = null;

    public CloudInstanceAnalyzer() throws IOException {
        loadInstanceInfoTable(EMRUtils.setOfCurrentGenEC2(), Double.MAX_VALUE, "", "", "", "");
    }

    public CloudInstanceAnalyzer(double maxInstancePrice) throws IOException {
        loadInstanceInfoTable(EMRUtils.setOfCurrentGenEC2(), maxInstancePrice, "", "", "", "");
    }

    /**
     * Hints for regex patterns:
     *  - '(p|g)' - p or g family;
     *  - '^$' - not additional capabilities
     *  -
     * @param familyIds: list of desired instance family ids to match
     * @param maxPrice: maximum price per instance per hour
     * @param familyPattern: regex to match desired families
     *                     Hint: '(p|g)' - p or g family
     * @param processorPattern: regex to match desired processors:
     *                        Hint: '^i?$' - for Intel only (Empty string stands also for intel for older generations)
     * @param generationPattern: regex to match instance generations
     *                         Hint: '^(?!7).' - excluding generation 7
     * @param capabilitiesPattern:regex to match desired additional capabilities
     *                           Hint: '^$' - not additional capabilities
     * @throws IOException
     */
    private void loadInstanceInfoTable(Set<EMRUtils.InstanceFamilyId> familyIds, double maxPrice, String familyPattern, String processorPattern, String generationPattern, String capabilitiesPattern) throws IOException {
        infoTable = new HashMap<>();
        int lineCount = 1;
        // try to open the file
        BufferedReader br = new BufferedReader(new FileReader(INSTANCE_TABLE_PATH));
        String parsedLine;
        // validate the file header
        parsedLine = br.readLine();
        if (!parsedLine.equals("API_Name,Memory,vCPUs,Family,Price"))
            throw new IOException("Invalid CSV header inside: " + INSTANCE_TABLE_PATH);

        Pattern[] filterPatterns = new Pattern[]{
                Pattern.compile(familyPattern),
                Pattern.compile(processorPattern),
                Pattern.compile(generationPattern),
                Pattern.compile(capabilitiesPattern)
        };

        while ((parsedLine = br.readLine()) != null) {
            String[] values = parsedLine.split(",");
            if (values.length != 5)
                throw new IOException(String.format("Invalid CSV line(%d) inside: %s", lineCount, INSTANCE_TABLE_PATH));
            EMRUtils.InstanceFamilyId currentFamilyId = EMRUtils.InstanceFamilyId.valueOf(values[3]);
            // filter the instance types to load
            if (familyIds != null && !familyIds.contains(currentFamilyId)) {
                continue;
            }
            float currentPrice = Float.parseFloat(values[4]);
            if (currentPrice > maxPrice) {
                continue;
            }
            InstanceTypeSpecification currentTypeSpec = null;
            try {
                currentTypeSpec = new InstanceTypeSpecification(values[0]);
            } catch (ValidationException e) {
                // ignore not recognized instances
                continue;
            }
            if (!familyPattern.equals("")) {
                if (!filterPatterns[0].matcher(currentTypeSpec.family).matches())
                    continue;
            }
            if (!generationPattern.equals("")) {
                if (!filterPatterns[1].matcher(currentTypeSpec.processor).matches())
                    continue;
            }
            if (!generationPattern.equals("")) {
                if (!filterPatterns[2].matcher(Integer.toString(currentTypeSpec.generation)).matches())
                    continue;
            }
            if (!capabilitiesPattern.equals("")) {
                if (!filterPatterns[3].matcher(currentTypeSpec.capabilities).matches())
                    continue;
            }
            infoTable.put(values[0], new CloudInstanceConfig(
                    values[0],                   // instanceType
                    1024*((long) Double.parseDouble(values[1])), // memory
                    Integer.parseInt(values[2]), // vCPUS
                    Double.parseDouble(values[4])  // price
            ));
        }
    }

    public String[] getListSupportedInstances() {
        return infoTable.keySet().toArray(new String[0]);
    }

    public CloudInstanceConfig getValueFromInfoTable(String key) {
        return infoTable.get(key);
    }

    public ArrayList<CloudInstanceConfig> getListInstances() {
        return new ArrayList<>(infoTable.values());
    }

    /**
     * Sorting the list values starting from the one with the lowest memory.
     * In case of equal memory, the prior one is the one with less virtual CPU cores.
     *
     * @return sorted list of {@link CloudInstanceConfig}
     */
    public ArrayList<CloudInstanceConfig> getListInstancesSorted() {
        ArrayList<CloudInstanceConfig> instanceList = getListInstances();
        instanceList.sort((o1, o2) -> {
            if (o1.getAvailableMemoryMB() > o2.getAvailableMemoryMB()) {
                return 1;
            } else if (o1.getAvailableMemoryMB() < o2.getAvailableMemoryMB()) {
                return -1;
            } else /* equal memory */ {
                return Integer.compare(o1.getVCPUCores(), o2.getVCPUCores());
            }
        });

        return instanceList;
    }
}
