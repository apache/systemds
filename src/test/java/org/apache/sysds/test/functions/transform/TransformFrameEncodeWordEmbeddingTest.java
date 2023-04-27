package org.apache.sysds.test.functions.transform;

        //import org.apache.sysds.common.Types;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class TransformFrameEncodeWordEmbeddingTest extends AutomatedTestBase
{
    private final static String TEST_NAME1 = "TransformFrameEncodeWordEmbeddings";
    private final static String TEST_DIR = "functions/transform/";
    private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeWordEmbeddingTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME1,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "result" }) );
    }

    @Test
    public void testTransformToWordEmbeddings() {
        runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE);
    }

    private void runTransformTest(String testname, ExecMode rt)
    {
        //set runtime platform
        ExecMode rtold = setExecMode(rt);
        try
        {
            int rows = 100;
            int cols = 100;
            TestConfiguration config = getTestConfiguration(testname); //availableTestConfigurations.get(testname);
            config.addVariable("rows", rows);
            config.addVariable("cols", cols);
            loadTestConfiguration(config);

            double[][] a = createRandomMatrix("a", rows, cols, 0, 10, 1, new Date().getTime());

            List<String> strings = generateRandomStrings(rows, 10);
            Map<String,Integer> map = writeDictToCsvFile(strings, baseDirectory + INPUT_DIR + "c");

            List<String> stringsColumn = shuffleAndMultiplyStrings(strings, 10);
            writeStringsToCsvFile(stringsColumn, baseDirectory + INPUT_DIR + "b");

            runTest(false);

            double[][] result = new double[stringsColumn.size()][cols];
            for (int i = 0; i < stringsColumn.size(); i++) {
                int rowMapped = map.get(stringsColumn.get(i));
                System.arraycopy(a[rowMapped], 0, result[i], 0, cols);
            }

            writeExpectedMatrix("result", result);
            compareResults();
        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
        finally {
            resetExecMode(rtold);
        }
    }

    public static List<String> shuffleAndMultiplyStrings(List<String> strings, int multiply){
        List<String> out = new ArrayList<>();
        Random random = new Random();
        for (int i = 0; i < strings.size()*multiply; i++) {
            out.add(strings.get(random.nextInt(strings.size())));
        }
        return out;
    }

    public static List<String> generateRandomStrings(int numStrings, int stringLength) {
        List<String> randomStrings = new ArrayList<>();
        Random random = new Random();
        String characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        for (int i = 0; i < numStrings; i++) {
            randomStrings.add(generateRandomString(random, stringLength, characters));
        }
        return randomStrings;
    }

    public static String generateRandomString(Random random, int stringLength, String characters){
        StringBuilder randomString = new StringBuilder();
        for (int j = 0; j < stringLength; j++) {
            int randomIndex = random.nextInt(characters.length());
            randomString.append(characters.charAt(randomIndex));
        }
        return randomString.toString();
    }

    public static void writeStringsToCsvFile(List<String> strings, String fileName) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))) {
            for (String line : strings) {
                bw.write(line);
                bw.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Map<String,Integer> writeDictToCsvFile(List<String> strings, String fileName) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))) {
            Map<String,Integer> map = new HashMap<>();
            for (int i = 0; i < strings.size(); i++) {
                map.put(strings.get(i), i);
                bw.write(strings.get(i) + Lop.DATATYPE_PREFIX + (i+1) + "\n");
            }
            return map;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
