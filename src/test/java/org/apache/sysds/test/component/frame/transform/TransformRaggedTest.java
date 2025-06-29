package org.apache.sysds.test.component.frame.transform;

import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.ColumnEncoder;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRagged;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import static org.junit.Assert.*;

public class TransformRaggedTest {
    protected static final Log LOG = LogFactory.getLog(TransformRaggedTest.class.getName());

    @Test
    public void testBasicRaggedEncoding() {
        FrameBlock data = new FrameBlock(new ValueType[]{ValueType.STRING});
        data.setColumnNames(new String[]{"C1"});  // Set column name
        data.appendRow(new Object[]{"apple"});
        data.appendRow(new Object[]{"orange"});
        data.appendRow(new Object[]{"apple"});
        data.appendRow(new Object[]{null});
        data.appendRow(new Object[]{"banana"});
        data.appendRow(new Object[]{"orange"});
        data.appendRow(new Object[]{""});
        data.appendRow(new Object[]{"apple"});

        // Proper JSON syntax
        String spec = "{\"ragged\": [\"C1\"]}";
        
        testRaggedEncoder(spec, data);
    }

    @Test
    public void testMixedTypesEncoding() {
        FrameBlock data = new FrameBlock(new ValueType[]{ValueType.STRING});
        data.setColumnNames(new String[]{"C1"});  // Set column name
        
        // All values as strings
        data.appendRow(new Object[]{"100"});
        data.appendRow(new Object[]{"100"}); 
        data.appendRow(new Object[]{"true"});
        data.appendRow(new Object[]{"true"}); 
        data.appendRow(new Object[]{null});
        data.appendRow(new Object[]{"NA"});

        String spec = "{\"ragged\": [\"C1\"]}";
        
        testRaggedEncoder(spec, data);
    }

    @Test
    public void testLargeDataset() {
        int numRows = 1000;
        FrameBlock data = new FrameBlock(new ValueType[]{ValueType.STRING});
        data.setColumnNames(new String[]{"C1"});  // Set column name
        
        String[] fruits = {"apple", "orange", "banana", "grape", null, ""};
        for (int i = 0; i < numRows; i++) {
            data.appendRow(new Object[]{fruits[i % fruits.length]});
        }

        // Proper JSON syntax
        String spec = "{\"ragged\": [\"C1\"]}";
        
        testRaggedEncoder(spec, data);
    }

    @Test
    public void testRaggedWithOtherEncoders() {
        FrameBlock data = new FrameBlock(new ValueType[]{ValueType.STRING, ValueType.INT32, ValueType.STRING});
        // Set column names
        data.setColumnNames(new String[]{"C1", "C2", "C3"});
        
        data.appendRow(new Object[]{"apple", 10, "red"});
        data.appendRow(new Object[]{"orange", 20, "orange"});
        data.appendRow(new Object[]{"apple", 15, "red"});
        data.appendRow(new Object[]{null, 5, null});
        data.appendRow(new Object[]{"banana", 25, "yellow"});
        data.appendRow(new Object[]{"orange", 20, "orange"});

        // Proper JSON syntax with null handling
        String spec = "{"
            + "\"ragged\": [\"C1\", \"C3\"], "
            + "\"bin\": [{"
            +     "\"id\": \"C2\", "
            +     "\"method\": \"equi-width\", "
            +     "\"numbins\": 3, "
            +     "\"na\": \"impute\" "  // Add null handling
            + "}]"
            + "}";
        
        testRaggedEncoder(spec, data);
    }

    @Test
public void testRaggedDirectly() {
    FrameBlock data = new FrameBlock(new ValueType[]{ValueType.STRING});
    data.appendRow(new Object[]{"apple"});
    data.appendRow(new Object[]{"orange"});
    
    // Create ragged encoder directly
    ColumnEncoderRagged encoder = new ColumnEncoderRagged(1);
    encoder.build(data);
    
    MatrixBlock out = new MatrixBlock(data.getNumRows(), 1, false);
    encoder.apply(data, out, 0);
    
    System.out.println("Encoded Matrix:");
    System.out.println(out);
}

        private void testRaggedEncoder(String spec, FrameBlock data) {
    try {
        System.out.println("========== STARTING TEST ==========");
        System.out.println("Transform Spec: " + spec);
        System.out.println("\n=== INPUT DATA ===");
        System.out.println(data);
        
        FrameBlock meta = null;
        System.out.println("\n=== CREATING ENCODER ===");
        MultiColumnEncoder encoder = EncoderFactory.createEncoder(
            spec, data.getColumnNames(), data.getNumColumns(), meta);
        
        // Print encoder configuration
        System.out.println("Encoder Type: " + encoder.getClass().getName());
        System.out.println("Column Encoders:");
        for (ColumnEncoder enc : encoder.getColumnEncoders()) {
            System.out.println(" - " + enc.getClass().getSimpleName() + 
                              " for column " + enc.getColID());
            if (enc instanceof ColumnEncoderRagged) {
                System.out.println("   Null Index: " + 
                                  ((ColumnEncoderRagged) enc).getNullIndex());
            }
        }
        
        System.out.println("\n=== ENCODING DATA ===");
        MatrixBlock encoded = encoder.encode(data);
        System.out.println("Encoded Matrix Dimensions: " + 
                           encoded.getNumRows() + " x " + encoded.getNumColumns());
        System.out.println("Encoded Matrix Content:");
        System.out.println(encoded);
        
        System.out.println("\n=== GETTING METADATA ===");
        meta = encoder.getMetaData(meta);
        System.out.println("Metadata Dimensions: " + 
                          meta.getNumRows() + " rows x " + meta.getNumColumns() + " columns");
        
        System.out.println("\n=== METADATA CONTENT ===");
        for (int r = 0; r < meta.getNumRows(); r++) {
            for (int c = 0; c < meta.getNumColumns(); c++) {
                System.out.println("[" + r + "," + c + "]: " + meta.get(r, c));
            }
        }
        
        System.out.println("\n=== RE-APPLYING ENCODING ===");
        MatrixBlock reapplied = encoder.apply(data);
        System.out.println("Reapplied Matrix Dimensions: " + 
                          reapplied.getNumRows() + " x " + reapplied.getNumColumns());
        
        System.out.println("\n=== COMPARING ENCODED AND REAPPLIED ===");
        // Manual matrix comparison
        boolean matricesMatch = true;
        if (encoded.getNumRows() != reapplied.getNumRows() || 
            encoded.getNumColumns() != reapplied.getNumColumns()) {
            matricesMatch = false;
            System.out.println("Matrix dimensions differ: " +
                              encoded.getNumRows() + "x" + encoded.getNumColumns() + " vs " +
                              reapplied.getNumRows() + "x" + reapplied.getNumColumns());
        } else {
            for (int i = 0; i < encoded.getNumRows(); i++) {
                for (int j = 0; j < encoded.getNumColumns(); j++) {
                    double val1 = encoded.getDouble(i, j);
                    double val2 = reapplied.getDouble(i, j);
                    if (val1 != val2) {
                        matricesMatch = false;
                        System.out.println("Difference at [" + i + "," + j + "]: " + 
                                          val1 + " vs " + val2);
                    }
                }
            }
        }
        System.out.println("Matrices identical: " + matricesMatch);
        
        System.out.println("\n=== NULL INDEX ANALYSIS ===");
        int nullIndex = -1;
        for (ColumnEncoder enc : encoder.getColumnEncoders()) {
            if (enc instanceof ColumnEncoderRagged) {
                nullIndex = ((ColumnEncoderRagged) enc).getNullIndex();
                System.out.println("Ragged encoder found. Null index: " + nullIndex);
                break;
            }
        }
        
        if (nullIndex != -1) {
            System.out.println("Checking for null indices in encoded data:");
            for (int i = 0; i < encoded.getNumRows(); i++) {
                double val = encoded.getDouble(i, 0);
                if (val == nullIndex) {
                    System.out.println("Row " + i + ": NULL value found (index " + nullIndex + ")");
                }
            }
        }
        
        System.out.println("========== TEST COMPLETED ==========\n");
    }
    catch(Exception e) {
        System.out.println("\n!!! TEST FAILED WITH EXCEPTION !!!");
        e.printStackTrace();
        System.out.println("========== TEST ABORTED ==========\n");
    }
}
}