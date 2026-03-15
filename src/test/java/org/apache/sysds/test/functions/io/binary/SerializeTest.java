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

package org.apache.sysds.test.functions.io.binary;

import com.google.crypto.tink.subtle.Random;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.data.DenseBlockFP64DEDUP;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderComposite;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderUDF;
import org.apache.sysds.runtime.util.LocalFileUtils;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.Collections;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;

public class SerializeTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "SerializeTest";
	private final static String TEST_DIR = "functions/io/binary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + SerializeTest.class.getSimpleName() + "/";
	
	public static int rows1 = 746;
	public static int cols1 = 586;
	public static int cols2 = 4;
	
	private final static double eps = 1e-14;

	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "X" }) );  
	}
	
	@Test
	public void testEmptyBlock() 
	{ 
		runSerializeTest( rows1, cols1, 0.0 ); 
	}
	
	@Test
	public void testDenseBlock() 
	{ 
		runSerializeTest( rows1, cols1, 1.0 ); 
	}
	@Test
	public void testDedupDenseBlock()
	{
		runSerializeDedupDenseTest( rows1, cols1 );
	}


	@Test
	public void testDenseSparseBlock() 
	{ 
		runSerializeTest( rows1, cols2, 0.3 ); 
	}
	
	@Test
	public void testDenseUltraSparseBlock() 
	{ 
		runSerializeTest( rows1, cols2, 0.1 ); 
	}
	
	@Test
	public void testSparseBlock() 
	{ 
		runSerializeTest( rows1, cols1, 0.1 ); 
	}
	
	@Test
	public void testSparseUltraSparseBlock() 
	{ 
		runSerializeTest( rows1, cols1, 0.0001 ); 
	}

	@Test
	public void testWEEncoderSerialization(){
		runSerializeWEEncoder();
	}

	@Test
	public void testUDFEncoderSerialization(){
		runSerializeUDFEncoder();
	}

	private void runSerializeTest( int rows, int cols, double sparsity ) 
	{
		try
		{	
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("X"), output("X") };
	
			//generate actual dataset 
			double[][] X = getRandomMatrix(rows, cols, -1.0, 1.0, sparsity, 7); 
			MatrixBlock mb = DataConverter.convertToMatrixBlock(X);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, 1000, 1000);
			DataConverter.writeMatrixToHDFS(mb, input("X"), FileFormat.BINARY, mc);
			HDFSTool.writeMetaDataFile(input("X.mtd"), ValueType.FP64, mc, FileFormat.BINARY);
			
			runTest(true, false, null, -1); //mult 7
			
			//compare matrices 
			MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(output("X"), FileFormat.BINARY, rows, cols, 1000, 1000);
			for( int i=0; i<mb.getNumRows(); i++ )
				for( int j=0; j<mb.getNumColumns(); j++ )
				{
					double val1 = mb.get(i, j) * 7;
					double val2 = mb2.get(i, j);
					Assert.assertEquals(val1, val2, eps);
				}
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}

	private void runSerializeWEEncoder(){
		try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
			 ObjectOutput out = new ObjectOutputStream(bos))
		{
			double[][] X = getRandomMatrix(5, 100, -1.0, 1.0, 1.0, 7);
			MatrixBlock emb = DataConverter.convertToMatrixBlock(X);
			FrameBlock data = DataConverter.convertToFrameBlock(new String[][]{{"A"}, {"B"}, {"C"}});
			FrameBlock meta = DataConverter.convertToFrameBlock(new String[][]{{"A" + Lop.DATATYPE_PREFIX + "1"},
					{"B" + Lop.DATATYPE_PREFIX + "2"},
					{"C" + Lop.DATATYPE_PREFIX + "3"}});
			MultiColumnEncoder encoder = EncoderFactory.createEncoder(
					"{ids:true, word_embedding:[1]}", data.getColumnNames(), meta.getSchema(), meta, emb);

			// Serialize the object
			encoder.writeExternal(out);
			out.flush();

			// Deserialize the object
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			ObjectInput in = new ObjectInputStream(bis);
			MultiColumnEncoder encoder_ser = new MultiColumnEncoder();
			encoder_ser.readExternal(in);
			in.close();
			MatrixBlock mout = encoder_ser.apply(data);
			for (int i = 0; i < mout.getNumRows(); i++) {
				for (int j = 0; j < mout.getNumColumns(); j++) {
					assert mout.get(i, j) == X[i][j];
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	private void runSerializeUDFEncoder(){
		try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
			 ObjectOutput out = new ObjectOutputStream(bos)) {
			final String udfName = "dummyUdf";
			final int colId = 2;
			final int domainSize = 5;

			ColumnEncoderUDF udf = createUdf(colId, udfName, domainSize);
			ColumnEncoderComposite composite = new ColumnEncoderComposite(Collections.singletonList(udf));
			MultiColumnEncoder encoder = new MultiColumnEncoder(Collections.singletonList(composite));

			encoder.writeExternal(out);
			out.flush();

			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			ObjectInput in = new ObjectInputStream(bis);
			MultiColumnEncoder encoderSer = new MultiColumnEncoder();
			encoderSer.readExternal(in);
			in.close();

			ColumnEncoderComposite decodedComposite = encoderSer.getColumnEncoders().get(0);
			ColumnEncoderUDF decodedUdf = decodedComposite.getEncoder(ColumnEncoderUDF.class);

			Assert.assertNotNull(decodedUdf);
			Assert.assertEquals(colId, decodedUdf.getColID());
			Assert.assertEquals(domainSize, decodedUdf._domainSize);
			Assert.assertEquals(udfName, getUdfName(decodedUdf));
		}
		catch(IOException | ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	private ColumnEncoderUDF createUdf(int colId, String name, int domainSize) {
		try {
			Constructor<ColumnEncoderUDF> ctor = ColumnEncoderUDF.class.getDeclaredConstructor(int.class, String.class);
			ctor.setAccessible(true);
			ColumnEncoderUDF udf = ctor.newInstance(colId, name);
			udf._domainSize = domainSize;
			return udf;
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	private String getUdfName(ColumnEncoderUDF udf) {
		try {
			Field f = ColumnEncoderUDF.class.getDeclaredField("_fName");
			f.setAccessible(true);
			return (String) f.get(udf);
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	private void runSerializeDedupDenseTest( int rows, int cols )
	{
		try
		{
			//generate actual dataset
			double[][] X = getRandomMatrix(rows, cols, -1.0, 1.0, 1.0, 7);
			double[][] X_duplicated = new double[rows*10][];
			MatrixBlock mb = new MatrixBlock(rows*10, cols, false, 0, true);
			mb.allocateDenseBlock(true, true);
			DenseBlockFP64DEDUP dedup = (DenseBlockFP64DEDUP) mb.getDenseBlock();
			dedup.setEmbeddingSize(cols);
			HashMap<double[], Integer > seen = new HashMap<>();
			for (int i = 0; i < rows*10; i++) {
				int row = Random.randInt(rows);
				Integer tmpPos = seen.get(X[row]);
				if(tmpPos == null) {
					tmpPos = seen.size();
					seen.put(X[row], tmpPos);
				}
				X_duplicated[i] = X[row];
				dedup.setDedupDirectly(i, X[row]);
			}

			String fname = SCRIPT_DIR + TEST_DIR + "dedupSerializedBlock.out";
			LocalFileUtils.writeCacheBlockToLocal(fname, mb);
			MatrixBlock mb2 = (MatrixBlock) LocalFileUtils.readCacheBlockFromLocal(fname, true);

			//compare matrices - values
			for( int i=0; i<mb.getNumRows(); i++ )
				for( int j=0; j<mb.getNumColumns(); j++ )
				{
					double val1 = mb.get(i, j);
					double val2 = mb2.get(i, j);
					Assert.assertEquals(val1, val2, eps);
				}

			//compare matrices - values
			DenseBlockFP64DEDUP dedup2 = (DenseBlockFP64DEDUP) mb2.getDenseBlock();
			HashMap<double[], Integer > seen2 = new HashMap<>();
			for( int i=0; i<mb.getNumRows()*dedup2.getNrEmbsPerRow(); i++ ){
				double[] row = dedup2.getDedupDirectly(i);
				Integer tmpPos = seen2.get(row);
				if(tmpPos == null) {
					tmpPos = seen2.size();
					seen2.put(row, tmpPos);
				}
				Integer posMb1 = seen.get(dedup.getDedupDirectly(i));
				Assert.assertEquals( (long) tmpPos, (long) posMb1);
			}
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
