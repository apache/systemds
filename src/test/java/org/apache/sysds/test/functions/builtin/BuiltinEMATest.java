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

package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;


import static org.junit.Assert.assertTrue;


public class BuiltinEMATest extends AutomatedTestBase {

	private final static String TEST_NAME = "exponentialMovingAverage";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinOutlierTest.class.getSimpleName() + "/";
	private static final double e = 0.00001;

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp() {
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@Test
	public void CompareToAirPython() {
		Double[] data= new Double[]{112.,118.,132.,129.,121.,135.,148.,148.,136.,119.,104.,118.,115.,126.,141.,135.,125.,
			149.,170.,170.,158.,133.,114.,140.,145.,150.,178.,163.,172.,178.,199.,199.,184.,162.,146.,166.,171.,180.,193.,
			181.,183.,218.,230.,242.,209.,191.,172.,194.,196.,196.,236.,235.,229.,243.,264.,272.,237.,211.,180.,201.,204.,
			188.,235.,227.,234.,264.,302.,293.,259.,229.,203.,229.,242.,233.,267.,269.,270.,315.,364.,347.,312.,274.,237.,
			278.,284.,277.,317.,313.,318.,374.,413.,405.,355.,306.,271.,306.,315.,301.,356.,348.,355.,422.,465.,467.,404.,
			347.,305.,336.,340.,318.,362.,348.,363.,435.,491.,505.,404.,359.,310.,337.,360.,342.,406.,396.,420.,472.,548.,
			559.,463.,407.,362.,405.,417.,391.,419.,461.,472.,535.,622.,606.,508.,461.,390.,432.};
		Double[] na_ma_ref = new Double[]{112.0,118.0,132.0,129.0,121.0,135.0,148.0,148.0,136.0,119.0,104.0,118.0,
			110.64316239316241,124.65882867132864,141.63607177544674,136.7354015394328,127.17767588262429,150.30329316196824,
			161.70161586161797,169.38854499320993,161.19601881138237,145.06448246981816,122.484478767933,127.39157601488053,
			129.72591491348797,152.03705561548736,169.0315776597689,174.0450155013305,161.71883176768532,197.38664830203192,
			200.16326105914544,199.3633290787375,186.7801706766464,166.84581558009475,147.01335928013515,158.28476700133308,
			158.16728220922437,177.39919242218832,197.8931822724927,190.7538466127484,176.3701663182143,203.6815674660317,
			230.97131101134238,237.12395417864016,235.69002588652017,203.9793963562213,177.70695462791417,182.8604180621578,
			181.7019278809486,198.65873966400525,211.81123478285303,224.81149965425675,234.03965829250822,264.637902643598,
			269.337433358522,268.2236823080142,258.90098063468133,226.18596531382858,194.3171221649824,188.13753099285515,
			180.73112166348903,197.19799095036475,202.00699270410672,215.3746071328909,221.7279582439683,263.37938956281806,
			291.04092902170333,315.2526229493343,299.24296207175007,258.8843893281736,215.75701045434437,208.30972791961293,
			204.6705448924415,230.79856526946912,251.66083493325718,257.19830790310965,265.79900618102994,302.134340294941,
			340.023092988112,377.22712842120563,361.68362912145847,319.1713492927022,265.9551211952107,243.11007047239005,
			243.40147848232402,266.55060745123546,292.2235041830092,305.63559628413805,313.4173919289333,352.29010792271606,
			400.1653619571387,430.1970897847026,419.06532694160865,367.4237601305769,299.03974216212833,272.31262029667437,
			264.33527736015446,286.58031520064947,311.91458026777696,337.82852013844627,351.57523114182044,395.0206150729328,
			451.2263986714124,486.29420474135196,484.7046653181514,427.5859319836733,347.2936931627886,307.5407332157359,
			286.9831427702983,298.6001849239334,319.6590363831499,334.1457260693061,341.82615134798755,393.9656366982082,
			460.0996934540272,511.6328818051482,527.2903638248666,447.9666064300616,361.5472947970979,313.6150739283058,
			282.8820652975183,304.4380438439978,341.65102241233495,379.3461886314897,403.40100350027996,467.25353238867564,
			515.0436188516387,566.7544915874026,576.6520376922236,507.8707884571211,417.21104490644575,369.75194100231744,
			350.34762676605754,363.74055663694884,392.5832813888661,391.0929206754841,442.5756334301262,517.5411050850213,
			582.8401346087848,647.5306590463075,635.3370387229387,555.6543956799871,467.86361741316273,403.15056379128157};
		Double[][] values = new Double[][]{data};
		FrameBlock f = generateBlock(data.length, 1, values);
		runTest(f, ExecType.CP,  100, "triple", 12, 0.5, 0.7, 0.1, na_ma_ref, e);
	}

	@Test
	public void CompareToAirGap() {
		Double[] data= new Double[]{112.,118.,132.,129., 131.,135.,148.,148.,149.,119.,104.,118.,115.,126.,141.,135.,125.,
			149.,170.,170.,Double.NaN,133.,Double.NaN,140.,145.,150.,178.,163.,172.,178.,199.,199.,184.,162.,146.,166.,171.,
			180.,193.,181.,183.,218.,230.,242.,209.,191.,172.,194.,196.,196.,236.,235.,229.,243.,264.,272.,237.,211.,180.,
			201.,204.,188.,235.,227.,234.,Double.NaN,302.,293.,259.,229.,203.,229.,242.,233.,267.,269.,270.,315.,364.,347.,
			312.,274.,237.,278.,284.,277.,Double.NaN,Double.NaN,Double.NaN,374.,413.,405.,355.,306.,271.,306.,315.,301.,356.,
			348.,355.,Double.NaN,465.,467.,404.,347.,Double.NaN,336.,340.,318.,Double.NaN,348.,363.,435.,491.,505.,404.,359.,
			310.,337.,360.,342.,406.,396.,420.,472.,548.,559.,463.,407.,362.,Double.NaN,417.,391.,419.,461.,Double.NaN,535.,
			622.,606.,508.,461.,390.,432.};
		Double[] na_ma_ref = new Double[]{112.,118.,132.,129.,133.1596639,135.,148.,148.,129.8606557,119.,104.,118.,115.,126.,
			141.,135.,125.,149.,170.,170.,151.7909091,133.,144.8090909,140.,145.,150.,178.,163.,172.,178.,199.,199.,184.,162.,
			146.,166.,171.,180.,193.,181.,183.,218.,230.,242.,209.,191.,172.,194.,196.,196.,236.,235.,229.,243.,264.,272.,
			237.,211.,180.,201.,204.,188.,235.,227.,234.,256.6349206,302.,293.,259.,229.,203.,229.,242.,233.,267.,269.,270.,
			315.,364.,347.,312.,274.,237.,278.,284.,277.,298.0641026,330.4516129,362.3589744,374.,413.,405.,355.,306.,271.,
			306.,315.,301.,356.,348.,355.,396.9677419,465.,467.,404.,347.,360.95,336.,340.,318.,354.1311475,348.,363.,435.,
			491.,505.,404.,359.,310.,337.,360.,342.,406.,396.,420.,472.,548.,559.,463.,407.,362.,410.766129,417.,391.,419.,
			461.,499.016129,535.,622.,606.,508.,461.,390.,432.};
		Double[][] values = new Double[][]{data};
		FrameBlock f = generateBlock(data.length, 1, values);
		runTest(f, ExecType.CP,  100, "triple", 4, Double.NaN, Double.NaN, Double.NaN,
			na_ma_ref, 200);
	}

	@Test
	public void checkSingleRData() {
		Double[] data= new Double[]{41.7275, 24.0418, 32.3281, 37.3287, 46.2132, 29.3463, 36.4829, 42.9777, 48.9015,
			31.1802, 37.7179, 40.4202, 51.2069, 31.8872, 40.9783, 43.7725, 55.5586, 33.8509, 42.0764, 45.6423, 59.7668,
			35.1919, 44.3197, 47.9137};
		Double[] na_ma_ref = new Double[]{41.7275, 32.88465, 32.606375, 34.9675375, 40.590368749999996, 34.968334375,
			35.7256171875, 39.35165859375, 44.126579296875, 37.653389648437496, 37.68564482421875, 39.052922412109375,
			45.129911206054686, 38.50855560302735, 39.743427801513675, 41.757963900756835, 48.65828195037842, 41.25459097518921,
			41.665495487594605, 43.6538977437973, 51.71034887189865, 43.451124435949325, 43.88541221797466, 45.899556108987326};
		Double[][] values = new Double[][]{data};
		FrameBlock f = generateBlock(data.length, 1, values);
		runTest(f, ExecType.CP,  100, "single", 4, 0.5, Double.NaN, Double.NaN, na_ma_ref, e);
	}

	@Test
	public void checkDoubleRData() {
		Double[] data= new Double[]{41.7275, 24.0418, 32.3281, 37.3287, 46.2132, 29.3463, 36.4829, 42.9777, 48.9015,
			31.1802, 37.7179, 40.4202, 51.2069, 31.8872, 40.9783, 43.7725, 55.5586, 33.8509, 42.0764, 45.6423, 59.7668,
			35.1919, 44.3197, 47.9137};
		Double[] na_ma_ref = new Double[]{41.7275, 6.356099999999998, 8.149399999999998, 18.841175, 35.47231874999999,
			33.82293593749999, 37.231535546874994, 43.619776464843746, 51.09622780761718, 40.99479652709961, 38.39370675506592,
			38.95093518028259, 47.68689059782028, 38.44509565713406, 39.003049272507425, 41.87148876206725, 52.620536316330345,
			42.448801014379306, 41.38258310980896, 43.697353380071554, 55.93435017018497, 44.57978602269542, 43.40138244327679,
			45.73726004274828};
		Double[][] values = new Double[][]{data};
		FrameBlock f = generateBlock(data.length, 1, values);
		runTest(f, ExecType.CP,  100, "double", 4, 0.5, 0.5, Double.NaN, na_ma_ref, e);
	}

	@Test
	public void checkTripleRData() {
		Double[] data= new Double[]{41.7275, 24.0418, 32.3281, 37.3287, 46.2132, 29.3463, 36.4829, 42.9777, 48.9015, 31.1802,
			37.7179, 40.4202, 51.2069, 31.8872, 40.9783, 43.7725, 55.5586, 33.8509, 42.0764, 45.6423, 59.7668, 35.1919,
			44.3197, 47.9137};
		Double[] na_ma_ref = new Double[]{41.7275, 24.0418, 32.3281, 37.3287, 43.73937749999999, 28.710109375, 37.62691046875001,
			42.5309708984375, 52.88580151367188, 33.15671534912109, 38.65758862487794, 42.44805077255249, 48.22574473186493,
			31.990608629751197, 38.66479375649214, 44.60066332877576, 53.321224908427446, 36.84788508769925, 42.86094393343228,
			45.86929170959124, 55.2140372941347, 38.72586172952057, 45.025358876239984, 48.23407483458373};
		Double[][] values = new Double[][]{data};
		FrameBlock f = generateBlock(data.length, 1, values);
		runTest(f, ExecType.CP,  100, "triple", 4, 0.5, 0.5, 0.5, na_ma_ref, e);
	}

	private static double calcRMSE(Double[] list1, Double[] list2) {
		double sum = 0;
		for(int i =0; i< list1.length; i++)
			sum += Math.pow(list1[i] - list2[i], 2);
		return Math.sqrt(sum / list1.length);
	}

	private void runTest(FrameBlock test_frame, ExecType et, Integer search_iterations, String mode, Integer freq,
		Double alpha, Double beta, Double gamma, Double[] reference, double max_error)
	{
		Types.ExecMode platformOld = setExecMode(et);

		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "F=" + input("F"), "O=" + output("O"), "search_iterations=" +
				search_iterations, "mode=" + mode, "freq=" + freq, "alpha=" + alpha, "beta=" + beta, "gamma=" + gamma};

			FrameWriterFactory.createFrameWriter(Types.FileFormat.CSV).
					writeFrameToHDFS(test_frame, input("F"), test_frame.getNumRows(), test_frame.getNumColumns());

			runTest(true, false, null, -1);

			FrameBlock outputFrame = readDMLFrameFromHDFS("O", Types.FileFormat.CSV);
			String[] values = (String[]) outputFrame.getColumnData(0);
			Double[] data = new Double[values.length];
			for (int i = 0; i< values.length; i++) data[i] = Double.valueOf(values[i]);


			assertTrue(calcRMSE(data, reference) <= max_error);
		}
		catch (Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private static FrameBlock generateBlock(int rows, int cols, Double[][] values)
	{
		Types.ValueType[] schema = new Types.ValueType[cols];
		for(int i = 0; i < cols; i++) {
			schema[i] = Types.ValueType.FP64;
		}

		String[] names = new String[cols];
		for(int i = 0; i < cols; i++)
			names[i] = schema[i].toString();
		FrameBlock frameBlock = new FrameBlock(schema, names);
		frameBlock.ensureAllocatedColumns(rows);
		for(int row = 0; row < rows; row++)
			for(int col = 0; col < cols; col++)
				frameBlock.set(row, col, values[col][row]);
		return frameBlock;
	}
}
