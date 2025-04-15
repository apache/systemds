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

package org.apache.sysds.test.component.federated;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.hops.fedplanner.FederatedPlanCostEnumerator;
import org.apache.sysds.utils.TeeOutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.File;

public class FederatedPlanCostEnumeratorTest extends AutomatedTestBase
{
	private static final String TEST_DIR = "functions/federated/privacy/";
	private static final String HOME = SCRIPT_DIR + TEST_DIR;
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedPlanCostEnumeratorTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {}

	@Test
	public void testFederatedPlanCostEnumerator1() { runTest("FederatedPlanCostEnumeratorTest1.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator2() { runTest("FederatedPlanCostEnumeratorTest2.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator3() { runTest("FederatedPlanCostEnumeratorTest3.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator4() { runTest("FederatedPlanCostEnumeratorTest4.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator5() { runTest("FederatedPlanCostEnumeratorTest5.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator6() { runTest("FederatedPlanCostEnumeratorTest6.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator7() { runTest("FederatedPlanCostEnumeratorTest7.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator8() { runTest("FederatedPlanCostEnumeratorTest8.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator9() { runTest("FederatedPlanCostEnumeratorTest9.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator10() { runTest("FederatedPlanCostEnumeratorTest10.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator11() { runTest("FederatedPlanCostEnumeratorTest11.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator12() { runTest("FederatedPlanCostEnumeratorTest12.dml"); }

	private void runTest(String scriptFilename) {
		int index = scriptFilename.lastIndexOf(".dml");
		String testName = scriptFilename.substring(0, index > 0 ? index : scriptFilename.length());
		TestConfiguration testConfig = new TestConfiguration(TEST_CLASS_DIR, testName, new String[] {});
		addTestConfiguration(testName, testConfig);
		loadTestConfiguration(testConfig);
		
		try {
			DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
			ConfigurationManager.setLocalConfig(conf);
			
			//read script
			String dmlScriptString = DMLScript.readDMLScript(true, HOME + scriptFilename);
		
			//parsing and dependency analysis
			ParserWrapper parser = ParserFactory.createParser();
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, new HashMap<>());
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);
			dmlt.rewriteHopsDAG(prog);

			// 출력을 파일과 터미널 모두에 저장
			String outputFile = testName + "_trace.txt";
			File outputFileObj = new File(outputFile);
			System.out.println("[INFO] Trace 파일: " + outputFileObj.getAbsolutePath());
			PrintStream fileOut = new PrintStream(new FileOutputStream(outputFile));
			TeeOutputStream teeOut = new TeeOutputStream(System.out, fileOut);
			PrintStream teePrintStream = new PrintStream(teeOut);
			
			// 원래 출력 스트림 저장
			PrintStream originalOut = System.out;
			
			// TeeOutputStream으로 출력 리다이렉션
			System.setOut(teePrintStream);
			
			// 테스트 실행
			FederatedPlanCostEnumerator.enumerateProgram(prog, true);
			
			// 원래 출력 스트림으로 복원
			System.setOut(originalOut);
			
			// 리소스 정리
			fileOut.close();
			teeOut.close();
			teePrintStream.close();

			// Python visualizer 실행 확인
			File visualizerDir = new File("visualization_output");
			if (!visualizerDir.exists()) {
				visualizerDir.mkdirs();
				System.out.println("[INFO] 시각화 출력 디렉토리 생성: " + visualizerDir.getAbsolutePath());
			}

			// Python visualizer 스크립트 경로 확인
			File scriptFile = new File("src/test/java/org/apache/sysds/test/component/federated/FederatedPlanVisualizer.py");
			System.out.println("[INFO] Python 스크립트 존재 여부: " + scriptFile.exists());
			System.out.println("[INFO] Python 스크립트 경로: " + scriptFile.getAbsolutePath());
			
			if (!scriptFile.exists()) {
				System.out.println("[오류] Python visualizer 스크립트를 찾을 수 없습니다: " + scriptFile.getAbsolutePath());
				Assert.fail("Python visualizer 스크립트를 찾을 수 없습니다: " + scriptFile.getAbsolutePath());
			}
			
			// Python 인터프리터 확인
			try {
				ProcessBuilder checkPython = new ProcessBuilder("python3", "--version");
				checkPython.redirectErrorStream(true);
				Process pythonCheck = checkPython.start();
				
				BufferedReader pythonReader = new BufferedReader(new InputStreamReader(pythonCheck.getInputStream()));
				String pythonVersion = pythonReader.readLine();
				System.out.println("[INFO] Python 버전: " + pythonVersion);
				
				pythonCheck.waitFor();
			} catch (Exception e) {
				System.out.println("[오류] Python 인터프리터를 확인할 수 없습니다: " + e.getMessage());
			}
			
			System.out.println("[INFO] Visualizer 실행 명령: python3 " + scriptFile.getAbsolutePath() + " " + outputFileObj.getAbsolutePath());
			ProcessBuilder pb = new ProcessBuilder("python3", scriptFile.getAbsolutePath(), outputFileObj.getAbsolutePath());
			pb.redirectErrorStream(true);
			Process p = pb.start();
			
			// Python 스크립트의 출력을 읽어서 표시
			BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
			String line;
			System.out.println("[INFO] Python 스크립트 출력:");
			while ((line = reader.readLine()) != null) {
				System.out.println("[Python] " + line);
			}
			
			// 프로세스 종료 코드 확인
			int exitCode = p.waitFor();
			System.out.println("[INFO] Python 프로세스 종료 코드: " + exitCode);
			
			if (exitCode == 0) {
				System.out.println("[INFO] Visualizer 실행 성공 (종료 코드: 0)");
				
				// 생성된 이미지 파일 확인
				System.out.println("[INFO] 생성된 시각화 파일:");
				File[] imageFiles = visualizerDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".png"));
				if (imageFiles != null && imageFiles.length > 0) {
					for (File imageFile : imageFiles) {
						System.out.println("  - " + imageFile.getAbsolutePath());
					}
				} else {
					System.out.println("[INFO] 시각화 파일이 생성되지 않았습니다.");
				}
			} else {
				System.out.println("[오류] Visualizer 실행 실패 (종료 코드: " + exitCode + ")");
				Assert.fail("Visualizer 실행 실패 (종료 코드: " + exitCode + ")");
			}
		}
		catch (IOException | InterruptedException e) {
			e.printStackTrace();
			Assert.fail(e.getMessage());
		}
	}
}
