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

package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.ifft;

import static org.junit.Assert.assertArrayEquals;

public class FourierTest {

	@Test
	public void test_fft_one_dim() {

		MatrixBlock re = new MatrixBlock(1, 4,  new double[]{0, 18, -15, 3});
		MatrixBlock im = new MatrixBlock(1, 4,  new double[4]);

		double[] expected_re = {6, 15, -36, 15};
		double[] expected_im = {0, -15, 0, 15};

		fft(re, im);

		assertArrayEquals(expected_re, re.getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, im.getDenseBlockValues(), 0.0001);

	}

	@Test
	public void test_fft_one_dim_2() {

		MatrixBlock re = new MatrixBlock(1, 8,  new double[]{0, 18, -15, 3, 5, 10, 5, 9});
		MatrixBlock im = new MatrixBlock(1, 8,  new double[8]);

		double[] expected_re = {35, 4.89949, 15, -14.89949, -45, -14.89949, 15, 4.89949};
		double[] expected_im = {0, 18.58579, -16, -21.41421, 0, 21.41421, 16, -18.58579};

		fft(re, im);

		assertArrayEquals(expected_re, re.getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, im.getDenseBlockValues(), 0.0001);
	}

	@Test
	public void test_fft_one_dim_matrixBlock() {

		MatrixBlock re = new MatrixBlock(1, 4,  new double[]{0, 18, -15, 3});
		MatrixBlock im = new MatrixBlock(1, 4,  new double[]{0, 0, 0, 0});

		double[] expected_re = {6, 15, -36, 15};
		double[] expected_im = {0, -15, 0, 15};

		MatrixBlock[] res = fft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

	@Test
	public void test_ifft_one_dim_matrixBlock_2() {

		double[] in_re = new double[]{1, -2, 3, -4};
		double[] in_im = new double[]{0, 0, 0, 0};

		MatrixBlock re = new MatrixBlock(1, 4, in_re);
		MatrixBlock im = new MatrixBlock(1, 4, in_im);

		MatrixBlock[] inter = fft(re, im);
		MatrixBlock[] res = ifft(inter[0], inter[1]);

		assertArrayEquals(in_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(in_im, res[1].getDenseBlockValues(), 0.0001);

	}

	@Test
	public void test_fft_two_dim_matrixBlock() {

		MatrixBlock re = new MatrixBlock(2, 2,  new double[]{0, 18, -15, 3});
		MatrixBlock im = new MatrixBlock(2, 2,  new double[]{0, 0, 0, 0});

		double[] expected_re = {6,-36, 30, 0};
		double[] expected_im = {0, 0, 0, 0};

		MatrixBlock[] res = fft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

	@Test
	public void test_ifft_two_dim_matrixBlock() {

		MatrixBlock re = new MatrixBlock(2, 2,  new double[]{6,-36, 30, 0});
		MatrixBlock im = new MatrixBlock(2, 2,  new double[]{0, 0, 0, 0});

		double[] expected_re = {0, 18, -15, 3};
		double[] expected_im = {0, 0, 0, 0};

		MatrixBlock[] res = ifft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

	@Test
	public void test_fft_two_dim_matrixBlock_row_1() {

		MatrixBlock re = new MatrixBlock(1, 2,  new double[]{0, 18});
		MatrixBlock im = new MatrixBlock(1, 2,  new double[]{0, 0});

		double[] expected_re = {18, -18};
		double[] expected_im = {0, 0};

		MatrixBlock[] res = fft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

	@Test
	public void test_fft_two_dim_matrixBlock_row_2() {

		MatrixBlock re = new MatrixBlock(1, 2,  new double[]{ -15, 3});
		MatrixBlock im = new MatrixBlock(1, 2,  new double[]{0, 0});

		double[] expected_re = {-12, -18};
		double[] expected_im = {0, 0};

		MatrixBlock[] res = fft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

	@Test
	public void test_ifft_with_complex_numpy_data() {

		// removed 0's at the end, not just real
		MatrixBlock re = new MatrixBlock(1, 16,  new double[]{
				0.5398705320215192, 0.1793355360736929, 0.9065254044489506, 0.45004385530909075,
				0.11128090341119468, 0.11742862805303522, 0.7574827475752481, 0.2778193170985158,
				0.9251562928110273, 0.9414429667551927, 0.45131569795507087, 0.9522067687409731,
				0.22491032260636257, 0.6579426733967295, 0.7021558730366062, 0.7861117825617701,
		});

		MatrixBlock im = new MatrixBlock(1, 16,  new double[16]);

		double[] expected_re = {
				0.5613143313659362, -0.020146500453061336, 0.07086545895481336, -0.05003801442765281,
				-0.06351635451036074, -0.03346768844048936, 0.07023899089706032, 0.007330763123826495,
				0.016022890367311193, 0.007330763123826495, 0.07023899089706032, -0.03346768844048936,
				-0.06351635451036074, -0.05003801442765281, 0.07086545895481336, -0.020146500453061336
		};

		double[] expected_im = {
				0.0, -0.07513090216687965, 0.023854392878864396, -0.018752997939582024,
				-0.035626994964481226, -0.07808216015046443, 0.036579082654843526, -0.10605270957897009,
				0.0, 0.10605270957897009, -0.036579082654843526, 0.07808216015046443,
				0.035626994964481226, 0.018752997939582024, -0.023854392878864396, 0.07513090216687965
		};

		MatrixBlock[] res = ifft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

	@Test
	public void test_ifft_2d_with_generated_data() {

		MatrixBlock re = new MatrixBlock(2, 2,  new double[]{
				0.6749989259154331, 0.6278845555308362, 0.995916990652601, 0.511472971081564
		});

		MatrixBlock im = new MatrixBlock(2, 2,  new double[]{
				0.8330832079105173, 0.09857986129294982, 0.6808883894146879, 0.28353782431047303
		});

		// adjusted the expected output
		double[] expected_re = {
				0.70256836, 0.1328896, -0.05112662, -0.10933241
		};

		double[] expected_im = {
				0.47402232, 0.28296348, -0.00819079, 0.0842882
		};

		MatrixBlock[] res = ifft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

	@Test
	public void test_ifft_with_real_numpy_data() {

		// not complex
		MatrixBlock re = new MatrixBlock(1, 16,  new double[]{
				0.17768499045697306, 0.3405035491673728, 0.9272906450602005, 0.28247504052271843,
				0.42775517613102865, 0.8338783039818357, 0.9813749624557385, 0.47224489612381737,
				0.7936831995784907, 0.5584182145651306, 0.5296113722056018, 0.6687593295928902,
				0.9630598447622862, 0.7130539473424196, 0.860081483892192, 0.8985058305053549
		});

		MatrixBlock im = new MatrixBlock(1, 16,  new double[16]);

		// adjusted the expected output
		double[] expected_re = {
				0.6517738, -0.0263837 , -0.03631354, -0.01644966,
				-0.05851095, -0.0849794, -0.01611732, -0.02618679,
				0.05579391, -0.02618679, -0.01611732, -0.0849794,
				-0.05851095, -0.01644966, -0.03631354, -0.0263837
		};

		double[] expected_im = {
				0, -0.04125649, -0.07121312, 0.02554502,
				0.00774181, -0.08723921, -0.02314382, -0.02021455,
				0, 0.02021455, 0.02314382, 0.08723921,
				-0.00774181, -0.02554502, 0.07121312, 0.04125649

		};

		MatrixBlock[] res = ifft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

	@Test
	public void test_fft_two_dim_8_times_8() {

		MatrixBlock re = new MatrixBlock(8, 8,  new double[]{
				0.8435874964408077, 0.3565209485970835, 0.6221038572251737, 0.05712418097055716,
				0.9301368966310067, 0.7748052735242277, 0.21117129518682443, 0.08407931152930459,
				0.5861235649815163, 0.45860122035396356, 0.6647476180103304, 0.9167930424492593,
				0.6310726270028377, 0.11110504251770592, 0.32369996452324756, 0.5790548902504138,
				0.5712310851880162, 0.5967356161025353, 0.6441861776319489, 0.14402445187596158,
				0.22642623625293545, 0.922443731897705, 0.9527667119829785, 0.2250880965427453,
				0.5755375055168817, 0.48898427237526954, 0.24518238824389693, 0.832292384016089,
				0.23789083930394805, 0.5558982102157535, 0.7220016080026206, 0.9666747522359772,
				0.20509423975210916, 0.23170117015755587, 0.7141206714718693, 0.2014158450611332,
				0.6486924358372994, 0.9044990419216931, 0.19849364935627056, 0.23340297110822106,
				0.46854050631969246, 0.10134155509558795, 0.5563200388698989, 0.2669820016661475,
				0.8889445005077763, 0.4273462470993935, 0.8269490075576963, 0.044351336481537995,
				0.3771564738915597, 0.11333723996854606, 0.6913138435759023, 0.062431275099310124,
				0.8003013976959878, 0.1276686539064662, 0.975167392001707, 0.44595301043682656,
				0.18401328301977316, 0.7158585484384759, 0.3240126702723025, 0.740836665073052,
				0.8890279623888511, 0.8841266040978419, 0.3058930798936259, 0.8987579873722049
		});

		MatrixBlock im = new MatrixBlock(1, 16,  new double[]{
				0.8572457113722648, 0.668182795310341, 0.9739416721141464, 0.8189153345383146,
				0.6425950286263254, 0.3569634253534639, 0.19715070300424575, 0.8915344479242211,
				0.39207930659031054, 0.1625193685179268, 0.2523438052868171, 0.30940628850519547,
				0.7461468672112159, 0.7123766750132684, 0.5261041429273977, 0.867155304805022,
				0.7207769261821749, 0.9139070611733158, 0.7638265842242135, 0.3508092733308539,
				0.6075639148195967, 0.9615531048215422, 0.719499617407839, 0.9616615941848492,
				0.2667126256574347, 0.8215093145949468, 0.4240476512138287, 0.5015798652459079,
				0.19784651066995873, 0.42315603332105356, 0.5575575283922164, 0.9051304828282485,
				0.30117855478511435, 0.14219967492505514, 0.32675429179906557, 0.04889894374947912,
				0.8338579676700041, 0.370201089804747, 0.06025987717830994, 0.9407970353033787,
				0.9871788482561391, 0.75984297199074, 0.414969247979073, 0.2453785474698862,
				0.06295683447294731, 0.40141192931768566, 0.19520663793867488, 0.3179027928938928,
				0.591138083168947, 0.5318366162549014, 0.04865894304644136, 0.5339043989658795,
				0.09892519435896363, 0.31616794516128466, 0.06702286400447643, 0.8466767273121609,
				0.8134875055724791, 0.6232554321597641, 0.21208039111457444, 0.25629831822305926,
				0.7373140896724466, 0.020486629088602437, 0.8666668269441752, 0.20094387974200512
		});

		double[] expected_re = {
				32.51214260297584, -3.4732779237490314, -0.7257760912890102, -1.9627494786611792,
				3.571671446098747, 1.0451692206901078, 0.8970702451384204, -1.3739767803210428,
				4.892442103095981, -1.1656855109832338, -0.5854908742291178, -1.3497699084098418,
				-1.377003693155216, -2.1698030461214923, 0.8172129683973663, -0.9259076518379679,
				-1.1343756245445045, -1.8734967800709579, 1.7367517585478862, 0.07349671655414491,
				-1.5933768052439223, 2.7965196291943983, 4.292588673604611, -1.1032899622026413,
				-2.4643093702874834, 2.109128987930992, 3.2834030498896456, 0.21371926254596152,
				-0.3107488550365316, 0.7293395030253796, -2.542403789759091, -1.8570654162590052,
				-2.325781245331303, 0.7963395911053484, -2.351990667205867, -2.4241188304485735,
				4.689766636746301, 3.4748121457306116, 0.5539071663846459, -0.950099313504134,
				2.122310975524349, 4.527637759721644, -2.125596093625001, 1.7676565539001592,
				5.748332643019926, 0.860140632830907, 2.9735142186218484, -1.7198774815194848,
				-0.18418859401548549, 1.1909629561342188, 0.21710627714418418, -1.5537184277268996,
				-0.5486540814747869, 0.14807346060743987, 2.4333154010438087, -3.2930077637380393,
				-2.3820067665775113, 2.5463581304688057, 2.5927580716559615, 1.8921802492721915,
				0.4957713559465988, -0.4983536537999108, 3.5808175362367676, 0.7530823235547575
		};

		double[] expected_im = {
				32.64565805549281, 5.177639365468945, 1.1792020104097647, -0.4850627423320939,
				-1.719468548169175, -3.064146170894837, 3.3226243586118906, 2.3819341640916107,
				1.5824429804361388, -2.192940882164737, 0.5774407122593543, 0.16873948200103983,
				4.297014293326352, -3.1712082122026883, 0.9741291131305898, -2.4929883795121235,
				1.111763301820595, -1.4012254390671657, -0.33687898317382636, 2.324190267133635,
				-2.8862969254091397, -4.7558982401265135, 1.8244587481290004, 0.5310550630270396,
				2.655726742689745, 2.510014260306531, 0.25589537824783704, 1.8720307201415736,
				-2.6458046644482884, 2.1732611302115585, -2.5162250969793227, -0.9103444457919911,
				2.2835527482590248, 0.5187392677625127, -3.335253420903965, 1.4668560670097441,
				-1.9681585205341436, -2.81914578771063, 4.818094364700921, -0.877636803126361,
				1.803174743823159, 3.1346192487664277, -3.564058675191744, -0.3391381913837902,
				-1.2897867384105863, 2.315065426377637, 1.5764817121472765, 2.412894091248795,
				-2.3182678917385218, -3.057303547366563, 0.033996764974414395, -0.5825423640666696,
				6.395088232674363, -0.5553659624089358, 1.1079219041153268, 0.1094531803830765,
				3.488182265163636, -1.5698242466218544, -0.1013387045518459, 0.9269290699615746,
				-0.699890233104248, 3.617209720991753, -0.5565163478425035, 3.502962737763559
		};

		MatrixBlock[] res = fft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

}
