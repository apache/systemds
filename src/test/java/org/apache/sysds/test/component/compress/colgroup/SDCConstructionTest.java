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
package org.apache.sysds.test.component.compress.colgroup;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.Bitmap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.junit.Test;

public class SDCConstructionTest {

	@Test
	public void test() {
		int[] colIndexes = new int[] {0};
		int[][] offsets = new int[][] {
			new int[] {14, 20, 21, 40, 41, 42, 43, 44, 47, 49, 52, 54, 209, 277, 281, 283, 285, 307, 308, 309, 310, 311,
				312, 314, 316, 317, 356, 472, 490, 543, 545, 548, 572, 573, 574, 575, 577, 579, 580, 581, 582, 714, 740,
				742, 766, 814, 837, 838, 840, 842, 843, 844, 845, 846, 847, 851, 852, 945},
			new int[] {28, 31, 174, 475, 476, 492, 496, 514, 557, 662, 744, 745, 746, 755, 756, 763, 767, 780, 855, 856,
				937, 944},
			new int[] {27, 56, 162, 211, 224, 228, 247, 324, 325, 385, 499, 589, 590, 591, 739, 759, 822, 854, 858,
				876},
			new int[] {34, 57, 77, 173, 234, 326, 556, 592, 596, 611, 743, 765, 776, 821, 834, 860, 877, 941},
			new int[] {10, 48, 50, 116, 178, 206, 233, 284, 313, 315, 322, 443, 546, 576, 578, 587, 839, 841},
			new int[] {11, 58, 124, 132, 172, 210, 227, 318, 319, 327, 444, 500, 554, 583, 588, 593, 666, 938},
			new int[] {37, 38, 39, 55, 304, 305, 306, 470, 498, 555, 569, 570, 571, 669, 785, 835, 836},
			new int[] {13, 51, 207, 401, 551, 553, 584, 594, 656, 663, 670, 715, 737, 817, 848, 853},
			new int[] {24, 59, 60, 61, 171, 175, 287, 289, 328, 329, 346, 473, 497, 612, 849},
			new int[] {35, 36, 64, 237, 294, 295, 303, 345, 477, 568, 597, 862, 943},
			new int[] {15, 17, 19, 45, 46, 446, 449, 469, 471, 542, 810, 811, 931},
			new int[] {22, 176, 254, 279, 288, 448, 549, 550, 758, 815, 816},
			new int[] {76, 256, 563, 564, 567, 824, 826, 827, 830, 831, 863},
			new int[] {16, 208, 544, 547, 741, 757, 809, 812, 813, 850, 939},
			new int[] {12, 62, 223, 290, 323, 330, 518, 552, 595, 819, 820},
			new int[] {83, 137, 143, 194, 352, 409, 412, 416, 679, 927},
			new int[] {29, 212, 293, 400, 510, 511, 747, 777, 781, 857},
			new int[] {129, 235, 250, 296, 297, 300, 332, 386, 558, 761},
			new int[] {79, 88, 161, 204, 348, 625, 665, 711, 994},
			new int[] {75, 219, 243, 343, 503, 608, 768, 775, 873},
			new int[] {248, 253, 299, 494, 495, 565, 566, 823, 833},
			new int[] {30, 214, 236, 331, 488, 501, 738, 764, 915},
			new int[] {151, 410, 411, 417, 618, 648, 680, 884, 953}, new int[] {215, 298, 302, 493, 515, 610, 762, 875},
			new int[] {388, 396, 408, 413, 415, 678, 952, 983}, new int[] {139, 152, 419, 624, 649, 683, 894, 912},
			new int[] {126, 320, 321, 585, 586, 661, 719, 807}, new int[] {78, 125, 133, 179, 347, 402, 450, 468},
			new int[] {117, 394, 519, 613, 713, 917, 918, 997}, new int[] {136, 191, 406, 461, 466, 617, 675, 734},
			new int[] {67, 73, 336, 338, 486, 602, 774, 867}, new int[] {26, 225, 232, 445, 474, 859, 936},
			new int[] {25, 177, 291, 489, 655, 668, 818}, new int[] {220, 221, 222, 242, 484, 754, 872},
			new int[] {190, 200, 351, 467, 647, 735, 891}, new int[] {23, 226, 278, 280, 286, 491, 935},
			new int[] {87, 380, 658, 659, 672, 880, 892}, new int[] {53, 282, 447, 654, 718, 808, 878},
			new int[] {32, 33, 213, 229, 246, 301, 760}, new int[] {189, 199, 201, 460, 465, 729},
			new int[] {141, 195, 196, 197, 682, 954}, new int[] {85, 155, 156, 373, 685, 926},
			new int[] {1, 266, 267, 361, 371, 530}, new int[] {230, 249, 398, 559, 561, 562},
			new int[] {86, 355, 405, 643, 882, 928}, new int[] {432, 657, 671, 930, 998, 999},
			new int[] {801, 802, 804, 805, 975}, new int[] {451, 722, 736, 985, 996}, new int[] {3, 92, 274, 630, 900},
			new int[] {82, 147, 462, 730, 883}, new int[] {144, 150, 414, 677, 913}, new int[] {93, 259, 365, 367, 522},
			new int[] {65, 217, 252, 479, 750}, new int[] {89, 360, 527, 635, 798}, new int[] {403, 614, 986, 991, 993},
			new int[] {146, 148, 192, 733, 950}, new int[] {81, 350, 395, 397, 616},
			new int[] {240, 508, 520, 607, 751}, new int[] {131, 238, 251, 478, 513},
			new int[] {269, 276, 433, 528, 799}, new int[] {72, 339, 604, 869, 870}, new int[] {504, 601, 752, 866},
			new int[] {372, 423, 687, 925}, new int[] {80, 381, 456, 645}, new int[] {6, 8, 9, 369},
			new int[] {123, 153, 420, 684}, new int[] {63, 292, 487, 861}, new int[] {181, 183, 453, 992},
			new int[] {110, 169, 704, 977}, new int[] {74, 342, 507, 600}, new int[] {97, 257, 357, 627},
			new int[] {193, 407, 921, 951}, new int[] {264, 535, 537, 788}, new int[] {384, 392, 404, 988},
			new int[] {265, 275, 533, 534}, new int[] {134, 393, 457, 929}, new int[] {244, 482, 769, 864},
			new int[] {716, 717, 916, 942}, new int[] {7, 358, 363, 637}, new int[] {188, 202, 383, 674},
			new int[] {145, 149, 653, 676}, new int[] {660, 664, 879, 946}, new int[] {128, 205, 644, 712},
			new int[] {218, 399, 481, 502}, new int[] {18, 255, 541, 806}, new int[] {102, 628, 897, 909},
			new int[] {69, 506, 603}, new int[] {119, 121, 964}, new int[] {231, 667, 914}, new int[] {263, 531, 790},
			new int[] {340, 505, 605}, new int[] {109, 696, 708}, new int[] {341, 773, 783}, new int[] {170, 428, 694},
			new int[] {66, 334, 865}, new int[] {454, 723, 987}, new int[] {268, 532, 536}, new int[] {691, 960, 982},
			new int[] {198, 431, 464}, new int[] {458, 710, 947}, new int[] {118, 424, 425}, new int[] {701, 797, 902},
			new int[] {127, 984, 995}, new int[] {459, 646, 948}, new int[] {130, 599, 782}, new int[] {111, 378, 623},
			new int[] {90, 619, 786}, new int[] {366, 521, 523}, new int[] {138, 140, 642}, new int[] {135, 382, 463},
			new int[] {390, 688, 924}, new int[] {640, 641, 908}, new int[] {335, 483, 485}, new int[] {239, 517, 778},
			new int[] {387, 721, 933}, new int[] {70, 606, 868}, new int[] {2, 270, 529}, new int[] {163, 480, 779},
			new int[] {186, 724, 727}, new int[] {434, 793, 795}, new int[] {720, 932, 934}, new int[] {184, 185, 725},
			new int[] {794, 803, 906}, new int[] {142, 418, 681}, new int[] {728, 920}, new int[] {379, 706},
			new int[] {120, 692}, new int[] {349, 726}, new int[] {159, 699}, new int[] {68, 71}, new int[] {157, 956},
			new int[] {971, 978}, new int[] {362, 898}, new int[] {94, 364}, new int[] {437, 970}, new int[] {965, 966},
			new int[] {540, 787}, new int[] {158, 430}, new int[] {974, 980}, new int[] {376, 979},
			new int[] {651, 922}, new int[] {422, 893}, new int[] {95, 96}, new int[] {375, 981}, new int[] {753, 784},
			new int[] {260, 524}, new int[] {792, 796}, new int[] {115, 919}, new int[] {354, 888},
			new int[] {167, 886}, new int[] {359, 899}, new int[] {695, 973}, new int[] {106, 967},
			new int[] {377, 707}, new int[] {273, 525}, new int[] {4, 636}, new int[] {160, 957}, new int[] {245, 516},
			new int[] {337, 770}, new int[] {702, 976}, new int[] {538, 800}, new int[] {731, 881},
			new int[] {104, 107}, new int[] {182, 452}, new int[] {697, 968}, new int[] {154, 421},
			new int[] {890, 962}, new int[] {241, 871}, new int[] {122, 693}, new int[] {560, 748},
			new int[] {632, 903}, new int[] {262, 271}, new int[] {689, 958}, new int[] {455, 990},
			new int[] {709, 959}, new int[] {389, 923}, new int[] {391, 955}, new int[] {626, 772},
			new int[] {791, 901}, new int[] {673, 989}, new int[] {5, 261}, new int[] {703}, new int[] {84},
			new int[] {969}, new int[] {961}, new int[] {99}, new int[] {789}, new int[] {168}, new int[] {940},
			new int[] {427}, new int[] {440}, new int[] {187}, new int[] {705}, new int[] {621}, new int[] {905},
			new int[] {258}, new int[] {686}, new int[] {629}, new int[] {108}, new int[] {101}, new int[] {907},
			new int[] {652}, new int[] {972}, new int[] {438}, new int[] {509}, new int[] {105}, new int[] {639},
			new int[] {634}, new int[] {112}, new int[] {370}, new int[] {436}, new int[] {638}, new int[] {732},
			new int[] {690}, new int[] {910}, new int[] {439}, new int[] {622}, new int[] {368}, new int[] {539},
			new int[] {91}, new int[] {650}, new int[] {0}, new int[] {911}, new int[] {620}, new int[] {633},
			new int[] {164}, new int[] {904}, new int[] {426}, new int[] {700}, new int[] {887}, new int[] {166},
			new int[] {114}, new int[] {771}, new int[] {896}, new int[] {526}, new int[] {963}, new int[] {435},
			new int[] {165}, new int[] {113}, new int[] {98}, new int[] {615}, new int[] {441}, new int[] {889},
			new int[] {103}, new int[] {353}, new int[] {631}, new int[] {100}, new int[] {272}, new int[] {885},
			new int[] {429}, new int[] {949}, new int[] {698}, new int[] {374}, new int[] {442}, new int[] {180},
			new int[] {203}};
		double[] values = new double[] {333.0, 347.0, 346.0, 344.0, 334.0, 339.0, 342.0, 338.0, 337.0, 351.0, 331.0,
			335.0, 358.0, 332.0, 340.0, 275.0, 349.0, 352.0, 315.0, 7.0, 354.0, 348.0, 274.0, 353.0, 276.0, 270.0,
			328.0, 326.0, 323.0, 282.0, 21.0, 343.0, 341.0, 18.0, 283.0, 336.0, 308.0, 330.0, 350.0, 285.0, 272.0,
			262.0, 81.0, 355.0, 288.0, 324.0, 100.0, 318.0, 80.0, 281.0, 277.0, 60.0, 2.0, 90.0, 313.0, 280.0, 291.0,
			15.0, 359.0, 89.0, 27.0, 16.0, 253.0, 304.0, 64.0, 266.0, 345.0, 312.0, 184.0, 14.0, 45.0, 278.0, 88.0,
			299.0, 86.0, 310.0, 8.0, 325.0, 63.0, 287.0, 279.0, 317.0, 320.0, 3.0, 329.0, 56.0, 24.0, 228.0, 356.0,
			85.0, 28.0, 198.0, 22.0, 214.0, 12.0, 311.0, 87.0, 240.0, 293.0, 300.0, 249.0, 94.0, 321.0, 289.0, 10.0,
			185.0, 135.0, 61.0, 271.0, 294.0, 252.0, 65.0, 17.0, 4.0, 322.0, 23.0, 83.0, 1.0, 309.0, 98.0, 327.0, 305.0,
			99.0, 273.0, 301.0, 197.0, 238.0, 303.0, 243.0, 26.0, 263.0, 186.0, 68.0, 54.0, 187.0, 227.0, 103.0, 255.0,
			212.0, 200.0, 254.0, 259.0, 53.0, 215.0, 13.0, 67.0, 96.0, 319.0, 143.0, 158.0, 72.0, 202.0, 217.0, 191.0,
			74.0, 77.0, 256.0, 5.0, 20.0, 97.0, 95.0, 297.0, 220.0, 314.0, 203.0, 264.0, 235.0, 25.0, 233.0, 357.0,
			102.0, 78.0, 250.0, 306.0, 246.0, 251.0, 268.0, 31.0, 91.0, 295.0, 71.0, 179.0, 269.0, 195.0, 237.0, 44.0,
			82.0, 153.0, 9.0, 219.0, 169.0, 298.0, 194.0, 128.0, 111.0, 55.0, 257.0, 70.0, 210.0, 50.0, 79.0, 244.0,
			192.0, 182.0, 6.0, 213.0, 57.0, 104.0, 204.0, 73.0, 178.0, 58.0, 286.0, 245.0, 49.0, 172.0, 139.0, 62.0,
			101.0, 113.0, 267.0, 76.0, 41.0, 140.0, 105.0, 166.0, 109.0, 239.0, 231.0, 156.0, 173.0, 181.0, 33.0, 36.0,
			84.0, 232.0, 170.0, 180.0, 193.0, 43.0, 302.0, 209.0, 157.0, 66.0, 147.0, 92.0, 47.0, 75.0, 150.0, 225.0,
			284.0, 218.0, 216.0, 241.0, 316.0, 292.0};
		// ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, null, compSettings.transposed);
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		ABitmap ubm = new Bitmap(gen(offsets), values, 1000);
		try {

			ColGroupFactory.compress(colIndexes, 1000, ubm, CompressionType.SDC, cs, null, 1.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	public IntArrayList[] gen(int[][] in) {
		IntArrayList[] res = new IntArrayList[in.length];
		int idx = 0;
		int totalLen = 0;
		for(int[] i : in) {
			totalLen += i.length;
			res[idx++] = new IntArrayList(i);
		}
		System.out.println(totalLen);
		return res;
	}
}
