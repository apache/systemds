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

package org.apache.sysds.test.component.compress.offset;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.junit.Test;

public class OffsetReverseTest {

	@Test
	public void reverse1() {
		AOffset off = OffsetFactory.createOffset(new int[] {1, 10, 13, 14});
		AOffset rev = AOffset.reverse(16, off);
		OffsetTests.compare(rev, new int[] {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15});
	}

	@Test
	public void reverse2() {
		AOffset off = OffsetFactory.createOffset(new int[] {1, 10});
		AOffset rev = AOffset.reverse(16, off);
		OffsetTests.compare(rev, new int[] {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15});
	}

	@Test
	public void reverse3() {
		AOffset off = OffsetFactory.createOffset(new int[] {1});
		AOffset rev = AOffset.reverse(16, off);
		OffsetTests.compare(rev, new int[] {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
	}

	@Test
	public void reverse4() {
		AOffset off = OffsetFactory.createOffset(new int[] {1, 10, 13, 14, 15});
		AOffset rev = AOffset.reverse(16, off);
		OffsetTests.compare(rev, new int[] {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12});
	}

	@Test
	public void reverse4_withCreateMethod() {
		int[] exp = new int[] {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12};
		AOffset off = OffsetFactory.createOffset(create(exp, 16));
		AOffset rev = AOffset.reverse(16, off);
		OffsetTests.compare(rev, exp);
	}

	@Test
	public void reverse1_withCreateMethod() {
		int[] exp = new int[] {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15};
		AOffset off = OffsetFactory.createOffset(create(exp, 16));
		AOffset rev = AOffset.reverse(16, off);
		OffsetTests.compare(rev, exp);
	}

	@Test
	public void reverseCreate1() {
		test(new int[] {100, 132, 520}, 1000);
	}

	@Test
	public void reverseCreate2() {
		test(new int[] {100, 132, 520, 999}, 1000);
	}

	@Test
	public void reverseCreate3() {
		test(new int[] {1, 999}, 1000);
	}

	@Test
	public void reverseCreate4() {
		test(new int[] {256, 512, 999}, 1000);
	}

	@Test
	public void reverse() {
		AOffset off = OffsetFactory
			.createOffset(
				new int[] {1, 3, 7, 8, 10, 11, 14, 15, 16, 23, 25, 26, 28, 31, 32, 34, 36, 38, 42, 43, 44, 46, 47, 52, 55,
					56, 57, 62, 63, 67, 68, 69, 70, 72, 74, 75, 79, 81, 82, 83, 84, 85, 87, 88, 92, 93, 94, 95, 96, 98, 100,
					105, 108, 109, 110, 111, 117, 120, 121, 124, 125, 126, 128, 129, 132, 135, 137, 139, 144, 147, 148, 149,
					150, 152, 155, 156, 157, 158, 159, 161, 165, 166, 167, 168, 170, 173, 176, 179, 180, 182, 183, 185, 187,
					188, 190, 191, 192, 194, 195, 196, 197, 200, 202, 203, 206, 209, 211, 215, 216, 217, 220, 221, 222, 223,
					224, 225, 226, 227, 228, 230, 234, 239, 240, 241, 246, 249, 253, 255, 256, 257, 261, 262, 263, 266, 268,
					269, 270, 271, 277, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 292, 293, 294, 297, 299, 302, 305,
					308, 313, 314, 318, 319, 323, 324, 325, 329, 330, 331, 332, 333, 338, 339, 341, 342, 343, 344, 345, 346,
					347, 350, 351, 352, 354, 355, 356, 358, 362, 363, 365, 367, 373, 374, 375, 376, 379, 380, 381, 382, 384,
					385, 387, 388, 390, 391, 392, 395, 397, 401, 402, 405, 406, 407, 411, 415, 416, 418, 419, 420, 423, 424,
					426, 427, 428, 429, 431, 435, 436, 438, 439, 440, 441, 445, 446, 447, 450, 451, 452, 456, 458, 461, 462,
					464, 465, 467, 468, 469, 470, 477, 481, 484, 485, 487, 488, 489, 494, 495, 500, 504, 505, 506, 508, 510,
					512, 513, 517, 518, 520, 524, 525, 526, 527, 528, 529, 531, 532, 534, 538, 540, 543, 544, 546, 548, 551,
					553, 554, 556, 560, 562, 563, 564, 567, 569, 570, 571, 575, 577, 578, 579, 581, 582, 585, 586, 587, 589,
					592, 593, 594, 598, 600, 605, 607, 613, 615, 617, 618, 623, 624, 629, 630, 632, 633, 634, 635, 636, 637,
					638, 639, 641, 644, 645, 646, 649, 651, 652, 654, 657, 659, 663, 664, 669, 671, 672, 673, 677, 678, 679,
					680, 683, 684, 685, 686, 687, 691, 692, 694, 696, 698, 700, 702, 705, 706, 713, 715, 720, 722, 723, 724,
					728, 729, 730, 733, 735, 736, 737, 739, 740, 741, 742, 743, 744, 745, 746, 747, 750, 751, 752, 758, 762,
					763, 764, 767, 768, 771, 772, 775, 776, 778, 779, 781, 785, 788, 789, 791, 792, 793, 794, 797, 804, 806,
					807, 809, 810, 811, 812, 813, 815, 816, 818, 819, 820, 821, 822, 824, 825, 827, 831, 833, 834, 835, 837,
					838, 839, 840, 841, 843, 848, 849, 851, 852, 853, 859, 862, 863, 864, 865, 866, 870, 871, 872, 873, 874,
					875, 877, 879, 880, 882, 883, 887, 889, 891, 892, 894, 896, 897, 898, 899, 901, 902, 903, 905, 906, 908,
					911, 912, 913, 917, 919, 920, 922, 926, 927, 931, 933, 935, 936, 938, 940, 941, 943, 944, 945, 946, 948,
					950, 953, 957, 959, 961, 967, 968, 974, 979, 980, 982, 983, 984, 986, 989, 990, 991, 993, 995, 996, 997});
		AOffset rev = AOffset.reverse(1000, off);
		// System.out.println(off);
		// System.out.println(rev);
		assertEquals(0, rev.getOffsetIterator().value());
	}

	private void test(int[] missing, int max) {
		AOffset off = OffsetFactory.createOffset(create(missing, max));
		AOffset rev = AOffset.reverse(max, off);
		OffsetTests.compare(rev, missing);
	}

	private static int[] create(int[] missing, int max) {
		int[] ret = new int[max - missing.length];
		int j = 0;
		int k = 0;
		for(int i = 0; i < max; i++) {
			if(j < missing.length && missing[j] == i)
				j++;
			else
				ret[k++] = i;
		}
		return ret;
	}
}
