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

package org.apache.sysds.utils;

/**
 * A fast double parser inspired from https://github.com/wrandelshofer/FastDoubleParser
 */
public interface DoubleParser {

	final static int MAX_EXPONENT_NUMBER = 1024;
	final static String ILLEGAL_OFFSET_OR_ILLEGAL_LENGTH = "offset < 0 or length > str.length";
	final static String SYNTAX_ERROR = "illegal syntax";
	final static long MINIMAL_NINETEEN_DIGIT_INTEGER = 1000_00000_00000_00000L;
	final static int DOUBLE_MIN_EXPONENT_POWER_OF_TEN = -325;
	final static int DOUBLE_MAX_EXPONENT_POWER_OF_TEN = 308;
	final static int DOUBLE_SIGNIFICAND_WIDTH = 53;
	final static double[] DOUBLE_POWERS_OF_TEN = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12,
		1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
	final static int DOUBLE_MAX_EXPONENT_POWER_OF_TWO = Double.MAX_EXPONENT;
	final static int DOUBLE_EXPONENT_BIAS = 1023;

	/**
	 * When mapping numbers from decimal to binary, we go from w * 10^q to m * 2^p, but we have 10^q = 5^q * 2^q, so
	 * effectively we are trying to match w * 2^q * 5^q to m * 2^p.
	 * <p>
	 * Thus, the powers of two are not a concern since they can be represented exactly using the binary notation, only
	 * the powers of five affect the binary significand.
	 * <p>
	 * The mantissas of powers of ten from -308 to 308, extended out to sixty-four bits. The array contains the powers of
	 * ten approximated as a 64-bit mantissa. It goes from 10^{@value #DOUBLE_MIN_EXPONENT_POWER_OF_TEN} to
	 * 10^{@value #DOUBLE_MAX_EXPONENT_POWER_OF_TEN} (inclusively). The mantissa is truncated, and never rounded up. Uses
	 * about 5 KB.
	 * 
	 * <pre>
	 * long getMantissaHigh(int q) {
	 *  MANTISSA_64[q - SMALLEST_POWER_OF_TEN];
	 * }
	 * </pre>
	 */
	static final long[] MANTISSA_64 = {0xa5ced43b7e3e9188L, 0xcf42894a5dce35eaL, 0x818995ce7aa0e1b2L,
		0xa1ebfb4219491a1fL, 0xca66fa129f9b60a6L, 0xfd00b897478238d0L, 0x9e20735e8cb16382L, 0xc5a890362fddbc62L,
		0xf712b443bbd52b7bL, 0x9a6bb0aa55653b2dL, 0xc1069cd4eabe89f8L, 0xf148440a256e2c76L, 0x96cd2a865764dbcaL,
		0xbc807527ed3e12bcL, 0xeba09271e88d976bL, 0x93445b8731587ea3L, 0xb8157268fdae9e4cL, 0xe61acf033d1a45dfL,
		0x8fd0c16206306babL, 0xb3c4f1ba87bc8696L, 0xe0b62e2929aba83cL, 0x8c71dcd9ba0b4925L, 0xaf8e5410288e1b6fL,
		0xdb71e91432b1a24aL, 0x892731ac9faf056eL, 0xab70fe17c79ac6caL, 0xd64d3d9db981787dL, 0x85f0468293f0eb4eL,
		0xa76c582338ed2621L, 0xd1476e2c07286faaL, 0x82cca4db847945caL, 0xa37fce126597973cL, 0xcc5fc196fefd7d0cL,
		0xff77b1fcbebcdc4fL, 0x9faacf3df73609b1L, 0xc795830d75038c1dL, 0xf97ae3d0d2446f25L, 0x9becce62836ac577L,
		0xc2e801fb244576d5L, 0xf3a20279ed56d48aL, 0x9845418c345644d6L, 0xbe5691ef416bd60cL, 0xedec366b11c6cb8fL,
		0x94b3a202eb1c3f39L, 0xb9e08a83a5e34f07L, 0xe858ad248f5c22c9L, 0x91376c36d99995beL, 0xb58547448ffffb2dL,
		0xe2e69915b3fff9f9L, 0x8dd01fad907ffc3bL, 0xb1442798f49ffb4aL, 0xdd95317f31c7fa1dL, 0x8a7d3eef7f1cfc52L,
		0xad1c8eab5ee43b66L, 0xd863b256369d4a40L, 0x873e4f75e2224e68L, 0xa90de3535aaae202L, 0xd3515c2831559a83L,
		0x8412d9991ed58091L, 0xa5178fff668ae0b6L, 0xce5d73ff402d98e3L, 0x80fa687f881c7f8eL, 0xa139029f6a239f72L,
		0xc987434744ac874eL, 0xfbe9141915d7a922L, 0x9d71ac8fada6c9b5L, 0xc4ce17b399107c22L, 0xf6019da07f549b2bL,
		0x99c102844f94e0fbL, 0xc0314325637a1939L, 0xf03d93eebc589f88L, 0x96267c7535b763b5L, 0xbbb01b9283253ca2L,
		0xea9c227723ee8bcbL, 0x92a1958a7675175fL, 0xb749faed14125d36L, 0xe51c79a85916f484L, 0x8f31cc0937ae58d2L,
		0xb2fe3f0b8599ef07L, 0xdfbdcece67006ac9L, 0x8bd6a141006042bdL, 0xaecc49914078536dL, 0xda7f5bf590966848L,
		0x888f99797a5e012dL, 0xaab37fd7d8f58178L, 0xd5605fcdcf32e1d6L, 0x855c3be0a17fcd26L, 0xa6b34ad8c9dfc06fL,
		0xd0601d8efc57b08bL, 0x823c12795db6ce57L, 0xa2cb1717b52481edL, 0xcb7ddcdda26da268L, 0xfe5d54150b090b02L,
		0x9efa548d26e5a6e1L, 0xc6b8e9b0709f109aL, 0xf867241c8cc6d4c0L, 0x9b407691d7fc44f8L, 0xc21094364dfb5636L,
		0xf294b943e17a2bc4L, 0x979cf3ca6cec5b5aL, 0xbd8430bd08277231L, 0xece53cec4a314ebdL, 0x940f4613ae5ed136L,
		0xb913179899f68584L, 0xe757dd7ec07426e5L, 0x9096ea6f3848984fL, 0xb4bca50b065abe63L, 0xe1ebce4dc7f16dfbL,
		0x8d3360f09cf6e4bdL, 0xb080392cc4349decL, 0xdca04777f541c567L, 0x89e42caaf9491b60L, 0xac5d37d5b79b6239L,
		0xd77485cb25823ac7L, 0x86a8d39ef77164bcL, 0xa8530886b54dbdebL, 0xd267caa862a12d66L, 0x8380dea93da4bc60L,
		0xa46116538d0deb78L, 0xcd795be870516656L, 0x806bd9714632dff6L, 0xa086cfcd97bf97f3L, 0xc8a883c0fdaf7df0L,
		0xfad2a4b13d1b5d6cL, 0x9cc3a6eec6311a63L, 0xc3f490aa77bd60fcL, 0xf4f1b4d515acb93bL, 0x991711052d8bf3c5L,
		0xbf5cd54678eef0b6L, 0xef340a98172aace4L, 0x9580869f0e7aac0eL, 0xbae0a846d2195712L, 0xe998d258869facd7L,
		0x91ff83775423cc06L, 0xb67f6455292cbf08L, 0xe41f3d6a7377eecaL, 0x8e938662882af53eL, 0xb23867fb2a35b28dL,
		0xdec681f9f4c31f31L, 0x8b3c113c38f9f37eL, 0xae0b158b4738705eL, 0xd98ddaee19068c76L, 0x87f8a8d4cfa417c9L,
		0xa9f6d30a038d1dbcL, 0xd47487cc8470652bL, 0x84c8d4dfd2c63f3bL, 0xa5fb0a17c777cf09L, 0xcf79cc9db955c2ccL,
		0x81ac1fe293d599bfL, 0xa21727db38cb002fL, 0xca9cf1d206fdc03bL, 0xfd442e4688bd304aL, 0x9e4a9cec15763e2eL,
		0xc5dd44271ad3cdbaL, 0xf7549530e188c128L, 0x9a94dd3e8cf578b9L, 0xc13a148e3032d6e7L, 0xf18899b1bc3f8ca1L,
		0x96f5600f15a7b7e5L, 0xbcb2b812db11a5deL, 0xebdf661791d60f56L, 0x936b9fcebb25c995L, 0xb84687c269ef3bfbL,
		0xe65829b3046b0afaL, 0x8ff71a0fe2c2e6dcL, 0xb3f4e093db73a093L, 0xe0f218b8d25088b8L, 0x8c974f7383725573L,
		0xafbd2350644eeacfL, 0xdbac6c247d62a583L, 0x894bc396ce5da772L, 0xab9eb47c81f5114fL, 0xd686619ba27255a2L,
		0x8613fd0145877585L, 0xa798fc4196e952e7L, 0xd17f3b51fca3a7a0L, 0x82ef85133de648c4L, 0xa3ab66580d5fdaf5L,
		0xcc963fee10b7d1b3L, 0xffbbcfe994e5c61fL, 0x9fd561f1fd0f9bd3L, 0xc7caba6e7c5382c8L, 0xf9bd690a1b68637bL,
		0x9c1661a651213e2dL, 0xc31bfa0fe5698db8L, 0xf3e2f893dec3f126L, 0x986ddb5c6b3a76b7L, 0xbe89523386091465L,
		0xee2ba6c0678b597fL, 0x94db483840b717efL, 0xba121a4650e4ddebL, 0xe896a0d7e51e1566L, 0x915e2486ef32cd60L,
		0xb5b5ada8aaff80b8L, 0xe3231912d5bf60e6L, 0x8df5efabc5979c8fL, 0xb1736b96b6fd83b3L, 0xddd0467c64bce4a0L,
		0x8aa22c0dbef60ee4L, 0xad4ab7112eb3929dL, 0xd89d64d57a607744L, 0x87625f056c7c4a8bL, 0xa93af6c6c79b5d2dL,
		0xd389b47879823479L, 0x843610cb4bf160cbL, 0xa54394fe1eedb8feL, 0xce947a3da6a9273eL, 0x811ccc668829b887L,
		0xa163ff802a3426a8L, 0xc9bcff6034c13052L, 0xfc2c3f3841f17c67L, 0x9d9ba7832936edc0L, 0xc5029163f384a931L,
		0xf64335bcf065d37dL, 0x99ea0196163fa42eL, 0xc06481fb9bcf8d39L, 0xf07da27a82c37088L, 0x964e858c91ba2655L,
		0xbbe226efb628afeaL, 0xeadab0aba3b2dbe5L, 0x92c8ae6b464fc96fL, 0xb77ada0617e3bbcbL, 0xe55990879ddcaabdL,
		0x8f57fa54c2a9eab6L, 0xb32df8e9f3546564L, 0xdff9772470297ebdL, 0x8bfbea76c619ef36L, 0xaefae51477a06b03L,
		0xdab99e59958885c4L, 0x88b402f7fd75539bL, 0xaae103b5fcd2a881L, 0xd59944a37c0752a2L, 0x857fcae62d8493a5L,
		0xa6dfbd9fb8e5b88eL, 0xd097ad07a71f26b2L, 0x825ecc24c873782fL, 0xa2f67f2dfa90563bL, 0xcbb41ef979346bcaL,
		0xfea126b7d78186bcL, 0x9f24b832e6b0f436L, 0xc6ede63fa05d3143L, 0xf8a95fcf88747d94L, 0x9b69dbe1b548ce7cL,
		0xc24452da229b021bL, 0xf2d56790ab41c2a2L, 0x97c560ba6b0919a5L, 0xbdb6b8e905cb600fL, 0xed246723473e3813L,
		0x9436c0760c86e30bL, 0xb94470938fa89bceL, 0xe7958cb87392c2c2L, 0x90bd77f3483bb9b9L, 0xb4ecd5f01a4aa828L,
		0xe2280b6c20dd5232L, 0x8d590723948a535fL, 0xb0af48ec79ace837L, 0xdcdb1b2798182244L, 0x8a08f0f8bf0f156bL,
		0xac8b2d36eed2dac5L, 0xd7adf884aa879177L, 0x86ccbb52ea94baeaL, 0xa87fea27a539e9a5L, 0xd29fe4b18e88640eL,
		0x83a3eeeef9153e89L, 0xa48ceaaab75a8e2bL, 0xcdb02555653131b6L, 0x808e17555f3ebf11L, 0xa0b19d2ab70e6ed6L,
		0xc8de047564d20a8bL, 0xfb158592be068d2eL, 0x9ced737bb6c4183dL, 0xc428d05aa4751e4cL, 0xf53304714d9265dfL,
		0x993fe2c6d07b7fabL, 0xbf8fdb78849a5f96L, 0xef73d256a5c0f77cL, 0x95a8637627989aadL, 0xbb127c53b17ec159L,
		0xe9d71b689dde71afL, 0x9226712162ab070dL, 0xb6b00d69bb55c8d1L, 0xe45c10c42a2b3b05L, 0x8eb98a7a9a5b04e3L,
		0xb267ed1940f1c61cL, 0xdf01e85f912e37a3L, 0x8b61313bbabce2c6L, 0xae397d8aa96c1b77L, 0xd9c7dced53c72255L,
		0x881cea14545c7575L, 0xaa242499697392d2L, 0xd4ad2dbfc3d07787L, 0x84ec3c97da624ab4L, 0xa6274bbdd0fadd61L,
		0xcfb11ead453994baL, 0x81ceb32c4b43fcf4L, 0xa2425ff75e14fc31L, 0xcad2f7f5359a3b3eL, 0xfd87b5f28300ca0dL,
		0x9e74d1b791e07e48L, 0xc612062576589ddaL, 0xf79687aed3eec551L, 0x9abe14cd44753b52L, 0xc16d9a0095928a27L,
		0xf1c90080baf72cb1L, 0x971da05074da7beeL, 0xbce5086492111aeaL, 0xec1e4a7db69561a5L, 0x9392ee8e921d5d07L,
		0xb877aa3236a4b449L, 0xe69594bec44de15bL, 0x901d7cf73ab0acd9L, 0xb424dc35095cd80fL, 0xe12e13424bb40e13L,
		0x8cbccc096f5088cbL, 0xafebff0bcb24aafeL, 0xdbe6fecebdedd5beL, 0x89705f4136b4a597L, 0xabcc77118461cefcL,
		0xd6bf94d5e57a42bcL, 0x8637bd05af6c69b5L, 0xa7c5ac471b478423L, 0xd1b71758e219652bL, 0x83126e978d4fdf3bL,
		0xa3d70a3d70a3d70aL, 0xccccccccccccccccL, 0x8000000000000000L, 0xa000000000000000L, 0xc800000000000000L,
		0xfa00000000000000L, 0x9c40000000000000L, 0xc350000000000000L, 0xf424000000000000L, 0x9896800000000000L,
		0xbebc200000000000L, 0xee6b280000000000L, 0x9502f90000000000L, 0xba43b74000000000L, 0xe8d4a51000000000L,
		0x9184e72a00000000L, 0xb5e620f480000000L, 0xe35fa931a0000000L, 0x8e1bc9bf04000000L, 0xb1a2bc2ec5000000L,
		0xde0b6b3a76400000L, 0x8ac7230489e80000L, 0xad78ebc5ac620000L, 0xd8d726b7177a8000L, 0x878678326eac9000L,
		0xa968163f0a57b400L, 0xd3c21bcecceda100L, 0x84595161401484a0L, 0xa56fa5b99019a5c8L, 0xcecb8f27f4200f3aL,
		0x813f3978f8940984L, 0xa18f07d736b90be5L, 0xc9f2c9cd04674edeL, 0xfc6f7c4045812296L, 0x9dc5ada82b70b59dL,
		0xc5371912364ce305L, 0xf684df56c3e01bc6L, 0x9a130b963a6c115cL, 0xc097ce7bc90715b3L, 0xf0bdc21abb48db20L,
		0x96769950b50d88f4L, 0xbc143fa4e250eb31L, 0xeb194f8e1ae525fdL, 0x92efd1b8d0cf37beL, 0xb7abc627050305adL,
		0xe596b7b0c643c719L, 0x8f7e32ce7bea5c6fL, 0xb35dbf821ae4f38bL, 0xe0352f62a19e306eL, 0x8c213d9da502de45L,
		0xaf298d050e4395d6L, 0xdaf3f04651d47b4cL, 0x88d8762bf324cd0fL, 0xab0e93b6efee0053L, 0xd5d238a4abe98068L,
		0x85a36366eb71f041L, 0xa70c3c40a64e6c51L, 0xd0cf4b50cfe20765L, 0x82818f1281ed449fL, 0xa321f2d7226895c7L,
		0xcbea6f8ceb02bb39L, 0xfee50b7025c36a08L, 0x9f4f2726179a2245L, 0xc722f0ef9d80aad6L, 0xf8ebad2b84e0d58bL,
		0x9b934c3b330c8577L, 0xc2781f49ffcfa6d5L, 0xf316271c7fc3908aL, 0x97edd871cfda3a56L, 0xbde94e8e43d0c8ecL,
		0xed63a231d4c4fb27L, 0x945e455f24fb1cf8L, 0xb975d6b6ee39e436L, 0xe7d34c64a9c85d44L, 0x90e40fbeea1d3a4aL,
		0xb51d13aea4a488ddL, 0xe264589a4dcdab14L, 0x8d7eb76070a08aecL, 0xb0de65388cc8ada8L, 0xdd15fe86affad912L,
		0x8a2dbf142dfcc7abL, 0xacb92ed9397bf996L, 0xd7e77a8f87daf7fbL, 0x86f0ac99b4e8dafdL, 0xa8acd7c0222311bcL,
		0xd2d80db02aabd62bL, 0x83c7088e1aab65dbL, 0xa4b8cab1a1563f52L, 0xcde6fd5e09abcf26L, 0x80b05e5ac60b6178L,
		0xa0dc75f1778e39d6L, 0xc913936dd571c84cL, 0xfb5878494ace3a5fL, 0x9d174b2dcec0e47bL, 0xc45d1df942711d9aL,
		0xf5746577930d6500L, 0x9968bf6abbe85f20L, 0xbfc2ef456ae276e8L, 0xefb3ab16c59b14a2L, 0x95d04aee3b80ece5L,
		0xbb445da9ca61281fL, 0xea1575143cf97226L, 0x924d692ca61be758L, 0xb6e0c377cfa2e12eL, 0xe498f455c38b997aL,
		0x8edf98b59a373fecL, 0xb2977ee300c50fe7L, 0xdf3d5e9bc0f653e1L, 0x8b865b215899f46cL, 0xae67f1e9aec07187L,
		0xda01ee641a708de9L, 0x884134fe908658b2L, 0xaa51823e34a7eedeL, 0xd4e5e2cdc1d1ea96L, 0x850fadc09923329eL,
		0xa6539930bf6bff45L, 0xcfe87f7cef46ff16L, 0x81f14fae158c5f6eL, 0xa26da3999aef7749L, 0xcb090c8001ab551cL,
		0xfdcb4fa002162a63L, 0x9e9f11c4014dda7eL, 0xc646d63501a1511dL, 0xf7d88bc24209a565L, 0x9ae757596946075fL,
		0xc1a12d2fc3978937L, 0xf209787bb47d6b84L, 0x9745eb4d50ce6332L, 0xbd176620a501fbffL, 0xec5d3fa8ce427affL,
		0x93ba47c980e98cdfL, 0xb8a8d9bbe123f017L, 0xe6d3102ad96cec1dL, 0x9043ea1ac7e41392L, 0xb454e4a179dd1877L,
		0xe16a1dc9d8545e94L, 0x8ce2529e2734bb1dL, 0xb01ae745b101e9e4L, 0xdc21a1171d42645dL, 0x899504ae72497ebaL,
		0xabfa45da0edbde69L, 0xd6f8d7509292d603L, 0x865b86925b9bc5c2L, 0xa7f26836f282b732L, 0xd1ef0244af2364ffL,
		0x8335616aed761f1fL, 0xa402b9c5a8d3a6e7L, 0xcd036837130890a1L, 0x802221226be55a64L, 0xa02aa96b06deb0fdL,
		0xc83553c5c8965d3dL, 0xfa42a8b73abbf48cL, 0x9c69a97284b578d7L, 0xc38413cf25e2d70dL, 0xf46518c2ef5b8cd1L,
		0x98bf2f79d5993802L, 0xbeeefb584aff8603L, 0xeeaaba2e5dbf6784L, 0x952ab45cfa97a0b2L, 0xba756174393d88dfL,
		0xe912b9d1478ceb17L, 0x91abb422ccb812eeL, 0xb616a12b7fe617aaL, 0xe39c49765fdf9d94L, 0x8e41ade9fbebc27dL,
		0xb1d219647ae6b31cL, 0xde469fbd99a05fe3L, 0x8aec23d680043beeL, 0xada72ccc20054ae9L, 0xd910f7ff28069da4L,
		0x87aa9aff79042286L, 0xa99541bf57452b28L, 0xd3fa922f2d1675f2L, 0x847c9b5d7c2e09b7L, 0xa59bc234db398c25L,
		0xcf02b2c21207ef2eL, 0x8161afb94b44f57dL, 0xa1ba1ba79e1632dcL, 0xca28a291859bbf93L, 0xfcb2cb35e702af78L,
		0x9defbf01b061adabL, 0xc56baec21c7a1916L, 0xf6c69a72a3989f5bL, 0x9a3c2087a63f6399L, 0xc0cb28a98fcf3c7fL,
		0xf0fdf2d3f3c30b9fL, 0x969eb7c47859e743L, 0xbc4665b596706114L, 0xeb57ff22fc0c7959L, 0x9316ff75dd87cbd8L,
		0xb7dcbf5354e9beceL, 0xe5d3ef282a242e81L, 0x8fa475791a569d10L, 0xb38d92d760ec4455L, 0xe070f78d3927556aL,
		0x8c469ab843b89562L, 0xaf58416654a6babbL, 0xdb2e51bfe9d0696aL, 0x88fcf317f22241e2L, 0xab3c2fddeeaad25aL,
		0xd60b3bd56a5586f1L, 0x85c7056562757456L, 0xa738c6bebb12d16cL, 0xd106f86e69d785c7L, 0x82a45b450226b39cL,
		0xa34d721642b06084L, 0xcc20ce9bd35c78a5L, 0xff290242c83396ceL, 0x9f79a169bd203e41L, 0xc75809c42c684dd1L,
		0xf92e0c3537826145L, 0x9bbcc7a142b17ccbL, 0xc2abf989935ddbfeL, 0xf356f7ebf83552feL, 0x98165af37b2153deL,
		0xbe1bf1b059e9a8d6L, 0xeda2ee1c7064130cL, 0x9485d4d1c63e8be7L, 0xb9a74a0637ce2ee1L, 0xe8111c87c5c1ba99L,
		0x910ab1d4db9914a0L, 0xb54d5e4a127f59c8L, 0xe2a0b5dc971f303aL, 0x8da471a9de737e24L, 0xb10d8e1456105dadL,
		0xdd50f1996b947518L, 0x8a5296ffe33cc92fL, 0xace73cbfdc0bfb7bL, 0xd8210befd30efa5aL, 0x8714a775e3e95c78L,
		0xa8d9d1535ce3b396L, 0xd31045a8341ca07cL, 0x83ea2b892091e44dL, 0xa4e4b66b68b65d60L, 0xce1de40642e3f4b9L,
		0x80d2ae83e9ce78f3L, 0xa1075a24e4421730L, 0xc94930ae1d529cfcL, 0xfb9b7cd9a4a7443cL, 0x9d412e0806e88aa5L,
		0xc491798a08a2ad4eL, 0xf5b5d7ec8acb58a2L, 0x9991a6f3d6bf1765L, 0xbff610b0cc6edd3fL, 0xeff394dcff8a948eL,
		0x95f83d0a1fb69cd9L, 0xbb764c4ca7a4440fL, 0xea53df5fd18d5513L, 0x92746b9be2f8552cL, 0xb7118682dbb66a77L,
		0xe4d5e82392a40515L, 0x8f05b1163ba6832dL, 0xb2c71d5bca9023f8L, 0xdf78e4b2bd342cf6L, 0x8bab8eefb6409c1aL,
		0xae9672aba3d0c320L, 0xda3c0f568cc4f3e8L, 0x8865899617fb1871L, 0xaa7eebfb9df9de8dL, 0xd51ea6fa85785631L,
		0x8533285c936b35deL, 0xa67ff273b8460356L, 0xd01fef10a657842cL, 0x8213f56a67f6b29bL, 0xa298f2c501f45f42L,
		0xcb3f2f7642717713L, 0xfe0efb53d30dd4d7L, 0x9ec95d1463e8a506L, 0xc67bb4597ce2ce48L, 0xf81aa16fdc1b81daL,
		0x9b10a4e5e9913128L, 0xc1d4ce1f63f57d72L, 0xf24a01a73cf2dccfL, 0x976e41088617ca01L, 0xbd49d14aa79dbc82L,
		0xec9c459d51852ba2L, 0x93e1ab8252f33b45L, 0xb8da1662e7b00a17L, 0xe7109bfba19c0c9dL, 0x906a617d450187e2L,
		0xb484f9dc9641e9daL, 0xe1a63853bbd26451L, 0x8d07e33455637eb2L, 0xb049dc016abc5e5fL, 0xdc5c5301c56b75f7L,
		0x89b9b3e11b6329baL, 0xac2820d9623bf429L, 0xd732290fbacaf133L, 0x867f59a9d4bed6c0L, 0xa81f301449ee8c70L,
		0xd226fc195c6a2f8cL, 0x83585d8fd9c25db7L, 0xa42e74f3d032f525L, 0xcd3a1230c43fb26fL, 0x80444b5e7aa7cf85L,
		0xa0555e361951c366L, 0xc86ab5c39fa63440L, 0xfa856334878fc150L, 0x9c935e00d4b9d8d2L, 0xc3b8358109e84f07L,
		0xf4a642e14c6262c8L, 0x98e7e9cccfbd7dbdL, 0xbf21e44003acdd2cL, 0xeeea5d5004981478L, 0x95527a5202df0ccbL,
		0xbaa718e68396cffdL, 0xe950df20247c83fdL, 0x91d28b7416cdd27eL, 0xb6472e511c81471dL, 0xe3d8f9e563a198e5L,
		0x8e679c2f5e44ff8fL};

	public static double parseFloatingPointLiteral(String str, int offset, int endIndex) {
		if(endIndex > 100)// long string
			return Double.parseDouble(str);
		// Skip leading whitespace
		int index = skipWhitespace(str, offset, endIndex);
		char ch = str.charAt(index);

		// Parse optional sign
		final boolean isNegative = ch == '-';
		if(isNegative || ch == '+') {
			ch = charAt(str, ++index, endIndex);
		}

		// Parse NaN or Infinity (this occurs rarely)
		// : is the first character after numbers.
		// 0 is the first number.
		// we use the last position, since this is not allowed to be other values than a number.
		if(str.charAt(endIndex - 1) > '9' || str.charAt(endIndex - 1) < '0') 
			return Double.parseDouble(str);

		final double val = parseDecFloatLiteral(str, index, offset, endIndex);
		if(Double.isNaN(val))
			return Double.parseDouble(str);
		return isNegative ? -val : val;
	}

	private static void illegal() {
		throw new NumberFormatException("illegal syntax");
	}

	private static int inc(int significand, char ch) {
		return 10 * significand + ch - '0';
	}

	private static long inc(long significand, char ch) {
		return 10 * significand + ch - '0';
	}

	private static double parseDecFloatLiteral(String str, int index, int startIndex, int endIndex) {

		long significand = 0;
		final int significandStartIndex = index;
		int virtualIndexOfPoint = -1;
		char ch = 0;
		for(; index < endIndex; index++) {
			ch = str.charAt(index);
			if(isDigit(ch)) {
				// This might overflow, we deal with it later.
				significand = inc(significand, ch);
			}
			else if(ch == '.') {
				if(virtualIndexOfPoint >= 0)
					illegal();
				virtualIndexOfPoint = index;
			}
			else if((ch | 0x20) == 'e') {
				break; // case of e
			}
			else
				illegal();
		}

		final int digitCount;
		final int significandEndIndex = index;
		int exponent;
		if(virtualIndexOfPoint < 0) {
			digitCount = significandEndIndex - significandStartIndex;
			virtualIndexOfPoint = significandEndIndex;
			exponent = 0;
		}
		else {
			digitCount = significandEndIndex - significandStartIndex - 1;
			exponent = virtualIndexOfPoint - significandEndIndex + 1;
		}

		if((ch | 0x20) == 'e')
			return handleExponent(str, index, startIndex, endIndex, significandStartIndex, significandEndIndex,
				virtualIndexOfPoint, significand, ch, exponent, digitCount);
		else
			return handleOverflow(str, index, startIndex, endIndex, digitCount, significand, significandStartIndex,
				significandEndIndex, exponent, 0, virtualIndexOfPoint);

	}

	private static Double handleExponent(String str, int index, int startIndex, int endIndex, int sigStart, int sigEnd,
		final int virtualIndexOfPoint, final long significand, char ch, int exponent, int digitCount) {

		int expNumber = 0;
		ch = charAt(str, ++index, endIndex);

		// skip + or - sign
		final boolean isExponentNegative = ch == '-';
		if(isExponentNegative || ch == '+')
			ch = charAt(str, ++index, endIndex);

		do {
			if(!isDigit(ch))
				illegal();
			// Guard against overflow
			if(expNumber < MAX_EXPONENT_NUMBER)
				expNumber = inc(expNumber, ch);

			ch = charAt(str, ++index, endIndex);
		}
		while(index < endIndex);

		exponent += isExponentNegative ? -expNumber : expNumber;

		return handleOverflow(str, index, startIndex, endIndex, digitCount, significand, sigStart, sigEnd, exponent,
			expNumber, virtualIndexOfPoint);
	}

	private static double handleOverflow(String str, int index, int startIndex, int endIndex, int digitCount,
		long significand, int sigStart, int sigEnd, int exponent, int expNumber, int virtualIndexOfPoint) {
		if(digitCount > 19) {
			char ch;
			int skipCountInTruncatedDigits = 0;// counts +1 if we skipped over the decimal point
			significand = 0;
			for(index = sigStart; index < sigEnd; index++) {
				ch = str.charAt(index);
				if(ch == '.') {
					skipCountInTruncatedDigits++;
				}
				else {
					if(Long.compareUnsigned(significand, MINIMAL_NINETEEN_DIGIT_INTEGER) < 0) {
						significand = inc(significand, ch);
					}
					else {
						break;
					}
				}
			}
			final boolean isSignificandTruncated = index < sigEnd;
			final int exponentOfTruncatedSignificand = virtualIndexOfPoint - index + skipCountInTruncatedDigits +
				expNumber;
			return tryDecFloatToDoubleTruncated(significand, exponent, isSignificandTruncated,
				exponentOfTruncatedSignificand);
		}
		else {
			return tryDecFloatToDoubleTruncated(significand, exponent, false, 0);
		}
	}

	private static int skipWhitespace(String str, int index, int endIndex) {
		while(index < endIndex && str.charAt(index) <= ' ') {
			index++;
		}
		return index;
	}

	private static double tryDecFloatToDoubleTruncated(long significand, int exponent, boolean isSignificandTruncated,
		final int exponentOfTruncatedSignificand) {

		final double result;
		if(isSignificandTruncated) {
			// We have too many digits. We may have to round up.
			// To know whether rounding up is needed, we may have to examine up to 768 digits.

			// There are cases, in which rounding has no effect.
			if(DOUBLE_MIN_EXPONENT_POWER_OF_TEN <= exponentOfTruncatedSignificand &&
				exponentOfTruncatedSignificand <= DOUBLE_MAX_EXPONENT_POWER_OF_TEN) {
				double withoutRounding = tryDecToDoubleWithFastAlgorithm(significand, exponentOfTruncatedSignificand);
				double roundedUp = tryDecToDoubleWithFastAlgorithm(significand + 1, exponentOfTruncatedSignificand);
				if(!Double.isNaN(withoutRounding) && roundedUp == withoutRounding) {
					return withoutRounding;
				}
			}

			// We have to take a slow path.
			result = Double.NaN;

		}
		else if(DOUBLE_MIN_EXPONENT_POWER_OF_TEN <= exponent && exponent <= DOUBLE_MAX_EXPONENT_POWER_OF_TEN) {
			result = tryDecToDoubleWithFastAlgorithm(significand, exponent);
		}
		else {
			result = Double.NaN;
		}
		return result;
	}

	private static double shortcut(long significand, int power) {
		// convert the integer into a double. This is lossless since
		// 0 <= i <= 2^53 - 1.
		double d = (double) significand;
		// The general idea is as follows.
		// If 0 <= s < 2^53 and if 10^0 <= p <= 10^22 then
		// 1) Both s and p can be represented exactly as 64-bit floating-point values
		// 2) Because s and p can be represented exactly as floating-point values,
		// then s * p and s / p will produce correctly rounded values.

		if(power < 0) {
			d = d / DOUBLE_POWERS_OF_TEN[-power];
		}
		else {
			d = d * DOUBLE_POWERS_OF_TEN[power];
		}
		return d;
	}

	private static double tryDecToDoubleWithFastAlgorithm(long significand, int power) {
		// we start with a fast path
		// It was described in Clinger WD (1990).
		if(-22 <= power && power <= 22 && Long.compareUnsigned(significand, (1L << DOUBLE_SIGNIFICAND_WIDTH) - 1) <= 0) {
			return shortcut(significand, power);
		}

		// The fast path has now failed, so we are falling back on the slower path.

		// We are going to need to do some 64-bit arithmetic to get a more precise product.
		// We use a table lookup approach.
		// It is safe because
		// power >= DOUBLE_MIN_EXPONENT_POWER_OF_TEN
		// and power <= DOUBLE_MAX_EXPONENT_POWER_OF_TEN
		// We recover the mantissa of the power, it has a leading 1. It is always
		// rounded down.
		long factorMantissa = MANTISSA_64[power - DOUBLE_MIN_EXPONENT_POWER_OF_TEN];

		// The exponent is 1023 + 64 + power + floor(log(5**power)/log(2)).
		//
		// 1023 is the exponent bias.
		// The 64 comes from the fact that we use a 64-bit word.
		//
		// Computing floor(log(5**power)/log(2)) could be
		// slow. Instead, we use a fast function.
		//
		// For power in (-400,350), we have that
		// (((152170 + 65536) * power ) >> 16);
		// is equal to
		// floor(log(5**power)/log(2)) + power when power >= 0,
		// and it is equal to
		// ceil(log(5**-power)/log(2)) + power when power < 0
		//
		//
		// The 65536 is (1<<16) and corresponds to
		// (65536 * power) >> 16 ---> power
		//
		// ((152170 * power ) >> 16) is equal to
		// floor(log(5**power)/log(2))
		//
		// Note that this is not magic: 152170/(1<<16) is
		// approximately equal to log(5)/log(2).
		// The 1<<16 value is a power of two; we could use a
		// larger power of 2 if we wanted to.
		long exponent = (((152170L + 65536L) * power) >> 16) + DOUBLE_EXPONENT_BIAS + 64;
		// We want the most significant bit of digits to be 1. Shift if needed.
		int lz = Long.numberOfLeadingZeros(significand);
		long shiftedSignificand = significand << lz;
		// We want the most significant 64 bits of the product. We know
		// this will be non-zero because the most significant bit of digits is
		// 1.
		UInt128 product = fullMultiplication(shiftedSignificand, factorMantissa);
		long upper = product.high;

		// The computed 'product' is always sufficient.
		// Mathematical proof:
		// Noble Mushtak and Daniel Lemire, Fast Number Parsing Without Fallback (to appear)

		// The final mantissa should be 53 bits with a leading 1.
		// We shift it so that it occupies 54 bits with a leading 1.
		long upperbit = upper >>> 63;
		long mantissa = upper >>> (upperbit + 9);
		lz += (int) (1 ^ upperbit);

		if(((upper & 0x1ff) == 0x1ff) || ((upper & 0x1ff) == 0) && (mantissa & 3) == 1) {
			// default to java double parser if we are unsure about rounding
			return Double.NaN;
		}

		mantissa += 1;
		mantissa >>>= 1;

		// Here we have mantissa < (1<<53), unless there was an overflow
		if(mantissa >= (1L << DOUBLE_SIGNIFICAND_WIDTH)) {
			// This will happen when parsing values such as 7.2057594037927933e+16
			mantissa = (1L << (DOUBLE_SIGNIFICAND_WIDTH - 1));
			lz--; // undo previous addition
		}

		mantissa &= ~(1L << (DOUBLE_SIGNIFICAND_WIDTH - 1));

		long realExponent = exponent - lz;
		// we have to check that realExponent is in range, otherwise we bail out
		if((realExponent < 1) || (realExponent > DOUBLE_MAX_EXPONENT_POWER_OF_TWO + DOUBLE_EXPONENT_BIAS)) {
			return Double.NaN;
		}

		long bits = mantissa | realExponent << (DOUBLE_SIGNIFICAND_WIDTH - 1);
		return Double.longBitsToDouble(bits);
	}

	private static boolean isDigit(char c) {
		return (char) (c - '0') < 10;
	}

	private static char charAt(String str, int i, int endIndex) {
		return i < endIndex ? str.charAt(i) : 0;
	}

	/**
	 * Computes {@code uint128 product = (uint64)x * (uint64)y}.
	 * <p>
	 * References:
	 * <dl>
	 * <dt>Getting the high part of 64 bit integer multiplication</dt>
	 * <dd><a href="https://stackoverflow.com/questions/28868367/getting-the-high-part-of-64-bit-integer-multiplication">
	 * stackoverflow</a></dd>
	 * </dl>
	 *
	 * @param x uint64 factor x
	 * @param y uint64 factor y
	 * @return uint128 product of x and y
	 */
	private static UInt128 fullMultiplication(long x, long y) {
		return new UInt128(unsignedMultiplyHigh(x, y), x * y);
	}

	public static long unsignedMultiplyHigh(long x, long y) { // if we update to jave 18 use Math internal.
		// Compute via multiplyHigh() to leverage the intrinsic
		long result = Math.multiplyHigh(x, y);
		result += (y & (x >> 63)); // equivalent to `if (x < 0) result += y;`
		result += (x & (y >> 63)); // equivalent to `if (y < 0) result += x;`
		return result;
	}

	static class UInt128 {
		final long high, low;

		private UInt128(long high, long low) {
			this.high = high;
			this.low = low;
		}
	}
}
