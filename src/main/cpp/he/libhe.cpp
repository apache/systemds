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

#include <cassert>
#include <algorithm>
#include <optional>
#include <gsl/span>

#include "libhe.h"

#include "seal/seal.h"
#include "seal/util/common.h"
#include "seal/util/rlwe.h"
#include "seal/util/polyarithsmallmod.h"

using namespace std;
using namespace seal;

RawPolynomData::RawPolynomData(const SEALContext& context) {
    // Extract encryption parameters
    auto &context_data = *context.key_context_data();
    auto &parms = context_data.parms();
    auto coeff_modulus = parms.coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t coeff_count = parms.poly_modulus_degree();
    _size = util::mul_safe(coeff_count, coeff_modulus_size);
};

void RawPolynomData::set_data(vector<uint64_t >& data) {
    assert(data.size() == _size);
    _data = move(data);
};


gsl::span<Ciphertext::ct_coeff_type > data_span(Ciphertext& c, size_t n) {
    size_t poly_size = util::mul_safe(c.poly_modulus_degree(), c.coeff_modulus_size());
    return { c.data(n), poly_size };
}

RawPolynomData generate_a(const SEALContext& context) {
    auto ciphertext_prng = UniformRandomGeneratorFactory::DefaultFactory()->create();

    auto &context_data = *context.key_context_data();
    auto &parms = context_data.parms();

    RawPolynomData rpd(parms);
    vector<uint64_t > a_poly_data(rpd.size());
    util::sample_poly_uniform(ciphertext_prng, parms, a_poly_data.data());
    rpd.set_data(a_poly_data);
    return rpd;
}

EncryptionParameters generateParameters() {
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 4096;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 54, 54 }));
    return parms;
}

size_t get_slot_count(const SEALContext& ctx) {
    // slot count is only half of it. but every slot can take one complex number or 2 doubles. so in the end we get twice
    // the space
    return ctx.first_context_data()->parms().poly_modulus_degree();
}

// returns a vector filled with random double values between 0 and 1
vector<double> random_plaintext_data(size_t count) {
    // this example is just copied from the CKKS example of SEAL
    vector<double> data;
    data.reserve(count);
    for (size_t i = 0; i < count; i++)
    {
        data.push_back(sqrt(static_cast<double>(rand()) / RAND_MAX));
    }
    return data;
}

GlobalState::GlobalState(double _scale) : context(generateParameters()), a(generate_a(context)), scale(_scale) {};


PublicKey Client::generate_partial_public_key(const SecretKey &secret_key, const SEALContext &context, RawPolynomData& a)
{
    PublicKey public_key;
    Ciphertext& destination = public_key.data();

    // We use a fresh memory pool with `clear_on_destruction' enabled.
    MemoryPoolHandle pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);

    auto &context_data = *context.key_context_data();
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t coeff_count = parms.poly_modulus_degree();
    auto ntt_tables = context_data.small_ntt_tables();
    size_t encrypted_size = 2;

    // If a polynomial is too small to store UniformRandomGeneratorInfo,
    // it is best to just disable save_seed. Note that the size needed is
    // the size of UniformRandomGeneratorInfo plus one (uint64_t) because
    // of an indicator word that indicates a seeded ciphertext.
    size_t poly_uint64_count = util::mul_safe(coeff_count, coeff_modulus_size);

    destination.resize(context, context.key_parms_id(), encrypted_size);
    destination.is_ntt_form() = true;
    destination.scale() = 1.0;

    // Create an instance of a random number generator. We use this for sampling
    // a seed for a second PRNG used for sampling u (the seed can be public
    // information. This PRNG is also used for sampling the noise/error below.
    auto bootstrap_prng = parms.random_generator()->create();

    // Sample a public seed for generating uniform randomness
    prng_seed_type public_prng_seed;
    bootstrap_prng->generate(prng_seed_byte_count, reinterpret_cast<seal_byte *>(public_prng_seed.data()));

    // Set up a new default PRNG for expanding u from the seed sampled above
    auto ciphertext_prng = UniformRandomGeneratorFactory::DefaultFactory()->create(public_prng_seed);

    // Generate ciphertext: (c[0], c[1]) = ([-(as+e)]_q, a)
    uint64_t *c0 = destination.data();
    uint64_t *c1 = destination.data(1);

    // copy a into c1
    assert(a.size() == poly_uint64_count);
    copy(a.data(), a.data()+poly_uint64_count, c1);

    // Sample e <-- chi
    auto noise(util::allocate_poly(coeff_count, coeff_modulus_size, pool));
    util::SEAL_NOISE_SAMPLER(bootstrap_prng, parms, noise.get());

    // Calculate -(a*s + e) (mod q) and store in c[0]
    for (size_t i = 0; i < coeff_modulus_size; i++)
    {
        util::dyadic_product_coeffmod(
                secret_key.data().data() + i * coeff_count, c1 + i * coeff_count, coeff_count, coeff_modulus[i],
                c0 + i * coeff_count);

        // Transform the noise e into NTT representation
        ntt_negacyclic_harvey(noise.get() + i * coeff_count, ntt_tables[i]);

        util::add_poly_coeffmod(
                noise.get() + i * coeff_count, c0 + i * coeff_count, coeff_count, coeff_modulus[i],
                c0 + i * coeff_count);
        util::negate_poly_coeffmod(c0 + i * coeff_count, coeff_count, coeff_modulus[i], c0 + i * coeff_count);
    }

    public_key.parms_id() = context.key_parms_id();
    return public_key;
}

Client::Client(GlobalState global_state) : _gs(move(global_state)), _encoder(_gs.context) {
    KeyGenerator keygen(_gs.context);
    _partial_secret_key = keygen.secret_key();
    _partial_public_key = generate_partial_public_key(_partial_secret_key, _gs.context, _gs.a);
};

Ciphertext Client::encrypted_data(gsl::span<const double> plain_data) {
    if (!_encryptor) {
        _encryptor = make_unique<Encryptor>(_gs.context, *_public_key);
    }

    // reinterpret plain data as complex<double>
    assert(plain_data.size() % 2 == 0);
    gsl::span complex_plain_data(reinterpret_cast<const complex<double>*>(plain_data.data()), plain_data.size() / 2);

    Plaintext plaintext;
    encoder().encode(complex_plain_data, _gs.scale, plaintext);
    Ciphertext ciphertext;
    encryptor().encrypt(plaintext, ciphertext);
    return ciphertext;
}

Plaintext Client::partial_decryption(const Ciphertext& encrypted) {
    using namespace seal::util;

    // c = (c0, c1)
    // dec(c) = c0+c1*s
    // we need: c0 + c1*sum(s[i])
    // so we return c1*s[i]*e[i] and add c0 at the server. e[i] is a noise term necessary for security

    // adapted from Decryptor::decrypt

    auto &context_data = *_gs.context.get_context_data(encrypted.parms_id());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    size_t coeff_count = parms.poly_modulus_degree();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t rns_poly_uint64_count = mul_safe(coeff_count, coeff_modulus_size);

    Plaintext plaintext;
    // Since we overwrite destination, we zeroize destination parameters
    // This is necessary, otherwise resize will throw an exception.
    plaintext.parms_id() = parms_id_zero;

    // Resize destination to appropriate size
    plaintext.resize(rns_poly_uint64_count);

    // Do the dot product of encrypted and the secret key array using NTT.
    RNSIter destination(plaintext.data(), coeff_count);
    ConstRNSIter secret_key_array(_partial_secret_key.data().data(), coeff_count);
    ConstRNSIter c1(encrypted.data(1), coeff_count);

    SEAL_ITERATE(
            iter(c1, secret_key_array, coeff_modulus, destination), coeff_modulus_size, [&](auto I) {
                // put < c_1 * s > mod q in destination
                dyadic_product_coeffmod(get<0>(I), get<1>(I), coeff_count, get<2>(I), get<3>(I));
            });

    // for security we need to introduce noise here
    // this part is based on rlwe.cpp:encrypt_zero_symmetric()
    auto prng = parms.random_generator()->create();
    MemoryPoolHandle pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);
    auto noise(allocate_poly(coeff_count, coeff_modulus_size, pool));
    SEAL_NOISE_SAMPLER(prng, parms, noise.get());
    auto ntt_tables = context_data.small_ntt_tables();

    for (size_t i = 0; i < coeff_modulus_size; i++)
    {
        // Transform the noise e into NTT representation
        ntt_negacyclic_harvey(noise.get() + i * coeff_count, ntt_tables[i]);

        add_poly_coeffmod(
                noise.get() + i * coeff_count, plaintext.data() + i * coeff_count, coeff_count, coeff_modulus[i],
                plaintext.data() + i * coeff_count);
    }

    // Set destination parameters as in encrypted
    plaintext.parms_id() = encrypted.parms_id();
    plaintext.scale() = encrypted.scale();
    return plaintext;
}

Server::Server(GlobalState global_state) : _gs(move(global_state)) {};

void Server::accumulate_partial_public_keys(gsl::span<const Ciphertext> partial_pub_keys) {
    // sum only the first poly of the ciphertexts
    // the second poly is always the same, see GlobalState.a
    Ciphertext sum = sum_first_polys(context(), partial_pub_keys);
    _public_key.data() = sum;
    assert(is_valid_for(_public_key, context()));
}

Ciphertext Server::sum_data(vector<Ciphertext>&& data) const {
    Evaluator e(_gs.context);
    Ciphertext result;
    e.add_many(data, result);
    return result;
}

vector<double> Server::average(const Ciphertext& encrypted_sum, gsl::span<const Plaintext> partial_decryptions) const {
    // the partial decryptions were of the form c1*s[i]. we need c0 + sum(c1+s[i])
    // so we need to add c0 once here.

    // FIXME: this copies encrypted_sum, which is unnecessary
    uint64_t num_coeffs = util::mul_safe(encrypted_sum.poly_modulus_degree(), encrypted_sum.coeff_modulus_size());
    gsl::span<const Plaintext::pt_coeff_type> es_data(encrypted_sum.data(0), num_coeffs);
    Plaintext c0(es_data);
    c0.parms_id() = context().first_parms_id();
    c0.scale() = encrypted_sum.scale();

    sum_first_polys_inplace(_gs.context, c0, partial_decryptions); // c0 + sum(c1+s[i])

    // decode sum
    size_t slot_count = context().first_context_data()->parms().poly_modulus_degree() >> 1;
    CKKSEncoder encoder(context());
    vector<double> result(slot_count * 2, 0.0);
    gsl::span<complex<double>> result_destination(reinterpret_cast<complex<double>*>(result.data()), slot_count);
    encoder.decode(c0, result_destination);

    // divide by N for average
    for (double& x : result) {
        x /= static_cast<double>(partial_decryptions.size());
    }
    return result;
}

