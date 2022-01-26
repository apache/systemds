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

#ifndef LIBHE_H
#define LIBHE_H

#include <cassert>
#include <algorithm>
#include <optional>
#include <gsl/span>

#include "seal/seal.h"
#include "seal/util/common.h"
#include "seal/util/rlwe.h"
#include "seal/util/polyarithsmallmod.h"

using namespace std;
using namespace seal;

class RawPolynomData {
    vector<uint64_t > _data;
    size_t _size;

public:
    explicit RawPolynomData(const SEALContext& context);

    SEAL_NODISCARD inline const size_t& size() const { return _size; };
    SEAL_NODISCARD inline uint64_t* data() { return _data.data(); };
    SEAL_NODISCARD inline gsl::span<uint64_t > data_span() { return { data(), size() }; };

    void set_data(vector<uint64_t >& data);
};

gsl::span<Ciphertext::ct_coeff_type > data_span(Ciphertext& c, size_t n);

RawPolynomData generate_a(const SEALContext& context);

EncryptionParameters generateParameters();

size_t get_slot_count(const SEALContext& ctx);

// returns a vector filled with random double values between 0 and 1
vector<double> random_plaintext_data(size_t count);

struct GlobalState {
    SEALContext context;
    RawPolynomData a;
    double scale;

    explicit GlobalState(double _scale);
};

class Client {
    GlobalState _gs;
    CKKSEncoder _encoder;
    SecretKey _partial_secret_key;
    PublicKey _partial_public_key;
    std::optional<PublicKey> _public_key = std::nullopt;
    std::unique_ptr<Encryptor> _encryptor = nullptr;

    SEAL_NODISCARD static PublicKey generate_partial_public_key(const SecretKey &secret_key, const SEALContext &context, RawPolynomData& a);

public:
    explicit Client(GlobalState global_state);

    SEAL_NODISCARD inline const SEALContext& context() const { return _gs.context; };
    SEAL_NODISCARD inline const PublicKey& partial_public_key() const { return _partial_public_key; };
    SEAL_NODISCARD inline const CKKSEncoder& encoder() const { return _encoder; };
    SEAL_NODISCARD inline CKKSEncoder& encoder() { return _encoder; };
    SEAL_NODISCARD inline const Encryptor& encryptor() const { assert(_encryptor != nullptr); return *_encryptor; };
    SEAL_NODISCARD inline const PublicKey& public_key() { return *_public_key; };
    inline void set_public_key(const PublicKey& pk) { _public_key = make_optional(pk); };

    Ciphertext encrypted_data(gsl::span<const double> plain_data);

    Plaintext partial_decryption(const Ciphertext& encrypted);
};

// adds b to a in place
template<typename T> void sum_first_poly_inplace(const SEALContext& context, T& a, const T& b) {
    auto &context_data = *context.get_context_data(a.parms_id());
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    size_t coeff_count = parms.poly_modulus_degree();
    size_t coeff_modulus_size = coeff_modulus.size();

    // by dereferencing we get only the first poly
    auto summand_iter = *util::ConstPolyIter(b.data(), coeff_count, coeff_modulus_size);
    auto sum_iter = *util::ConstPolyIter(a.data(), coeff_count, coeff_modulus_size);
    auto result_iter = *util::PolyIter(a.data(), coeff_count, coeff_modulus_size);
    // see Evaluator::add_inplace
    util::add_poly_coeffmod(sum_iter, summand_iter, coeff_modulus_size, coeff_modulus, result_iter);
}

// This function adds the first polys in summands to sum (either Ciphertext or Plaintext).
template<typename T> T sum_first_polys_inplace(const SEALContext& context, T& sum, gsl::span<const T> summands) {
    for (size_t i = 0; i < summands.size(); i++) {
        sum_first_poly_inplace(context, sum, summands[i]);
    }
    return sum;
}

// This function sums the first polys in summands (either Ciphertext or Plaintext).
template<typename T> T sum_first_polys(const SEALContext& context, gsl::span<const T> summands) {
    T sum = summands[0];
    sum_first_polys_inplace(context, sum, gsl::span(&summands.data()[1], summands.size() - 1));
    return sum;
}

class Server {
    GlobalState _gs;
    PublicKey _public_key;

public:
    explicit Server(GlobalState global_state);

    SEAL_NODISCARD inline RawPolynomData& a() { return _gs.a; };
    SEAL_NODISCARD inline const SEALContext& context() const { return _gs.context; };
    SEAL_NODISCARD inline const PublicKey& public_key() const { return _public_key; };

    void accumulate_partial_public_keys(gsl::span<const Ciphertext> partial_pub_keys);

    Ciphertext sum_data(vector<Ciphertext>&& data) const;

    vector<double> average(const Ciphertext& encrypted_sum, gsl::span<const Plaintext> partial_decryptions) const;
};

#endif //LIBHE_H
