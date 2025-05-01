#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include "lz4.h"

// --------------------- CSR 정의 ---------------------
struct CSR {
    std::vector<float> values;
    std::vector<int>   col_idx;
    std::vector<int>   row_ptr;
};

// CSR 변환: 밀집 행렬 → CSR
CSR to_csr(const std::vector<float>& dense,
           int rows, int cols, float zero_tol = 0.0f)
{
    CSR csr;
    csr.row_ptr.reserve(rows + 1);
    csr.row_ptr.push_back(0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float v = dense[r*cols + c];
            if (std::abs(v) > zero_tol) {
                csr.values.push_back(v);
                csr.col_idx.push_back(c);
            }
        }
        csr.row_ptr.push_back((int)csr.values.size());
    }
    return csr;
}

// 직렬화 (CSR 버전)
std::vector<char> serialize_csr(const CSR& csr) {
    size_t sz_vals = csr.values.size() * sizeof(float);
    size_t sz_idx  = csr.col_idx.size() * sizeof(int);
    size_t sz_ptr  = csr.row_ptr.size() * sizeof(int);
    std::vector<char> buf(sz_vals + sz_idx + sz_ptr);
    size_t off = 0;
    std::memcpy(buf.data()+off, csr.values.data(), sz_vals); off += sz_vals;
    std::memcpy(buf.data()+off, csr.col_idx.data(), sz_idx);   off += sz_idx;
    std::memcpy(buf.data()+off, csr.row_ptr.data(), sz_ptr);
    return buf;
}

// 직렬화 (밀집 행렬 버전)
std::vector<char> serialize_dense(const std::vector<float>& dense) {
    const char* p = reinterpret_cast<const char*>(dense.data());
    return std::vector<char>(p, p + dense.size()*sizeof(float));
}

// 측정용 헬퍼
using Clock = std::chrono::high_resolution_clock;
template<typename F>
double measure_ms(F fn) {
    auto t0 = Clock::now();
    fn();
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    // 예제: 1000x1000 크기의 희소 행렬 생성
    const int ROWS = 1000, COLS = 1000;
    std::vector<float> dense(ROWS*COLS, 0.0f);
    // 임의의 희소 패턴: 대각선+몇 개 랜덤
    for (int i = 0; i < ROWS; i += 50) {
        dense[i*COLS + (i%COLS)] = i * 1.0f;
    }

#ifdef USE_CSR
    // 1) CSR 변환
    CSR csr = to_csr(dense, ROWS, COLS);
    auto buf       = serialize_csr(csr);
    std::cout << "[MODE] CSR + LZ4\n";
    std::cout << "  CSR values count: "   << csr.values.size()
              << ", col_idx count: "     << csr.col_idx.size()
              << ", row_ptr count: "     << csr.row_ptr.size() << "\n";
#else
    // 2) 밀집 그대로
    auto buf       = serialize_dense(dense);
    std::cout << "[MODE] Dense + LZ4\n";
    std::cout << "  Dense elements count: " << dense.size() << "\n";
#endif

    std::cout << "  Raw size: " << buf.size() << " bytes\n";

    // 3) 압축
    int max_dst = LZ4_compressBound((int)buf.size());
    std::vector<char> comp(max_dst);
    double t_comp = measure_ms([&](){
        int cs = LZ4_compress_default(buf.data(), comp.data(),
                                      (int)buf.size(), max_dst);
        comp.resize(cs);
    });
    std::cout << "  Compressed size: " << comp.size()
              << " bytes, time: " << t_comp << " ms\n";

    // 4) 복호화
    std::vector<char> decomp(buf.size());
    double t_decomp = measure_ms([&](){
        LZ4_decompress_safe(comp.data(), decomp.data(),
                            (int)comp.size(), (int)decomp.size());
    });
    std::cout << "  Decompressed time: " << t_decomp << " ms\n";

    // (선택) 복원 확인: 첫 몇 값 출력
    float* fptr = reinterpret_cast<float*>(decomp.data());
    std::cout << "  Sample values: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << fptr[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
