#include <iostream>
#include <vector>
#include <cstring>
#include "lz4.h"

// Simple CSR data
struct CSR {
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
};

// 직렬화
std::vector<char> serialize(const CSR& csr) {
    size_t total_size = csr.values.size() * sizeof(float)
                      + csr.col_indices.size() * sizeof(int)
                      + csr.row_ptr.size() * sizeof(int);

    std::vector<char> buffer(total_size);
    size_t offset = 0;

    std::memcpy(buffer.data() + offset, csr.values.data(), csr.values.size() * sizeof(float));
    offset += csr.values.size() * sizeof(float);

    std::memcpy(buffer.data() + offset, csr.col_indices.data(), csr.col_indices.size() * sizeof(int));
    offset += csr.col_indices.size() * sizeof(int);

    std::memcpy(buffer.data() + offset, csr.row_ptr.data(), csr.row_ptr.size() * sizeof(int));

    return buffer;
}

// 역직렬화
CSR deserialize(const std::vector<char>& buffer, size_t val_len, size_t col_len, size_t row_len) {
    CSR csr;
    csr.values.resize(val_len);
    csr.col_indices.resize(col_len);
    csr.row_ptr.resize(row_len);

    size_t offset = 0;

    std::memcpy(csr.values.data(), buffer.data() + offset, val_len * sizeof(float));
    offset += val_len * sizeof(float);

    std::memcpy(csr.col_indices.data(), buffer.data() + offset, col_len * sizeof(int));
    offset += col_len * sizeof(int);

    std::memcpy(csr.row_ptr.data(), buffer.data() + offset, row_len * sizeof(int));

    return csr;
}

int main() {
    // 예제 희소 행렬: values: [3, 22, 7, 5]
    CSR csr = {
        {3.0f, 22.0f, 7.0f, 5.0f},    // values
        {2, 0, 1, 3},                 // col_indices
        {0, 1, 2, 4}                  // row_ptr
    };

    auto serialized = serialize(csr);

    // 압축
    int max_dst_size = LZ4_compressBound(serialized.size());
    std::vector<char> compressed(max_dst_size);
    int compressed_size = LZ4_compress_default(
        serialized.data(), compressed.data(),
        serialized.size(), max_dst_size
    );
    compressed.resize(compressed_size);
    std::cout << "Original size: " << serialized.size() << ", Compressed size: " << compressed_size << "\n";

    // 복호화
    std::vector<char> decompressed(serialized.size());
    int decompressed_size = LZ4_decompress_safe(
        compressed.data(), decompressed.data(),
        compressed.size(), decompressed.size()
    );

    // 역직렬화
    CSR restored = deserialize(decompressed, 4, 4, 4);

    std::cout << "Restored values: ";
    for (float v : restored.values) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
