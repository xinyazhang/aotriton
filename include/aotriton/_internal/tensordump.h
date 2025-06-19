// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_FRAMEWORKS_TENSORDUMP_H
#define AMD_FRAMEWORKS_TENSORDUMP_H

#include <aotriton/config.h>
#include <string>
#include <fstream>

// Put libtensordump inside AOTRITON_NS to avoid potential conflicts when debugging PyTorch
namespace AOTRITON_NS::libtensordump {

template<typename DType>
inline const char* cast_dtype(DType t_dtype)
{
#define CAST_TYPE(aname, dtname) if (t_dtype == DType::aname) return #dtname;
  // CAST_TYPE(kByte, uint8);
  // CAST_TYPE(kChar, int8);
  // CAST_TYPE(kShort, int16);
  // CAST_TYPE(kInt, int32);
  // CAST_TYPE(kLong, int64);
  CAST_TYPE(kFloat16, float16);
  CAST_TYPE(kBFloat16, bfloat16);
  CAST_TYPE(kFloat32, float32);
  CAST_TYPE(kUInt8, uint8);
  CAST_TYPE(kUInt16, uint16);
  CAST_TYPE(kUInt32, uint32);
  CAST_TYPE(kUInt64, uint64);
  CAST_TYPE(kInt8, int8);
  CAST_TYPE(kInt16, int16);
  CAST_TYPE(kInt32, int32);
  CAST_TYPE(kInt64, int64);
  return "unknown";
#undef CAST_TYPE
}

template<typename DType>
inline size_t sizeof_dtype(DType t_dtype)
{
#define SIZEOF_TYPE(aname, dtname) if (t_dtype == DType::aname) return sizeof(dtname);
  // SIZEOF_TYPE(kByte, uint8_t);
  // SIZEOF_TYPE(kChar, int8_t);
  SIZEOF_TYPE(kFloat16, uint16_t);
  SIZEOF_TYPE(kBFloat16, uint16_t);
  SIZEOF_TYPE(kFloat32, uint32_t);
  SIZEOF_TYPE(kUInt8, uint8_t);
  SIZEOF_TYPE(kUInt16, uint16_t);
  SIZEOF_TYPE(kUInt32, uint32_t);
  SIZEOF_TYPE(kUInt64, uint64_t);
  SIZEOF_TYPE(kInt8, int8_t);
  SIZEOF_TYPE(kInt16, int16_t);
  SIZEOF_TYPE(kInt32, int32_t);
  SIZEOF_TYPE(kInt64, int64_t);
  return 0;
#undef SIZEOF_TYPE
}

template<typename Tensor>
size_t nbytes_tensor(const Tensor& t)
{
  size_t max_s = 0;
  size_t max_i = 0;
  const auto strides = t.strides();
  const auto sizes = t.sizes();
  for (size_t i = 0; i < strides.size(); i++) {
    auto s = strides[i];
    if (sizes[i] > 0 && max_s < s) {
      max_s = s;
      max_i = i;
    }
  }
  auto off = sizes[max_i];
  // can be (off * max_s) but the trailing space may be inaccesible
  size_t nelements = (off - 1) * max_s + off;
  return nelements * sizeof_dtype(t.dtype());
}

template<typename Tensor>
void metadump(const Tensor& t, std::fstream& fout)
{
  fout << R"zzz({ "sizes":[)zzz";
  const char* spacer = "";
  for (auto s : t.sizes()) {
    fout << spacer << s;
    spacer = ", ";
  }
  fout << R"zzz(], "strides": [)zzz";
  spacer = "";
  for (int s : t.strides()) {
    fout << spacer << s;
    spacer = ", ";
  }
  fout << R"zzz(], "dtype": ")zzz";
  fout << cast_dtype(t.dtype());
  fout << R"zzz(", "offset": )zzz";
  // fout << reinterpret_cast<const char*>(t.data_ptr()) - reinterpret_cast<const char*>(t.storage().data_ptr().get());
  fout << 0;  // TensorView does not have .storage(), we kept the line above in case of at::Tensor support
  fout << R"zzz(, "nbytes": )zzz";
  // fout << t.nbytes();  // at::Tensor. It is not implemented in TensorView because it is useless.
  fout << nbytes_tensor(t);
  fout << R"zzz(})zzz";
}

template<typename Tensor>
void datadump(const Tensor& t, std::fstream& fout)
{
#if 0
  fout.write(reinterpret_cast<char*>(t.storage().data_ptr().get()),
             t.storage().nbytes());
#endif
  fout.write(reinterpret_cast<const char*>(t.data_ptr()), nbytes_tensor(t));
}

template<typename Tensor>
void tensordump(const Tensor& t, std::string basename, int gpu_index, int call_index)
{
  basename += "-";
  basename += std::to_string(gpu_index);
  basename += ".";
  basename += std::to_string(call_index);
  {
    std::fstream fout(basename + ".json", std::ios::out | std::ios::trunc);
    metadump(t, fout);
  }
  {
    std::fstream fout(basename + ".tdata", std::ios::out | std::ios::binary | std::ios::trunc);
    datadump(t, fout);
  }
}

template<typename Scalar>
void scalardump(Scalar scalar, std::string basename, int gpu_index, int call_index)
{
  basename += "-";
  basename += std::to_string(gpu_index);
  basename += ".";
  basename += std::to_string(call_index);
  std::fstream fout(basename + ".scalar", std::ios::out | std::ios::trunc);
  fout.precision(19);
  fout << scalar;
}

}

#endif
