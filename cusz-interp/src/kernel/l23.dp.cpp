#include "detail/l23.dp.inl"

#include <chrono>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "cusz/type.h"
#include "kernel/lrz.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

template <typename T, typename Eq, typename FP>
pszerror psz_comp_l23(
    T* const data, sycl::range<3> const len3, PROPER_EB const eb,
    int const radius, Eq* const eq, T* const outlier, f4* time_elapsed,
    void* stream)
{
  auto divide3 = [](sycl::range<3> len, sycl::range<3> subl) {
    return sycl::range<3>(
        (len[2] - 1) / subl[2] + 1,  //
        (len[1] - 1) / subl[1] + 1,  //
        (len[0] - 1) / subl[0] + 1);
  };

  auto ndim = [&]() {
    if (len3[0] == 1 and len3[1] == 1)
      return 1;
    else if (len3[0] == 1 and len3[1] != 1)
      return 2;
    else
      return 3;
  };

  constexpr auto Tile1D = 256;
  constexpr auto Seq1D = 4;  // x-sequentiality == 4
  // x-sequentiality == 4
  auto Block1D = sycl::range<3>(1, 1, Tile1D / 4);
  auto Grid1D = divide3(len3, sycl::range<3>(1, 1, Tile1D));

  auto Tile2D = sycl::range<3>(1, 16, 16);
  // y-sequentiality == 8
  auto Block2D = sycl::range<3>(1, 2, 16);
  auto Grid2D = divide3(len3, Tile2D);

  auto Tile3D = sycl::range<3>(8, 8, 32);
  // y-sequentiality == 8
  // auto Block3D = dim3(32, 1, 8);  // for v0
  auto Block3D = sycl::range<3>(1, 8, 32);  // for v0::r1_shfl
  auto Grid3D = divide3(len3, Tile3D);

  auto d = ndim();

  auto queue = (sycl::queue*)stream;
  sycl::event e;

  // error bound
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = sycl::range<3>(len3[2] * len3[1], len3[2], 1);

  if (d == 1) {
    // dpct::has_capability_or_fail(queue->get_device(), {sycl::aspect::fp64});
    e = queue->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<T, 1> scratch(sycl::range<1>(Tile1D), cgh);
      sycl::local_accessor<Eq, 1> s_eq(sycl::range<1>(Tile1D), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(Grid1D * Block1D, Block1D),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            psz::dpcpp::__kernel::c_lorenzo_1d1l<T, Eq, FP, Tile1D, Seq1D>(
                data, len3, leap3, radius, ebx2_r, eq, outlier, item_ct1,
                scratch.get_pointer(), s_eq.get_pointer());
          });
    });
  }
  else if (d == 2) {
    // dpct::has_capability_or_fail(queue->get_device(), {sycl::aspect::fp64});
    e = queue->parallel_for(
        sycl::nd_range<3>(Grid2D * Block2D, Block2D),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          psz::dpcpp::__kernel::c_lorenzo_2d1l<T, Eq, FP>(
              data, len3, leap3, radius, ebx2_r, eq, outlier, item_ct1);
        });
  }
  else if (d == 3) {
    // dpct::has_capability_or_fail(queue->get_device(), {sycl::aspect::fp64});
    e = queue->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<T, 2> s_acc_ct1(sycl::range<2>(9, 33), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(Grid3D * Block3D, Block3D),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            psz::dpcpp::__kernel::c_lorenzo_3d1l<T, Eq, FP>(
                data, len3, leap3, radius, ebx2_r, eq, outlier, item_ct1,
                s_acc_ct1);
          });
    });
  }

  e.wait();
  SYCL_TIME_DELTA(e, *time_elapsed);

  return CUSZ_SUCCESS;
}

template <typename T, typename Eq, typename FP>
pszerror psz_decomp_l23(
    Eq* eq, sycl::range<3> const len3, T* outlier, PROPER_EB const eb,
    int const radius, T* xdata, f4* time_elapsed, void* stream)
{
  auto divide3 = [](sycl::range<3> l, sycl::range<3> subl) {
    return sycl::range<3>(
        (l[2] - 1) / subl[2] + 1, (l[1] - 1) / subl[1] + 1,
        (l[0] - 1) / subl[0] + 1);
  };

  auto ndim = [&]() {
    if (len3[0] == 1 and len3[1] == 1)
      return 1;
    else if (len3[0] == 1 and len3[1] != 1)
      return 2;
    else
      return 3;
  };

  constexpr auto Tile1D = 256;
  constexpr auto Seq1D = 8;  // x-sequentiality == 8
  auto Block1D = sycl::range<3>(1, 1, 256 / 8);
  auto Grid1D = divide3(len3, sycl::range<3>(1, 1, Tile1D));

  auto Tile2D = sycl::range<3>(1, 16, 16);
  // auto Seq2D    = dim3(1, 8, 1);  // y-sequentiality == 8
  auto Block2D = sycl::range<3>(1, 2, 16);
  auto Grid2D = divide3(len3, Tile2D);

  auto Tile3D = sycl::range<3>(8, 8, 32);
  // auto Seq3D    = dim3(1, 8, 1);  // y-sequentiality == 8
  auto Block3D = sycl::range<3>(8, 1, 32);
  auto Grid3D = divide3(len3, Tile3D);

  // error bound
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = sycl::range<3>(len3[2] * len3[1], len3[2], 1);

  auto d = ndim();

  auto queue = (sycl::queue*)stream;
  sycl::event e;

  if (d == 1) {
    // dpct::has_capability_or_fail(queue->get_device(), {sycl::aspect::fp64});
    e = queue->submit([&](sycl::handler& cgh) {
      constexpr auto NTHREAD = Tile1D / Seq1D;
      sycl::local_accessor<T, 1> scratch(sycl::range<1>(Tile1D), cgh);
      sycl::local_accessor<Eq, 1> s_eq(sycl::range<1>(Tile1D), cgh);
      sycl::local_accessor<T, 1> exch_in(sycl::range<1>(NTHREAD / 32), cgh);
      sycl::local_accessor<T, 1> exch_out(sycl::range<1>(NTHREAD / 32), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(Grid1D * Block1D, Block1D),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            psz::dpcpp::__kernel::x_lorenzo_1d1l<T, Eq, FP, Tile1D, Seq1D>(
                eq, outlier, len3, leap3, radius, ebx2, xdata, item_ct1,
                scratch.get_pointer(), s_eq.get_pointer(),
                exch_in.get_pointer(), exch_out.get_pointer());
          });
    });
  }
  else if (d == 2) {
    // dpct::has_capability_or_fail(queue->get_device(), {sycl::aspect::fp64});
    e = queue->submit([&](sycl::handler& cgh) {
      /*
      DPCT1101:106: 'BLOCK' expression was replaced with a value.
      Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<T, 1> scratch(sycl::range<1>(16 /*BLOCK*/), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(Grid2D * Block2D, Block2D),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            psz::dpcpp::__kernel::x_lorenzo_2d1l<T, Eq, FP>(
                eq, outlier, len3, leap3, radius, ebx2, xdata, item_ct1,
                scratch.get_pointer());
          });
    });
  }
  else if (d == 3) {
    /*
    DPCT1049:74: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::has_capability_or_fail(queue->get_device(), {sycl::aspect::fp64});
    e = queue->submit([&](sycl::handler& cgh) {
      /*
      DPCT1101:107: 'BLOCK' expression was replaced with a value.
      Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<T, 3> scratch(
          sycl::range<3>(8 /*BLOCK*/, 4, 8), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(Grid3D * Block3D, Block3D),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            psz::dpcpp::__kernel::x_lorenzo_3d1l<T, Eq, FP>(
                eq, outlier, len3, leap3, radius, ebx2, xdata, item_ct1,
                scratch);
          });
    });
  }

  e.wait();

  SYCL_TIME_DELTA(e, *time_elapsed);

  return CUSZ_SUCCESS;
}

#define CPP_INS(T, Eq)                                                    \
  template pszerror psz_comp_l23<T, Eq>(                                  \
      T* const data, sycl::range<3> const len3, PROPER_EB const eb,       \
      int const radius, Eq* const eq, T* const outlier, f4* time_elapsed, \
      void* stream);                                                      \
                                                                          \
  template pszerror psz_decomp_l23<T, Eq>(                                \
      Eq * eq, sycl::range<3> const len3, T* outlier, PROPER_EB const eb, \
      int const radius, T* xdata, f4* time_elapsed, void* stream);

CPP_INS(f4, u1);
CPP_INS(f4, u2);
CPP_INS(f4, u4);
CPP_INS(f4, f4);

// CPP_INS(f8, u1);
// CPP_INS(f8, u2);
// CPP_INS(f8, u4);
// CPP_INS(f8, f4);

CPP_INS(f4, i4);
// CPP_INS(f8, i4);

#undef CPP_INS
