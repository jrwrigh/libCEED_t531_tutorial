// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q,
                      const CeedScalar *const *in,
                      CeedScalar *const *out) {
  // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.

  // in[0] is Jacobians with shape [2, nc=2, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar (*J)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar *qw = in[1];

  // out[0] is qdata, size (Q)
  CeedScalar (*qd)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature point loop
  for (CeedInt q=0; q<Q; q++) {
    // J: 0 2   qd: 0 2   adj(J):  J22 -J12
    //    1 3       2 1           -J21  J11
    const CeedScalar J11 = J[0][q];
    const CeedScalar J21 = J[1][q];
    const CeedScalar J12 = J[2][q];
    const CeedScalar J22 = J[3][q];
    const CeedScalar w = qw[q] / (J11*J22 - J21*J12);
    qd[0][q] =   w * (J12*J12 + J22*J22);
    qd[2][q] =   w * (J11*J11 + J21*J21);
    qd[1][q] = - w * (J11*J12 + J21*J22);
  }

  return 0;
}

CEED_QFUNCTION(diff)(void *ctx, const CeedInt Q, const CeedScalar *const *in,
                     CeedScalar *const *out) {
  // in[0] is gradient u, shape [2, nc=1, Q]
  // in[1] is quadrature data, size (3*Q)
  const CeedScalar (*du)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar (*qd)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // out[0] is output to multiply against gradient v, shape [2, nc=1, Q]
  CeedScalar (*dv)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature point loop
  for (CeedInt q=0; q<Q; q++) {
    const CeedScalar du0 = du[0][q];
    const CeedScalar du1 = du[1][q];
    dv[0][q] = qd[0][q]*du0 + qd[2][q]*du1;
    dv[1][q] = qd[2][q]*du0 + qd[1][q]*du1;
  }

  return 0;
}

CEED_QFUNCTION(diff_lin)(void *ctx, const CeedInt Q,
                         const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is gradient u, shape [2, nc=1, Q]
  // in[1] is quadrature data, size (4*Q)
  const CeedScalar (*du)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar (*qd)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // out[0] is output to multiply against gradient v, shape [2, nc=1, Q]
  CeedScalar (*dv)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature point loop
  for (CeedInt q=0; q<Q; q++) {
    const CeedScalar du0 = du[0][q];
    const CeedScalar du1 = du[1][q];
    // Linearized Qdata is provided column-major
    //  0 2
    //  1 3
    dv[0][q] = qd[0][q]*du0 + qd[2][q]*du1;
    dv[1][q] = qd[1][q]*du0 + qd[3][q]*du1;
  }

  return 0;
}
