/// @file
/// Test assembly of Poisson operator QFunction
/// \test Test assembly of Poisson operator QFunction
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t531-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction elem_restr_x, elem_restr_u,
                      elem_restr_qd_i;
  CeedBasis basis_x, basis_u;
  CeedQFunction qf_setup, qf_laplacian;
  CeedOperator op_setup, op_laplacian;
  CeedVector q_data, X, A, u, v;

  CeedInt P = 3, Q = 4; // Give in 1D, before tensor product
  CeedInt dim = 2;
  CeedInt nx = 3, ny = 2, num_elem=nx*ny;
  CeedInt num_dofs_x = nx*2+1, num_dofs_y=ny*2+1;
  CeedInt num_dofs   = num_dofs_x*num_dofs_y;
  CeedInt num_qpts   = num_elem*Q*Q;
  CeedInt ind_x[num_elem*P*P];
  CeedScalar x[dim*num_dofs];

  CeedInit(argv[1], &ceed); // Select libCEED backend from argument

  // DoF Coordinates
  for (CeedInt i=0; i<num_dofs_x; i++)
    for (CeedInt j=0; j<num_dofs_y; j++) {
      x[i+j*num_dofs_x+0*num_dofs] = (CeedScalar) i / (2*nx);
      x[i+j*num_dofs_x+1*num_dofs] = (CeedScalar) j / (2*ny);
    }
  CeedVectorCreate(ceed, dim*num_dofs, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vector
  CeedVectorCreate(ceed, num_qpts*dim*(dim+1)/2, &q_data);

  // Element Setup
  for (CeedInt i=0; i<num_elem; i++) {
    CeedInt col, row, offset;
    col = i % nx;
    row = i / nx;
    offset = col*(P-1) + row*num_dofs_x*(P-1);
    for (CeedInt j=0; j<P; j++)
      for (CeedInt k=0; k<P; k++)
        ind_x[P*(P*i+k)+j] = offset + k*num_dofs_x + j;
  }

  // Restrictions
  CeedElemRestrictionCreate(ceed,          num_elem,         P*P,   dim, num_dofs, dim*num_dofs,
                            CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_x);
  CeedElemRestrictionCreate(ceed,          num_elem,         P*P,   1,   1,        num_dofs,
                            CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_u);

  CeedInt strides_qd[3] = {1, Q*Q, Q*Q*dim*(dim+1)/2};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q, dim*(dim+1)/2,
                                   dim*(dim+1)/2*num_qpts, strides_qd,
                                   &elem_restr_qd_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1,   P, Q, CEED_GAUSS, &basis_u);

  // QFunction - setup
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup,  "dx",     dim*dim,       CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup,  "weight", 1,             CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata",  dim*(dim+1)/2, CEED_EVAL_NONE);

  // Operator - setup
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "dx",     elem_restr_x,              basis_x,               CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x,               CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "qdata",  elem_restr_qd_i,           CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply setup Operator
  CeedOperatorApply(op_setup, X, q_data, CEED_REQUEST_IMMEDIATE);

  // QFunction - Laplacian
  CeedQFunctionCreateInterior(ceed, 1, laplacian, laplacian_loc, &qf_laplacian);
  CeedQFunctionAddInput(qf_laplacian,  "du",    dim,           CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_laplacian,  "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_laplacian, "dv",    dim,           CEED_EVAL_GRAD);

  // Operator - Laplacian
  CeedOperatorCreate(ceed, qf_laplacian, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_laplacian);
  CeedOperatorSetField(op_laplacian, "du",    elem_restr_u,    basis_u,               CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_laplacian, "qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_laplacian, "dv",    elem_restr_u,    basis_u,               CEED_VECTOR_ACTIVE);

  // Initialize input and output vectors
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorSetValue(u, 1.0);
  CeedVectorCreate(ceed, num_dofs, &v);
  CeedVectorSetValue(v, 0.0);

  // Apply Laplacian Operator
  CeedOperatorApply(op_laplacian, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  const CeedScalar *vv;
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &vv);
  for (CeedInt i=0; i<num_dofs; i++)
    if (fabs(vv[i]) > 100.*CEED_EPSILON)
      // LCOV_EXCL_START
      printf("Error: Operator computed v[%" CeedInt_FMT "] = %f != 0.0\n", i, vv[i]);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(v, &vv);

  // Cleanup
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_laplacian);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_laplacian);
  CeedElemRestrictionDestroy(&elem_restr_u);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_qd_i);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&A);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedDestroy(&ceed);
  return 0;
}
