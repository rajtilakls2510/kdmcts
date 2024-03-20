cdef extern from "gsl/gsl_matrix.h" nogil:
     ctypedef struct gsl_matrix

     gsl_matrix * gsl_matrix_alloc (const size_t n1, const size_t n2);
     void gsl_matrix_free (gsl_matrix * m);
     int gsl_matrix_memcpy(gsl_matrix * dest, const gsl_matrix * src);

cdef extern from "gsl/gsl_cblas.h" nogil:

    enum CBLAS_ORDER:
        CblasRowMajor=101,
        CblasColMajor=102,

    enum CBLAS_TRANSPOSE:
        CblasNoTrans=111,
        CblasTrans=112,
        CblasConjTrans=113

    void cblas_dscal(const int N, const double alpha, double *X, const int incX); # x = alpha * x
    void cblas_dcopy(const int N, const double *X, const int incX,
                 double *Y, const int incY); # y = x
    double cblas_ddot(const int N, const double *X, const int incX,
                  const double *Y, const int incY); # = x^T @ y

    double cblas_dnrm2(const int N, const double *X, const int incX); # = ||x||_2
    void cblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY); # y = y + alpha * x

    void cblas_dgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);   # y = alpha * A @ x + beta * y

    void cblas_dgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc); # C = alpha * A @ B + beta * C

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng

    gsl_rng_type* gsl_rng_default
    gsl_rng* gsl_rng_alloc(gsl_rng_type* T)
    void gsl_rng_set (const gsl_rng * r, unsigned long int seed);
    void gsl_rng_free(gsl_rng* r)
    double gsl_rng_uniform(gsl_rng* r)

cdef extern from "gsl/gsl_randist.h" nogil:
    double gsl_ran_gaussian (const gsl_rng * r, const double sigma);
    double gsl_ran_flat(const gsl_rng *r, double a, double b)