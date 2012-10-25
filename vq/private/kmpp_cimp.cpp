/********************************************************************
 *
 * C++ mex implementation of kmpp seeding
 *
 ********************************************************************/

#include <light_mat/matlab/matlab_port.h>

using namespace lmat;
using namespace lmat::matlab;

typedef cref_matrix<double> cmat_t;
typedef ref_matrix<double> mat_t;

typedef cref_matrix<double, 0, 1> cvec_t;
typedef ref_matrix<double, 0, 1> vec_t;


LMAT_ENSURE_INLINE
inline double calc_cost(const cvec_t& x, const cvec_t& y)
{
    double s = 0;
    const index_t n = x.nrows();
    for (index_t i = 0; i < n; ++i)
    {
        double dx = x[i] - y[i];
        s += (dx * dx);
    }
    return s;
}


LMAT_ENSURE_INLINE
inline index_t sample_idx(const index_t n, double u)
{
    return static_cast<index_t>(std::floor(u * double(n)));
}

LMAT_ENSURE_INLINE
inline index_t sample_idx(const index_t n, double u, 
        const dense_col<double>& costs, const double total_cost)
{    
    const double v = u * total_cost;
    
    index_t i = 0;
    double x = costs[0];
    
    const index_t nm1 = n - 1;
    
    while(x < v && i < nm1)
    {
        x += costs[++i];
    }  
    
    return i;
}


inline double init_costs(const cmat_t& X, const index_t i, 
        dense_col<double>& min_costs)
{
    const index_t n = X.ncolumns();
    
    cvec_t sx = X.column(i);
    double tcost = 0;
    
    for (index_t j = 0; j < n; ++j)
    {
        double c = i == j ? 0.0 : calc_cost(sx, X.column(j));
        tcost += (min_costs[j] = c);
    }
    
    return tcost;
}


inline double update_costs(const cmat_t& X, const index_t i, 
        dense_col<double>& min_costs)
{
    const index_t n = X.ncolumns();
    
    cvec_t sx = X.column(i);
    double tcost = 0;
    
    for (index_t j = 0; j < n; ++j)
    {
        double c = i == j ? 0.0 : calc_cost(sx, X.column(j));
        
        if (c < min_costs[j]) min_costs[j] = c;        
        tcost += min_costs[j];
    }
    
    return tcost;
}


/**
 * Inputs
 * [0] X :      The sample matrix
 * [1] u :      a sequence of uniform random numbers
 *
 * Outputs
 * [0] S :      A vector of selected seeds [1 x K double]
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // take inputs
    
    const_marray _X(prhs[0]);
    const_marray _u(prhs[1]);
    
    cmat_t X = view2d<double>(_X);
    const index_t d = X.nrows();
    const index_t n = X.ncolumns();
    
    cvec_t u = view_as_col<double>(_u);
    const index_t K = u.nelems();
    
    // prepare output
    
    marray _S = marray::numeric_matrix<double>(1, K);    
    vec_t S = view_as_col<double>(_S);
    
    // main
    
    dblock<bool> visited(n, zero());
    dense_col<double> min_costs(n);
    
    // randomly select
    
    // initial round
    
    index_t i = sample_idx(n, u[0]);
    S[0] = i + 1;
    
    cvec_t sx = X.column(i);
    double tcost = init_costs(X, i, min_costs);
    
    // middle rounds
    
    for (index_t k = 1; k < K - 1; ++k)
    {
        i = sample_idx(n, u[k], min_costs, tcost);
        S[k] = i + 1;
        
        cvec_t sx = X.column(i);
        tcost = update_costs(X, i, min_costs);
    }    
    
    // final round
    
    i = sample_idx(n, u[K-1], min_costs, tcost);
    S[K-1] = i + 1;    
    
    // output
    
    plhs[0] = _S;
}

