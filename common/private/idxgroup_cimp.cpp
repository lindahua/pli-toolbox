/**********************************************************
 *
 *  idxgroup_cimp.cpp
 *
 *  C++ mex implementation of find_bin 
 *
 **********************************************************/

#include <light_mat/matlab/matlab_port.h>

using namespace lmat;
using namespace lmat::matlab;

void idx_group(const cref_matrix<int32_t>& L, 
        ref_col<int32_t>& sx,
        ref_col<int32_t>& b, 
        ref_col<int32_t>& e)
{
    const int32_t K = (int32_t)b.nelems();
    const int32_t n = (int32_t)L.nelems();
    
    // first pass: count labels (use e as temporary store)
    
    for (int32_t i = 0; i < n; ++i)
    {
        int32_t k = L[i];
        if (k >= 0 && k < K) ++ e[k];
    }
    
    // set b and e
    
    int32_t pre_e = 0;
    for (int32_t k = 0; k < K; ++k)
    {
        b[k] = pre_e;
        pre_e += e[k];
        e[k] = b[k];
        
    }
    
    // second pass: fill in indices (and advance e)
    
    for (int32_t i = 0; i < n; ++i)
    {
        int32_t k = L[i];
        if (k >= 0 && k < K)
        {
            sx[e[k]++] = (i + 1);
        }
    }    
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const int32_t K = const_marray(prhs[0]).scalar<int32_t>();
    const_marray mL(prhs[1]);
    
    // prepare inputs
    
    cref_matrix<int32_t> L = view2d<int32_t>(mL);
    
    // prepare outputs
    
    marray mS = marray::numeric_matrix<int32_t>(L.nrows(), L.ncolumns());       
    marray mB = marray::numeric_matrix<int32_t>(K, 1);
    marray mE = marray::numeric_matrix<int32_t>(K, 1);
    
    ref_col<int32_t> sx = view_as_col<int32_t>(mS);
    ref_col<int32_t> b = view_as_col<int32_t>(mB);    
    ref_col<int32_t> e = view_as_col<int32_t>(mE);
    
    // main
    
    idx_group(L, sx, b, e);
    
    // return outputs
    
    plhs[0] = mS;
    plhs[1] = mB;
    plhs[2] = mE;
}


