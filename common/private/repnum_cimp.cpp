/********************************************************************
 *
 *  repnum_cimp.cpp
 *
 *  C++ mex implementation of pli_repnum
 *
 ********************************************************************/

#include <light_mat/matlab/matlab_port.h>


using namespace lmat;
using namespace lmat::matlab;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const_marray _n(prhs[0]);
    const_marray _r(prhs[1]);
    
    const index_t n = (index_t)_n.get_scalar();
    const uint32_t *r = (const uint32_t*)(_r.data<uint32_t>());
    
    uint32_t N = 0;
    for (index_t i = 0; i < n; ++i)
    {
        N += r[i];
    }
    
    marray _y = marray::numeric_matrix<int32_t>(1, (index_t)N);
    int32_t *y = (int32_t*)(_y.data<int32_t>());
    
    for (index_t i = 0; i < n; ++i)
    {
        uint32_t ri = r[i];
        int32_t v = int32_t(i + 1);
        
        for (uint32_t j = 0; j < ri; ++j) *y++ = v;
    }
    
    plhs[0] = _y;
}

