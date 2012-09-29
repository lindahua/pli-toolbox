/**********************************************************
 *
 *  intcount1_cimp.cpp
 *
 *  C++ mex implementation of intcount (one-subscript case)
 *
 **********************************************************/

#include <light_mat/matlab/matlab_port.h>

using namespace lmat;
using namespace lmat::matlab;


struct Program
{
    int32_t m;
    const_marray mI;
    
    marray mC;
   
    Program(const mxArray *prhs[])
    : m(const_marray(prhs[0]).scalar<int32_t>())
    , mI(prhs[1])
    , mC(0)
    {}
    
    mxClassID class_id() const
    {
        return mI.class_id();
    }
    
    template<typename T>
    void run()
    {
        // prepare input
        
        cref_col<T> I = view_as_col<T>(mI);
        index_t len = I.nelems();
        
        // prepare output
        
        mC = marray::numeric_matrix<int32_t>(m, 1);
        ref_col<int32_t> C = view_as_col<int32_t>(mC);
        
        // main work
        
        for (index_t idx = 0; idx < len; ++idx)
        {
            int32_t i = I[idx] - 1;
            if (i >= 0 && i < m)
            {
                ++ C[i];
            }
        }
    }
};


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    Program prg(prhs);
    
    dispatch_for_numtypes(prg);
    
    plhs[0] = prg.mC;
}

