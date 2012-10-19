/**********************************************************
 *
 *  C++ mex of the core part of pli_streamvq
 *
 **********************************************************/

#include <light_mat/matlab/matlab_port.h>

using namespace lmat;
using namespace lmat::matlab;

typedef cref_matrix<double> cmat_t;
typedef cref_col<double> cvec_t;
typedef ref_matrix<double> mat_t;
typedef ref_col<double> vec_t;


/**
 * Input
 *   [0] C0:    initial set of centers
 *   [1] w0:    initial center weights
 *   [2] cbnd:  cost bound
 *
 *   [3] X:     sample matrix
 *   [4] xw:    sample weights
 *   [5] u:     rand numbers (one for a sample)
 *
 *   [6] kmax:  maximum number of centers
 *   [7] i0:    The index of starting sample (i.e. # processed samples)
 *   [8] vb_intv:   interval for displaying
 *
 * Output
 *   [0] C:     updated centers
 *   [1] w:     updated center weights
 *   [2] i:     the (zero-based) index of last processed samples
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // input arguments
    
    const_marray _C0 = prhs[0];
    const_marray _w0 = prhs[1];
    const_marray _cbnd = prhs[2];
    
    const_marray _X = prhs[3];
    const_marray _xw = prhs[4];
    const_marray _u = prhs[5];
    
    const_marray _kmax = prhs[6];
    const_marray _i0 = prhs[7];
    const_marray _vb_intv = prhs[8];
    
    // extract inputs
    
    cmat_t C0 = view2d<double>(_C0);
    cvec_t w0 = view_as_col<double>(_w0);
    double cbnd = _cbnd.get_scalar();
    
    cmat_t X = view2d<double>(_X);
    cvec_t xw = view_as_col<double>(_xw);
    cvec_t u = view_as_col<double>(_u);
    
    const index_t kmax = (index_t)(_kmax.get_scalar());
    const index_t i0 = (index_t)(_i0.get_scalar());
    const index_t vb_intv = (index_t)(_vb_intv.get_scalar());
    
    
    
}
