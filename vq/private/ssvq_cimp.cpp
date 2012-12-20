/**********************************************************
 *
 *  C++ mex of the core part of pli_ssvq
 *
 **********************************************************/

#include <light_mat/matlab/matlab_port.h>

using namespace lmat;
using namespace lmat::matlab;

typedef cref_matrix<double, 0, 0> cmat_t;
typedef cref_matrix<double, 0, 1> cvec_t;
typedef ref_matrix<double, 0, 0> mat_t;
typedef ref_matrix<double, 0, 1> vec_t;


LMAT_ENSURE_INLINE
inline void set_center(mat_t& C, index_t k, const cvec_t& x)
{
    vec_t c = C.column(k);
    c = x;
}


LMAT_ENSURE_INLINE
inline double calc_cost(const cvec_t& cen, const cvec_t& x)
{
    double s = 0;
    const index_t d = cen.nrows();
    
    for (index_t i = 0; i < d; ++i)
    {
        double dx = cen[i] - x[i];
        s += dx * dx;
    }
    
    return s;
}


LMAT_ENSURE_INLINE
inline index_t find_closest_center(
        const mat_t& C, const index_t k, const cvec_t& x, double& cost)
{
    index_t best_i = 0;
    cost = calc_cost(C.column(0), x);
    
    for (index_t i = 1; i < k; ++i)
    {
        double cc = calc_cost(C.column(i), x);
        if (cc < cost) 
        {
            cost = cc;
            best_i = i;
        }
    }
    
    return best_i;
}


LMAT_ENSURE_INLINE
inline void update_center(vec_t& c, double& cw, const cvec_t& x, double xw)
{
    cw += xw;
    
    double a = xw / cw;
    double b = 1.0 - a;
    
    const index_t d = c.nrows();
    for (index_t i = 0; i < d; ++i)
    {
        c[i] = a * x[i] + b * c[i];
    }
}


LMAT_ENSURE_INLINE
inline void see_print(const index_t vb_intv, index_t& vbcount, index_t i)
{
    if (vb_intv)
    {
        if (--vbcount == 0)
        {
            vbcount = vb_intv;
            mexPrintf("    %ld samples processed\n", long(i));
        }        
    }
}


void ssvq(
        mat_t& C,           /* centers */
        vec_t& w,           /* center weights */
        const double cbnd,  /* cost bound */
        const cmat_t& X,    /* samples */
        const cvec_t& xw,   /* sample weights */
        const cvec_t& u,    /* random number sequence */
        index_t& k,         /* number of centers */
        index_t& i,         /* index of last processed sample */
        const index_t vb_intv   /* displaying interval */
        )
{
    const index_t kmax = C.ncolumns();
    const index_t n = X.ncolumns();
    
    index_t vbcount = 0;
    
    if (vb_intv)
    {
        vbcount = vb_intv - (i % vb_intv);
    }
    
    
    if (k == 0 && i < n)
    {
        cvec_t x = X.column(i);
        
        w[k] = xw[i];        
        set_center(C, k++, x);
        
        ++i;
        see_print(vb_intv, vbcount, i);
    }
    
    
    while (i < n)
    {
        cvec_t x = X.column(i);
        
        double cost;
        index_t s = find_closest_center(C, k, x, cost);
        
        bool to_add = !(xw[i] * cost < u[i] * cbnd);
                
        if (xw[i] * cost < u[i] * cbnd)
        {
            // update to s-th center
            
            vec_t cen = C.column(s);            
            update_center(cen, w[s], x, xw[i]);
        }
        else
        {
            if (k >= kmax) return;
            
            // add new center
            w[k] = xw[i];
            set_center(C, k++, x);                        
        }
         
        ++i;
        see_print(vb_intv, vbcount, i);
    }
}



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
    
    double cbnd = _cbnd.get_scalar();
    
    cmat_t X = view2d<double>(_X);
    cvec_t xw = view_as_col<double>(_xw);
    cvec_t u = view_as_col<double>(_u);
    
    const index_t kmax = (index_t)(_kmax.get_scalar());
    index_t i = (index_t)(_i0.get_scalar());
    const index_t vb_intv = (index_t)(_vb_intv.get_scalar());
    
    const index_t d = X.nrows();
    
    // prepare outputs
    
    marray _C = marray::numeric_matrix<double>(d, kmax);
    marray _w = marray::numeric_matrix<double>(1, kmax);
    
    mat_t C = view2d<double>(_C);
    vec_t w = view_as_col<double>(_w);
    
    index_t k = 0;
    
    if (_C0.is_empty())
    {
        k = _C0.ncolumns();
        
        copy_vec(d * k, _C0.data<double>(), C.ptr_data());
        copy_vec(k, _w0.data<double>(), w.ptr_data());        
    }
    
    
    // call main function
    
    ssvq(C, w, cbnd, X, xw, u, k, i, vb_intv);
    
    
    // output
    
    if (k < kmax)
    {
        marray _Ctmp = _C;
        marray _wtmp = _w;
        
        _C = marray::numeric_matrix<double>(d, k);
        _w = marray::numeric_matrix<double>(1, k);
        
        copy_vec(d * k, C.ptr_data(), _C.data<double>());
        copy_vec(k, w.ptr_data(), _w.data<double>());
        
        _Ctmp.destroy();
        _wtmp.destroy();
    }
    
    plhs[0] = _C;
    plhs[1] = _w;
    plhs[2] = marray::from_double_scalar(double(i));
    
    
}
