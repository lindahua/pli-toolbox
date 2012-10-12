/********************************************************************
 *
 *  C++ mex implementation on SGD-QN for SVM
 *
 ********************************************************************/

#include "svm_sgdx_common.h"


// without bias
class SGD_QN
{
private:
    // parameters
    
    vec_t& _w;
    
    const index_t _dim;
    const double _lambda;
    const index_t _skip;
    const double _r_lb;
    const double _r_ub;
    
    // involved variables
    
    dense_col<double> _v;
    dense_col<double> _B;    
    
    // states
    
    double _time;
    index_t _count;
    bool _updateB;
    
public:
    
    SGD_QN(vec_t& w, double lambda, index_t skip, double t0)
    : _w(w), _dim(w.nrows()), _lambda(lambda), _skip(skip)
    , _r_lb(lambda), _r_ub(100.0 * lambda)
    , _v(_dim), _B(_dim)
    , _time(t0), _count(skip), _updateB(false)
    {
        // initialize B
        
        zero(_v);
        
        double vB0 = 1.0 / (_lambda * (t0 + 1));
        fill(_B, vB0);
    }
        
    // k is always equal to 1
    
    void learn(const double *pX, const double *pY, const index_t )
    {
        cvec_t x(pX, _dim, 1);
        double y = *pY;
        
        if (_updateB)
        {
            double uw = lmat::dot(_w, x);
            double uv = lmat::dot(_v, x);
            
            bool w_sv = (uw * y < 1.0);
            bool v_sv = (uv * y < 1.0);
            
            for (index_t i = 0; i < _dim; ++i)
            {
                double gw_i = _lambda * _w[i];
                if (w_sv) gw_i += y * x[i];
                
                double gv_i = _lambda * _v[i];
                if (v_sv) gv_i += y * x[i];
                
                double r_i = (gw_i - gv_i) / (_w[i] - _v[i]);
                
                if (r_i < _r_lb) r_i = _r_lb;
                else if (r_i > _r_ub) r_i = _r_ub;
                
                _B[i] /= (1.0 + _skip * _B[i] * r_i);
            }
            
            _updateB = false;
        }
        
        double u = lmat::dot(_w, x);
        
        if (-- _count <= 0)
        {
            _count = _skip;
            _updateB = true;
            copy(_w, _v);
            
            for (index_t i = 0; i < _dim; ++i)
                _w[i] *= (1.0 - _skip * _lambda * _B[i]);
        }
        
        if (u * y < 1.0)
        {
            for (index_t i = 0; i < _dim; ++i)
                _w[i] += y * (_B[i] * x[i]);
        }
    }        
    
};


// with bias
class SGD_QN_A
{
private:
    // parameters
    
    vec_t& _w;
    double& _w0;
    
    const index_t _dim;
    const double _lambda;
    const double _aug;
    
    const index_t _skip;
    const double _r_lb;
    const double _r_ub;
    
    // involved variables
    
    dense_col<double> _v;
    double _v0;
    
    dense_col<double> _B;    
    double _B0;
    
    // states
    
    double _time;
    index_t _count;
    bool _updateB;
    
public:
    
    SGD_QN_A(vec_t& w, double& w0, double lambda, double aug, index_t skip, double t0)
    : _w(w), _w0(w0), _dim(w.nrows()), _lambda(lambda), _aug(aug), _skip(skip)
    , _r_lb(lambda), _r_ub(100.0 * lambda)
    , _v(_dim), _v0(0), _B(_dim), _B0(0)
    , _time(t0), _count(skip), _updateB(false)
    {
        // initialize B
        
        zero(_v);
        
        double vB0 = 1.0 / (_lambda * (t0 + 1));
        fill(_B, vB0);
        _B0 = vB0;
    }
        
    // k is always equal to 1
    
    void learn(const double *pX, const double *pY, const index_t )
    {
        cvec_t x(pX, _dim, 1);
        double y = *pY;
        
        if (_updateB)
        {
            double uw = lmat::dot(_w, x) + _w0 * _aug;
            double uv = lmat::dot(_v, x) + _v0 * _aug;
            
            bool w_sv = (uw * y < 1.0);
            bool v_sv = (uv * y < 1.0);
            
            for (index_t i = 0; i < _dim; ++i)
            {
                double gw_i = _lambda * _w[i];
                if (w_sv) gw_i += y * x[i];
                
                double gv_i = _lambda * _v[i];
                if (v_sv) gv_i += y * x[i];
                
                double r_i = 0;
                if (_w[i] != _v[i])
                {
                    r_i = (gw_i - gv_i) / (_w[i] - _v[i]);
                }
                else
                {
                    r_i = 0;
                }
                
                if (r_i < _r_lb) r_i = _r_lb;
                else if (r_i > _r_ub) r_i = _r_ub;
                
                _B[i] /= (1.0 + _skip * _B[i] * r_i);
            }
            
            double gw_0 = _lambda * _w0;
            if (w_sv) gw_0 += y * _aug;
            
            double gv_0 = _lambda * _v0;
            if (v_sv) gv_0 += y * _aug;
            
            double r_0;
            if (_w0 != _v0)
            {
                r_0 = (gw_0 - gv_0) / (_w0 - _v0);
            }
            else
            {
                r_0 = 0;
            }
            
            if (r_0 < _r_lb) r_0 = _r_lb;
            else if (r_0 > _r_ub) r_0 = _r_ub;
            
            _B0 /= (1.0 + _skip * _B0 * r_0);
                        
            _updateB = false;
        }
        
        double u = lmat::dot(_w, x) + _w0 * _aug;
        
        if (-- _count <= 0)
        {
            _count = _skip;
            _updateB = true;
            
            copy(_w, _v); _v0 = _w0;
            
            for (index_t i = 0; i < _dim; ++i)
                _w[i] *= (1.0 - _skip * _lambda * _B[i]);
            
            _w0 *= (1.0 - _skip * _lambda * _B0);
        }
        
        if (u * y < 1.0)
        {
            for (index_t i = 0; i < _dim; ++i)
                _w[i] += y * (_B[i] * x[i]);
            
            _w0 += y * (_B0 * _aug);
        }
    }        
    
};




void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // arguments
    
    const_marray mX(prhs[0]);
    const_marray mY(prhs[1]);
    const_marray mLam(prhs[2]);
    const_marray mAug(prhs[3]);
    const_marray mT0(prhs[4]);
    const_marray mSkip(prhs[5]);
    
    marray mW = duplicate(prhs[6]);
    marray mW0 = duplicate(prhs[7]);
    
    // take inputs
    
    cmat_t X = view2d<double>(mX);
    cvec_t Y = view_as_col<double>(mY);
    
    double lambda = mLam.get_scalar();
    double aug = mAug.get_scalar();
    
    double t0 = mT0.get_scalar();
    index_t skip = (index_t)mSkip.get_scalar();
    
    const index_t d = mX.nrows();
    
    // main
    
    if (aug == 0)
    {
        vec_t w(mW.ptr_real(), d, 1);
        SGD_QN trainer(w, lambda, skip, t0);
        run_sgdx(trainer, X, Y, 1);
    }
    else
    {
        vec_t w(mW.ptr_real(), d, 1);
        double& w0 = *(mW0.ptr_real());
        
        SGD_QN_A trainer(w, w0, lambda, aug, skip, t0);
        run_sgdx(trainer, X, Y, 1); 
    }
    
    // return
    
    plhs[0] = mW;
    plhs[1] = mW0;
}




