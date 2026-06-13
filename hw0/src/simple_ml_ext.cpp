#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

int index(int i,int j,size_t n) { return i*n+j; }

void matrix_mul(const float *x,const float *y,float *r,
                size_t m,size_t n,size_t k){
    for(int ri=0;ri<m;ri++){
        for(int rj=0;rj<k;rj++){
            int idx=index(ri,rj,k);
            r[idx]=0;
            for(int ni=0;ni<n;ni++){
                r[idx]+=x[index(ri,ni,n)]*y[index(ni,rj,k)];
            }
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    // * softmax回归epoch代码的C++版本。
    // 该函数应对由X和y（以及尺寸m、n、k）定义的数据执行单个epoch，
    // 并原地更新theta。你的函数可能需要分配（并在之后删除）
    // 一些辅助数组来存储logits和梯度。
    // * 参数：
    // *     X (const float *): 指向X数据的指针，尺寸为 m*n，以行主序（C）
    // 格式存储
    // // *     y (const unsigned char *): 指向y数据的指针，尺寸为 m
    // // *     theta (float *): 指向theta数据的指针，尺寸为 n*k，
    // 以行主序（C）格式存储
    // *     m (size_t): 样本数
    // *     n (size_t): 输入维度
    // *     k (size_t): 类别数
    // *     lr (float): 学习率 / SGD步长
    // *     batch (int): SGD小批量大小
    // * 返回值：
    // *     (无)

    /// BEGIN YOUR CODE
    
    for(int ba=0;ba<m;ba+=batch){
        float logits[batch*k],g[n*k],y_hat[batch*k],logits_sum[batch],logits_exp[batch*k],y_err[batch*k],X_T[n*batch];
        const float *X_batch=X+ba*n;
        auto *y_batch=y+ba;
        matrix_mul(X_batch,theta,logits,batch,n,k);

        for(int i=0;i<batch*k;i++){
            logits_exp[i]=std::exp(logits[i]);
        }

        for (int i=0;i<batch;i++){
            logits_sum[i]=0;
            for (int j=0;j<k;j++){
                logits_sum[i]+=logits_exp[index(i,j,k)];
            }
        }

        for (int i=0;i<batch;i++){
            for (int j=0;j<k;j++){
                int idx=index(i,j,k);
                y_hat[idx]=logits_exp[idx]/logits_sum[i];
                y_err[idx]=y_hat[idx]-(j==y_batch[i]?1.0:0.0);
            }
        }

        for(int i=0;i<batch;i++){
            for(int j=0;j<n;j++){
                X_T[j*batch+i]=X_batch[i*n+j];
            }
        }

        matrix_mul(X_T,y_err,g,n,batch,k);

        for(int i=0;i<n*k;i++){
            theta[i]-=g[i]*lr/batch;
        }
    }

    /// END YOUR CODE
}


/**
 * 这是用于封装上述函数的 pybind11 代码。它唯一的作用  
 * 是将上述函数封装到一个 Python 模块中，您无需  
 * 对代码做任何修改。
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
