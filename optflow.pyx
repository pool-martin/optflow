# -*- coding: utf-8 -*-

import numpy as np  # Import Python functions, attributes, submodules of numpy
cimport numpy as np  # Import numpy C/C++ API
from libcpp cimport bool
from cpython.ref cimport PyObject
from libc.stdlib cimport malloc, free

np.import_array()

cdef extern from 'src/pyopencv_converter.cpp':
    #mrc689 April 20,2017
    cdef PyObject*pyopencv_from(const Mat& m)
    cdef bool pyopencv_to(PyObject*o, Mat& m)

cdef extern from 'opencv2/core/core.hpp':
    cdef int CV_8UC1
    cdef int CV_32FC1
    cdef int CV_32FC2
    cdef int CV_32FC3

cdef extern from 'opencv2/core/core.hpp' namespace 'cv':
    cdef cppclass Mat:
        Mat() except +
        Mat(int, int, int) except +
        int cols
        int rows

cdef extern from 'opencv2/core/cvstd.hpp' namespace 'cv':
    cdef cppclass Ptr[T]:
        Ptr() except +
        T*get()

cdef extern from 'opencv2/core/cuda.hpp' namespace 'cv::cuda':
    cdef cppclass GpuMat:
        GpuMat() except +
        GpuMat(int, int, int) except +
        void upload(Mat arr) except +
        void download(Mat dst) const
        int cols
        int rows

cdef extern from 'opencv2/cudaoptflow.hpp' namespace 'cv::cuda':
    cdef cppclass BroxOpticalFlow:
        @ staticmethod
        Ptr[BroxOpticalFlow] create(double, double, double, int, int, int)

        void calc(GpuMat, GpuMat, GpuMat) except +

cpdef brox(np.ndarray[np.float64_t, ndim=2, mode="c"] im0_d, np.ndarray[np.float64_t, ndim=2, mode="c"] im1_d,
           alpha = 0.197, gamma = 50.0, scale_factor = 0.8, inner_iterations = 5, outer_iterations = 150,
           solver_iterations = 10):
    """
    Calculates the optical flow between images im0 and im1 using the CUDA accelerated OpenCV code of
    
    Brox, Thomas, et al. High accuracy optical flow estimation based on a theory for warping.
    European conference on computer vision. 2004.
    
    :param im0: Image 0 (NxM)
    :param im1: Image 1 (NxM)
    :param alpha: alpha = 0.197
    :param gamma: gamma = 50.0
    :param scale_factor: scale_factor = 0.8
    :param inner_iterations: inner_iterations = 5
    :param outer_iterations: outer_iterations = 150
    :param solver_iterations: solver_iterations = 10
    :return: Flow field (NxMx2)
    """
    cdef np.ndarray[np.float32_t, ndim=2] im0 = im0_d.astype(np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] im1 = im1_d.astype(np.float32)

    assert im0.shape[0] == im1.shape[0] and im0.shape[1] == im1.shape[1]

    cdef unsigned int h = im0.shape[0]
    cdef unsigned int w = im0.shape[1]

    cdef Mat m_im0
    cdef GpuMat g_im0
    cdef Mat m_im1
    cdef GpuMat g_im1

    cdef Mat flow = Mat(h, w, CV_32FC2)
    cdef GpuMat g_flow = GpuMat(h, w, CV_32FC2)

    pyopencv_to(<PyObject*> im0, m_im0)
    g_im0.upload(m_im0)
    pyopencv_to(<PyObject*> im1, m_im1)
    g_im1.upload(m_im1)

    cdef Ptr[BroxOpticalFlow] brox = BroxOpticalFlow.create(
        alpha, gamma, scale_factor, inner_iterations, outer_iterations, solver_iterations)

    brox.get().calc(g_im0, g_im1, g_flow)

    g_flow.download(flow)
    return <object> pyopencv_from(flow)
