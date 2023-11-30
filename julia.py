
import numpy as np
from pyopencl import clcreate, clprogram

def compute_julia_opencl(width, height, c, max_iter):
    # Create a CL context and command queue
    ctx = cl.create_context()
    queue = ctx.get_command_queue()

    # Define the kernel code for computing the Julia set
    kernel_code = """
        __kernel void julia_set(__global float *x, __global float *y,
                                 __global int *julia, const int max_iter) {
            int idx = get_global_id(0);
            int idy = get_global_id(1);

            // Compute the Julia set value for this pixel
            double z_re = x[idx] + y[idy];
            double z_im = 0;
            int iter = 0;

            while (iter < max_iter && np.abs(z_re) <= 2.0) {
                z_re *= z_re - c.re;
                z_im *= z_im - c.im;
                iter++;
            }

            // Store the Julia set value in the output array
            julia[idx * height + idy] = iter;
        }
    """

    # Create a CL program and kernel from the kernel code
    prg = clprogram.Program(ctx, kernel_code)
    prg.build()

    # Allocate memory for the input arrays (x, y) and output array (julia)
    x = np.random.rand(width * height).astype(np.float32)
    y = np.random.rand(height * width).astype(np.float32)
    julia = np.zeros((width, height), dtype=np.int)

    # Enqueue a kernel execution for the Julia set computation
    prg.julia_set(queue, (width, height), None,
                  x.astype(np.float32),
                  y.astype(np.float32),
                  julia.astype(np.int))

    return julia

def plot_julia(julia):
    plt.imshow(julia)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    width, height = 500, 500
    c = complex(-0.8, 0.156)
    max_iter = 255

    julia = compute_julia_opencl(width, height, c, max_iter)
    plot_julia(julia)

