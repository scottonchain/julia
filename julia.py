
def compute_julia_opencl(width, height, c, max_iter):
    ctx = cl.create_context()
    queue = ctx.get_command_queue()

    kernel_code = """
        __kernel void julia_set(__global float2 *x,
                                  __global uint8 *julia, const int max_iter) {
            int idx = get_global_id(0);
            int idy = get_global_id(1);

            double z_re = x[idx].s0 + y[idy].s0;
            double z_im = x[idx].s1 - y[idy].s1;

            int iter = 0;

            while (iter < max_iter && np.abs(z_re) <= 2.0) {
                z_re *= z_re - c.re;
                z_im *= z_im - c.im;
                iter++;
            }

            julia[idx * height + idy] = (uint8)(iter);
        }
    """

    prg = clprogram.Program(ctx, kernel_code)
    prg.build()

    x = np.random.rand(width * height).astype(np.float32)
    y = np.random.rand(height * width).astype(np.float32)

    julia_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.HOST_WRITE,
                            (width, height) * 1 * np.uint8.dtype.itemsize)

    prg.julia_set(queue, (width, height), None, x.astype(np.float32),
                  y.astype(np.float32), julia_buf)

    return julia_buf

