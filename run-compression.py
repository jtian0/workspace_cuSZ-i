import subprocess as sp

## all modes are
# eb = 1e-3
# fname = ""
# x, y, z = 1, 1, 1
# rate = 1.0


class len3:
    x: int = 1
    y: int = 1
    z: int = 1

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class run_compression:

    @staticmethod
    def get_value_range(fname: str) -> float:
        """
        Use qcat to extract the value range, which requires proper setup of
        LD_LIBRARY_PATH
        """
        value_range = sp.check_output(
            f"echo $(printProperty -f {fname} 0) | grep valueRange | "
            "awk -F'= ' ' {print $NF} '",
            shell=True,
        )
        return float(value_range)

    @staticmethod
    def get_compressor_command(
        compressor: str,
        fname: str,
        nd_size: len3,
        dowhat: str = "compress",
        mode: str = "relative",
        eb: str = "1e-3",
        rate: float = "16.0",
    ) -> str:
        x, y, z = nd_size.x, nd_size.y, nd_size.z

        if dowhat in ["compress", "c"]:
            if compressor in ["cuszi", "cuszinterp"]:
                comp_mode = "r2r" if mode == "relative" else "abs"
                return f"cuszi -z -t f32 -m {comp_mode} -e {eb} -i {fname} -l {x}x{y}x{z} --predictor spline"
            elif compressor == "cusz":
                return f"cuszi -z -t f32 -m {comp_mode} -e {eb} -i {fname} -l {x}x{y}x{z} --predictor lorenzo"
            elif compressor in ["fzgpu", "fz-gpu"]:
                ## round trip
                val_rng = run_compression.get_value_range(fname)
                rel_eb = float(eb) * val_rng
                return f"fz-gpu {fname} {x} {y} {z} {rel_eb}"
            elif compressor in ["cuszp"]:
                ## round trip
                comp_mode = "REL" if mode == "relative" else "ABS"
                return f"cuSZp_gpu_f32_api {fname} {comp_mode} {eb}"
            elif compressor in ["zfp", "cuzfp"]:
                ## rate is equivalence based on other compressors
                return f"zfp -f -i {fname} -3 {z} {y} {x} -r {rate} -x cuda"
            ## TODO SZX
        elif dowhat in ["decompress", "decomp", "x"]:
            if compressor in ["cuszi", "cuszinterp", "cusz"]:
                return "cuszi -x -i {fname}.cusza --compare {fname}"
            ## TODO SZX
            ## TODO cuZFP
