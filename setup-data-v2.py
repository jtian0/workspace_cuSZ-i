import subprocess as sp
import os, sys
import numpy as np


db: dict = {
    "nyx": {
        "type": "f4",
        "url": "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz",
        "tar_gz": "SDRBENCH-EXASKY-NYX-512x512x512.tar.gz",
        "untar_dir": "SDRBENCH-EXASKY-NYX-512x512x512",
        "file_list": [
            "baryon_density.f32",
            "dark_matter_density.f32",
            "temperature.f32",
            "template_data.txt",
            "velocity_x.f32",
            "velocity_y.f32",
            "velocity_z.f32",
        ],
    },
    "qmc": {
        "type": "f4",
        "url": "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/QMCPack/SDRBENCH-QMCPack.tar.gz",
        "tar_gz": "SDRBENCH-QMCPack.tar.gz",
        "untar_dir": "dataset",
        "supposed_untar_dir": "SDRBENCH-QMCPack",
        "file_list": [
            # "115x69x69x288/einspline_115_69_69_288.f32",
            "288x115x69x69/einspline_288_115_69_69.pre.f32",
            # "einspline_115_69_69_288.f32",
            "einspline_288_115_69_69.pre.f32",
        ],
    },
    "miranda": {
        "type": "f8",
        "url": "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Miranda/SDRBENCH-Miranda-256x384x384.tar.gz",
        "tar_gz": "SDRBENCH-Miranda-256x384x384.tar.gz",
        "untar_dir": "SDRBENCH-Miranda-256x384x384",
        "file_list": [
            "density.d64",
            "diffusivity.d64",
            "pressure.d64",
            "velocityx.d64",
            "velocityy.d64",
            "velocityz.d64",
            "viscocity.d64",
        ],
        "file_list_converted": [
            "density.f32",
            "diffusivity.f32",
            "pressure.f32",
            "velocityx.f32",
            "velocityy.f32",
            "velocityz.f32",
            "viscocity.f32",
        ],
    },
    "s3d": {
        "type": "f8",
        "url": "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/S3D/SDRBENCH-S3D.tar.gz",
        "tar_gz": "SDRBENCH-S3D.tar.gz",
        "untar_dir": "SDRBENCH-S3D",
        "file_list": [
            "stat_planar.1.1000E-03.field.d64",
            "stat_planar.1.7000E-03.field.d64",
            "stat_planar.2.3500E-03.field.d64",
            "stat_planar.2.9000E-03.field.d64",
            "stat_planar.2.9950E-03.field.d64",
        ],
        "file_list_converted": [
            "stat_planar.1.1000E-03.CH4.f32",
            "stat_planar.1.7000E-03.CH4.f32",
            "stat_planar.2.3500E-03.CH4.f32",
            "stat_planar.2.9000E-03.CH4.f32",
            "stat_planar.2.9950E-03.CH4.f32",
            "stat_planar.1.1000E-03.CO.f32",
            "stat_planar.1.7000E-03.CO.f32",
            "stat_planar.2.3500E-03.CO.f32",
            "stat_planar.2.9000E-03.CO.f32",
            "stat_planar.2.9950E-03.CO.f32",
            "stat_planar.1.1000E-03.CO2.f32",
            "stat_planar.1.7000E-03.CO2.f32",
            "stat_planar.2.3500E-03.CO2.f32",
            "stat_planar.2.9000E-03.CO2.f32",
            "stat_planar.2.9950E-03.CO2.f32",
            "stat_planar.1.1000E-03.H2O.f32",
            "stat_planar.1.7000E-03.H2O.f32",
            "stat_planar.2.3500E-03.H2O.f32",
            "stat_planar.2.9000E-03.H2O.f32",
            "stat_planar.2.9950E-03.H2O.f32",
            "stat_planar.1.1000E-03.N2.f32",
            "stat_planar.1.7000E-03.N2.f32",
            "stat_planar.2.3500E-03.N2.f32",
            "stat_planar.2.9000E-03.N2.f32",
            "stat_planar.2.9950E-03.N2.f32",
            "stat_planar.1.1000E-03.O2.f32",
            "stat_planar.1.7000E-03.O2.f32",
            "stat_planar.2.3500E-03.O2.f32",
            "stat_planar.2.9000E-03.O2.f32",
            "stat_planar.2.9950E-03.O2.f32",
            "stat_planar.1.1000E-03.PRES.f32",
            "stat_planar.1.7000E-03.PRES.f32",
            "stat_planar.2.3500E-03.PRES.f32",
            "stat_planar.2.9000E-03.PRES.f32",
            "stat_planar.2.9950E-03.PRES.f32",
            "stat_planar.1.1000E-03.T.f32",
            "stat_planar.1.7000E-03.T.f32",
            "stat_planar.2.3500E-03.T.f32",
            "stat_planar.2.9000E-03.T.f32",
            "stat_planar.2.9950E-03.T.f32",
            "stat_planar.1.1000E-03.U.f32",
            "stat_planar.1.7000E-03.U.f32",
            "stat_planar.2.3500E-03.U.f32",
            "stat_planar.2.9000E-03.U.f32",
            "stat_planar.2.9950E-03.U.f32",
            "stat_planar.1.1000E-03.V.f32",
            "stat_planar.1.7000E-03.V.f32",
            "stat_planar.2.3500E-03.V.f32",
            "stat_planar.2.9000E-03.V.f32",
            "stat_planar.2.9950E-03.V.f32",
            "stat_planar.1.1000E-03.W.f32",
            "stat_planar.1.7000E-03.W.f32",
            "stat_planar.2.3500E-03.W.f32",
            "stat_planar.2.9000E-03.W.f32",
            "stat_planar.2.9950E-03.W.f32",
        ],
        "file_list_converted_del": [
            "stat_planar.1.1000E-03.CH4.f32",
            "stat_planar.1.1000E-03.CO2.f32",
            "stat_planar.1.1000E-03.CO.f32",
            "stat_planar.1.1000E-03.H2O.f32",
            "stat_planar.1.1000E-03.N2.f32",
            "stat_planar.1.1000E-03.O2.f32",
            "stat_planar.1.1000E-03.PRES.f32",
            "stat_planar.1.1000E-03.T.f32",
            "stat_planar.1.1000E-03.U.f32",
            "stat_planar.1.1000E-03.V.f32",
            "stat_planar.1.1000E-03.W.f32",
            "stat_planar.1.7000E-03.CH4.f32",
            "stat_planar.1.7000E-03.CO2.f32",
            "stat_planar.1.7000E-03.CO.f32",
            "stat_planar.1.7000E-03.H2O.f32",
            "stat_planar.1.7000E-03.N2.f32",
            "stat_planar.1.7000E-03.O2.f32",
            "stat_planar.1.7000E-03.PRES.f32",
            "stat_planar.1.7000E-03.T.f32",
            "stat_planar.1.7000E-03.U.f32",
            "stat_planar.1.7000E-03.V.f32",
            "stat_planar.1.7000E-03.W.f32",
            "stat_planar.2.3500E-03.N2.f32",
            "stat_planar.2.3500E-03.PRES.f32",
            "stat_planar.2.9000E-03.CH4.f32",
            "stat_planar.2.9000E-03.CO2.f32",
            "stat_planar.2.9000E-03.CO.f32",
            "stat_planar.2.9000E-03.H2O.f32",
            "stat_planar.2.9000E-03.N2.f32",
            "stat_planar.2.9000E-03.O2.f32",
            "stat_planar.2.9000E-03.PRES.f32",
            "stat_planar.2.9000E-03.T.f32",
            "stat_planar.2.9000E-03.U.f32",
            "stat_planar.2.9000E-03.V.f32",
            "stat_planar.2.9000E-03.W.f32",
            "stat_planar.2.9950E-03.CH4.f32",
            "stat_planar.2.9950E-03.CO2.f32",
            "stat_planar.2.9950E-03.CO.f32",
            "stat_planar.2.9950E-03.H2O.f32",
            "stat_planar.2.9950E-03.N2.f32",
            "stat_planar.2.9950E-03.O2.f32",
            "stat_planar.2.9950E-03.PRES.f32",
            "stat_planar.2.9950E-03.T.f32",
            "stat_planar.2.9950E-03.U.f32",
            "stat_planar.2.9950E-03.V.f32",
            "stat_planar.2.9950E-03.W.f32",
        ],
    },
}

db_keys = db.keys()

RED = "\033[0;31m"
GRAY = "\033[0;37m"
NOCOLOR = "\033[0m"
BOLDRED = "\033[1;31m"


def validate_url(url: str):
    if not url.startswith("https"):
        raise ValueError("[fn::basename] ILLEGAL URL: not start with `https`")
    if not url.endswith("tar.gz"):
        raise ValueError("[fn::basename] ILLEGAL URL: not end with `tar.gz`")


def download(key: str):
    url = db[key]["url"]
    tar_gz = db[key]["tar_gz"]
    validate_url(url)

    target_tar_gz = os.path.join(datapath, tar_gz)
    if os.path.exists(target_tar_gz):
        print(f"[{key}::wget]\t{target_tar_gz} exists...skip downloading")
        pass
    else:
        print(f"[{key}::wget]\tdownloading {tar_gz}")
        cmd = f"wget {url} -P {datapath}"
        # print(cmd)
        sp.check_call(cmd, shell=True)


def untar(key: str):
    tar_gz = db[key]["tar_gz"]
    untar_dir = db[key]["untar_dir"]

    target_tar_gz = os.path.join(datapath, tar_gz)

    if not os.path.exists(target_tar_gz):
        raise FileNotFoundError(f"[untar {key}]\t{target_tar_gz} (for {key}) does not exists...")
    
    if key == "qmc" and any([os.path.exists(f"{datapath}/{untar_dir}/{i}") for i in db[key]['file_list']]):
        print(f"[{key}::untar]\tneeded files previously untar'ed ({key})...skip")
    elif all([os.path.exists(f"{datapath}/{untar_dir}/{i}") for i in db[key]['file_list']]):
        print(f"[{key}::untar]\tall files previously untar'ed ({key})...skip")
    else:
        cmd = f"tar zxvf {target_tar_gz} --directory {datapath}"
        # print(cmd)
        sp.check_call(cmd, shell=True)


# special fix to QMC: nested dir
def fix_qmc():
    ori_dir = os.path.join(datapath, "dataset") if datapath.startswith("/") else "dataset"
    untar_dir = db["qmc"]["supposed_untar_dir"]
    supposed_dir = os.path.join(datapath, untar_dir)

    # create a symbolic link to "dataset"
    try:
        os.symlink(ori_dir, supposed_dir)
    except FileExistsError:
        os.remove(supposed_dir)
        os.symlink(ori_dir, supposed_dir)
        print("[qmc::fix]\tcreated a symbolic link for QMC dir")
        pass

    # flatten qmc dir
    try:
        os.rename(
            f"{supposed_dir}/288x115x69x69/einspline_288_115_69_69.pre.f32",
            f"{supposed_dir}/einspline_288_115_69_69.pre.f32",
        )
    except FileNotFoundError:
        if os.path.exists(
            f"{supposed_dir}/einspline_288_115_69_69.pre.f32",
        ):
            pass
        else:
            raise FileNotFoundError(
                f"[qmc::fix]\tPlease go to {datapath}, manually run `tar zxvf SDRBENCH-QMCPack.tar.gz`, and come back to this director, and rerun by `python setup-data-v2.py`"
            )


def convert_miranda():
    fdir = db["miranda"]["untar_dir"]
    flist_d64 = db["miranda"]["file_list"]
    flist_f32 = db["miranda"]["file_list_converted"]
    if all([os.path.exists(f"{datapath}/{fdir}/{i}") for i in flist_d64]):
        ## exists .d64
        if not all([os.path.exists(f"{datapath}/{fdir}/{i}") for i in flist_f32]):
            print("[miranda::convert]\tconverting from f8 to f4...")
            for i, (src, dst) in enumerate(zip(flist_d64, flist_f32)):
                src = f"{datapath}/{fdir}/{src}"
                dst = f"{datapath}/{fdir}/{dst}"
                sp.check_call(
                    f"convertDoubleToFloat {src} {dst} >/dev/null", shell=True
                )
            print()
        else:
            print("[miranda::convert]\tall .f32 files ready, skip")
    else:
        print(
            f"[miranda::convert]\tPlease go to {datapath}, manually run `tar zxvf SDRBENCH-Miranda-256x384x384.tar.gz`, and come back to this director, and rerun by `python setup-data-v2.py`"
        )


def convert_s3d_helper():
    s3d_path = os.path.join(datapath, "SDRBENCH-S3D")
    files = [f for f in os.listdir(s3d_path) if ".d64" in f]
    fields = ["CH4", "O2", "CO", "CO2", "H2O", "N2", "T", "PRES", "U", "V", "W"]
    for f in files:
        f = os.path.join(s3d_path, f)
        print(f"    spliting and converting {f}")
        a = np.fromfile(f, dtype=np.double).reshape((11, 500, 500, 500))
        spl = f.split(".")
        print("        ", end="")
        for i in range(11):
            print(f"{i}/11..", end="\n" if i == 10 else "")
            outname = ".".join([spl[0], spl[1], spl[2], fields[i], "f32"])
            if outname not in db["s3d"]["file_list_converted_del"]:
                outname = os.path.join(s3d_path, outname)
                a[i].astype(np.float32).tofile(outname)


def convert_s3d():
    fdir = db["s3d"]["untar_dir"]
    flist_d64 = db["s3d"]["file_list"]
    flist_f32 = db["s3d"]["file_list_converted"]
    flist_f32_del = db["s3d"]["file_list_converted_del"]
    if all([os.path.exists(f"{datapath}/{fdir}/{i}") for i in flist_d64]):
        ## exists .d64
        feature_list = [i for i in flist_f32 if i not in flist_f32_del]
        if not all([os.path.exists(f"{datapath}/{fdir}/{i}") for i in feature_list]):
            print("[s3d::convert]\textacting S3D data (from f8 to f4) and converting...")
            convert_s3d_helper()

        else:
            print("[s3d::convert]\tall .f32 files ready, skip")
    else:
        print(
            f"[s3d::conver]\tPlease go to {datapath}, manually run `tar zxvf SDRBENCH-S3D.tar.gz`, and come back to this director, and rerun by `python setup-data-v2.py`"
        )


##################
## run this script
##################

try:
    datapath = os.environ["DATAPATH"]
except KeyError as e:
    print(
        "[setup::data]\tShell variable `DATAPATH` is not set. Please `source setup-all.sh <CUDA VER> <WHERE_TO_PUT_DATA_DIRS>`."
    )
    exit(1)

if not os.path.exists(datapath):
    os.makedirs(datapath)
    print(f"[setup::data]\tcreating DATAPATH -> {datapath}")
else:
    print(f"[setup::data]\tDATAPATH -> {datapath} exists, skip")

# print(datapath)

for k in db_keys:
    download(k)
    untar(k)

fix_qmc()
convert_miranda()
convert_s3d()
