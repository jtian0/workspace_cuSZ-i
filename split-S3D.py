import numpy as np
import os

files=[f for f in os.listdir() if ".d64" in f]
fields=["CH4","O2","CO","CO2","H2O","N2","T","PRES","U","V","W"]
for f in files:
 print(f"    spliting and converting {f}")
 a=np.fromfile(f,dtype=np.double).reshape((11,500,500,500))
 spl=f.split(".")
 print("        ", end="")
 for i in range(11):
  print(f"{i}/11..", end="\n" if i == 10 else "")
  outname=".".join([spl[0],spl[1],spl[2],fields[i],"f32"])
  a[i].astype(np.float32).tofile(outname)
