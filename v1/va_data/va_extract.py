import numpy as np

panels = np.genfromtxt("20240126_21617082-SOLAT_M1_Installed_Panels.csv", usecols=(1,), dtype=str, delimiter=',', skip_header=1)
data = np.genfromtxt("20240126_21617082-SOLAT_M1_Installed_Panels.csv", usecols=(8,9,10), delimiter=',', skip_header=1)

uniq_panels = np.unique(panels)

for up in uniq_panels:
    msk = panels == up
    dat = data[msk]
    np.savetxt(f"{up}.txt", dat)
