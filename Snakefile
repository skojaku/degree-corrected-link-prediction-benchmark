import os
import numpy as np

# ALPHAS = np.round(np.linspace(0.1,2,5),2) #weights of social influence 
# BETAS = np.round(np.linspace(0.1,2,5),2) #weights of internal coherence
ALPHAS = [1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.8,2.9,3.1,3.2,3.3,3.5]
INI_STRING = ["splus","sminus"]
START = [5,10,25,50,100]
SD = [5,25,45,100]
ENSEMBLES = [1,2,3,4,5]

data_dir = "/data/sg/racball/link-prediction/notebooks/rachith/metropolis-hastings"

DATA = os.path.join(data_dir, "{ini_string}_alpha{alpha}_start{start}_normalscale{sd}_ensembles{e}.pkl")

rule all:
    input:
        expand(DATA,ini_string = INI_STRING, alpha=ALPHAS, start=START,sd = SD,e=ENSEMBLES)
  
rule generate_data:
    input:
        "workflow/generate-samples-metropolis-hastings.py"
    output:
        DATA
    shell:
        "python {input} -a {wildcards.alpha} -st {wildcards.start} -s {wildcards.sd} -i {wildcards.ini_string} -e {wildcards.e}"