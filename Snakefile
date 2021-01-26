import tempfile
from os.path import join as j

import numpy as np
import pandas as pd


configfile: "workflow/config.yaml"


#
# Directories
#
DATA_DIR = config["data_dir"]
PAPER_DIR = config["paper_dir"]
FIG_DIR = config["fig_dir"]
SHARED_DIR = config["shared_dir"]
INTB_CONT_RES_DIR = j(DATA_DIR, "res-intb-cont")
INTB_CONT_SEIR_RES_DIR = j(INTB_CONT_RES_DIR, "seir")
DTU_DIR = j(DATA_DIR, "res-dtu/seir")
BA_CONT_RES_DIR = j(DATA_DIR, "res-ba-cont")
BA_CONT_SEIR_RES_DIR = j(BA_CONT_RES_DIR, "seir")


#
# Paper
#
PAPER_SRC, SUPP_SRC = [j(PAPER_DIR, f) for f in ("main.tex", "supp.tex")]
PAPER, SUPP = [j(PAPER_DIR, f) for f in ("main.pdf", "supp.pdf")]


#
# Figures
#
FIGS = [
    j(FIG_DIR, f)
    for f in (
        "schematic-ctrace.pdf",
        "deg-ccdf-new.pdf",
        "sim_results.pdf",
        "sim_dtu_results.pdf",
    )
]


#
# Simulations on people-gathering network
#
# Parameter for networks
INTB_CONT_N = 250000  # fraction of initial infected individuas
INTB_CONT_GFRAC = ["0.2"]  # fraction of gathering
INTB_CONT_GAMMA = ["3.0"]  # Transmission rate
INTB_CONT_SAMPLE_NUM = 100  # Number of simulations
INTB_CONT_SEIR_E2I_RATE = ["0.1", "0.25", "0.5", "1.0", "2.0"]  # incubation rate
INTB_CONT_TRNS_RATE = ["0.25"]  # transmission rate
INTB_CONT_RECOV_RATE = ["0.25"]  # recovery rate

# Parameter for contact tracing
INTB_CONT_PT = [1.0]  # probability of tracing
INTB_CONT_PS_LIST = [
    "0.05",
    "0.25",
    "0.5",
]  # probability of detecting infections

INTB_CONT_MAX_TRACE_NODE = [
    10,
    30,
    50,
    100,
    99999999,
]  # maximum number of tracing contacts
INTB_CONT_INTERV_START_DAY = [0.1]  # intervention starting time
INTB_CONT_TRACE_MODE = ["frequency"]  # type of contact tracing
INTB_CONT_INCUBATION_PERIOD = ["0"]  # time lag between infection and isolation
INTB_CONT_INTERV_CYCLE = ["1.0"]  # Inter-tracing time
INTB_CONT_INTERV_MEMORY = ["0"]  # set 0. This is no longer used
INTB_CONT_PARAMS = {
    "ps": INTB_CONT_PS_LIST,
    "maxnode": INTB_CONT_MAX_TRACE_NODE,
    "gamma": INTB_CONT_GAMMA,
    "gfrac": INTB_CONT_GFRAC,
    "sample": np.arange(INTB_CONT_SAMPLE_NUM),
    "cycle": INTB_CONT_INTERV_CYCLE,
    "memory": INTB_CONT_INTERV_MEMORY,
    "start_day": INTB_CONT_INTERV_START_DAY,
    "incubation": INTB_CONT_INCUBATION_PERIOD,
    "tracemode": INTB_CONT_TRACE_MODE,
}

# Parameter for plotting
INTB_CONT_NUM_TIME_POINTS = 100  # number of time points at which we measure the status

# Log files for simulations
INTB_CONT_SEIR_LOG_FILE = j(
    INTB_CONT_SEIR_RES_DIR,
    "output",
    "log-g{gamma}-gfrac{gfrac}-e2i{E2I_rate}-trans{trans_rate}-recov{recov_rate}-s{sample}.csv",
)
INTB_CONT_SEIR_LOG_FILE_ALL = expand(
    INTB_CONT_SEIR_LOG_FILE,
    gamma=INTB_CONT_GAMMA,
    gfrac=INTB_CONT_GFRAC,
    E2I_rate=INTB_CONT_SEIR_E2I_RATE,
    trans_rate=INTB_CONT_TRNS_RATE,
    recov_rate=INTB_CONT_RECOV_RATE,
    sample=np.arange(INTB_CONT_SAMPLE_NUM),
)

# Network files
INTB_CONT_SEIR_NET_FILE = j(
    INTB_CONT_SEIR_RES_DIR,
    "output",
    "net-g{gamma}-gfrac{gfrac}-e2i{E2I_rate}-trans{trans_rate}-recov{recov_rate}-s{sample}.gexf",
)
INTB_CONT_SEIR_NET_FILE_ALL = expand(
    INTB_CONT_SEIR_NET_FILE,
    gamma=INTB_CONT_GAMMA,
    gfrac=INTB_CONT_GFRAC,
    E2I_rate=INTB_CONT_SEIR_E2I_RATE,
    trans_rate=INTB_CONT_TRNS_RATE,
    recov_rate=INTB_CONT_RECOV_RATE,
    sample=np.arange(INTB_CONT_SAMPLE_NUM),
)

# Result files
INTB_CONT_SEIR_RESULT_FILE = j(
    INTB_CONT_SEIR_RES_DIR,
    "results",
    "res-g{gamma}-grac{gfrac}-e2i{E2I_rate}-trans{trans_rate}-recov{recov_rate}-s{sample}-ps{ps}-maxnode{maxnode}-cycle{cycle}-memory={memory}-start={start_day}-incuvation={incubation}-tracemode={tracemode}.csv.gz",
)
INTB_CONT_SEIR_RESULT_FILE_ALL = expand(
    INTB_CONT_SEIR_RESULT_FILE,
    E2I_rate=INTB_CONT_SEIR_E2I_RATE,
    trans_rate=INTB_CONT_TRNS_RATE,
    recov_rate=INTB_CONT_RECOV_RATE,
    **INTB_CONT_PARAMS
)
INTB_CONT_SEIR_RESULT_EVENT_FILE = j(
    INTB_CONT_SEIR_RES_DIR,
    "results",
    "event-g{gamma}-grac{gfrac}-e2i{E2I_rate}-trans{trans_rate}-recov{recov_rate}-s{sample}-ps{ps}-maxnode{maxnode}-cycle{cycle}-memory={memory}-start={start_day}-incuvation={incubation}-tracemode={tracemode}.csv.gz",
)
INTB_CONT_SEIR_RESULT_EVENT_FILE_ALL = expand(
    INTB_CONT_SEIR_RESULT_EVENT_FILE,
    E2I_rate=INTB_CONT_SEIR_E2I_RATE,
    trans_rate=INTB_CONT_TRNS_RATE,
    recov_rate=INTB_CONT_RECOV_RATE,
    **INTB_CONT_PARAMS
)

# Files used for plot
INTB_CONT_SEIR_PLOT_TIME_INFECTED_FILE_LIST = expand(
    INTB_CONT_SEIR_RESULT_EVENT_FILE,
    E2I_rate="0.25",
    trans_rate=INTB_CONT_TRNS_RATE,
    recov_rate=INTB_CONT_RECOV_RATE,
    **INTB_CONT_PARAMS
)
INTB_CONT_SEIR_PLOT_TIME_INFECTED_DATA = j(
    INTB_CONT_SEIR_RES_DIR, "plot-data-time-vs-infected.csv"
)
INTB_CONT_SEIR_PLOT_PS_INFECTED_FILE_LIST = expand(
    INTB_CONT_SEIR_RESULT_FILE,
    E2I_rate="0.25",
    trans_rate=INTB_CONT_TRNS_RATE,
    recov_rate=INTB_CONT_RECOV_RATE,
    **INTB_CONT_PARAMS
)
INTB_CONT_SEIR_PLOT_PS_INFECTED_DATA = j(
    INTB_CONT_SEIR_RES_DIR, "plot-data-ps-vs-infected.csv"
)


#
# Simulations on student contact networks
#
# Input for contact data
DTU_CONT_CONTACT_DATA = j(
    SHARED_DIR, "shared_data/sensible-dtu/input/bluetooth-short-q60.csv"
)
# Retrieve the simulation results
DTU_CONT_SIMULATION_DATA_BETA = ["0.50"]  # Transmission rate
DTU_CONT_SIMULATION_DATA = j(
    SHARED_DIR,
    "shared_data/sensible-dtu/output/%s/beta{beta}_T5.1_logs.csv" % DTU_MODEL,
)
DTU_CONT_SIMULATION_DATA_ALL = expand(
    DTU_CONT_SIMULATION_DATA, beta=DTU_CONT_SIMULATION_DATA_BETA
)
DTU_CONT_SIMULATION_META_DATA = j(
    SHARED_DIR,
    "shared_data/sensible-dtu/output/%s/beta{beta}_T5.1_meta.csv" % DTU_MODEL,
)
DTU_CONT_SIMULATION_META_DATA_ALL = expand(
    DTU_CONT_SIMULATION_META_DATA, beta=DTU_CONT_SIMULATION_DATA_BETA
)

# Parameter for contact tracing
DTU_CONT_TRACE_TIME_WINDOW = [
    7
]  # length of tracing window within which we count the number of contacts
DTU_CONT_CLOSE_CONTACT_THRESHOLD_PER_DAY = [
    0.1,
    1,
]  # Threshold between light and close contacts

# detection and tracing probability. See INTB for details.
DTU_CONT_PS_LIST = [
    "0.05",
    "0.25",
    "0.5",
]

DTU_CONT_MAX_TRACE_NODE = [1, 3, 10, 9999]
DTU_CONT_INTERV_START_DAY = [3]
DTU_CONT_INCUBATION_PERIOD = [0]
DTU_CONT_INTERV_CYCLE = ["1.0"]
DTU_CONT_INTERV_MEMORY = ["0"]
DTU_CONT_TRACE_MODE = ["frequency"]
DTU_CONT_RESULT_FILE = j(
    DTU_DIR,
    "res_beta={beta}_cont_ttwindow={ttwindow}_ccontact={ccontact}_ps={ps}_maxnode={maxnode}_cycle={cycle}_memory={memory}_start={start_day}_incubation={incubation}_tracemode={tracemode}.csv",
)
DTU_CONT_PARAMS = {
    "ttwindow": DTU_CONT_TRACE_TIME_WINDOW,
    "ccontact": DTU_CONT_CLOSE_CONTACT_THRESHOLD_PER_DAY,
    "ps": DTU_CONT_PS_LIST,
    "maxnode": DTU_CONT_MAX_TRACE_NODE,
    "beta": DTU_CONT_SIMULATION_DATA_BETA,
    "cycle": DTU_CONT_INTERV_CYCLE,
    "memory": DTU_CONT_INTERV_MEMORY,
    "start_day": DTU_CONT_INTERV_START_DAY,
    "incubation": DTU_CONT_INCUBATION_PERIOD,
    "tracemode": ["frequency", "random"],
}
DTU_CONT_RESULT_FILE_ALL = expand(DTU_CONT_RESULT_FILE, **DTU_CONT_PARAMS)
DTU_CONT_RESULT_EVENT_FILE = j(
    DTU_DIR,
    "event_beta={beta}_cont_ttwindow={ttwindow}_ccontact={ccontact}_ps={ps}_maxnode={maxnode}_cycle={cycle}_memory={memory}_start={start_day}_incubation={incubation}_tracemode={tracemode}.csv",
)
DTU_CONT_RESULT_EVENT_FILE_ALL = expand(DTU_CONT_RESULT_EVENT_FILE, **DTU_CONT_PARAMS)

# Parameter for plot
DTU_CONT_NUM_TIME_POINTS = 100
DTU_CONT_PLOT_DATA_PARAM = {
    "ttwindow": "7",
    "ccontact": "0.00595238095",  # 1 hour for seven days
    "ps": DTU_CONT_PS_LIST,
    "maxnode": DTU_CONT_MAX_TRACE_NODE,
    "beta": "0.50",
    "cycle": ["1.0"],
    "memory": "0",
    "start_day": DTU_CONT_INTERV_START_DAY,
    "incubation": 0,
    "tracemode": ["frequency", "random"],
}
DTU_CONT_PLOT_TIME_INFECTED_FILE_LIST = expand(
    DTU_CONT_RESULT_EVENT_FILE, **DTU_CONT_PLOT_DATA_PARAM
)
DTU_CONT_PLOT_TIME_INFECTED_DATA = j(DTU_DIR, "plot-data-time-vs-infected.csv")
DTU_CONT_PLOT_PS_INFECTED_FILE_LIST = expand(
    DTU_CONT_RESULT_FILE, **DTU_CONT_PLOT_DATA_PARAM
)
DTU_CONT_PLOT_PS_INFECTED_DATA = j(DTU_DIR, "plot-data-ps-vs-infected.csv")

#
# Simulations on Barabashi-Albert Net
#
# Parameter for networks
BA_CONT_N = 250000  # number of nodes
BA_CONT_M = 2

# Parameter for simulations. See INTB for details.
BA_CONT_NUM_SAMPLE = 100
BA_CONT_PS_LIST = [
    "0.05",
    "0.25",
    "0.5",
]  # detection probability
BA_CONT_T_LIST = ["0.25"]
BA_CONT_R_LIST = ["0.25"]
BA_CONT_SEIR_E2I_RATE = ["0.1", "0.25", "0.5", "1.0", "2.0"]

# Parameter for contact tracing
BA_CONT_INTERV_START_DAY = ["0.5"]
BA_CONT_INTERV_CYCLE = ["1.0"]
BA_CONT_INTERV_MEMORY = ["0"]
BA_CONT_INCUBATION_PERIOD = ["0"]
BA_CONT_MAX_TRACE_NODE = [10, 20, 30, 50, 99999999]
BA_CONT_TRACE_MODE = ["frequency"]
BA_CONT_PARAMS = {
    "ps": BA_CONT_PS_LIST,
    "trans": BA_CONT_T_LIST,
    "recov": BA_CONT_R_LIST,
    "sample": np.arange(BA_CONT_NUM_SAMPLE),
    "start_day": BA_CONT_INTERV_START_DAY,
    "incubation": BA_CONT_INCUBATION_PERIOD,
    "maxnode": BA_CONT_MAX_TRACE_NODE,
    "cycle": BA_CONT_INTERV_CYCLE,
    "memory": BA_CONT_INTERV_MEMORY,
    "tracemode": ["frequency"],
}

# Log files for simulations
BA_CONT_SEIR_LOG_FILE = j(
    BA_CONT_SEIR_RES_DIR,
    "output",
    "log-e2i{E2I_rate}-trans{trans}-recov{recov}-s{sample}.csv",
)
BA_CONT_SEIR_LOG_FILE_ALL = expand(
    BA_CONT_SEIR_LOG_FILE,
    trans=BA_CONT_T_LIST,
    recov=BA_CONT_R_LIST,
    E2I_rate=BA_CONT_SEIR_E2I_RATE,
    sample=np.arange(BA_CONT_NUM_SAMPLE),
)

# Network files
BA_CONT_SEIR_NET_FILE = j(
    BA_CONT_SEIR_RES_DIR,
    "output",
    "net-e2i{E2I_rate}-trans{trans}-recov{recov}-s{sample}.edgelist",
)
BA_CONT_SEIR_NET_FILE_ALL = expand(
    BA_CONT_SEIR_NET_FILE,
    trans=BA_CONT_T_LIST,
    recov=BA_CONT_R_LIST,
    E2I_rate=BA_CONT_SEIR_E2I_RATE,
    sample=np.arange(BA_CONT_NUM_SAMPLE),
)

# Result files
BA_CONT_SEIR_RESULT_FILE = j(
    BA_CONT_SEIR_RES_DIR,
    "results",
    "res_e2i{E2I_rate}_trans{trans}_recov{recov}_s{sample}_ps{ps}_maxnode{maxnode}_cycle{cycle}_memory={memory}_start={start_day}_incuvation={incubation}_tracemode={tracemode}.csv.gz",
)
BA_CONT_SEIR_RESULT_FILE_ALL = expand(
    BA_CONT_SEIR_RESULT_FILE, E2I_rate=BA_CONT_SEIR_E2I_RATE, **BA_CONT_PARAMS
)
BA_CONT_SEIR_RESULT_EVENT_FILE = j(
    BA_CONT_SEIR_RES_DIR,
    "results",
    "event_e2i{E2I_rate}_trans{trans}_recov{recov}_s{sample}_ps{ps}_maxnode{maxnode}_cycle{cycle}_memory={memory}_start={start_day}_incuvation={incubation}_tracemode={tracemode}.csv.gz",
)

# Files for plotting
BA_CONT_NUM_TIME_POINTS = 100
BA_CONT_SEIR_PLOT_TIME_INFECTED_DATA = j(
    BA_CONT_SEIR_RES_DIR, "plot-data-time-vs-infected.csv"
)
BA_CONT_SEIR_PLOT_PS_INFECTED_DATA = j(
    BA_CONT_SEIR_RES_DIR, "plot-data-ps-vs-infected.csv"
)
BA_CONT_SEIR_PLOT_PS_INFECTED_FILE_LIST = expand(
    BA_CONT_SEIR_RESULT_FILE, E2I_rate="0.25", **BA_CONT_PARAMS
)
BA_CONT_SEIR_PLOT_TIME_INFECTED_FILE_LIST = expand(
    BA_CONT_SEIR_RESULT_EVENT_FILE, E2I_rate="0.25", **BA_CONT_PARAMS
)

#
# Degree distribution for nodes, infected and traced nodes
#
# BA
BA_CONT_SEIR_DEG_DIST = j(BA_CONT_SEIR_RES_DIR, "deg-dist.csv")

# people gathering
INTB_CONT_SEIR_DEG_DIST = j(INTB_CONT_SEIR_RES_DIR, "deg-dist.csv")

# Student contact
DTU_CONT_TIME_RESOL = [1, 3, 6, 12, 12 * 6, 12 * 12, 12 * 24]
DTU_CONT_DEG_DIST = j(DTU_DIR, "%s-deg-dist-{resol}.csv" % DTU_MODEL)
DTU_CONT_DEG_DIST_ALL = expand(DTU_CONT_DEG_DIST, resol=DTU_CONT_TIME_RESOL)


rule all:
    input:
        PAPER,
        SUPP,


rule paper:
    input:
        PAPER_SRC,
        SUPP_SRC,
        FIGS,
    params:
        paper_dir=PAPER_DIR,
    output:
        PAPER,
        SUPP,
    shell:
        "cd {params.paper_dir}; make"


#
# Rules for generating and simulating SEIR models
#
rule intb_generate_networks_seir:
    output:
        INTB_CONT_SEIR_LOG_FILE,
        INTB_CONT_SEIR_NET_FILE,
    params:
        E2I_rate=lambda wildcards: wildcards.E2I_rate,
        trans_rate=lambda wildcards: wildcards.trans_rate,
        recov_rate=lambda wildcards: wildcards.recov_rate,
        gamma=lambda wildcards: wildcards.gamma,
        frac=lambda wildcards: wildcards.gfrac,
    shell:
        "python3 workflow/generate-synthe-people-gathering-nets-seir.py {INTB_CONT_N} {params.gamma} {params.frac} {params.E2I_rate} {params.trans_rate} {params.recov_rate} {output}"


rule ba_generate_networks_seir:
    output:
        BA_CONT_SEIR_LOG_FILE,
        BA_CONT_SEIR_NET_FILE,
    params:
        E2I_rate=lambda wildcards: wildcards.E2I_rate,
        trans=lambda wildcards: wildcards.trans,
        recov=lambda wildcards: wildcards.recov,
    shell:
        "python3 workflow/generate-ba-net-seir.py {BA_CONT_N} {BA_CONT_M} {params.E2I_rate} {params.trans} {params.recov} {output}"


rule ba_seir_continuous_ct:
    input:
        BA_CONT_SEIR_NET_FILE,
        BA_CONT_SEIR_LOG_FILE,
    output:
        BA_CONT_SEIR_RESULT_FILE,
        BA_CONT_SEIR_RESULT_EVENT_FILE,
    params:
        sample=lambda wildcards: wildcards.sample,
        ps=lambda wildcards: wildcards.ps,
        maxnode=lambda wildcards: wildcards.maxnode,
        cycle=lambda wildcards: wildcards.cycle,
        start_day=lambda wildcards: wildcards.start_day,
        incubation=lambda wildcards: wildcards.incubation,
        memory=lambda wildcards: wildcards.memory,
        trace_mode=lambda wildcards: wildcards.tracemode,
    shell:
        "python3 workflow/simulate_continuous_contact_tracing.py {input} {params.ps} {params.maxnode} {params.start_day} {params.cycle} {params.memory} {params.incubation} {params.trace_mode} {output}"


rule intb_seir_continuous_ct:
    input:
        INTB_CONT_SEIR_NET_FILE,
        INTB_CONT_SEIR_LOG_FILE,
    output:
        INTB_CONT_SEIR_RESULT_FILE,
        INTB_CONT_SEIR_RESULT_EVENT_FILE,
    params:
        sample=lambda wildcards: wildcards.sample,
        ps=lambda wildcards: wildcards.ps,
        maxnode=lambda wildcards: wildcards.maxnode,
        cycle=lambda wildcards: wildcards.cycle,
        start_day=lambda wildcards: wildcards.start_day,
        incubation=lambda wildcards: wildcards.incubation,
        memory=lambda wildcards: wildcards.memory,
        trace_mode=lambda wildcards: wildcards.tracemode,
    shell:
        "python3 workflow/simulate_continuous_contact_tracing.py {input} {params.ps} {params.maxnode} {params.start_day} {params.cycle} {params.memory} {params.incubation} {params.trace_mode} {output}"


#
# Rules for DTU data
#
rule dtu_continuous_interv_simulation:
    input:
        DTU_CONT_CONTACT_DATA,
        DTU_CONT_SIMULATION_DATA,
        DTU_CONT_SIMULATION_META_DATA,
    output:
        DTU_CONT_RESULT_FILE,
        DTU_CONT_RESULT_EVENT_FILE,
    params:
        ttwindow=lambda wildcards: wildcards.ttwindow,
        ccontact=lambda wildcards: wildcards.ccontact,
        ps=lambda wildcards: wildcards.ps,
        maxnode=lambda wildcards: wildcards.maxnode,
        cycle=lambda wildcards: wildcards.cycle,
        memory=lambda wildcards: wildcards.memory,
        start_day=lambda wildcards: wildcards.start_day,
        incubation=lambda wildcards: wildcards.incubation,
        tracemode=lambda wildcards: wildcards.tracemode,
    shell:
        "python3 workflow/dtu_sir_with_continuous_ct.py {input} {params.ttwindow} {params.ccontact} {params.ps} {params.maxnode} {params.start_day} {params.cycle} {params.memory} {params.incubation} {params.tracemode} {output}"


rule dtu_continuous_interv_deg_plot:
    input:
        DTU_CONT_CONTACT_DATA,
        DTU_CONT_SIMULATION_DATA_ALL,
        DTU_CONT_SIMULATION_META_DATA_ALL,
    output:
        DTU_CONT_DEG_DIST,
    params:
        ttwindow=7,
        ccontact=0.00595238095,
        ps=1,
        cycle=0.5,
        incubation=0,
        resol=lambda wildcards: wildcards.resol,
    shell:
        "python3 workflow/calc-deg-dist-dtu.py {input} {params.ttwindow} {params.ccontact} {params.ps} {params.cycle} {params.incubation} {params.resol} {output}"


rule ba_ct_degree_dist_seir:
    output:
        BA_CONT_SEIR_DEG_DIST,
    params:
        E2I_rate=0.25,
        trans_rate=0.25,
        recov_rate=0.25,
        num_samples=30,
        p_s=0.05,
        p_t=0.5,
        interv_t=10,
    shell:
        "python3 workflow/calc-deg-dist-seir-ba.py {BA_CONT_N} {BA_CONT_M} {params.E2I_rate} {params.trans_rate} {params.recov_rate} {params.num_samples} {params.p_s} {params.p_t} {params.interv_t} {output}"


rule intb_ct_degree_dist_seir:
    output:
        INTB_CONT_SEIR_DEG_DIST,
    params:
        E2I_rate=0.25,
        gamma=3.0,
        gfrac=0.2,
        trans_rate=0.25,
        recov_rate=0.25,
        num_samples=30,
        p_s=0.05,
        p_t=0.5,
        interv_t=5,
    shell:
        "python3 workflow/calc-deg-dist-seir-intb.py {INTB_CONT_N} {params.gamma} {params.gfrac} {params.E2I_rate} {params.trans_rate} {params.recov_rate} {params.num_samples} {params.p_s} {params.p_t} {params.interv_t} {output}"


# This is a remedy for preventing snakemake to stop due to passing too many files as commandline arguments.
# To get around this, I create a list of file names and save it as a csv file. The csv file is then passed to the program.
def make_file_list(files):
    filename = tempfile.NamedTemporaryFile(delete=False).name
    pd.DataFrame(files).to_csv(filename, index=False, header=None)
    return filename


#
# Preprocess data for plotting
#
# DTU data
rule prep_plot_data_time_vs_infected_dtu:
    input:
        DTU_CONT_PLOT_TIME_INFECTED_FILE_LIST,
    output:
        DTU_CONT_PLOT_TIME_INFECTED_DATA,
    params:
        filelist=temp(make_file_list(DTU_CONT_PLOT_TIME_INFECTED_FILE_LIST)),
    shell:
        "python3 workflow/calc_time_vs_infected_nodes.py {params.filelist}  {DTU_CONT_NUM_TIME_POINTS} {output}"


rule prep_plot_data_ps_vs_infected_dtu:
    input:
        DTU_CONT_PLOT_PS_INFECTED_FILE_LIST,
    output:
        DTU_CONT_PLOT_PS_INFECTED_DATA,
    params:
        filelist=temp(make_file_list(DTU_CONT_PLOT_PS_INFECTED_FILE_LIST)),
    shell:
        "python3 workflow/calc_ps_vs_infected.py {params.filelist} {output}"


# People-Gathering net
rule prep_plot_data_time_vs_infected_intb_seir:
    input:
        INTB_CONT_SEIR_PLOT_TIME_INFECTED_FILE_LIST,
    output:
        INTB_CONT_SEIR_PLOT_TIME_INFECTED_DATA,
    params:
        filelist=temp(make_file_list(INTB_CONT_SEIR_PLOT_TIME_INFECTED_FILE_LIST)),
    shell:
        "python3 workflow/calc_time_vs_infected_nodes.py {params.filelist} {INTB_CONT_NUM_TIME_POINTS} {output}"


rule prep_plot_data_ps_vs_infected_intb_seir:
    input:
        INTB_CONT_SEIR_PLOT_PS_INFECTED_FILE_LIST,
    output:
        INTB_CONT_SEIR_PLOT_PS_INFECTED_DATA,
    params:
        filelist=temp(make_file_list(INTB_CONT_SEIR_PLOT_PS_INFECTED_FILE_LIST)),
    shell:
        "python3 workflow/calc_ps_vs_infected.py {params.filelist} {output}"


# Barabasi-Albert net
rule prep_plot_data_time_vs_infected_ba_seir:
    input:
        BA_CONT_SEIR_PLOT_TIME_INFECTED_FILE_LIST,
    output:
        BA_CONT_SEIR_PLOT_TIME_INFECTED_DATA,
    params:
        filelist=temp(make_file_list(BA_CONT_SEIR_PLOT_TIME_INFECTED_FILE_LIST)),
    shell:
        "python3 workflow/calc_time_vs_infected_nodes.py {params.filelist} {BA_CONT_NUM_TIME_POINTS} {output}"


rule prep_plot_data_ps_vs_infected_ba_seir:
    input:
        BA_CONT_SEIR_PLOT_PS_INFECTED_FILE_LIST,
    output:
        BA_CONT_SEIR_PLOT_PS_INFECTED_DATA,
    params:
        filelist=temp(make_file_list(BA_CONT_SEIR_PLOT_PS_INFECTED_FILE_LIST)),
    shell:
        "python3 workflow/calc_ps_vs_infected.py {params.filelist} {output}"


rule plot_fig_degree_dist:
    input:
        ba_deg_dist_file=BA_CONT_SEIR_DEG_DIST,
        intb_deg_dist_file=INTB_CONT_SEIR_DEG_DIST,
        dtu_deg_dist_file=DTU_CONT_DEG_DIST.format(resol=12),
    output:
        fig=j(FIG_DIR, "deg-ccdf-new.pdf"),
        data=j(FIG_DIR, "deg-ccdf-new.csv"),
    shell:
        "papermill workflow/plot_fig_degree_dist.ipynb -r ba_deg_dist_file {input.ba_deg_dist_file} -r intb_deg_dist_file {input.intb_deg_dist_file} -r dtu_deg_dist_file {input.dtu_deg_dist_file} -r outputfile {output.fig} -r outputfile_data {output.data} $(mktemp)"


rule plot_fig_sim_result:
    input:
        int_time=BA_CONT_SEIR_PLOT_TIME_INFECTED_DATA,
        int_ps=BA_CONT_SEIR_PLOT_PS_INFECTED_DATA,
        intb_time=INTB_CONT_SEIR_PLOT_TIME_INFECTED_DATA,
        intb_ps=INTB_CONT_SEIR_PLOT_PS_INFECTED_DATA,
    output:
        fig=j(FIG_DIR, "sim_results.pdf"),
        data=j(FIG_DIR, "sim_results.csv"),
    shell:
        "papermill workflow/plot-sim-result.ipynb -r int_time {input.int_time} -r int_ps {input.int_ps} -r intb_time {input.intb_time} -r intb_ps {input.intb_ps} -r outputfile {output.fig} -r outputfile_data {output.data} $(mktemp)"


rule plot_fig_sim_dtu_result:
    input:
        dtu_time=DTU_CONT_PLOT_TIME_INFECTED_DATA,
        dtu_ps=DTU_CONT_PLOT_PS_INFECTED_DATA,
    output:
        fig=j(FIG_DIR, "sim_dtu_results.pdf"),
        data=j(FIG_DIR, "sim_dtu_results.csv"),
    shell:
        "papermill workflow/plot-sim-dtu-result.ipynb -r dtu_time {input.dtu_time} -r dtu_ps {input.dtu_ps} -r outputfile {output.fig} -r outputfile_data {output.data} $(mktemp)"
