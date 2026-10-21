#!/bin/bash


#SBATCH --qos=regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -q shared
#SBATCH -J CCElibrary
#SBATCH -A m2676
#SBATCH -t 24:00:00
#SBATCH --mail-user=rita.feriozzi@gssi.it
#SBATCH --mail-type=ALL
#SBATCH --image=legendexp/legend-software:latest
#SBATCH --output=slurm_output/CCElibrary_%A.out
#SBATCH --error=slurm_output/CCElibrary_%A.err

date


WORKING_DIR="/global/u2/r/ritaferi/CTM/CCE/tau(x)"

LOGDIR="${WORKING_DIR}/logs"
OUTDIR="${WORKING_DIR}/plot"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"
mkdir -p "${WORKING_DIR}/slurm_output"


cd "${WORKING_DIR}"


# Numero di cariche simulate
N_charges=10000


echo "Charges generated:"
echo "${N_charges}"


CODE="${WORKING_DIR}/job_CCE_lab.py"


LOGFILE="${LOGDIR}/CCElibrary.out"


echo "Running:"
echo "shifter --image=docker:legendexp/legend-software:latest python3 ${CODE} --N_charges ${N_charges}"


shifter --image=docker:legendexp/legend-software:latest \
python3 "${CODE}" \
--N_charges "${N_charges}" \
&> "${LOGFILE}"


echo "Finished"
date