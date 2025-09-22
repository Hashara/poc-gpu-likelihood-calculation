nattempts=10
execution_file=""
queue="normal"
time="00:02:00"
ncpus=1
mem="4GB"
test_type="iqtree"
working_dir=$(pwd)


for ((attempt=1; attempt <= $nattempts; attempt +=1 ));
do
    echo "Submitting attempt $attempt"
    qsub -q$queue -Pdx61 -lwalltime=$time,ncpus=$ncpus,mem=$mem,jobfs=20GB,storage=scratch/dx61,wd -N $test_type.attempt -vARG1=$attempt,ARG2=${working_dir} ${working_dir}/test_script_iqtree.sh
done