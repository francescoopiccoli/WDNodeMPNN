# [TODO-F] Fix this file to run the supervised training for 10 iterations and save the results in the RUNS folder
 
# Create Models Weights and Figures Directories and clear them
mkdir Models_Weights
rm -rf Models_Weights/*
mkdir Figures
rm -rf Figures/*


# Now make the RUNS directory in which we perform save all the results from the runs
mkdir RUNS
cd RUNS


echo "!!! - STARTING SUPERVISED RUNS - !!!"
for i in {1..10}
do 
    # make a directory for that run and a weights and figures directory to which to copy the results from the run
    mkdir RUN_$i
    cd RUN_$i
    mkdir Models_Weights
    mkdir Figures

    # Run the supervised train python file
    cd ../../src
    echo "RUNNING ITERATION $i"
    echo "_____________________"
    echo "Starting Supervised"
    python training.py
    
    # copy the model weights and figures to the folder in run
    cp -r Models_Weights/* RUNS/RUN_$i/Models_Weights
    cp -r Figures/* RUNS/RUN_$i/Figures

    cd RUNS
done

echo 'DONE!!'
