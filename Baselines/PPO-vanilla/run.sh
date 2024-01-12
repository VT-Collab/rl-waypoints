runs=(1 2 3 4 5)
for run in ${runs[@]}; do
    python3 main.py --env Stack --num_wp 2 --run_num t$run 
    wait
done
