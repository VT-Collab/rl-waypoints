runs=(1 2 3 4 5)
for run in ${runs[@]}; do
    python3 main_door.py --env Door --num_wp 2 --run_num t$run
    wait
done
