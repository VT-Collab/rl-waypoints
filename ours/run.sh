runs=(2 3 4 5)
for run in ${runs[@]}; do
    python3 oat_random.py --env PickPlace --object can --num_wp 3 --run_num t$run
    wait
done