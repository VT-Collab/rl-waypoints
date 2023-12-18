runs=(1 2 3 4 5)
for run in ${runs[@]}; do
    python3 oat_stack_random.py --run_num oat_stack_random$run --num_wp 5
    # python3 main_lift.py --run_num test_1init_$run --n_inits 1 --num_wp 3
    wait
done

# runs=(1 2 3 4 5)
# for run in ${runs[@]}; do
#     python main_stack.py --run_num $run --n_inits 1 --num_wp 5
#     wait

# runs=(1 2 3 4 5)
# for run in ${runs[@]}; do
#     python main_door.py --run_num $run --n_inits 1 --num_wp 2
#     wait