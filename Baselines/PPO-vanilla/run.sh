# runs=(1 2 3 4 5)
# for run in ${runs[@]}; do
#     python3 eval_door.py --env Door --num_wp 2 --run_num t$run 
#     wait
# done

# python3 eval.py --env Lift --num_eval 100 --run_num test --num_wp 2
# wait
# python3 eval.py --env Stack --num_eval 100 --run_num test --num_wp 2
# wait
# python3 eval.py --env NutAssembly --num_eval 100 --run_num test --num_wp 2
# wait
# python3 eval.py --env PickPlace --object bread --num_eval 100 --run_num test --num_wp 2
# wait
# python3 eval.py --env PickPlace --object milk --num_eval 100 --run_num test --num_wp 2
# wait
# python3 eval.py --env PickPlace --object can --num_eval 100 --run_num test --num_wp 2
python3 main_door.py --env Door --num_episodes 599 --run_num without_latch_test --num_wp 2
wait
python3 main_door.py --env Door --num_episodes 599 --run_num with_latch_test --use_latch --num_wp 2
wait
python3 eval_door.py --env Door --num_eval 100 --run_num without_latch_test --num_wp 2
wait
python3 eval_door.py --env Door --num_eval 100 --use_latch --run_num with_latch_test --num_wp 2