from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster
from run_MULTI_VANILLA import train


if __name__ == '__main__':
    # Set up our argparser and make the y_val tunable.
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=32, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--dataset_list", type=str, default="SGD,TM19,TM20,MWOZ", help="Path for saving")
    parser.add_argument("--setting", type=str, default="single", help="Path for saving")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="lenght of the generation")
    parser.add_argument("--max_history", type=int, default=5, help="max number of turns in the dialogue")
    parser.add_argument("--percentage_LAML", type=float, default=0.2, help="LAML percentage of augmented data used")
    parser.add_argument("--continual", action='store_true', help="continual baseline")
    parser.add_argument("--multi", action='store_true', help="multitask baseline")
    parser.add_argument("--debug", action='store_true', help="continual baseline")
    parser.add_argument("--superposition", action='store_true', help="multitask baseline")
    # parser.add_argument("--reg", type=float, default=0.01, help="CL regularization term")
    # parser.add_argument("--episodic_mem_size", type=int, default=20, help="number of batch put in the episodic memory")
    # parser.add_argument("--CL", type=str, default="REPLAY", help="CL strategy used")
    # parser.add_argument("--task_type", type=str, default="E2E", help="Select the kind of task to run")

    parser.add_argument('--test_tube_exp_name', default='my_test')
    parser.add_argument('--log_path', default='.')

    parser.opt_list('--task_type', type=str, default="DST", options=["DST"], tunable=True)
    # parser.opt_list('--task_type', type=str, default="E2E", options=["E2E","DST","NLG","INTENT"], tunable=True)
    # parser.opt_list('--CL', type=str, default="EWC", options=["EWC"], tunable=True)
    parser.opt_list('--CL', type=str, default="EWC", options=["EWC","L2"], tunable=True)
    parser.opt_list('--episodic_mem_size', default=1, type=int, options=[100], tunable=True)
    parser.opt_list('--reg', default=0.1, type=float, options=[0.01], tunable=True)
    parser.opt_list('--seed', default=1, type=int, options=[1,2,3,4,5], tunable=True)


    hyperparams = parser.parse_args()


    # # Enable cluster training.
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.log_path,
        python_cmd='python'
    )

    # Email results if your hpc supports it.
    cluster.notify_job_status(
        email='andreamad8@fb.com', on_done=False, on_fail=True)

    # SLURM Module to load.
    cluster.load_modules([
        'cuda/10.1',
        'cudnn/v7.6.5.32-cuda.10.1',
        'anaconda3'
    ])

    # # Add commands to the non-SLURM portion.
    cluster.add_command('source activate CLTOD')

    # Add custom SLURM commands which show up as:
    # #comment
    # #SBATCH --cmd=value
    # ############
    cluster.add_slurm_cmd(cmd='cpus-per-task', value='10', comment='CPUS per task.')
    cluster.add_slurm_cmd(cmd='mem', value='0', comment='Memory')
    cluster.add_slurm_cmd(cmd='constraint', value='volta32gb', comment='GPU type per task.')
    cluster.add_slurm_cmd(cmd='time', value='72:0:0', comment='GPU type per task.')

    # Set job compute details (this will apply PER set of hyperparameters.)
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1

    # Each hyperparameter combination will use 8 gpus.
    cluster.optimize_parallel_cluster_gpu(
        # Function to execute:
        train,
        nb_trials=10,
        # This is what will display in the slurm queue:
        job_name='CL_EWCL2')