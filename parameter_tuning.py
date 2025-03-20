import itertools
from main import arg_parse, main

hyperparameter_grid = {
    'lr': [0.0001],  # Learning rates to test
    'batch_size': [32,128,512,1000, 2000, 4000],    # Batch sizes to test
    'hidden_dim': [128, 256, 512],       # Hidden dimensions to test
    'num_gc_layers': [2, 3, 4],         # Number of graph convolution layers to test
    'DS' : ['Tox21_HSE'], # Dataset to test
    'feature' : ['default'], # Feature to test options are 'default' for using node attributes or 'deg-num'
    'num_epochs' : [10000], # Number of epochs to train for Early stopping and Reduce on LR is implemented
}

def override_args(args, **kwargs):
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args

def hyperparameter_tuning():
    results = []
    default_args = arg_parse()  

    keys, values = zip(*hyperparameter_grid.items())
    for combination in itertools.product(*values):
        hyperparams = dict(zip(keys, combination))
        
        args = override_args(default_args, **hyperparams)
        
        print(f"Testing Hyperparameters: {hyperparams}\n")
        
        roc_auc, pr_auc = main(args)
        
        results.append({
            'hyperparameters': hyperparams,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
        })
    
    print("Hyperparameter Tuning Results:\n")
    for result in results:
        print(result)

if __name__ == "__main__":
    hyperparameter_tuning()