import json
import subprocess

scale_sets = [
    (1.25, 0.875),
    (1.50, 0.75),
    (1.75, 0.625),
    (2.00, 0.50),
]

entropy_sets = [
    (4, 1),
    (2, 1),
    (1, 2),
    (1, 4),
]

def run_job(scene, lmbda, voxel_size, mask_lr_final):
    for gs, ps in scale_sets:
        grad_json  = json.dumps({"0": gs, "1": ps})
        prune_json = json.dumps({"0": gs, "1": ps})
        for ew0, ew1 in entropy_sets:
            entropy_json = json.dumps({"0": ew0, "1": ew1})

            args = [
                "python", "train.py",
                "-s", f"../3DGSDATASETS/360_v2/{scene}",
                "--eval",
                "--lod", "0",
                "--voxel_size", str(voxel_size),
                "--update_init_factor", "16",
                "--iterations", "30_000",
                "--test_iterations", "30_000",
                "-m", f"outputs/{scene}/{lmbda}",
                "--lmbda", str(lmbda),
                "--mask_lr_final", str(mask_lr_final),
                "--segIter", "1590",
                "--segReq", "0.1",
                "--segSpread", "0.05",
                "--grad_scale_by_obj", grad_json,
                "--prune_scale_by_obj", prune_json,
                "--entropy_bias_by_obj", entropy_json,  # new entropy sampling weights
            ]

            print("Launching:", " ".join(args))
            # Use check=True so you'll see immediately if argparse rejects something
            subprocess.run(args, check=True)

# Set 1
for lmbda in [0.004]:  # optionally: 0.003, 0.002, 0.001, 0.0005
    for scene in ['bonsai', 'bicycle', 'kitchen', 'garden']:
        mask_lr_final = min(0.0005 * lmbda / 0.001, 0.0015)
        run_job(scene, lmbda, voxel_size=0.001, mask_lr_final=mask_lr_final)

# Set 2
for lmbda in [0.004]:
    for scene in ['Truck', 'Train', 'Ignatius', 'Caterpillar', 'Horse', 'Family', 'Francis']:
        mask_lr_final = 0.0001 * lmbda / 0.001
        run_job(scene, lmbda, voxel_size=0.01, mask_lr_final=mask_lr_final)
