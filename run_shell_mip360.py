import os


# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['bonsai', 'bicycle']): #, 'kitchen' 'garden'
#         mask_lr_final = 0.0005 * lmbda / 0.001
#         mask_lr_final = min(mask_lr_final, 0.0015)
#         one_cmd = (
#             f'python train_split.py -s C:/Users/Nicholas/Desktop/3DGSDATASETS/360_v2/{scene} '
#             f'--eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 1_600 --test_iterations 1_600 '
#             f'-m outputs/mipnerf360/{scene}/{lmbda} '
#             f'--lmbda {lmbda} --mask_lr_final {mask_lr_final} '
#             f'--memMB 80 --segIter 1590 --segReq 0.1 --segSpread 0.05'
#         )   
#     os.system(one_cmd)

for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['bonsai', 'bicycle']): #, 'garden', 'kitchen'
        mask_lr_final = 0.0005 * lmbda / 0.001
        mask_lr_final = min(mask_lr_final, 0.0015)
        one_cmd = (
            f'python train.py -s C:/Users/Nicholas/Desktop/3DGSDATASETS/360_v2/{scene} '
            f'--eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 7_000 --test_iterations 7_000 '
            f'-m outputs/mipnerf360/{scene}/{lmbda} '
            f'--lmbda {lmbda} --mask_lr_final {mask_lr_final} '
            f'--memMB 80 --segIter 1590 --segReq 0.1 --segSpread 0.05'
        )   
        os.system(one_cmd)