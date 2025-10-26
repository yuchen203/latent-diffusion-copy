from cleanfid import fid

score = fid.compute_fid(
    "./dataset/celebahq256_imgs",
    "./outputs/sampling/celeba256/samples/00490000/2025-10-25-15-34-14/img",
    num_workers=4,
    batch_size=50,
    mode="clean", 
    #model_name="clip_vit_b_32"
)
print(f"FID: {score}")