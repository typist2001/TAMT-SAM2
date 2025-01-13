CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/media/sda/typ/SAM2-UNet/check_pths_moe_fem_5/SAM2-UNet-125.pth" \
--test_image_path "/media/sda/typ/Dataset/RMAS/test/img/" \
--test_gt_path "/media/sda/typ/Dataset/RMAS/test/label/" \
--save_path "/media/sda/typ/SAM2-UNet/results/RMAS/"

#/media/sda/typ/Dataset/MSD/test
#/media/sda/typ/Dataset/Polyp/TestDataset/TestDataset
#/media/sda/typ/Dataset/MSD/test
#/media/sda/typ/Dataset/MAS3K/test
#/media/sda/typ/Dataset/SOD/ECSSD/
#/media/sda/typ/Dataset/SBU-shadow/SBU-shadow/SBU-Test/ShadowImages/
#185 110 125