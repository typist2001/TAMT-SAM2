CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "/media/sda/typ/SAM2-UNet/hiera_path/sam2_hiera_large.pt" \
--train_image_path "/media/sda/typ/Dataset/CoTask/Train/Imgs/" \
--train_mask_path "/media/sda/typ/Dataset/CoTask/Train/GT/" \
--save_path "/media/sda/typ/SAM2-UNet/check_pths_moe_fem_5_moe/" \
--epoch 200 \
--lr 0.001 \
--batch_size 10