REM python validate.py --model_file D:/dlwater/train_data/wat_nj_rgb/out/trained_models/cmgfnet/CMGFNet_best.pt --data_dir D:/dlwater/train_data/wat_nj_rgb/val/images --arct CMGFNet --conf 0.75 --img_sz 256 --plot 1

python validate.py --model_file D:/dlwater/train_data/wat_nj_rgb/out/trained_models/fusenet/FuseNet_best.pt --data_dir D:/dlwater/train_data/wat_nj_rgb/val/images --arct FuseNet --conf 0.5 --img_sz 256 --plot 1
 
python validate.py --model_file D:/dlwater/train_data/wat_nj_rgb/out/trained_models/gfbunet/GFBUNet_best.pt --data_dir D:/dlwater/train_data/wat_nj_rgb/val/images --arct GFBUNet --conf 0.5 --img_sz 256 --plot 1

python validate.py --model_file D:/dlwater/train_data/wat_nj_rgb/out/trained_models/gfunet/GFUNet_best.pt --data_dir D:/dlwater/train_data/wat_nj_rgb/val/images --arct GFUNet --conf 0.5 --img_sz 256 --plot 1

python validate.py --model_file D:/dlwater/train_data/wat_nj_rgb/out/mfsegnet/MFSegNet_best.pt --data_dir D:/dlwater/train_data/wat_nj_rgb/val/images --arct MFSegNet --conf 0.5 --img_sz 256 --plot 1

