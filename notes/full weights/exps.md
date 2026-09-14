# python launch.py --cfg experiment/omega260526/first_frame.yaml
python launch.py --cfg default_24.yaml

python process_kaggle.py --add-node --cfg ocfgs/first_cam.yaml --run


python process_kaggle.py --add-node --cfg ocfgs/pts_to_gt/pts_to_gt.yaml --run