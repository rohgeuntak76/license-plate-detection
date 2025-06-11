from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import os
import argparse
import time

def update_token(trainer):
    with open('/etc/secrets/ezua/.auth_token','r') as file:
        AUTH_TOKEN = file.read()
        os.environ['MLFLOW_TRACKING_TOKEN']=AUTH_TOKEN
        print("Token successfully synced!!")

def main(epochs,batch):
    # Backbone Model
    model = YOLO("yolo11s")
    model.add_callback("on_model_save",update_token)
    model.callbacks["on_model_save"]
    
    # Datasets
    datasets = "./datasets/data.yaml"
    
    # MLFLOW configurations
    run_name = model.model_name.name.replace('.pt','-') +time.strftime("%Y%m%d-%H%M%S", time.localtime())
    os.environ['MLFLOW_RUN'] = run_name
    os.environ['MLFLOW_TRACKING_URI'] = "https://mlflow.ingress.pcai0305.sg2.hpecolo.net"
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://local-s3-service.ezdata-system.svc.cluster.local:30000"
    os.environ['MLFLOW_EXPERIMENT_NAME'] = 'license_plate_yolo11s'

    # Set the Token
    update_token('dummy')

    # Train
    results = model.train(data=datasets, epochs=epochs, batch=batch)  # train the model
    
    # Save the Overall Results as a plot. ( results.png )
    plot_results(results.save_dir.joinpath("results.csv"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO training CLI argument parser')
    
    # Add arguments
    parser.add_argument('--epochs', type=int, default=1, 
                       help='Number of epochs (default: 1)')
    parser.add_argument('--batch', type=float, default=16.0,
                       help='Batch size (default: 16)')

    # Parse arguments
    args = parser.parse_args()
    if args.batch > 1:
        args.batch = int(args.batch)

    main(args.epochs,args.batch)