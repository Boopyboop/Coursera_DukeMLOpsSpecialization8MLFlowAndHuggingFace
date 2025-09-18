import mlflow
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="MLflow run ID")
    parser.add_argument("--model_name", required=True, help="Registered model name")
    args = parser.parse_args()

    mlflow.register_model(f"runs:/{args.run_id}/model", args.model_name)
    print(f"✅ Registered model {args.model_name} from run {args.run_id}")
