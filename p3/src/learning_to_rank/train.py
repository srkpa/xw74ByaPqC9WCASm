from src.learning_to_rank.utils import *


def main(file_path: Path, debug: bool, embbeder_config: EmbedderConfig):
    X, y, qIds = get_features(
        file_path=file_path, debug=debug, embbeder_config=embbeder_config
    )
    print()
    most_recent_model = find_latest_xgboost_model(MODELS_DIR)
    fitted_model = train(
        X=X, y=y, qIDs=qIds, xgb_model=most_recent_model
    )  # incremental learning
    params = fitted_model.get_xgb_params()
    non_null_params = list(filter(lambda x: params[x] is not None, params))
    print("Model fitted with : ")
    for p in non_null_params:
        print(f"\t{p} : {params[p]}")
    print()

    score = evaluate(fitted_model, X, y)
    print(f"Result: {params['objective']} -> {score}\n")

    save_filepath = save(fitted_model)
    print(f"Model saved at : {save_filepath}")
