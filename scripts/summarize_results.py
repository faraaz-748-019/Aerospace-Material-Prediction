import json

with open('models/ml/training_results.json') as f:
    d = json.load(f)

print('Best Model:', d.get('best_model'))
print('Feature Columns:', d.get('feature_cols'))
print('Target Columns:', d.get('target_cols'))
print('\n' + '='*95)
print(f"{'Model':<22} | {'MAE':<8} | {'RMSE':<8} | {'R2':<8} | {'Acc%':<8} | {'UTS_RMSE':<10} | {'YS_RMSE':<10}")
print('='*95)
for m, v in d.get('results', {}).items():
    print(f"{m:<22} | {v.get('MAE', 0):<8.4f} | {v.get('RMSE', 0):<8.4f} | {v.get('R2', 0):<8.4f} | {v.get('Accuracy%', 0):<8.2f} | {v.get('UTS_RMSE', 0):<10.4f} | {v.get('YS_RMSE', 0):<10.4f}")
print('='*95)

if 'results' in d:
    print("\nDetailed Per-Target Breakdown for Top Models:")
    for m in ['StackingEnsemble', 'ExtraTrees', 'XGBoost', 'PINN', 'CNN_LSTM_Hybrid', '1D_CNN', 'BiLSTM']:
        if m in d['results']:
            print(f"\n--- {m} ---")
            tm = d['results'][m].get('Target_Metrics', {})
            for tname, tmetrics in tm.items():
                print(f"  {tname:<22}: MAE={tmetrics.get('MAE',0):.4f}, RMSE={tmetrics.get('RMSE',0):.4f}, R2={tmetrics.get('R2',0):.4f}, Acc={tmetrics.get('Accuracy%',0):.2f}%")
