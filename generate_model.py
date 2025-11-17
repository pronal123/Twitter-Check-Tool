import joblib
from sklearn.linear_model import LinearRegression

# 1. Create a simple model to represent the trained ML algorithm
# 実際のシナリオでは、このモデルは過去のBTC市場データで訓練されます。
model = LinearRegression()

# 2. Save the model to a joblib file
# このファイル (ml_model.joblib) がプロジェクトのルートディレクトリに存在する必要があります。
filename = 'ml_model.joblib'
joblib.dump(model, filename)

print(f"実戦用モデルファイル '{filename}' を作成しました。")
print("このファイルがプロジェクトのルートディレクトリに存在することを確認してデプロイしてください。")
