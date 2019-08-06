from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

boston = load_boston()
# print(type(boston))

# print(boston.data.shape)
# print(boston.keys())
# print(boston.target)
# print(boston.target.shape)

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, test_size=0.2)

from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 모델
model1 = LinearRegression()
model2 = Ridge()
model3 = Lasso()

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)

linear_score = model1.score(x_test, y_test)
ridge_score = model2.score(x_test, y_test)
lasso_score = model3.score(x_test, y_test)

# 평가
print('linear_score: ', linear_score)
print('ridge_score: ', ridge_score)
print('lasso_score: ', lasso_score)

# y_pred = model1.predict(x_test)
# print(y_pred)
