#Импортируем библиотеки, и сокращаем к ней обращение
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, GammaRegressor


pd.set_option('display.max_columns', None) #Убираем ограничение с колонок, при выводе БД
rd = pd.read_csv('housing.csv', delimiter=',') #Открываем БД, и ставим разделитель, что бы можно было вывести БД
db = pd.DataFrame(rd) #Переменная, при обращении к которой, будет выводится БД

print(db.describe()) #Вывод информации по БД, для того что бы найти пустые значения если они есть.
db.fillna(0) #Заменяем пусты значения на 0
db.sort_values('total_rooms') #Сортировка таблицы по колонке 'total_rooms'
print(db.describe()) #Перепроверяем, убрались ли пустые значения

#Выводим график БД
plt.scatter(db['latitude'], db['longitude']) #Указываем тип графика, и выбираем колонки из БД
#plt.xlabel('Ширина') #Указываем название оси X
#plt.ylabel('Длинна') #Указываем название оси Y
plt.show() #Выводим точечный график - (scatter)

#Выводим графики регрессии
model = LinearRegression() #Инициализация метода регрессии
x = rd.iloc[:, 1].values.reshape(-1, 1) #Присваивание значений к переменной X из колонки с индексом 1
y = rd.iloc[:, 0].values.reshape(-1, 1) #Присваивание значений к переменной Y из колонки с индексом 1
model.fit(x,y) #Выполняем подгонку линейной регрессии
y_predict = model.predict(x) #Создаем переменную, которая будет равна предсказанному 'y'

plt.scatter(x,y) #Выбираем тип графика, и берем значения x, y
plt.plot(x,y_predict, color='red') #Создаем линию на графике, по которой будет проводится регрессионный анализ
plt.show() #Выводим линейную регрессиюы

x = rd.iloc[:, 5].values.reshape(-1, 1)
y = rd.iloc[:, 6].values.reshape(-1, 1)
model.fit(x,y)
y_predict = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_predict, color='purple')
plt.show()

model2 = GammaRegressor()
x = rd.iloc[:, 0].values.reshape(-1, 1)
y = rd.iloc[:, 1].values.reshape(-1, 1)
model2.fit(x,y)
y_predict = model2.predict(x)

plt.scatter(x, y)
plt.plot(x, y, color="yellow")
plt.plot(x, y_predict, color="red")
plt.show()
