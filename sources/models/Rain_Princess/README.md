#### Конвертация модели

`python <Директория с openvino>\deployment_tools\model_optimizer\mo.py --input_model <Путь к репозиторию>\sources\models\Rain_Princess\Rain_Princess.onnx`

#### Запуск модели

`python <Путь к репозиторию>\sources\models\Rain_Princess\Rain_Princess.py --model <Путь к репозиторию>\sources\models\Rain_Princess\Rain_Princess.xml --weights <Путь к репозиторию>\sources\models\Rain_Princess\Rain_Princess.bin --image <Путь к изображению>`

#### Пример работы модели

![Исходное изображение](img_before.jpg)
![Полученное изображение](img_after.jpg)